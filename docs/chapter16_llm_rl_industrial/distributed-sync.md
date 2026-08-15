# 18.4 分布式 RL 训练

上一节我们说了训练系统必须保证采样策略、模型版本和数值计算一致——那是在你已经知道要查什么的前提下。现在把一次训练真正分到多张 GPU 上，你会看到最朴素的问题：数据怎么流？

假设一组 rollout GPU 生成了一批回答。奖励进程读取回答、调用验证器或奖励模型算分，训练 GPU 再用这些回答算梯度、更新 Actor。更新完了，新参数必须立刻同步回 rollout GPU，否则下一批回答还是从旧策略生成的，训练就乱了：

```text
Rollout GPU 生成回答
        ↓
奖励进程计算分数
        ↓
训练 GPU 更新 Actor
        ↓
把新参数同步给 Rollout GPU
        └──────────────► 下一批回答
```

单机代码里这就是四个先后调用的函数，到多机系统中变成了不同进程之间的数据传输和等待。系统设计主要解决三个问题：

1. **模型放在哪里。** Actor、Reference、Reward Model 和 rollout 引擎加起来几十上百亿参数，可能根本塞不进一组 GPU。
2. **谁在等待谁。** 生成一条回答要几秒钟，参数更新只要几百毫秒，慢的那一步会让其他所有设备闲置。
3. **何时交换数据与参数。** 同步太频繁通信开销太大，间隔太久 rollout 又会用过时策略生成回答。

---

## 分布式 RL 需要做的三项系统决策

- **训练与生成是否共享 GPU**：共享部署节省 GPU；分离部署减少切换并提高并行吞吐。显存不足时先解决模型切分；GPU 大量等待时再考虑生成与训练分离。
- **是否等待整批 rollout 完成**：同步训练保持数据较新；异步训练减少等待但会产生陈旧经验。数学和代码题的生成时长较接近，通常先用同步方案；工具调用、浏览器操作和长时间环境交互耗时差别大，更容易从异步方案受益。
- **一个模型怎样分到多张 GPU**：FSDP、张量并行、流水线并行分摊显存，同时增加通信与调度复杂度。模型采用 MoE 时，还要处理 token 在专家之间的路由与通信。

::: tip 第一次阅读到这里即可
记住数据顺序：**生成 → 奖励 → 更新 → 同步参数**。veRL、slime 和 OpenRLHF 的实现不同，都在安排这四步使用哪些 GPU、何时交换数据。
:::

下面继续沿着这批回答的路径展开。首先确认生成端和训练端计算的是同一个策略；然后用框架安排各个模型；生成速度不足时优化 rollout，显存不足时切分训练状态；任务耗时差别变大后再引入异步；模型换成 MoE 后，还要处理专家路由和通信。这个顺序也是排查多机训练问题的顺序。

---

## 训推一致性：生成端和训练端必须使用同一策略

参数同步解决了模型版本问题。rollout GPU 和训练 GPU 拿到同一组权重后，还要经过各自的计算引擎才能得到 token 概率。推理侧通常使用 vLLM 或 SGLang，并采用 KV Cache 和低精度计算；训练侧常用 FSDP 或 Megatron，并保留反向传播所需的计算图。两条计算路径不同，最终得到的概率也可能不同。

把 rollout 引擎实际执行的策略记为 $\pi_{\text{rollout}}$，把训练端记录的旧策略记为 $\pi_{\text{old}}$。理想情况下二者应当相同；出现模型版本滞后、浮点精度差异、MoE 路由差异或 log-probability 重算误差时，二者就会产生偏差。这就是训推不一致（Training-Inference Mismatch）。

**生成策略与训练策略的偏差来源**：

- rollout 侧通常使用 vLLM 或 SGLang，以 FP8/BF16 生成回答，并启用 KV Cache 优化。
- 训练侧通常使用 FSDP 或 Megatron，以 BF16/FP32 计算 log-probability 和梯度，并可能启用激活重计算。

模型权重相同，只能保证两边从同一组参数出发。计算精度、算子实现和 MoE 专家路由仍会改变 token 的 log-probability。高概率 token 的微小误差通常影响有限；低概率 token 数量多，误差累积后可能明显改变梯度估计。

**PPO Clipping 为什么无法修正训推偏差**：PPO 使用下面的重要性采样比率限制一次更新的幅度：

$$
\mathcal{L}^{\text{CLIP}} = \mathbb{E}\left[\min\left( r_t(\theta) \hat{A}_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right)\right],
$$

其中

$$
r_t(\theta) = \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\text{old}}(a_t\mid s_t)}.
$$

$\hat A_t$ 表示动作 $a_t$ 比当前平均水平好多少，$\epsilon$ 决定允许比率偏离 1 的范围。假设旧策略生成某个 token 的概率是 $0.20$，新策略把它提高到 $0.24$，那么 $r_t=0.24/0.20=1.2$。当 $\epsilon=0.2$ 时，这次变化刚好到达 clipping 的上边界。

这个计算有一个前提：分母 $\pi_{\text{old}}$ 必须是生成动作时真正使用的策略。如果回答来自 $\pi_{\text{rollout}}$，训练端却用另一条计算路径重算 $\pi_{\text{old}}$，那么 $r_t$ 在更新开始以前就已经不准确。clipping 只能限制参数更新，不能修正两个引擎算出的概率差异。

**训推偏差的排查顺序**：

1. 核对 rollout 使用的模型版本，确认参数同步已经完成。
2. 用同一批 token 分别记录 rollout 侧和训练侧的 log-probability，观察误差集中在哪些位置。
3. 对齐两侧的浮点精度和算子实现，再比较误差是否缩小。
4. 对 MoE 模型额外记录专家路由，确认训练端是否复现了生成时的路由。

**训推偏差的修正方法**：

- **统一计算精度**：先用 FP16/BF16 替代 rollout 侧的 FP8，判断低精度计算是否是主要误差来源。需要继续使用 FP8 时，应同时加入偏差监控和重要性采样修正。
- **记录真实行为策略**：直接保存 rollout 时的 log-probability，避免训练端把重新计算的结果当作真实行为概率。
- **重新计算并校验**：训练前用训练引擎重算 log-probability，并与 rollout 记录逐 token 对比。重算本身不能恢复真实行为策略，但能暴露差异的位置和大小。
- **限制极端比率**：Truncated IS（TIS）等方法会截断过大的重要性采样比率，降低少量异常 token 对梯度的影响。
- **处理长尾 token**：动态词表剪枝等方法会过滤偏差最大的低概率区域，减少误差在长序列中的累积。
- **回放 MoE 路由**：R3（Rollout Routing Replay）在训练时复现 rollout 的专家选择，减少路由变化造成的概率偏差。

工程中的 On-policy 程度取决于 $\pi_{\text{rollout}}$ 与当前训练策略之间的距离。参数同步控制模型版本，精度对齐、概率记录和重要性采样修正继续控制计算路径带来的偏差。

---

## 模型与 GPU 的资源安排

**veRL 的 HybridFlow：五类角色的编排**：训推一致性解决了"数据能不能正确更新模型"。下一步是让 Actor、Critic、Reference Model、Reward Model 和 rollout 引擎在不同 GPU 上协作。veRL（Volcano Engine Reinforcement Learning）把这个问题拆成算法主循环、模型计算和资源分配三个层级。

HybridFlow 的核心设计是把 RLHF/GRPO/PPO 训练抽象成 single-controller 多模型编排：Driver 作为单控制器运行算法主循环和资源调度，下面分设 Actor Worker（FSDP 训练）、Critic Worker（FSDP 训练）、Reference Worker（冻结模型）、Reward Model Worker 和 Rollout Engine Worker（vLLM/SGLang 推理），它们共享同一个 ResourcePool（GPU 集合）。

### 三个核心抽象

**ResourcePool：GPU 资源分组**——把 GPU 分组，每组可以放一个或多个模型。不同模型可以共享 GPU（colocate）或独占 GPU（disaggregated）。例如 Actor 和 Rollout 可以共享同一组 GPU，也可以分开设池。

**Worker：模型实例封装**——每个 Worker 是一个独立的模型实例，封装了具体的训练/推理逻辑。ActorWorker 负责 loss 计算、反向传播和优化器更新；RolloutWorker 负责批量生成和权重同步。

**Driver：单控制器编排**——Driver 是 RL 算法的主循环，顺序执行：同步权重给 Rollout → 采样 responses → 用 Reward Model 算分 → 用 Critic 算 value → 算 advantage 更新 Actor → 更新 Critic。

**HybridFlow 的混合并行策略**：这里的 Hybrid 指统一的混合并行策略——同一个框架内可以组合 3D Parallelism（TP×PP×DP）、Colocate vs Disaggregated、多种训练后端（FSDP、Megatron、DeepSpeed ZeRO）和多种推理后端（vLLM、SGLang、HuggingFace generate）。

### 主流框架架构对比

| 框架 | 编排模式 | 训练后端 | 推理后端 | 典型规模 | 代表使用者 | 适用场景 |
|------|---------|---------|---------|---------|-----------|---------|
| [veRL (HybridFlow)](https://arxiv.org/abs/2409.19256) | Single-controller | FSDP、[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)、[DeepSpeed ZeRO](https://arxiv.org/abs/1910.02054) | [vLLM](https://arxiv.org/abs/2309.06180)、[SGLang](https://arxiv.org/abs/2312.07104)、HF generate | 8–1024 GPU | Qwen、DeepSeek、字节跳动 | 大规模生产训练、需要灵活资源组合 |
| [OpenRLHF](https://arxiv.org/abs/2405.11143) | Single-controller（Ray Actor 隔离） | FSDP、DeepSpeed | vLLM | 8–256 GPU | 社区、研究团队 | 研究实验、中等规模训练 |
| [NeMo-Aligner](https://github.com/NVIDIA/NeMo-Aligner) | Multi-controller | [Megatron](https://github.com/NVIDIA/Megatron-LM) | [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | 8–512 GPU | NVIDIA 生态、企业集群 | 已采用 NVIDIA NeMo 栈的生产环境 |
| [TRL](https://github.com/huggingface/trl) | Single-process | [HuggingFace Accelerate](https://huggingface.co/docs/accelerate) | HF generate | 1–8 GPU | 入门学习、快速原型 | 学习算法、小规模实验验证 |

选型可以从当前规模和已有技术栈出发：学习和原型用 TRL；研究和中等规模用 OpenRLHF 或 veRL；大规模生产用 veRL 或 NeMo-Aligner（看硬件栈）。

---

## 生成吞吐优化与显存控制

**Rollout 引擎的生成吞吐优化**：在许多 LLM RL 任务中，生成回答占用的时间长于一次参数更新。Rollout 引擎因此直接决定训练进程能否持续拿到新数据。

### vLLM 的核心优化技术

| 技术 | 解决的问题 | 核心原理 | 典型收益 | GRPO 中的重要性 |
|------|-----------|---------|---------|----------------|
| [PagedAttention](https://arxiv.org/abs/2309.06180) | KV cache 连续分配导致显存碎片、利用率低（50-70%） | 借鉴操作系统虚拟内存分页，把 KV cache 分成固定大小 block，按需分配与回收 | 显存利用率提升至 95%+，有效 batch size 提升 2–4 倍 | 基础优化，所有场景必需 |
| Continuous Batching | 传统 static batching 要等整批序列全部生成完才换，导致 GPU 空等 | 某条序列生成 EOS 后立刻在同一 iteration 填入新序列，实现迭代级动态调度 | 整体生成吞吐提升 5–10 倍 | 长回答场景收益最大 |
| Speculative Decoding | 自回归生成逐 token 解码，计算密度低 | 用小模型（draft model）先预测多个 token，大模型并行验证，accept 匹配的、reject 后重采样 | 典型 LLM 推理吞吐提升 2–3 倍 | 短回答、对延迟敏感场景 |
| Prefix Caching | 同一 prompt 重复生成时，前缀 KV 重复计算浪费 | 对 prompt 部分的 KV cache 做哈希复用，相同前缀直接命中缓存 | GRPO 中同一 prompt 生成 $G=8$ 条回答时，前缀计算节省 70–80% | **GRPO 核心优化**，veRL 默认启用 |

### SGLang 的生成与调度优化

[SGLang](https://arxiv.org/abs/2312.07104) 由 LMSYS 团队开发，在 agentic 场景下比 vLLM 更快：RadixAttention 用基数树管理 KV cache 支持跨请求复用，Programmatic Frontend 支持复杂的控制流（多轮调用、分支、循环），Constrained Decoding 内置 JSON、regex 约束生成。

工业实践中推理引擎选择：

| 引擎 | 核心优势 | 最适合场景 |
|------|---------|-----------|
| [vLLM](https://arxiv.org/abs/2309.06180) | PagedAttention、Continuous Batching 生态成熟 | 通用 rollout、单轮生成、GRPO 数学/代码训练 |
| [SGLang](https://arxiv.org/abs/2312.07104) | RadixAttention、多轮控制流、结构化输出 | Agentic rollout、多轮工具调用、需要 constrained decoding |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | NVIDIA GPU 深度优化、FP8 支持最好 | NVIDIA 硬件栈上的最高吞吐生产部署 |

### 多 GPU 显存分摊技术

生成速度解决以后，训练端仍要容纳权重、梯度、优化器状态和激活。一个 70B 模型的 BF16 全参数训练远超单张 80GB H100 的容量。

**训练显存的构成**：以常见的 BF16 权重与梯度、FP32 主权重和 Adam 一阶、二阶动量为例，每个参数大约需要 16 字节：

| 组件 | 数据类型 | 单参数字节数 | 70B 模型占用 | 说明 |
|------|---------|------------|-------------|------|
| 模型权重（Weight） | BF16/FP16 | 2 B | 140 GB | 训练时需要保留当前参数 |
| 梯度（Gradient） | BF16/FP16 | 2 B | 140 GB | 反向传播后累积 |
| 优化器主权重（Master Weight） | FP32 | 4 B | 280 GB | Adam 需要 FP32 副本做稳定更新 |
| Adam 一阶动量（m） | FP32 | 4 B | 280 GB | 梯度的指数移动平均 |
| Adam 二阶动量（v） | FP32 | 4 B | 280 GB | 梯度平方的指数移动平均 |
| 激活值（Activation） | BF16/FP16 | 动态 | ~100 GB | 与 batch size、序列长度正相关 |
| **合计（全参数训练）** | - | **16 B/param** | **~1.22 TB** | 远单单张 80GB H100 容量 |

**ZeRO：零冗余优化器级别对比**——[DeepSpeed ZeRO](https://arxiv.org/abs/1910.02054) 按分片内容分为三个级别：

| ZeRO 级别 | 分片 Optimizer State | 分片 Gradient | 分片 Weight | 单卡显存节省倍数 | 通信开销 | 适用场景 |
|----------|---------------------|--------------|------------|----------------|---------|---------|
| ZeRO-1 | ✅ | ❌ | ❌ | ~4× | 低 | 中小模型、通信受限场景 |
| ZeRO-2 | ✅ | ✅ | ❌ | ~8× | 中 | 大多数训练场景的默认选择 |
| ZeRO-3 | ✅ | ✅ | ✅ | $N$×（$N$=GPU数） | 高 | 大模型全参数训练，需要额外 all-gather |

ZeRO-3 把权重也切分到各 GPU，每个 GPU 只存 $1/N$ 的权重，但前向反向传播时需要通过 all-gather 临时聚合所需参数。

[FSDP（Fully Sharded Data Parallel）](https://pytorch.org/docs/stable/fsdp.html) 是 PyTorch 原生实现，等价于 ZeRO-3，与 PyTorch 生态兼容性更好，是 veRL 默认的训练后端。

**Gradient Checkpointing：梯度检查点**——不切分模型，而是用计算换显存：前向时不保存全部中间激活，反向传播时重新计算所需激活。激活显存从 $O(L)$ 降到 $O(\sqrt{L})$（$L$ 是 Transformer 层数），代价是训练速度降低 20–30%。

**显存优化方案组合**（以 70B 模型、单张 80GB H100 为例）：

| 方案组合 | 单卡显存需求 | 训练速度 | 可行性 |
|---------|------------|---------|-------|
| 全参数 + Adam（不做任何分片） | ~940 GB | 基准（最快） | ❌ 不可行，远超单卡容量 |
| ZeRO-3（仅分片训练状态） | ~118 GB | 比基准慢 10–15%（通信开销） | ❌ 单卡仍然 OOM |
| ZeRO-3 + Gradient Checkpointing | ~30 GB | 比基准慢 30–40% | ✅ 可行 |
| ZeRO-3 + Gradient Checkpointing + LoRA | ~8 GB | 比基准慢 ~40%（但参数量少） | ✅ 最快的工业方案 |

工业级 70B RL 训练通常使用 LoRA + FSDP 的组合，在显存、速度和训练效果之间取得平衡。

---

## 多机流水线的持续运行

**异步调度：减少流水线等待**：同步训练需要等一批 rollout 全部结束以后再更新，耗时较长的轨迹会让训练 GPU 等待。异步训练让生成和更新分别推进，再通过队列交换轨迹。

### 三种异步框架对比

| 框架 | 发布方 | 核心设计 | 陈旧度处理 | 典型规模 | 公开速度ups | 代表场景 |
|------|-------|---------|-----------|---------|-----------|---------|
| [LlamaRL](https://arxiv.org/abs/2505.24034) | Meta（2025） | 去中心化，无主节点；Rollout worker 持续取 prompt 生成，Train worker 持续取 batch 更新，权重异步广播 | 不做显式重要性修正，靠持续版本更新覆盖 | 4096+ GPU | Llama-3-70B GRPO 比同步快 **10.4×** | 超大规模同步瓶颈明显的推理任务 |
| [AReaL](https://arxiv.org/abs/2505.24298) | 蚂蚁集团+清华（2025） | 完全异步 rollout，每条轨迹记录生成时的策略版本和 logprob | 显式计算 token-level 重要性权重 $\exp(\text{current\_logprob} - \text{gen\_logprob})$，截断到 $[0.8, 1.2]$ | 1024 GPU | 671B MoE GRPO 比同步快 **2.77×** | MoE 模型、需要显式控制偏差的任务 |
| [AgentRL](https://arxiv.org/abs/2510.04206) | THUDM/智谱（2025） | 异步生成训练流水线 + 统一环境接口；训练侧分 rollout/Actor/Reference worker 池，环境侧通过 Controller/Task Worker 管理异构任务 | 异步队列 + 任务隔离，多轮环境会话单独维护 | 多机多环境 | 支撑 AutoGLM 训练 | 多轮 Agent（SWE、Computer Use、Deep Research） |

**LlamaRL（Meta，2025）**：去中心化设计，不设置统一的主节点，每个 worker 根据自己的角色持续取任务并提交结果。Rollout worker 从队列取 prompts 生成 responses 并推送到训练队列；Train worker 从 rollout 队列取 batch 更新并异步广播权重。特性是无单点故障、横向扩展容易、适合超大规模（10k+ GPU）。

**AReaL（Ant Group 和清华，2025）**：完全异步 rollout，并在 PPO 更新中显式处理策略陈旧度。Rollout worker 持续生成样本，training worker 拿到 batch 后立即更新；每条轨迹记录生成时的策略版本和 logprob，训练端计算重要性权重 $\exp(\text{current\_logprob} - \text{gen\_logprob})$ 并截断到 [0.8, 1.2] 避免旧样本造成过大梯度。

**AgentRL（THUDM/智谱，2025）**：把异步生成训练流水线与统一环境接口放在一起。训练侧使用 rollout、Actor 和 Reference 三类 worker 池；环境侧通过函数调用接口、容器、Controller 和 Task Worker 管理异构任务。支持 multi-turn、multi-task agentic RL，被用于构建 AutoGLM，适合 SWE-Agent、Computer Use 和 Deep Research Agent 等多轮环境任务。

---

### MoE 带来的额外系统复杂度

DeepSeek V3、Qwen3 和 GLM-4.5 都采用 MoE 架构。每个 token 只激活少量专家，参数能够分散到更多 GPU；与此同时，RL 的样本分布会改变专家负载，训练系统还要记录路由和通信状态。

以 DeepSeek V3 为例：Dense 部分（attention 等）约 20B 参数，MoE 部分有 256 个 expert × 5B 参数 = 1.28T，每条样本激活 8 个 expert（实际激活 40B），总参数 1.3T，激活参数 60B。

**MoE RL 的三项系统挑战**：

| 挑战 | 现象 | 解决方案 | 代表工作 |
|------|------|---------|---------|
| Expert 负载不均 | 某些 expert 被频繁激活（hot expert），其他 expert 闲置；部分 expert 训练不充分 | Expert Balancing Loss，鼓励激活频率接近均匀分布 $1/\text{num\_experts}$；动态路由调整 | [DeepSeek-V3](https://arxiv.org/abs/2412.19437)、[GShard](https://arxiv.org/abs/2006.16668) |
| 跨卡通信开销 | Expert 分布在多 GPU（Expert Parallelism），每条样本都需要 all-to-all 路由 token | 优化 all-to-all 通信 kernel，计算与通信重叠 | [DeepEP](https://github.com/deepseek-ai/DeepEP) |
| Token-level IS 方差过大 | MoE 路由差异导致 token 级重要性采样比率波动剧烈，梯度方差高 | 将 IS 比率从 token 级改为序列级（整个序列共享一个 ratio） | [GSPO](https://arxiv.org/abs/2507.18071)（Qwen3 全系采用） |

**减少流水线空闲时间**：模型切到多张 GPU 后，每张卡并不会自动保持忙碌。

| 技术 | 解决的问题 | 核心原理 | 收益 |
|------|-----------|---------|------|
| [DualPipe](https://arxiv.org/abs/2412.19437) | 传统流水线并行存在大量气泡（bubble），前向/反向无法重叠 | 双向流水线调度，让前向 stage N 和反向 stage N-1 在同一 GPU 上重叠执行 | 气泡比例从传统 $\frac{P-1}{M}$ 降到 $\frac{P-1}{2M}$（$P$=PP stage 数，$M$=micro-batch 数） |
| Best-Fit Packing | Micro-batch 大小不均导致部分 GPU 提前完成后空等 | 用装箱算法（bin packing）把不同大小的 micro-batch 分配到 GPU，平衡负载 | DeepSeek V3 中 GPU 利用率从 70% 提升到 95% |

### 性能瓶颈定位方法

前面的每项技术都可能把瓶颈推到下一处：生成加快以后，权重同步可能变慢；模型切分以后，跨卡通信可能占据主要时间；样本装箱改善以后，数据读取又可能跟不上。

**常用性能分析工具**：

| 工具 | 用途 | 能看到什么 | 适用阶段 |
|------|------|-----------|---------|
| [PyTorch Profiler](https://pytorch.org/docs/stable/profiler.html) | 通用 PyTorch 训练性能分析 | CPU/CUDA 活动时间线、显存占用、kernel 执行时间、top 耗时操作 | 训练阶段优化 |
| [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems) | 系统级 GPU 性能可视化 | 每个 CUDA kernel 执行时间、CPU-GPU 同步点、NCCL 通信开销、多流重叠 | 通信/调度瓶颈定位 |
| veRL 内置 Profiler | RL 训练流程时间分解 | Rollout 生成、Actor 更新、Critic 更新、权重同步、通信各占总时间比例 | **RL 训练首选**，直接定位 pipeline 瓶颈 |

**常见瓶颈与优化方向**：

| 瓶颈现象 | 判断标准（时间占比） | 优化方向 |
|---------|---------------------|---------|
| Rollout 生成慢 | rollout 占总时间 80%+ | 增加 rollout GPU 数量、启用 vLLM Prefix Caching、增大 batch size、考虑分离部署 |
| 权重同步慢 | weight sync 占 5%+ | 使用 LoRA（只同步 adapter 权重，而非完整模型）、NCCL 打包传输、减少同步频率 |
| 跨卡通信开销大 | all-reduce/all-gather 占 10%+ | 增大 micro-batch size、使用 gradient accumulation、优化并行策略切分 |
| 激活显存爆炸（OOM） | 训练中途 CUDA out of memory | 启用 Gradient Checkpointing、降低 max sequence length、减小 batch size |
| Expert 负载不均 | 部分 GPU 利用率 90%+、部分 30% | 开启 Expert Balancing Loss、调整 MoE 路由策略、使用 EP 负载均衡 |
| 慢人问题（Straggler） | batch 内最长序列决定整批完成时间 | 长度分桶（length bucketing）、Seer divided rollout（按预测生成长度分组）|

**MFU（Model FLOPs Utilization）**：用实际执行的浮点运算量除以硬件在同一时间内能够提供的峰值运算量：

$$\text{MFU} = \frac{\text{实际 FLOPs}}{\text{峰值 FLOPs} \times \text{时间}}$$

不同训练配置的典型 MFU 参考：

| 训练配置 | 典型 MFU 范围 | 瓶颈来源 |
|---------|--------------|---------|
| Dense 模型 + FSDP + Gradient Checkpointing（同步） | 35–45% | 激活重计算、跨卡 all-reduce |
| MoE 模型 + Expert Parallelism + DualPipe | 50–60% | Expert all-to-all 通信、负载不均 |
| 异步 RL（生成/训练分离部署，rollout 用 vLLM） | 训练端 40–50%，rollout 端 70–80% | 权重同步、队列等待 |

MFU 低于 30% 时，应结合时间分解继续检查通信、数据加载和 rollout 等待。

---

## 本节小结

- 多机 RL 仍然沿着生成、奖励、更新和参数同步四步运行。
- 模型切分解决显存问题，训练与生成的资源安排解决等待问题，权重同步保证 rollout 使用正确的策略版本。
- 训推不一致是分布式 RL 特有的隐患：即使权重相同，推理引擎和训练引擎的计算路径差异仍可能导致 log-probability 偏差。
- vLLM 的 PagedAttention、Continuous Batching 和 Prefix Caching 是生成吞吐优化的核心；FSDP/ZeRO 和 Gradient Checkpointing 解决训练显存问题。
- 同步训练更容易保证数据较新；异步训练减少等待，同时必须处理经验陈旧和重要性修正。
- MoE 在普通并行之外还要处理专家负载与路由通信；DualPipe 和 Best-Fit Packing 减少流水线空闲。

多机系统解决了算力怎样协同，训练仍然需要持续供应可执行、可验证的数据。[18.5 大规模 RL 数据工程](./data-engineering) 将沿着一条轨迹的生命周期，说明任务、环境、奖励和失败样本怎样进入下一轮训练。

## 延伸阅读

- [HybridFlow: A Flexible and Efficient RLHF Framework (veRL)](https://arxiv.org/abs/2409.19256)
- [OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework](https://arxiv.org/abs/2405.11143)
- [Efficient Memory Management for Large Language Model Serving with PagedAttention (vLLM)](https://arxiv.org/abs/2309.06180)
- [SGLang](https://arxiv.org/abs/2312.07104)
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- [LlamaRL: A Distributed Asynchronous Reinforcement Learning Framework](https://arxiv.org/abs/2505.24034)
- [AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning](https://arxiv.org/abs/2505.24298)
- [AgentRL: Scaling Agentic Reinforcement Learning with a Multi-Turn, Multi-Task Framework](https://arxiv.org/abs/2510.04206)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [GSPO: Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071)
