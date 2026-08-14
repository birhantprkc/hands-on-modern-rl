# 18.4 分布式 RL 训练

上一节我们说了训练系统必须保证**采样策略、模型版本和数值计算**一致——那是在你已经知道要查什么的前提下。现在把一次训练真正分到多张 GPU 上，你会看到最朴素的问题：**数据怎么流？**

假设一组 rollout GPU 生成了一批回答。奖励进程读取回答、调用验证器或奖励模型算分，训练 GPU 再用这些回答算梯度、更新 Actor。更新完了，**新参数必须立刻同步回 rollout GPU**，否则下一批回答还是从旧策略生成的，训练就乱了：

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

单机代码里这就是四个先后调用的函数，到多机系统中变成了不同进程之间的**数据传输和等待**。系统设计主要解决三个问题：

1. **模型放在哪里。** Actor、Reference、Reward Model 和 rollout 引擎加起来几十上百亿参数，可能根本塞不进一组 GPU。
2. **谁在等待谁。** 生成一条回答要几秒钟，参数更新只要几百毫秒，慢的那一步会让其他所有设备闲置。
3. **何时交换数据与参数。** 同步太频繁通信开销太大，间隔太久 rollout 又会用过时策略生成回答。

## 分布式 RL 需要做的三项系统决策

| 系统问题                  | 常见选择                   | 主要取舍                                               |
| ------------------------- | -------------------------- | ------------------------------------------------------ |
| 训练与生成是否共享 GPU    | 共享部署 / 分离部署        | 共享部署节省 GPU；分离部署减少切换并提高并行吞吐       |
| 是否等待整批 rollout 完成 | 同步 / 异步                | 同步训练保持数据较新；异步训练减少等待但会产生陈旧经验 |
| 一个模型怎样分到多张 GPU  | FSDP、张量并行、流水线并行 | 分摊显存，同时增加通信与调度复杂度                     |

先根据瓶颈做选择：显存不足时先解决模型切分；GPU 大量等待时再考虑生成与训练分离或异步；模型采用 MoE 时，还要处理 token 在专家之间的路由与通信。

::: tip 第一次阅读到这里即可
记住数据顺序：**生成 → 奖励 → 更新 → 同步参数**。veRL、slime 和 OpenRLHF 的实现不同，都在安排这四步使用哪些 GPU、何时交换数据。
:::

下面继续沿着这批回答的路径展开。首先确认生成端和训练端计算的是同一个策略；然后用框架安排各个模型；生成速度不足时优化 rollout，显存不足时切分训练状态；任务耗时差别变大后再引入异步；模型换成 MoE 后，还要处理专家路由和通信。这个顺序也是排查多机训练问题的顺序。

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

$\hat A_t$ 表示动作 $a_t$ 比当前平均水平好多少，$\epsilon$ 决定允许比率偏离 1 的范围。假设旧策略生成某个 token 的概率是 $0.20$，新策略把它提高到 $0.24$，那么 $r_t=0.24/0.20=1.2$。当 $\epsilon=0.2$ 时，这次变化刚好到达 clipping 的上边界；继续提高概率也不会继续放大这一项的优化收益。

这个计算有一个前提：分母 $\pi_{\text{old}}$ 必须是生成动作时真正使用的策略。如果回答来自 $\pi_{\text{rollout}}$，训练端却用另一条计算路径重算 $\pi_{\text{old}}$，那么 $r_t$ 在更新开始以前就已经不准确。clipping 只能限制参数更新，不能修正两个引擎算出的概率差异。

**训推偏差的排查顺序**：排查时沿着回答的生成和训练路径逐项对齐：

1. 核对 rollout 使用的模型版本，确认参数同步已经完成。
2. 用同一批 token 分别记录 rollout 侧和训练侧的 log-probability，观察误差集中在哪些位置。
3. 对齐两侧的浮点精度和算子实现，再比较误差是否缩小。
4. 对 MoE 模型额外记录专家路由，确认训练端是否复现了生成时的路由。

**训推偏差的修正方法**：

- **统一计算精度。** 先用 FP16/BF16 替代 rollout 侧的 FP8，判断低精度计算是否是主要误差来源。需要继续使用 FP8 时，应同时加入偏差监控和重要性采样修正。
- **记录真实行为策略。** 直接保存 rollout 时的 log-probability，避免训练端把重新计算的结果当作真实行为概率。
- **重新计算并校验。** 训练前用训练引擎重算 log-probability，并与 rollout 记录逐 token 对比。重算本身不能恢复真实行为策略，但能暴露差异的位置和大小。
- **限制极端比率。** Truncated IS（TIS）等方法会截断过大的重要性采样比率，降低少量异常 token 对梯度的影响。
- **处理长尾 token。** 动态词表剪枝等方法会过滤偏差最大的低概率区域，减少误差在长序列中的累积。
- **回放 MoE 路由。** R3（Rollout Routing Replay）在训练时复现 rollout 的专家选择，减少路由变化造成的概率偏差。

**相关研究工作**：下面几类工作分别从数值精度、分布修正和系统调度处理这一问题：

- _When Speed Kills Stability: Demystifying RL Collapse from the Training-Inference Mismatch_（Liu et al., 2025）集中分析训推不一致与训练崩溃的关系。
- _Defeating the Training-Inference Mismatch via FP16_（Qi et al., 2025）考察浮点精度对两侧 log-probability 偏差的影响。
- _Taming the Tail: Stable LLM Reinforcement Learning via Dynamic Vocabulary Pruning_（arXiv:2512.23087）处理低概率 token 上更明显的偏差。
- _Stabilizing Reinforcement Learning with LLMs: Formulation and Practices_（Zheng et al., arXiv:2512.01374）讨论训推一致、策略时效性和 MoE 路由回放。
- FP8-RL（Qiu et al., arXiv:2601.18150）在 veRL 中结合 W8A8 低精度训练与重要性采样修正。
- TIS（Yao et al., NeurIPS 2025）和 MinPRO（Lei et al., arXiv:2601.22718）限制策略偏移后产生的极端重要性采样比率。
- 动态优化方法（Zhang et al., arXiv:2602.01826）根据回答长度等训练信号调整优化过程。

工程中的 On-policy 程度取决于 $\pi_{\text{rollout}}$ 与当前训练策略之间的距离。参数同步控制模型版本，精度对齐、概率记录和重要性采样修正继续控制计算路径带来的偏差。[第 4 章算法分类](../chapter03_mdp/algorithm-taxonomy)介绍了 On-policy 与 Off-policy 的算法区别；这里关注的是同一概念落到分布式系统后怎样被测量和修正。

## 模型与 GPU 的资源安排

**veRL 的 HybridFlow：五类角色的编排**：训推一致性解决了"数据能不能正确更新模型"。下一步是让 Actor、Critic、Reference Model、Reward Model 和 rollout 引擎在不同 GPU 上协作。veRL（Volcano Engine Reinforcement Learning）把这个问题拆成算法主循环、模型计算和资源分配三个层级，对应论文为 [HybridFlow](https://arxiv.org/abs/2409.19256)。

**HybridFlow 的核心设计**：把 RLHF/GRPO/PPO 训练抽象成 **single-controller 多模型编排**：

```
┌─────────────────────────────────────────────────────────┐
│              Single Controller (Driver)                  │
│  - 算法逻辑（PPO/GRPO 算法主循环）                       │
│  - 资源调度（哪些 GPU 跑哪个模型）                       │
└──────────┬──────────────────────────────────────────────┘
           │
   ┌───────┼───────┬─────────────┬─────────────┐
   │       │       │             │             │
   ▼       ▼       ▼             ▼             ▼
┌──────┐ ┌──────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│Actor │ │Critic│ │Reference │ │Reward    │ │Rollout   │
│(FSDP)│ │(FSDP)│ │(Frozen)  │ │Model     │ │Engine    │
│      │ │      │ │          │ │          │ │(vLLM)    │
└──────┘ └──────┘ └──────────┘ └──────────┘ └──────────┘
   ▲       ▲       ▲             ▲             ▲
   │       │       │             │             │
   └───────┴───────┴─────────────┴─────────────┘
              ResourcePool (GPU 集合)
```

**HybridFlow 的三个核心抽象**：

**ResourcePool：GPU 资源分组**：把 GPU 分组，每组可以放一个或多个模型：

```python
# veRL 配置示例（简化）
resource_pools = {
    "actor_pool": num_gpus=8,    # Actor 用 8 张卡
    "critic_pool": num_gpus=4,   # Critic 用 4 张卡
    "rollout_pool": num_gpus=8,  # Rollout 用 8 张卡
    "ref_pool": num_gpus=2,      # Reference 模型 2 张卡
}
```

不同模型可以**共享 GPU**（colocate）或**独占 GPU**（disaggregated）：

```python
# Colocate：Actor 和 Rollout 共享同一组 GPU
mapping = {
    "actor": "actor_rollout_pool",
    "rollout": "actor_rollout_pool",  # 共享！
    "critic": "critic_pool",
    "ref": "ref_pool",
}
```

**Worker：模型实例封装**：每个 Worker 是一个独立的模型实例，封装了具体的训练/推理逻辑：

```python
class ActorWorker:
    def __init__(self, model_config):
        self.model = FSDPActor(model_config)

    def update(self, batch):
        # PPO/GRPO loss 计算 + 反向传播
        loss = compute_ppo_loss(batch, self.model)
        loss.backward()
        self.optimizer.step()

    def get_weights(self):
        # 给 Rollout Engine 同步权重
        return self.model.state_dict()

class RolloutWorker:
    def __init__(self, model_config):
        self.engine = vLLMEngine(model_config)

    def generate(self, prompts):
        return self.engine.generate(prompts)

    def sync_weights(self, new_weights):
        self.engine.load_weights(new_weights)
```

**Driver：单控制器编排**：Driver 是 RL 算法的主循环，编排所有 Worker：

```python
class PPODriver:
    def train(self, num_epochs):
        for epoch in range(num_epochs):
            # 1. 让 Actor 暴露当前权重给 Rollout
            weights = self.actor_worker.get_weights()
            self.rollout_worker.sync_weights(weights)

            # 2. 用当前策略采样
            prompts = sample_prompts(self.dataset)
            responses = self.rollout_worker.generate(prompts)

            # 3. 用 Reward Model 算 reward
            rewards = self.reward_worker.score(prompts, responses)

            # 4. 用 Critic 算 value
            values = self.critic_worker.value(prompts, responses)

            # 5. 算 advantage + PPO loss 更新 Actor
            advantages = compute_gae(rewards, values)
            self.actor_worker.update(prompts, responses, advantages)

            # 6. 更新 Critic
            self.critic_worker.update(prompts, responses, rewards)
```

**HybridFlow 的混合并行策略**：这里的 Hybrid 指**统一的混合并行策略**——同一个框架内可以组合：

- **3D Parallelism**：TP（张量并行）× PP（流水线并行）× DP（数据并行）
- **Colocate vs Disaggregated**：模型可共享或独占 GPU
- **多种训练后端**：FSDP、Megatron、DeepSpeed ZeRO
- **多种推理后端**：vLLM、SGLang、HuggingFace generate

这些配置决定 Actor、Critic、Reference Model、Reward Model 和 rollout 引擎能否共享资源。框架之间的主要差别也在这里：有些允许灵活组合资源池，有些要求各模型使用独立进程或固定后端。

**主流框架架构对比**：

| 维度         | veRL (HybridFlow) | OpenRLHF          | NeMo-Aligner     | TRL            |
| ------------ | ----------------- | ----------------- | ---------------- | -------------- |
| **编排方式** | Single-controller | Single-controller | Multi-controller | Single-process |
| **资源分配** | 任意组合          | 严格分离          | NVIDIA 栈        | 单 GPU         |
| **训练后端** | FSDP + Megatron   | FSDP/DeepSpeed    | Megatron         | Accelerate     |
| **推理后端** | vLLM/SGLang       | vLLM              | TRT-LLM          | HF generate    |
| **典型规模** | 8-1024 GPU        | 8-256 GPU         | 8-512 GPU        | 1-8 GPU        |

[第 15 章 GRPO 实践](../chapter18_grpo/grpo-practice-and-mechanism) 用的就是 veRL。

**其他主流框架的实现方式**：veRL 使用一个 Driver 编排多个角色，但这并非唯一实现。OpenRLHF 更强调 Ray 进程之间的角色分离，NeMo-Aligner 围绕 NVIDIA 的 Megatron 与 TRT-LLM 构建，TRL 则把复杂度压缩到适合单机实验的 Trainer 接口。比较它们时，仍然看同三个问题：模型怎样放置，生成使用什么后端，参数怎样同步。

**OpenRLHF**：[OpenRLHF, arXiv:2405.11143](https://arxiv.org/abs/2405.11143) 由 OpenLLMAI 团队维护，是最早的开源 RLHF 框架之一。

OpenRLHF 采用以下结构：

- 基于 **Ray** 做分布式调度
- 严格的 **Actor/Critic/Ref/RM 分离**——每个模型在独立的 Ray Actor 进程
- 用较直接的配置接口组织这些 Ray Actor

```python
# OpenRLHF 训练 PPO（伪代码）
from openrlhf import PPOTrainer, ModelGroup

actor = ModelGroup(num_gpus=8, backend="deepspeed")
critic = ModelGroup(num_gpus=8, backend="deepspeed")
ref = ModelGroup(num_gpus=4)
reward = ModelGroup(num_gpus=4)
vllm = VLLMRollout(num_gpus=8)

trainer = PPOTrainer(actor, critic, ref, reward, vllm)
trainer.train(dataset, num_epochs=100)
```

它适合研究用途和中等规模训练（8–256 GPU）。模型角色严格分离，便于独立扩缩容，也会增加角色之间的数据传输。

**NeMo-Aligner**：[NeMo-Aligner](https://github.com/NVIDIA/NeMo-Aligner) 是 NVIDIA 官方栈，深度集成 Megatron-LM 和 TRT-LLM。

NeMo-Aligner 采用以下结构：

- **Megatron** 训练后端，负责张量、流水线和数据并行。
- **TRT-LLM** 推理后端，负责 NVIDIA GPU 上的生成优化。
- 训练、推理和通信都围绕 NVIDIA 软件栈配置。

它适合已经使用 NVIDIA NeMo 与 Megatron 的集群，尤其是 70B 以上模型。采用其他训练后端的团队需要评估迁移成本。

**TRL（Transformer Reinforcement Learning）**：[TRL](https://github.com/huggingface/trl) 是 HuggingFace 出品的轻量级框架。

TRL 采用以下结构：

- 基于 **Accelerate**（HuggingFace 的分布式抽象）
- 单进程模型，靠 Accelerate 自动切分
- 以 Trainer 接口降低小规模实验的配置成本

```python
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

model = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")
config = PPOConfig(batch_size=8)
trainer = PPOTrainer(config, model)
trainer.train(dataset)
```

它适合学习、原型验证和 1–8 张 GPU 的小规模实验。模型数量、生成吞吐和跨节点调度成为瓶颈后，需要换用专门的分布式 RL 框架。

**四种框架的选型建议**：

| 框架             | 易用性 | 性能 | 规模上限  | 工业采用                 |
| ---------------- | ------ | ---- | --------- | ------------------------ |
| **veRL**         | 中     | 高   | 1024+ GPU | Qwen、DeepSeek、字节内部 |
| **OpenRLHF**     | 高     | 中   | 256 GPU   | SimpleRL、部分开源       |
| **NeMo-Aligner** | 低     | 极高 | 512+ GPU  | NVIDIA 客户、Nemotron    |
| **TRL**          | 极高   | 低   | 8 GPU     | 研究、教学               |

选型可以从当前规模和已有技术栈出发：

- 学习、原型：TRL
- 研究、中等规模：OpenRLHF 或 veRL
- 大规模生产：veRL 或 NeMo-Aligner（看硬件栈）

## 生成吞吐优化与显存控制

**Rollout 引擎的生成吞吐优化**：在许多 LLM RL 任务中，生成回答占用的时间长于一次参数更新，具体成本见[附录 A.2](../appendix_industrial_training/rl-infrastructure)。Rollout 引擎因此直接决定训练进程能否持续拿到新数据。下面以 vLLM 为例说明生成端的三项优化。

**vLLM 的三项核心技术**：

**PagedAttention：分页式 KV 缓存**：传统 KV cache 是连续分配，导致显存碎片严重。vLLM 借鉴 OS 的分页机制，把 KV cache 分成固定大小的 block：

```python
# 传统：KV cache 连续分配
seq_len = 2048
kv_cache = torch.empty(batch_size, seq_len, num_heads, head_dim)
# 显存利用率 50-70%

# vLLM PagedAttention：分块
block_size = 16
blocks = allocate_blocks(num_blocks)
# 显存利用率 95%+
```

显存利用率从 50-70% 提升到 95%+，batch size 提升 2-4 倍。

**Continuous Batching：动态批处理**：传统 batching 是"等一个 batch 全部生成完才换"。vLLM 是**动态 batching**——某条序列生成完后立刻换上新序列：

```
时间:  ──────────────────────────────────────►
序列A: [tok][tok][tok][tok][EOS]
序列B: [tok][tok][tok][tok][tok][tok][EOS]
序列C:           [tok][tok][tok][tok][EOS]  ← A 结束后立刻加入
序列D:                    [tok][tok][tok][EOS]  ← C 结束后加入
```

吞吐提升 5-10 倍 vs 静态 batching。

**Speculative Decoding：投机解码**：用小模型先 draft 几个 token，大模型并行验证：

```python
def speculative_decode(prompt, draft_model, target_model, num_draft=4):
    while not done:
        # 1. 小模型生成 num_draft 个 token
        draft_tokens = draft_model.generate(prompt, max_tokens=num_draft)

        # 2. 大模型并行验证
        target_logits = target_model.forward(prompt + draft_tokens)

        # 3. 接受匹配的 token，拒绝后重新生成
        for i, token in enumerate(draft_tokens):
            if target_logits[i].argmax() == token:
                prompt.append(token)
            else:
                prompt.append(target_logits[i].argmax())
                break
```

吞吐提升 2-3 倍（典型 LLM 推理）。

**vLLM 在 RL 数据流中的作用**：veRL 中 vLLM 作为 RolloutWorker：

```python
class VLLMRolloutWorker:
    def __init__(self, model_path, tensor_parallel_size=8):
        from vllm import LLM
        self.engine = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            enable_prefix_caching=True,  # 关键：GRPO 同 prompt 多采样时复用 KV
            gpu_memory_utilization=0.9,
        )

    def generate(self, prompts, sampling_params):
        # 批量生成
        return self.engine.generate(prompts, sampling_params)

    def sync_weights(self, new_weights):
        # vLLM 0.5+ 支持在线权重更新
        self.engine.load_weights(new_weights)
```

**Prefix Caching** 对 GRPO 特别重要——同一个 prompt 生成 $G=8$ 条回答，前缀（prompt 部分）的 KV cache 可以复用，节省 70-80% 的显存和时间。

**SGLang 的生成与调度优化**：[SGLang](https://github.com/sgl-project/sglang) 由 LMSYS 团队开发，在 agentic 场景下比 vLLM 更快：

- **RadixAttention**：用基数树管理 KV cache，跨请求复用
- **Programmatic Frontend**：支持复杂的控制流（多轮调用、分支、循环）
- **Constrained Decoding**：内置 JSON、regex 约束生成

工业实践中：

- **vLLM**：通用 rollout、单轮生成
- **SGLang**：agentic rollout、多轮、结构化输出
- **TRT-LLM**：针对 NVIDIA GPU 的推理优化

**多 GPU 显存分摊技术**：生成速度解决以后，训练端仍要容纳权重、梯度、优化器状态和激活。一个 70B 模型的 BF16 全参数训练远超单张 80GB H100 的容量，因此这些状态必须切分或重算。

**训练显存的构成分析**：训练显存包含权重、梯度、优化器状态和激活。以常见的 BF16 权重与梯度、FP32 主权重和 Adam 一阶、二阶动量为例，每个参数大约需要：

$$
\begin{aligned}
M \approx {}& \underbrace{2N}_{\text{BF16 权重}}
+ \underbrace{2N}_{\text{BF16 梯度}}
+ \underbrace{4N}_{\text{FP32 主权重}} \\
&+ \underbrace{8N}_{\text{Adam 的 }m\text{ 和 }v}
+ M_{\text{act}},
\end{aligned}
$$

其中 $N$ 是参数数量，前四项的单位都是字节，$M_{\text{act}}$ 是激活占用。也就是说，在不考虑激活时，这种配置约需每个参数 $16$ 字节。不同优化器和精度配置会改变这个数字，例如不保存 FP32 主权重时会少 $4N$ 字节。

对 70B 模型：

- 权重：140 GB
- 梯度：140 GB
- FP32 主权重：280 GB
- Adam 的 $m$、$v$：560 GB
- 激活：~100 GB（取决于 batch size 和 seq len）
- **总计**：约 1.22 TB

这个估算的作用是确定量级，并非精确预测显存。激活检查点、优化器实现、序列长度和 batch size 都会改变实际占用；但它已经足以说明 70B 全参数训练无法放进一张 80GB GPU。

**ZeRO：零冗余优化器**：[DeepSpeed ZeRO, arXiv:1910.02054](https://arxiv.org/abs/1910.02054) 把训练状态切分到多个 GPU：

| 阶段       | 切分内容                      | 节省倍数         | 通信开销 |
| ---------- | ----------------------------- | ---------------- | -------- |
| **ZeRO-1** | Optimizer state               | 4×               | 低       |
| **ZeRO-2** | Optimizer + Gradient          | 8×               | 中       |
| **ZeRO-3** | Optimizer + Gradient + Weight | $N$×（N=GPU 数） | 高       |

ZeRO-3 把权重也切分，每个 GPU 只存 $1/N$ 的权重，但前向反向时需要 all-gather 还原。

```python
# DeepSpeed ZeRO-3 配置
config = {
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "sub_group_size": 1e9,
        "reduce_bucket_size": 5e8,
    },
    "bf16": {"enabled": True}
}
```

**FSDP：完全分片数据并行**：PyTorch 原生的 ZeRO-3 等价物，比 DeepSpeed 更易用：

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = LlamaForCausalLM(config)
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # 等价 ZeRO-3
    mixed_precision=MixedPrecision(param_dtype=torch.bfloat16),
    cpu_offload=CPUOffload(offload_params=False),  # 可选 CPU offload
)
```

veRL 默认用 FSDP——比 DeepSpeed 更稳定、与 PyTorch 生态更兼容。

**Gradient Checkpointing：梯度检查点**：不切分模型，而是用计算换显存——前向时不保存中间激活，反向时重新计算：

```python
from torch.utils.checkpoint import checkpoint

class CheckpointedBlock(nn.Module):
    def forward(self, x):
        # 用 checkpoint 包裹 transformer block
        return checkpoint(self._forward, x, use_reentrant=False)

    def _forward(self, x):
        return self.transformer_block(x)
```

激活显存从 $O(L)$ 降到 $O(\sqrt{L})$（$L$ 是层数），代价是前向计算两次——训练慢 20-30%。

**显存优化组合配置估算**：对 70B 模型（8 张 H100 80GB）：

| 配置                                   | 单卡显存           | 训练速度 |
| -------------------------------------- | ------------------ | -------- |
| 全参 + Adam（baseline）                | 940 GB（超出显存） | -        |
| ZeRO-3                                 | 118 GB（超出显存） | -        |
| ZeRO-3 + Gradient Checkpointing        | 30 GB              | 1×       |
| ZeRO-3 + Gradient Checkpointing + LoRA | 8 GB               | 1.2×     |

LoRA（[第 18 章](./industrial-post-training)）只训少量参数，显存需求大幅降低。工业级 70B RL 训练通常用 LoRA + FSDP。

## 多机流水线的持续运行

### 异步调度：减少流水线等待

同步训练需要等一批 rollout 全部结束以后再更新，耗时较长的轨迹会让训练 GPU 等待。异步训练让生成和更新分别推进，再通过队列交换轨迹。下面比较 LlamaRL、AReaL 和 AgentRL 的调度方法。

#### LlamaRL

[LlamaRL, Meta arXiv:2505.24034](https://arxiv.org/abs/2505.24034) Meta 2025 年 5 月发布的分布式 RL 框架：

LlamaRL 使用去中心化设计，不设置统一的主节点，每个 worker 根据自己的角色持续取任务并提交结果。

```python
# LlamaRL 架构（简化）
class LlamaRLWorker:
    def run(self):
        while True:
            # 每个 worker 自己决定做什么
            if self.role == "rollout":
                prompts = self.fetch_from_queue()
                responses = self.generate(prompts)
                self.push_to_train_queue(responses)

            elif self.role == "train":
                batch = self.fetch_from_rollout_queue()
                self.update(batch)
                self.broadcast_weights()  # 异步广播
```

这种设计带来三项系统特性：

- 无单点故障
- 横向扩展容易（加 worker 即可）
- 适合超大规模（10k+ GPU）

**实测**：在 4096 GPU 上跑 Llama-3-70B GRPO，比同步训练快 **10.4×**。

#### AReaL（Asynchronous RL）

[AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning, arXiv:2505.24298](https://arxiv.org/abs/2505.24298) 是 Ant Group 和清华 2025 年开源的大规模异步 LLM RL 系统：

AReaL 采用完全异步的 rollout，并在 PPO 更新中显式处理策略陈旧度。Rollout worker 持续生成样本，training worker 拿到 batch 后立即更新；每条轨迹记录生成时的策略版本和概率，训练端据此修正旧策略数据。

```python
# AReaL 关键算法（简化）
def staleness_aware_update(batch, current_weights):
    # batch 记录了 rollout 时的 policy version 与 logprob
    gen_log_probs = batch["gen_log_probs"]
    current_log_probs = compute_log_probs(batch, current_weights)
    importance_weights = torch.exp(current_log_probs - gen_log_probs)

    # 截断重要性权重，避免旧样本造成过大梯度
    clipped_weights = torch.clamp(importance_weights, 0.8, 1.2)
    loss = -(clipped_weights * advantages).mean()

    return loss
```

这种设计允许：

- 允许训练用旧数据，不要求严格 on-policy
- 缓冲区可以积累大量数据
- 训练和生成完全解耦

**实测**：在 1024 GPU 上跑 671B MoE GRPO，比同步快 **2.77×**。

#### AgentRL

[AgentRL: Scaling Agentic Reinforcement Learning with a Multi-Turn, Multi-Task Framework, arXiv:2510.04206](https://arxiv.org/abs/2510.04206) 是 2025 年 10 月发布的多轮、多任务 Agentic RL 框架，代码在 [THUDM/AgentRL](https://github.com/THUDM/AgentRL)：

AgentRL 把异步生成训练流水线与统一环境接口放在一起。训练侧使用 rollout、Actor 和 Reference 三类 worker 池；环境侧通过函数调用接口、容器、Controller 和 Task Worker 管理异构任务。Cross-Policy Sampling 增加多轮探索，Task Advantage Normalization 则对齐不同任务的优势尺度。

```python
# AgentRL 异步训练结构（简化）
rollout_workers.stream_trajectories(task_manager)
actor_workers.update_policy(buffer.sample())
reference_workers.compute_kl(buffer.sample())
controller.route_function_calls(task_workers)
```

它主要处理以下需求：

- 支持 multi-turn、multi-task agentic RL
- 异步解耦 trajectory 采集和 policy 更新
- 通过 controller / task worker / transport layer 管理环境部署
- 被用于构建 AutoGLM

这套结构适合 SWE-Agent、Computer Use 和 Deep Research Agent 等多轮环境任务。

#### 三种异步框架对比

| 框架        | 主要贡献者       | 核心机制                             | 加速比     | 适用场景       |
| ----------- | ---------------- | ------------------------------------ | ---------- | -------------- |
| **LlamaRL** | Meta             | 完全去中心化                         | 10.4×      | 超大规模 Dense |
| **AReaL**   | Ant Group 和清华 | 全异步 rollout + staleness-aware PPO | 2.77×      | 大规模 LLM RL  |
| **AgentRL** | THUDM/智谱       | 多轮多任务 + 统一环境接口            | 论文未标注 | Agent 训练     |

### MoE 带来的额外系统复杂度

DeepSeek V3、Qwen3 和 GLM-4.5 都采用 MoE 架构。每个 token 只激活少量专家，参数能够分散到更多 GPU；与此同时，RL 的样本分布会改变专家负载，训练系统还要记录路由和通信状态。

#### MoE 训练的数据流特点

MoE 模型的参数分布不均匀——大多数参数在 expert 里，每条样本只激活少数 expert：

```
MoE 模型结构（DeepSeek V3）:
┌─────────────────────────────────────┐
│ Dense 部分（attention 等）: 20B 参数 │
├─────────────────────────────────────┤
│ MoE 部分:                            │
│  - 256 个 expert × 5B 参数 = 1.28T   │
│  - 每条样本激活 8 个 expert           │
│  - 实际激活参数: 40B                  │
└─────────────────────────────────────┘
总参数: 1.3T，激活参数: 60B
```

#### MoE RL 的三项系统挑战

##### Expert 负载不均

某些 expert 被频繁激活，其他 expert 闲置。导致：

- 计算负载不均（部分 GPU 过载）
- 训练数据分布偏（部分 expert 训练不充分）

**解决**：**Expert Balancing Loss**：

```python
def expert_balancing_loss(router_logits, num_experts):
    # 计算每个 expert 的激活频率
    router_probs = torch.softmax(router_logits, dim=-1)
    expert_freq = router_probs.mean(dim=0)  # [num_experts]

    # 鼓励均匀分布
    target_freq = 1.0 / num_experts
    balance_loss = ((expert_freq - target_freq) ** 2).mean()

    return balance_loss
```

##### 通信开销

MoE 的 expert 分布在多个 GPU（Expert Parallelism），每条样本都要 all-to-all 通信：

```
GPU 0: expert 0,1,2     ──┐
GPU 1: expert 3,4,5     ──┼── all-to-all ── 处理完后 all-to-all 回去
GPU 2: expert 6,7,8     ──┤
GPU 3: expert 9,10,11   ──┘
```

**解决**：**DeepEP**（DeepSeek Expert Parallelism），优化 all-to-all 通信模式。

##### Token 级重要性采样方差

[GRPO 家族](../chapter18_grpo/grpo-family) 提到——MoE 下不同 token 路由到不同 expert，token 级 importance sampling 比率波动剧烈，梯度方差大。

**解决**：**GSPO（Group Sequence Policy Optimization）**——把 IS 比率从 token 级改成序列级：

```python
# PPO/GRPO: token 级 IS
token_ratio = exp(log_prob_new - log_prob_old)  # 每个 token 独立

# GSPO: 序列级 IS
sequence_log_prob_new = sum(log_prob_new_per_token)
sequence_log_prob_old = sum(log_prob_old_per_token)
sequence_ratio = exp(sequence_log_prob_new - sequence_log_prob_old)
# 整个序列用同一个 ratio
```

Qwen3 全系（包括 235B-A22B）都基于 GSPO 训练。

#### DeepSeek V3 的 MoE RL 方案

DeepSeek V3（671B MoE，37B 激活）的 RL 训练实践：

- **DualPipe**：流水线并行优化（详见 36.7）
- **FP8 训练**：用 FP8 减少显存和计算（[arXiv:2412.19437](https://arxiv.org/abs/2412.19437)）
- **MTP (Multi-Token Prediction)**：一次预测多个 token，提升训练信号密度

#### Step Flash 的 MoE 方案

Step Flash 是阶跃星辰 2025 年发布的 MoE RL 优化：

- **Dynamic Expert Allocation**：根据 batch 内 token 分布动态调整 expert 数量
- **Sparse Gradient Sync**：只同步被激活的 expert 的梯度
- **Cache-aware Routing**：路由时考虑 KV cache 局部性

#### GLM-4.5 的 MoE 方案

GLM-4.5 用 **slime** 框架训练（[THUDM/slime](https://github.com/THUDM/slime)）：

- Megatron 训练后端
- SGLang 推理后端
- 原生 MoE 优化（DeepEP 通信、fp8 rollout）

### 减少流水线空闲时间

模型切到多张 GPU 后，每张卡并不会自动保持忙碌。流水线阶段可能等待前一阶段，长度不同的样本也会在同一 batch 中留下空位。DualPipe 减少前向与反向之间的等待，Best-Fit Packing 则把长度接近的序列装入同一批次。它们分别处理计算时间线和样本形状造成的空闲。

#### DualPipe

[DeepSeek V3 论文 arXiv:2412.19437](https://arxiv.org/abs/2412.19437) 提出 **DualPipe**——双向流水线并行。

传统流水线并行（PP）的气泡（bubble）问题：

```
GPU 0: [F0][F1][F2][F3]              [B3][B2][B1][B0]
GPU 1:       [F0][F1][F2][F3]   [B3][B2][B1][B0]
GPU 2:             [F0][F1][F2][F3][B3][B2][B1][B0]
                   ↑                ↑
                   前向              反向
                   气泡很大
```

DualPipe 让前向和反向**同时跑**——前向 stage N 和反向 stage N-1 在同一 GPU 上重叠：

```
GPU 0: [F0|B0][F1|B1][F2|B2][F3|B3]  ← 前向和反向重叠
GPU 1:       [F0|B0][F1|B1][F2|B2][F3|B3]
GPU 2:             [F0|B0][F1|B1][F2|B2][F3|B3]
                                    几乎没有气泡
```

气泡比例从传统的 $\frac{P-1}{M}$（$P$ 是 PP stage 数，$M$ 是 micro-batch 数）降到 $\frac{P-1}{2M}$。

```python
# DualPipe 伪代码
class DualPipeScheduler:
    def schedule(self, num_stages, num_micro_batches):
        schedule = []
        for step in range(num_micro_batches + num_stages - 1):
            for stage in range(num_stages):
                # 同一 stage 同一 step 既做前向又做反向
                fwd_mb = step - stage
                bwd_mb = step - (num_stages - 1 - stage)
                if fwd_mb >= 0 and fwd_mb < num_micro_batches:
                    schedule.append(("forward", stage, fwd_mb))
                if bwd_mb >= 0 and bwd_mb < num_micro_batches:
                    schedule.append(("backward", stage, bwd_mb))
        return schedule
```

#### Best-Fit Packing

传统 micro-batch 分配是均匀的——每个 GPU 拿相同数量。但 MoE 下不同 expert 负载不同，均匀分配导致不均衡。

**Best-Fit Packing**：用装箱算法（bin packing）把不同大小的 micro-batch 分配到 GPU：

```python
def best_fit_pack(items, bin_capacity):
    """items 是不同大小的 micro-batch, bin_capacity 是单 GPU 容量"""
    bins = [[]]
    for item in sorted(items, reverse=True):  # 从大到小
        # 找到能放下且最满的 bin
        best_bin = None
        best_remaining = float('inf')
        for bin in bins:
            remaining = bin_capacity - sum(bin)
            if item <= remaining < best_remaining:
                best_bin = bin
                best_remaining = remaining
        if best_bin is None:
            bins.append([item])
        else:
            best_bin.append(item)
    return bins
```

DeepSeek V3 用 Best-Fit Packing 让 GPU 利用率从 70% 提升到 95%。

### 性能瓶颈定位方法

前面的每项技术都可能把瓶颈推到下一处：生成加快以后，权重同步可能变慢；模型切分以后，跨卡通信可能占据主要时间；样本装箱改善以后，数据读取又可能跟不上。性能分析先记录每一步实际耗时，再决定增加 GPU、修改并行方式还是调整 batch。

#### 常用性能分析工具

##### PyTorch Profiler

```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
) as prof:
    trainer.train_step()

# 打印 top 10 耗时操作
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

##### NVIDIA Nsight Systems

```bash
# 用 nsys 跑训练
nsys profile -o rl_train_profile python train.py

# 用 Nsight Systems GUI 查看时间线
nsys-ui rl_train_profile.qdrep
```

可视化每个 CUDA kernel 的执行时间、CPU-GPU 同步、通信开销。

##### veRL 内置 Profiler

veRL 提供了 RL 特定的 profiling：

```python
from verl.utils.profiler import RLProfiler

with RLProfiler() as p:
    trainer.train()
    p.print_summary()
# 输出：
#   rollout time: 3500s (85%)
#   actor update time: 120s (3%)
#   critic update time: 80s (2%)
#   weight sync time: 30s (0.7%)
#   communication: 400s (10%)
```

#### 常见瓶颈与优化方向

| 瓶颈                | 症状                     | 优化                                      |
| ------------------- | ------------------------ | ----------------------------------------- |
| **Rollout 慢**      | rollout 占 80%+ 时间     | 增加 rollout GPU、用 vLLM prefix caching  |
| **Weight Sync 慢**  | sync 占 5%+ 时间         | 用 LoRA、NCCL 打包传输                    |
| **通信开销**        | all-reduce 占 10%+ 时间  | 增大 batch size、用 gradient accumulation |
| **激活显存爆炸**    | OOM                      | Gradient checkpointing                    |
| **Expert 负载不均** | 部分 GPU 90%+、部分 30%  | Expert balancing loss、动态路由           |
| **慢人问题**        | batch 内最长序列决定时间 | 长度分桶、Seer divided rollout            |

#### MFU（Model FLOPs Utilization）

MFU 用实际执行的浮点运算量除以硬件在同一时间内能够提供的峰值运算量：

$$\text{MFU} = \frac{\text{实际 FLOPs}}{\text{峰值 FLOPs} \times \text{时间}}$$

例如，8 张 GPU 的理论峰值都是 1000 TFLOPS，连续运行 10 秒最多可完成 $8\times1000\times10=80{,}000$ TFLOP 的计算。若模型实际完成了 32,000 TFLOP，那么 MFU 为 $32{,}000/80{,}000=40\%$。剩余时间可能消耗在通信、等待数据或生成轨迹上。

H100 bf16 峰值 ~1000 TFLOPS。典型 LLM RL 训练 MFU：

| 配置                         | MFU                                |
| ---------------------------- | ---------------------------------- |
| Dense + FSDP + checkpointing | 35-45%                             |
| MoE + EP + DualPipe          | 50-60%                             |
| 异步 RL（生成/训练分离）     | 70-80%（rollout 部分用 vLLM 加速） |

MFU 低于 30% 时，应结合时间分解继续检查通信、数据加载和 rollout 等待。MFU 本身不能说明具体瓶颈位于哪一步。

### 大规模集群配置实践

模型并行、异步 rollout、显存优化和故障恢复最终要在同一套集群配置中协作。下面用一份大规模 MoE 训练配置说明这些参数之间的关系。

#### 典型配置

以 Qwen3-235B-A22B（235B 总参，22B 激活 MoE）的 GRPO 训练为例：

```yaml
# 集群配置
total_gpus: 12288 # 12k H100
intra_node_bandwidth: 900 GB/s # NVLink
inter_node_bandwidth: 50 GB/s # InfiniBand

# 模型并行
tensor_parallel: 8 # TP=8（节点内）
pipeline_parallel: 4 # PP=4（跨节点）
expert_parallel: 16 # EP=16
data_parallel: 24 # DP=24

# 训练配置
algorithm: GSPO # MoE 优化的 GRPO 变体
batch_size_per_gpu: 1
gradient_accumulation: 32
seq_len: 32768
group_size: 8 # GRPO 每个 prompt 生成 8 条

# 异步配置
async_mode: disaggregated
rollout_buffer_size: 100000
weight_sync: lora # 只同步 LoRA adapter
weight_sync_method: nccl_packed
```

#### 性能测量

```text
训练 1 epoch (10B tokens):
  Total time: 24 小时
  GPU hours: 294912

分项时间:
  Rollout: 18 小时 (75%)
  Actor update: 3 小时 (12.5%)
  Critic update: 2 小时 (8%)
  Weight sync: 0.5 小时 (2%)
  Other: 0.5 小时 (2.5%)

MFU: 52%（MoE + DualPipe + FP8）
```

#### 大规模训练的故障与瓶颈

##### 故障恢复

12288 张卡，平均每天有 5-10 张故障。必须：

- **Checkpoint 频率**：每 30 分钟存一次，故障时回滚
- **冗余设计**：每 1024 张卡配 8 张备份
- **自动重启**：故障检测后自动从最近 checkpoint 恢复

##### 通信瓶颈

跨节点通信慢，万卡集群网络设计：

- **Topology-aware**：相邻 GPU 优先组成 tensor parallel group
- **Overlap 通信与计算**：反向传播时同时启动梯度 all-reduce
- **Gradient Bucket**：合并小梯度，减少通信次数

##### MoE 路由稳定性

MoE 训练中 expert 路由可能突然塌缩——所有 token 都路由到少数 expert。监控：

```python
# 实时监控 expert 负载
def monitor_expert_balance(model):
    while training:
        for layer in model.moe_layers:
            router_probs = layer.router.get_recent_probs()
            entropy = -torch.sum(router_probs * torch.log(router_probs + 1e-10))
            if entropy < threshold:  # 路由熵过低
                alert(f"Layer {layer.id}: expert routing collapse!")
        time.sleep(60)
```

##### 数据流水线瓶颈

万卡集群每秒消费数百万 token，数据加载本身可能成为瓶颈：

- **预取**：提前准备未来 10 个 batch 的数据
- **数据压缩**：用更紧凑的格式存储
- **分布式存储**：数据分布在多个 SSD，避免单点 I/O 瓶颈

## 本节小结

- 多机 RL 仍然沿着生成、奖励、更新和参数同步四步运行。
- 模型切分解决显存问题，训练与生成的资源安排解决等待问题，权重同步保证 rollout 使用正确的策略版本。
- 同步训练更容易保证数据较新；异步训练减少等待，同时必须处理经验陈旧和重要性修正。
- MoE 在普通并行之外还要处理专家负载与路由通信。

多机系统解决了算力怎样协同，训练仍然需要持续供应可执行、可验证的数据。[18.5 大规模 RL 数据工程](./data-engineering) 将沿着一条轨迹的生命周期，说明任务、环境、奖励和失败样本怎样进入下一轮训练。

## 延伸阅读

- [Sheng et al. 2024 "HybridFlow: A Flexible and Efficient RLHF Framework"](https://arxiv.org/abs/2409.19256)
- [Hu et al. 2024 "OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework"](https://arxiv.org/abs/2405.11143)
- [Kwon et al. 2023 "Efficient Memory Management for Large Language Model Serving with PagedAttention" (vLLM)](https://arxiv.org/abs/2309.06180)
- [Zheng et al. 2023 "SGLang"](https://arxiv.org/abs/2312.07104)
- [Rajbhandari et al. 2020 "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"](https://arxiv.org/abs/1910.02054)
- [LlamaRL (Meta GenAI) 2025 "LlamaRL: A Distributed Asynchronous Reinforcement Learning Framework"](https://arxiv.org/abs/2505.24034)
- [Fu et al. 2025 "AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning"](https://arxiv.org/abs/2505.24298)
- [Zhang et al. 2025 "AgentRL: Scaling Agentic Reinforcement Learning with a Multi-Turn, Multi-Task Framework"](https://arxiv.org/abs/2510.04206)
- [DeepSeek-AI 2024 "DeepSeek-V3 Technical Report"](https://arxiv.org/abs/2412.19437)
- [DeepSeek-AI 2025 "DeepSeek-R1: Incentivizing Reasoning Capability via RL"](https://arxiv.org/abs/2501.12948)
- [Qwen Team 2025 "Qwen3 Technical Report"](https://arxiv.org/abs/2505.09388)
- [Zheng et al. 2025 "GSPO: Group Sequence Policy Optimization"](https://arxiv.org/abs/2507.18071)
- [Qin et al. 2025 "Seer: Online Context Learning for Fast Synchronous LLM Reinforcement Learning"](https://arxiv.org/abs/2511.14617)
