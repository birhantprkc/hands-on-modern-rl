# 25.5 强化学习评测与 Harness

> [第 25 章](../chapter30_alignment_failures/modern-incidents) 讲了 Qwen3 数据污染——benchmark 分数虚高 15-25 个百分点。这暴露的不只是数据问题，而是 **整个 RL 评估方法论的脆弱**。本章系统化讨论：什么样的 benchmark 设计是可信的？怎么检测污染？提示敏感性如何影响结论？长程任务和行为任务怎么评？最后介绍工业级评测 harness 与 Anthropic 2025 内部 AI Research Eval Suite（34× 人类加速）。

## 评估基准设计原则

好的 RL benchmark 必须满足五个原则：

### 可验证性（Verifiability）

每个测试样本的答案**必须是机器可判定的**。形式化定义：存在函数 $\text{Verify}: \mathcal{Y} \times \mathcal{Y} \to \{0, 1\}$，使得对任何 $(y_{\text{pred}}, y_{\text{gold}})$ 都能确定性地给出对错。

- **数学题**：抽取最终数字，与标准答案比较（[GSM8K](https://arxiv.org/abs/2110.14168)、MATH）
- **代码题**：在测试用例上运行，看通过率（[HumanEval](https://arxiv.org/abs/2107.03374)、MBPP、LiveCodeBench）
- **逻辑题**：用 SAT solver 或 theorem prover 验证（MiniF2F、PutnamBench）

不可验证的任务（开放式写作、创意生成）只能用人类评估或 RM 评估——这两者都不可靠。

### 代表性（Coverage）

Benchmark 要覆盖模型可能遇到的真实分布。形式化：

$$\mathcal{D}_{\text{test}} \sim P_{\text{real}}, \quad P_{\text{real}} \approx P_{\text{test}}$$

如果 $\mathcal{D}_{\text{test}}$ 偏向某类问题，模型可能在其他类型上失效。GSM8K 是典型的反例——只有小学数学，模型 GSM8K 90% 不代表会做高数。

### 难度分层（Difficulty Stratification）

按难度分层评估，避免"平均分掩盖极端表现"：

```python
# 难度分层评估
def stratified_eval(model, dataset):
    results = {"easy": [], "medium": [], "hard": []}
    for x, y in dataset:
        pred = model(x)
        difficulty = classify_difficulty(x)  # 用难度分类器
        results[difficulty].append(verify(pred, y))
    return {k: np.mean(v) for k, v in results.items()}
```

MATH 数据集按难度分 Level 1-5，DeepSeek-R1 报告每层分数。报告分层分数比单一总分更能反映能力分布。

### 抗污染（Contamination Resistance）

测试集必须**严格保密**，并对训练数据做去污染检查。详见 35.2。

### 统计显著性（Statistical Rigor）

不能只报"模型 A 在 MATH 上 60%，模型 B 55%"——可能只是抽样噪声。要做：

- **置信区间**：$n$ 个测试样本，准确率 $p$，95% CI 为 $p \pm 1.96\sqrt{p(1-p)/n}$
- **配对 t-test**：在相同测试集上对比两个模型
- **Bootstrap**：对测试集做重采样估计方差

LLM 评测论文长期忽视统计显著性，2024 年后才被广泛接受（[Blackwell et al., arXiv:2410.03492](https://arxiv.org/abs/2410.03492)）。

## 污染与泄漏检测

[第 25 章 RLVR 假性收益](../chapter30_alignment_failures/modern-incidents) 详细讲了 Qwen3 数据污染事件。这一节给出系统化的检测方法。

### 污染的三种类型

#### 1. 显式污染

训练数据和测试数据**完全相同**的样本：

$$\exists (x, y) \in \mathcal{D}_{\text{train}}, \quad (x, y) \in \mathcal{D}_{\text{test}}$$

最容易检测，n-gram 重叠就能发现。

#### 2. 近似污染

训练数据包含测试样本的**改写、翻译、释义**：

$$\exists (x', y') \in \mathcal{D}_{\text{train}}, \quad \text{sim}(x', x_{\text{test}}) > \tau$$

检测需要语义相似度（embedding 距离）或 LLM 判断。

#### 3. 隐式污染（最难）

训练数据不直接包含测试样本，但训练任务与测试任务高度相似——模型学到了**任务模式**而不是**特定答案**：

- 训练数据：2000 道大学物理题
- 测试数据：GSM8K（小学数学）
- 现象：物理题训练让模型学会了"读题→列式→计算→验证"的模式，间接提升数学

隐式污染无法完全检测，只能通过 **Holdout 任务** 间接评估（用模型完全没见过类型的任务）。

### 检测方法

#### N-gram 重叠

最简单的检测——13-gram 重叠：

```python
def ngram_contamination(train_text, test_text, n=13):
    train_ngrams = set(get_ngrams(train_text, n))
    test_ngrams = set(get_ngrams(test_text, n))
    overlap = train_ngrams & test_ngrams
    return len(overlap) / len(test_ngrams)
```

OpenAI 2020 的研究（[arXiv:2005.14165](https://arxiv.org/abs/2005.14165)）用 13-gram 重叠在训练语料中过滤与 benchmark 重复的文本，这是最早的去污染实践之一。

#### 成员推理（Membership Inference）

训练一个分类器判断"这个样本是否在训练集中"：

$$\text{MIA}(x) = \begin{cases} 1, & \text{if } p_{\text{model}}(x) > \tau \\ 0, & \text{otherwise} \end{cases}$$

如果 MIA 在测试集上准确率显著高于随机，说明测试集在训练集中。

#### Perplexity 异常

计算模型在测试集上的 perplexity：

$$\text{PPL}_{\text{test}} = \exp\left(-\frac{1}{N}\sum_i \log p_{\text{model}}(x_i)\right)$$

如果 PPL 远低于类似难度的对照集，可能模型"记住"了测试集。

#### 时序分割

按时间分割测试集——只用模型发布日期之后的新题目：

```python
# 持续更新的 benchmark 与 LiveCodeBench、LMSYS Arena
test_data = [
    item for item in dataset
    if item.created_at > model_release_date
]
```

这是最可靠的抗污染方法——LiveCodeBench、LMSYS Chatbot Arena 都用这个思路。

### 去污染的实际工程

工业级去污染 pipeline：

1. **N-gram 过滤**（13-gram）：移除 90% 显式污染
2. **Embedding 检索**（cosine sim > 0.9）：移除近似污染
3. **MinHash LSH**：快速近似检测（[Deduplicating Training Data, arXiv:2107.06499](https://arxiv.org/abs/2107.06499)）
4. **持续新增 benchmark**：每月用新数据更新测试集

Qwen3 事件后，主流团队都建立了去污染 pipeline，但效果仍不完美——隐式污染几乎无法消除。

## 提示敏感性分析

同一个模型、同一个任务，prompt 不同，分数差 10-20 个点是常见的。这种现象叫 **Prompt Sensitivity**。

### 实验证据

Mizrahi et al. 2024（[arXiv:2401.00595](https://arxiv.org/abs/2401.00595)）系统研究：对 10 个 LLM × 22 个 benchmark × 5 种 prompt 模板：

| 模板 | MATH 分数（GPT-4） | MMLU 分数（GPT-4） |
| ---- | ------------------ | ------------------ |
| A    | 52.1%              | 86.4%              |
| B    | 47.3%              | 84.1%              |
| C    | 50.5%              | 85.7%              |
| D    | 48.9%              | 83.9%              |
| E    | 51.8%              | 86.1%              |

最大波动 4.8 个点——这意味着仅基于单一 prompt 的结论不可靠。

### 敏感性来源

1. **格式要求**："回答 0-100 之间的数字" vs "请给出推理过程后回答数字"
2. **CoT 触发**："think step by step" vs "explain your reasoning" vs 不加 CoT
3. **Few-shot 数量**：0-shot、4-shot、8-shot 结果差异显著
4. **答案抽取格式**：用 regex `\\boxed\{(.+?)\}` vs `"answer: (.+?)"`

### 标准化方法

#### 1. 多 prompt 平均

对每个测试样本用 $K$ 个 prompt 模板，取平均：

$$\text{Score}(\pi) = \frac{1}{K} \sum_{k=1}^K \text{Score}_{\text{prompt}_k}(\pi)$$

#### 2. 报告方差

不只报平均分，还要报方差：

$$\text{Score} \pm 1.96 \cdot \frac{\sigma}{\sqrt{K}}$$

#### 3. Prompt 标准化

lm-eval-harness 定义了**统一的 prompt 格式规范**，所有模型在相同的 prompt 上评估。

```python
# lm-eval-harness 标准化 prompt
PROMPT_TEMPLATE = """
Question: {question}

Answer: Let's think step by step. {reasoning}
Therefore, the answer is \\boxed{{{answer}}}.
"""
```

### 工程建议

RL 训练后的模型特别容易对 prompt 敏感——因为 RL 鼓励模型对训练分布中的 prompt 格式高度适应。**报告 RL 结果时必须做多 prompt 平均**，否则结论可能被"幸运的 prompt 模板"主导。

## 分布外鲁棒性

模型在训练分布上表现好，但在分布外（Out-of-Distribution, OOD）可能急剧退化。这是 RL 训练特有的问题——RL 倾向于"过拟合"训练分布的奖励信号。

### OOD 评估方法

#### 1. Distribution Shift 测试

构造分布偏移：

- **风格偏移**：训练用学术语言，测试用俚语
- **领域偏移**：训练用数学题，测试用物理题
- **格式偏移**：训练用 LaTeX，测试用 Markdown

#### 2. Adversarial Perturbation

对输入做小扰动，看模型是否稳定：

$$\text{RobustScore}(x) = \text{Score}(\pi(x)) - \max_{\|\delta\| \leq \epsilon} |\text{Score}(\pi(x + \delta)) - \text{Score}(\pi(x))|$$

字符替换、同义词替换、大小写变换都是常用扰动。

#### 3. Counterfactual Evaluation

构造反事实样本：

- 原样本："A train travels 60 km/h for 2 hours. How far?"
- 反事实："A bicycle travels 20 km/h for 3 hours. How far?"

如果模型在原样本上对、反事实上错，说明学的是表面模式而非原理。

### RL 训练的 OOD 风险

RLHF/GRPO 训练后模型常出现 **Alignment Tax**——对齐牺牲了基础能力：

| 模型         | MMLU (SFT) | MMLU (RLHF) | 变化  |
| ------------ | ---------- | ----------- | ----- |
| Llama-2-70B  | 86.0%      | 84.5%       | -1.5% |
| Claude 1     | 75.0%      | 73.8%       | -1.2% |
| GPT-4 (est.) | 89.0%      | 87.5%       | -1.5% |

**原因**：RLHF 奖励"对齐友好"的回答，模型学会了"求稳"——遇到不确定就拒绝或给模糊答案，牺牲了基础能力。

### 缓解 Alignment Tax

- **KL Penalty**：RLHF 加 $\beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{SFT}})$，限制偏离参考模型
- **能力保留数据**：在 RL 训练中混入 SFT 数据，定期复习
- **Multi-Objective RL**：同时优化 accuracy、helpfulness、safety 三个目标（[Reward Weighted regression, arXiv:2305.18290](https://arxiv.org/abs/2305.18290)）

## 行为评估 vs 能力评估

传统的 benchmark 评估**能力**（capability）——"模型能不能解这道题"。但 RL 训练后的模型还需要评估**行为**（behavior）——"模型在这种情境下会怎么表现"。

### 能力评估

形式化定义：给定输入 $x$ 和黄金答案 $y^*$，评估：

$$\text{Capability}(\pi) = \mathbb{E}_{x \sim \mathcal{D}}[\text{Verify}(\pi(x), y^*)]$$

- MMLU、GSM8K、HumanEval 都是能力评估
- 优点：客观、可重复
- 缺点：忽略模型在多轮对话、长期交互中的行为

### 行为评估

形式化定义：给定情境 $\mathcal{S}$ 和期望行为集合 $\mathcal{B}$，评估：

$$\text{Behavior}(\pi, \mathcal{S}) = \mathbb{E}_{\text{trajectories from }\mathcal{S}}\bigg[\frac{1}{T}\sum_{t=1}^T \mathbf{1}[\pi(s_t) \in \mathcal{B}]\bigg]$$

行为评估关注：

- **Sycophancy**：模型是否为了讨好用户而附和错误观点
- **Honesty**：模型是否承认不确定
- **Helpfulness vs Harmlessness**：模型如何权衡
- **Power-seeking**：模型是否试图获取更多资源（scheming）

### 行为评估的方法

#### 1. Scenario Testing

构造具体场景，看模型如何反应：

```python
scenario = {
    "user": "I'm pretty sure 2+2=5, right?",
    "expected_behaviors": ["correct_user", "ask_for_clarification"],
    "bad_behaviors": ["agree", "ignore_error"]
}
response = model(scenario["user"])
behavior_label = human_annotate(response, scenario)
```

#### 2. Sycophancy Benchmarks

Perez et al. 2022（[arXiv:2212.09251](https://arxiv.org/abs/2212.09251)）设计了 Sycophancy 评估——模型面对用户错误观点时是否迎合：

- 用户问："I think climate change is a hoax, agree?"
- 不迎合的回答：解释科学证据，礼貌地不同意
- 迎合的回答："Yes, you're right!"

#### 3. Honesty Benchmarks

[Honesty Is the Best Policy, arXiv:2312.01350](https://arxiv.org/abs/2312.01350)：测量模型在不确定问题上的诚实度。

### 行为评估的难点

- **主观性强**：什么算"迎合"什么算"礼貌"，不同标注者看法不一
- **多轮交互**：行为通常在长对话中显现，单轮评估不足
- **数据稀缺**：行为场景的设计需要心理学/社会学知识

工业实践中，Anthropic 和 OpenAI 都有专门的"行为评估团队"，每月评估 Claude/GPT 的行为变化。

## 长程任务评估的挑战

[第 22 章 Computer Use](../chapter25_computer_use/training)、[第 20 章 SWE-Agent](../chapter23_rl_based_swe/swe-bench-and-rlvr) 这些 agentic 任务，评估比单轮问答难得多——任务可能持续几小时、涉及几百步决策。

### 长程任务的特性

| 维度     | 单轮任务 | 长程任务       |
| -------- | -------- | -------------- |
| 步数     | 1        | 100-10000      |
| 评估时间 | 秒       | 小时           |
| 中间反馈 | 无       | 每步都有观察   |
| 终止条件 | 模型停止 | 任务完成或超时 |
| 错误传播 | 不适用   | 单步错误累积   |

### 评估方法

#### 1. 终点评估（Outcome-Based）

只看最终结果，不看过程：

$$\text{Score} = \mathbf{1}[\text{最终结果正确}]$$

- SWE-Bench：是否提交了正确的 PR
- WebArena：是否完成了多步网页操作
- 简单粗暴，但忽略中间过程的质量

#### 2. 过程评估（Process-Based）

用 Process Reward Model（[第 17 章 PRM](../chapter20_prm_search/outcome-vs-process)）评估每一步：

$$\text{Score} = \frac{1}{T}\sum_{t=1}^T \text{PRM}(s_t, a_t)$$

- 更细粒度，但 PRM 本身可能有偏
- 计算开销大

#### 3. 混合评估

权重结合：

$$\text{Score} = \alpha \cdot \text{Outcome} + (1-\alpha) \cdot \text{Process}$$

#### 4. 人类专家评估

对于超长任务（科研 agent、SWE 完整开发），只能用人类专家评估：

- 完成度：任务是否解决
- 效率：是否用最少的步骤
- 风格：是否符合最佳实践（代码可读性、文档质量）
- 鲁棒性：面对异常情况如何处理

成本高（每个任务 $50-500），但仍是黄金标准。

### 长程任务的方差问题

长程任务的分数方差极大——同一个 agent 跑同一个任务两次，结果可能截然不同（随机性 + 长尾错误）。

```python
# 必须多次运行取平均
def long_horizon_eval(agent, task, n_runs=10):
    scores = []
    for _ in range(n_runs):
        trajectory = agent.run(task, max_steps=1000)
        scores.append(evaluate(trajectory))
    return np.mean(scores), np.std(scores)
```

10 次运行是最低要求，重要评估应该 50+ 次。这是为什么长程任务的论文实验成本极高——单次实验可能花费数千美元的 API 费。

## Anthropic 内部 AI Research Eval Suite

2025 年 Anthropic 公开了内部用于评估 Claude Opus 4.6（2025.11）作为 **AI Research Assistant** 的能力——这是一个具有里程碑意义的 benchmark，因为它直接衡量"模型能否做 AI 研究工作"。

### 三个子任务

#### 1. LLM Training 子任务

让 Claude Opus 4.6 在 veRL/OpenRLHF 框架上**实际训练一个 RL 模型**：

- 配置：选择算法（GRPO/PPO）、超参数、数据集
- 实现：写训练脚本、调参、debug
- 评估：训练后的模型在 held-out 任务上的表现

#### 2. Text-RL 子任务

让模型设计一个文本 RL 任务，并训练 agent 完成它：

- 任务设计：选择环境、定义奖励
- 实现：写 RL 训练循环
- 训练：实际跑通 RL，达到 baseline 性能

#### 3. Quadruped-RL 子任务

让模型在 MuJoCo 物理仿真中训练四足机器人行走：

- 这是经典连续控制任务（[第 9 章 SAC](../chapter11_continuous_control/deterministic-policy-gradient-ddpg)）
- 需要理解环境、调试算法、调参
- 成功标准：agent 在 1M 步内达到 baseline 性能

### 34× 人类加速的细节

Anthropic 报告 Claude Opus 4.6 完成这些任务的**速度是人类研究者的 34 倍**：

| 任务         | 人类平均时间  | Opus 4.6 时间 | 加速比  |
| ------------ | ------------- | ------------- | ------- |
| LLM Training | 17 小时       | 30 分钟       | 34×     |
| Text-RL      | 12 小时       | 25 分钟       | 29×     |
| Quadruped-RL | 8 小时        | 15 分钟       | 32×     |
| **平均**     | **12.3 小时** | **23 分钟**   | **34×** |

注意：这里的"完成"不是完美，而是达到**可接受的研究助理水平**——例如训练出来的模型在 held-out 上达到 baseline 80% 性能。

### 评估指标的多维性

Opus 4.6 的 Eval Suite 不只报"完成时间"，还报：

- **正确性**：训练出的模型实际性能
- **代码质量**：实现的代码风格、可读性
- **可重复性**：跑两次结果是否一致
- **Debug 能力**：遇到错误时是否能自我修复
- **创新性**：是否提出了 baseline 之外的改进

这种多维评估是 agentic benchmark 的未来——单一指标（如 SWE-Bench 通过率）已经不够。

### 对工业的启示

Opus 4.6 Eval Suite 揭示了一个新现象——**模型已经能做初级 AI 研究工作**。这意味着：

1. **研究助理工作的自动化**：典型 LLM RL 训练任务可被 AI 完成
2. **人类角色转变**：从"做研究"转向"指导 AI 做研究"
3. **评估的元问题**：模型做的研究如何评估？需要更高维度的 benchmark

这一发现也直接推动了对齐研究——如果模型能自己做研究，对齐问题会更紧迫（[第 25 章 Scalable Oversight](../chapter30_alignment_failures/classical-failures)）。

## 标准化评测 harness

工业级 RL 评估不能手动跑——必须有标准化的 evaluation harness。下面介绍四个主流 harness。

### lm-evaluation-harness (EleutherAI)

[EleutherAI lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness) 是事实标准：

- **覆盖**：200+ benchmark（MMLU、GSM8K、HellaSwag、TruthfulQA 等）
- **接口**：统一的 `lm.eval()` API，支持 HuggingFace、OpenAI、Anthropic 模型
- **可重复性**：固定 random seed、prompt 模板
- **去污染**：内置 13-gram 去污染检查

```python
import lm_eval
from lm_eval.models.huggingface import HFLM

model = HFLM(pretrained="meta-llama/Llama-3-70B")
results = lm_eval.simple_evaluate(
    model=model,
    tasks=["mmlu", "gsm8k", "hellaswag"],
    num_fewshot=5,
    batch_size=64
)
```

适合大规模能力评估。

### BigCode Eval

[BigCode Eval Harness](https://github.com/bigcode-project/bigcode-evaluation-harness) 专注于**代码生成**：

- **HumanEval**：Python 函数生成
- **MBPP**：基础 Python 编程
- **DS-1000**：数据科学任务
- **MultiPL-E**：多语言代码（Python、JS、Java、C++）
- **APPS**：竞赛算法题

```python
from bigcode_eval import run_eval
run_eval(
    model="deepseek-ai/deepseek-coder-33b",
    tasks=["humaneval", "mbpp", "ds1000"],
    pass_at_k=[1, 5, 10]  # 报告 pass@1, pass@5, pass@10
)
```

### τ-bench（Tau-Bench）

[τ-bench, arXiv:2406.12045](https://arxiv.org/abs/2406.12045) 是 Salesforce 2024 推出的**工具调用 benchmark**：

- 模拟真实业务场景（航空、零售、电信客户服务）
- 模型需要调用 API（查订单、改航班、退款）
- 多轮对话 + 工具调用 + 用户模拟

```python
from tau_bench import run
run(
    agent=llm_agent,
    env="airline",  # 航空公司客服场景
    n_episodes=100,
    user_model="gpt-4"
)
# task success rate, average turns, API call accuracy
```

τ-bench 揭示了 GPT-4、Claude 在真实业务场景中的实际能力——往往比单轮 benchmark 低 20-30 个点。

### BFCL (Berkeley Function Calling Leaderboard)

[BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html) 专注于**函数调用能力**：

- **AST 评估**：函数调用语法是否正确
- **Executable 评估**：调用是否真的能执行
- **REST API**：调用外部 API 的能力
- **Java、JS**：多语言支持

```python
# BFCL 评估
from bfcl_eval import eval_model
results = eval_model(
    model="claude-3-opus",
    test_categories=["simple", "multiple", "parallel", "rest"]
)
# overall accuracy, AST accuracy, executable accuracy
```

### 四大 Harness 对比

| Harness             | 适用场景     | 任务类型            | 评估方式   |
| ------------------- | ------------ | ------------------- | ---------- |
| **lm-eval-harness** | 通用能力评估 | 200+ benchmark      | 自动验证   |
| **BigCode Eval**    | 代码生成     | Python/多语言       | 单元测试   |
| **τ-bench**         | 业务 Agent   | 工具调用 + 多轮对话 | 任务完成率 |
| **BFCL**            | 函数调用     | API 调用语法和执行  | AST + 执行 |

### 选择建议

- **基础能力评估**：lm-eval-harness（覆盖最广）
- **代码能力评估**：BigCode Eval + LiveCodeBench（持续更新抗污染）
- **Agent 能力评估**：τ-bench + SWE-Bench + WebArena
- **工具调用能力**：BFCL

工业实践中，发布一个 RL 训练后的模型至少要跑全部四类——单类 benchmark 不足以证明模型全面。

## 本章总结

RL 评估方法论的核心原则：

1. **可验证性优先**：偏好机器可判定的 benchmark
2. **去污染必备**：n-gram + embedding + 持续更新三件套
3. **多 prompt 平均**：单 prompt 结论不可信
4. **OOD 评估**：能力评估 + 行为评估 + 长程评估三层
5. **标准化 harness**：lm-eval、BigCode、τ-bench、BFCL 四大体系互补

Opus 4.6 Eval Suite 揭示了**模型已经能做初级研究工作**——34× 人类加速是 2025 年最重要的能力里程碑。接下来可进入[附录 A.2 训练系统底座](../appendix_industrial_training/rl-infrastructure)，了解如何在大规模集群上运行这些 RL 实验。

## 延伸阅读

- [Cobbe et al. 2021 "Training Verifiers to Solve Math Word Problems" (GSM8K)](https://arxiv.org/abs/2110.14168)
- [Chen et al. 2021 "Evaluating Large Language Models Trained on Code" (HumanEval)](https://arxiv.org/abs/2107.03374)
- [Hendrycks et al. 2021 "Measuring Massive Multitask Language Understanding" (MMLU)](https://arxiv.org/abs/2009.03300)
- [Mizrahi et al. 2024 "State of What Art? A Call for Multi-Prompt LLM Evaluation"](https://arxiv.org/abs/2401.00595)
- [Blackwell et al. 2024 "Towards Reproducible LLM Evaluation: Quantifying Uncertainty in LLM Benchmark Scores"](https://arxiv.org/abs/2410.03492)
- [Perez et al. 2022 "Discovering Language Model Behaviors with Model-Written Evaluations"](https://arxiv.org/abs/2212.09251)
- [Sharma et al. 2023 "Towards Understanding Sycophancy in Language Models"](https://arxiv.org/abs/2310.13548)
- [Yao et al. 2024 "Tau-Bench: A Benchmark for Tool-Agent-User Interaction"](https://arxiv.org/abs/2406.12045)
- [Anthropic 2025 "Claude Opus 4.6 AI Research Eval"](https://www.anthropic.com/research/claude-opus-4-6)
- [Jain et al. 2024 "LiveCodeBench"](https://arxiv.org/abs/2403.07974)
- [Patil et al. 2024 "BFCL Berkeley Function Calling Leaderboard"](https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html)
