# 18.3 Training Stability

[18.2](./industrial-post-training) introduces data preparation, sampling, reward calculation, and model updates. After the training process truly begins, the first thing to do is usually to look at the reward curve: if the reward keeps increasing, it seems to indicate that the model is making progress.

Suppose the training reward of a mathematical model increases from 0.4 to 0.7, but the accuracy on an independent test set remains unchanged, while the average length of the answers doubles. The model may have learned that "writing longer answers leads to higher scores." Another scenario is more direct: after an update, the loss and gradients suddenly become NaN, and the parameters after that point are no longer usable. Some issues arise from the system itself—such as the version of the model that generates answers being out of sync with the training side, and the recorded probabilities not matching the current policy.

These issues will be reflected in different curves. Looking only at the reward curve cannot distinguish whether the model has truly learned to solve problems, exploited a reward loophole, or whether the parameter updates have already gone wrong. Therefore, during training, it is necessary to compare the loss, gradient norms, KL divergence, entropy, and independent evaluations on the same timeline.

## First Use Four Categories of Signals to Locate the Fault

Place the curves on the same timeline to first determine the source of the anomaly, and then choose the appropriate tools. There is a clear causal order between the four layers: data and rewards determine what the model learns, strategy indicators describe how the model changes, numerical indicators reflect whether the parameters can be updated normally, and the training system determines whether the probabilities in the logs are truly from the policy that generates the answers.

| Layer            | Main Signals                                                 | First Check What                                                            |
| ---------------- | ------------------------------------------------------------ | --------------------------------------------------------------------------- |
| Data & Reward    | Reward increases, independent evaluation remains unchanged   | Reward rules, data duplication, evaluation contamination, and answer length |
| Strategy Change  | KL divergence rapidly increases or entropy rapidly decreases | Update magnitude, KL constraints, sampling difficulty, and strategy version |
| Numerical Update | Loss, gradient norms explode or become NaN                   | Learning rate, computational precision, abnormal batch, and clipping        |
| Training System  | Rollout and training-side probabilities, versions mismatch   | Weight synchronization, token, precision, operator, and MoE routing         |

## 1. Trustworthiness of Data and Rewards

First, examine whether the training rewards and independent evaluations improve together. If the reward increases and the average response length also increases, while the independent test set accuracy remains unchanged, this suggests that the model has learned a form of output that is easier to obtain rewards, rather than improving its goal-oriented capability. The model may have simply memorized the training tasks, or it may have exploited vulnerabilities in the answer parser and test procedures.

At this level, we first check the training samples, answer parser, test environment, and reward direction, then check for training set repetition and evaluation contamination. Once the goal signal is wrong, the more stable the subsequent optimization becomes, the more the model will consistently deviate from the true goal.

## 2. Whether the Strategy is Updating Too Quickly

After ensuring the trustworthiness of data and rewards, we then observe KL divergence and entropy. KL divergence measures how far the current model's output distribution is from the reference model's; a rapid increase indicates that one or multiple updates have pushed the model out of its original capability range. Entropy represents how dispersed the output distribution is; a rapid decrease suggests that a few tokens or response patterns have taken up most of the probability, and the model is prematurely stopping exploration.

At this point, we should compare the old policy during sampling with the updated new policy, and check whether the learning rate, clipping range, KL constraints, and experience have become outdated. When the task is too simple, the model may quickly concentrate on a few high-reward answers, so the difficulty of the data should be considered together with the strategy metrics.

## 3. Whether Parameters Can Be Updated Normally

Loss reflects the change in the current training objective, and the gradient norm reflects how far this step is going to push the parameters. A sudden spike in either, or the direct appearance of NaN, indicates a numerical issue in the forward, backward, or parameter update step. When troubleshooting, first fix a batch and run it repeatedly, then check the learning rate, low-precision computation, gradient clipping, abnormal samples, and optimizer state.

AdamW, Muon, and various clipping methods all operate at this level: they convert gradients into more controllable parameter updates. Optimizers cannot fix erroneous rewards or corrupted data, nor can they allow the generative end and training end to automatically use the same model version. Therefore, when seeing a loss spike, we can check the optimizer; when seeing a separation between rewards and evaluations, continuing to switch optimizers is meaningless.

## 4. Is the System Training the Same Policy?

When the first three layers are functioning correctly, the final step is to verify the alignment between the rollout engine and the training engine. The generation side may use an earlier model version, FP8 precision, and a set of inference operators, while the training side uses the updated weights, BF16/FP32 precision, and another set of operators. Even if both sides read the same parameters, the log probability of tokens may still differ; MoE models are further affected by differences in expert routing.

At this level, we must align the model versions, tokens, log probabilities, computational precision, and expert routing to confirm that the old policy described in the training logs is indeed the policy that generates the trajectories. [18.4](./distributed-sync) will continue to elaborate on weight synchronization and training-inference consistency.

The four-layer troubleshooting must be carried out in sequence: first, confirm that the model is learning the correct objective, then check whether the policy is moving too fast, then inspect the per-step updates, and finally verify the distributed system. This ensures that we avoid using numerical tools to fix reward issues or using data cleaning to mask weight synchronization errors.

## 5. Understanding the Four-Layer Failures with Public Cases

Below are the complete public cases for GLM, Llama, Seed, and Kimi. These are not four consecutive training steps, but rather illustrate different issues within the four layers: GLM demonstrates the system constraints of MoE and multi-stage training; Llama 4 shows why evaluations must also change after architectural changes; Seed-Thinking illustrates how data difficulty affects the reward signal; and Kimi K2 directly addresses anomalies in parameter updates and attention calculations.

| Case              | Key Observation                                           | Corresponding Instability Signal                            |
| ----------------- | --------------------------------------------------------- | ----------------------------------------------------------- |
| GLM-4.5 / GLM-4.6 | MoE routing, multi-stage training, and mode switching     | Expert load imbalance, capability regression between stages |
| Llama 4           | Multimodal, long context, and evaluation version          | Training scores inconsistent with real task performance     |
| Seed-Thinking     | Data difficulty, curriculum learning, and self-validation | Reward sparsity, group internal advantage close to zero     |
| Kimi K2           | Optimizer update and attention score                      | Loss spike, gradient or attention value anomalies           |

## 6. GLM: How Multi-Stage Training Affects System Stability

[GLM-4.5](https://github.com/zai-org/GLM-4.5) (Zai AI, released in July 2025) and GLM-4.6 (released in October 2025) both adopt multi-stage training. This case is suitable for observing how MoE, reasoning RL, general RLHF, and dual-mode output can be integrated into a single training pipeline.

### Model Architecture and Capability Alignment

First, let's examine the constraints that the model itself imposes on subsequent training:

- **MoE Architecture**: GLM-4.5 has a total of 355B parameters, with 32B parameters activated during each forward pass. RL updates also need to pay attention to expert load and routing stability.
- **Thinking and Non-Thinking Dual Modes**: The same model needs to learn when to perform reasoning and when to directly answer.
- **Open Data**: Public weights, training methods, and some data are available, providing an entry point for reproducing experiments.
- **Code and Agent Capabilities**: Training data must cover code generation, tool invocation, and multi-step execution.

### Multi-Stage Training of GLM-4.5

```text
┌──────────────────────────────────────────────────────────┐
│ Phase 1: Base Pre-training (MoE Architecture)             │
│   - 15T tokens of high-quality data                      │
│   - MoE: 355B total / 32B active                         │
│   - RoPE scaling supports long context                   │
├──────────────────────────────────────────────────────────┤
│ Phase 2: General SFT                                       │
│   - Multilingual dialogue data                           │
│   - Training on tool invocation format                   │
├──────────────────────────────────────────────────────────┤
│ Phase 3: Reasoning RL                                     │
│   - Math, code, and reasoning tasks                      │
│   - GRPO + Rule-based Rewards                            │
│   - Integration of Self-validation                       │
├──────────────────────────────────────────────────────────┤
│ Phase 4: General RLHF                                     │
│   - Dialogue quality and safety                          │
│   - Helpfulness / Harmlessness dual objectives           │
├──────────────────────────────────────────────────────────┤
│ Phase 5: Unified Thinking / Non-Thinking                 │
│   - Mixed data SFT                                        │
│   - Let the model learn to switch modes                  │
└──────────────────────────────────────────────────────────┘
```

This process first uses SFT to establish basic behavior, then employs reasoning RL to improve accuracy on verifiable tasks, followed by general RLHF to refine dialogue quality and safety, and finally unifies the two response modes. [The training process of DeepSeek-R1](../chapter18_grpo/deepseek-dapo) adopts a similar stage division.

### Training Improvements for GLM-4.6

GLM-4.6 continues to refine the reasoning length, tool usage, and mode control:

- **Longer Thinking**: Supports reasoning trajectories of 100K+ tokens.
- **More Agent Tools**: Covers search, code execution, and file operations.
- **Multimodal Coordination**: Works in conjunction with the GLM-4.5V visual model.
- **Finer Thinking Budget**: Users can control the reasoning budget.

### Benchmark Results for GLM-4.6

| Benchmark     | GLM-4.5 | GLM-4.6 |
| ------------- | ------- | ------- |
| AIME 2025     | 75.3    | 83.6    |
| MATH-500      | 92.1    | 95.4    |
| LiveCodeBench | 56.2    | 62.7    |
| GPQA Diamond  | 68.5    | 72.4    |

These benchmarks evaluate mathematical, coding, and scientific reasoning capabilities. When reading the results, one should also consider the specific evaluation settings, and not judge the training stability solely based on a single average score.

### Engineering Insights from GLM

This case provides three engineering insights:

1. **MoE and Inference RL should be jointly debugged.** In addition to rewards and KL divergence, it is also necessary to monitor expert load, routing, and cross-card communication.
2. **Dual modes require separate evaluation.** The accuracy, length, and cost of the Thinking mode cannot substitute for the response quality of the Non-Thinking mode.
3. **Code and tool tasks require real execution.** Static answer scores cannot cover environmental states, tool returns, and long trajectory failures.

## 7. Llama 4: How to Maintain Evaluation Consistency After Architectural Changes

[Llama 4](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) (Meta, released in April 2025) places MoE, native multimodality, and long context within the same series. It demonstrates how changes in model architecture continue to affect post-training data and evaluation methods.

### Model Series and Parameter Scale

Llama 4 includes three variants:

- **Llama 4 Scout**: 109B total / 17B active (MoE), 10M context
- **Llama 4 Maverick**: 400B total / 17B active, 1M context
- **Llama 4 Behemoth** (unreleased): 2T total / 288B active, trained internally by Meta

### Architecture and Training Improvements

**Native Multimodality.**

Llama 4 processes text and image tokens simultaneously from the pretraining stage. As a result, post-training requires the simultaneous preparation of text tasks, multimodal tasks, and cross-modal consistency evaluations.

**Early Fusion.**

Early Fusion allows text and image information to interact at earlier layers of the model. Multimodal rewards must also assess whether the response genuinely utilizes image evidence.

**MoE Architecture.**

Llama 4 employs MoE across its entire series. Each token is processed by only a subset of experts, enabling an increase in total parameters while keeping the activation parameters during a single forward pass at a low level; the training system must also handle additional responsibilities such as expert routing and communication.

**Long Context.**

Llama 4 Scout supports a context of up to 10M tokens and utilizes iRoPE (interleaved RoPE) and sparse attention. Long-context evaluations must simultaneously check evidence retrieval, answer accuracy, and inference cost.

### Multi-Stage Training Methods

Meta has not publicly disclosed the full training details. The following stage divisions are based on publicly available papers and blogs, and the content is inferred and should be distinguished from official disclosures:

```text
┌─────────────────────────────────────────────────────────┐
│ Phase 1: Multimodal Pretraining                         │
│   - Joint training of text, image, and video            │
│   - Level of 22T tokens (estimated)                     │
│   - Early fusion architecture                           │
├─────────────────────────────────────────────────────────┤
│ Phase 2: Mid-training (Intermediate SFT)                │
│   - General instruction following                       │
│   - Tool call format                                    │
├─────────────────────────────────────────────────────────┤
│ Phase 3: Post-training RL                                │
│   - Hybrid of RLHF + RLVR                              │
│   - Multi-objective: Helpfulness / Safety / Reasoning   │
└─────────────────────────────────────────────────────────┘
```

### Public Evaluation and Controversy

The public scores of Llama 4 and the actual user experience have sometimes diverged, with the main issues concentrated in two areas.

**Benchmark Scores versus Real-World Task Performance.**

Llama 4 Maverick achieved high scores on multiple benchmarks, but users found it to be less effective in practice compared to Claude 3.5 / GPT-5. Meta later acknowledged that there was a gap between benchmark evaluations and real-world experience.

**Evaluation Version versus Open-Source Version.**

The Maverick version running on LM Arena is a **specialized optimized version** — it uses adjusted chat templates and prompt engineering. The open-source Maverick version differs from the Arena version.

Such version differences highlight the necessity of recording the weight version, chat templates, system prompts, and sampling parameters when comparing models. [Modern Model Failures](../chapter30_alignment_failures/modern-incidents) also discuss evaluation risks such as data contamination.

### Engineering Insights from Llama 4

From the perspective of training systems, this case leaves three reusable lessons:

1. **MoE requires independent routing monitoring.** Even when the total loss is normal, local expert load imbalance can still occur.
2. **Early Fusion changes the training samples.** Text and images must form a verifiable correspondence within the same task.
3. **Long context increases the evaluation dimensions.** In addition to whether the model can accommodate the input, it must also check whether the model can find evidence and control the generation cost.

## 8. Seed-Thinking: How Data Difficulty Affects Policy Signals

[Seed1.5-Thinking](https://arxiv.org/abs/2504.13914) (Byte Seed, April 2025) combines data curation, policy optimization, self-verification, and curriculum learning into a single reasoning training pipeline. Below, we explain the role of each component in sequence.

### Core Components of the Training Framework

Seed-Thinking focuses on combining multiple existing components and enabling them to collaborate within the same training process.

**Data Curation.**

```text
Mathematical Data:
  - High-quality math problems (AIME, Putnam historical problems)
  - Automatically generated problems (using strong LLMs to create new problems)
  - Difficulty grading (based on the pass rate of the base model)

Code Data:
  - Codeforces problems (with test cases)
  - SWE-bench / SWE-smith (with PR data)
  - Function generation (extension of HumanEval)
```

**Improvements to GRPO and DAPO.**

Seed-Thinking incorporates four engineering improvements from [DAPO](../chapter18_grpo/deepseek-dapo) plus some new enhancements:

- **Dynamic KL:** Strong KL in the early training phase, weakened later
- **Adaptive Clip:** Adjust clip range based on training progress
- **Group Size Scheduling:** Large groups early, small groups later

**Self-Verification.**

The model performs self-verification after generating an answer:

```python
def self_verification_reward(response, ground_truth):
    answer = extract_answer(response)

    # Let the model re-read the question and verify the answer
    verification_prompt = f"Check if this answer is correct: {answer}"
    verification = model.generate(verification_prompt)

    if "correct" in verification and answer == ground_truth:
        return 1.0  # Answer is correct and verification passes
    elif "incorrect" in verification and answer != ground_truth:
        return 0.5  # Answer is incorrect but the model detects the error
    else:
        return 0.0  # Answer is incorrect and the model fails to detect the error
```

This reward simultaneously checks the final answer against the self-verification result. The model retains partial signals even when it answers incorrectly but can identify the error, thereby ensuring the training objective covers both the "answering" and "checking" steps.

**Curriculum Learning.**

Training data is sorted by difficulty, allowing the current model to first gain effective rewards on easier tasks before gradually increasing the difficulty. This approach reduces the issue of all samples in the initial training phase failing, leading to near-zero advantage signals.

### Public Evaluation Results

Seed-Thinking 1.5 achieves the following scores on multiple benchmarks:

| Benchmark         | Score |
| ----------------- | ----- |
| AIME 2024         | 86.4% |
| MATH-500          | 96.2% |
| GPQA Diamond      | 75.1% |
| Codeforces Rating | 1822  |

These results reflect the performance of the entire training setup across mathematical, scientific reasoning, and coding tasks. Further product deployment requires continued checks on answer length, inference latency, and the regression of general capabilities.

## 9. Kimi K2: How to Limit Abnormal Parameter Updates

[Kimi K2](https://arxiv.org/abs/2507.20534) (Moonshot, July 2025) simultaneously uses MuonClip and QK-clip to control abnormal updates during training. The former acts on parameter updates, while the latter acts on attention scores.

### Muon Optimizer

[Muon](https://kellerjordan.github.io/posts/muon/) (February 2025) combines momentum with orthogonalization:

- **Momentum**: Accumulates directional information across consecutive updates.
- **Orthogonalization**: Orthogonalizes the update matrix.

Orthogonalization adjusts the singular values of the update matrix, limiting certain directions from being overly amplified. It addresses the shape of updates at the optimizer level.

### MuonClip Update Constraints

The following simplified code illustrates where clipping occurs:

```python
def muon_clip_update(grad, momentum, clip_threshold=1.0):
    # Muon main process
    momentum = beta * momentum + (1 - beta) * grad
    orthogonalized = orthogonalize(momentum)

    # Clip to prevent explosion
    norm = torch.norm(orthogonalized)
    if norm > clip_threshold:
        orthogonalized = orthogonalized * (clip_threshold / norm)

    return -lr * orthogonalized
```

Clipping restricts the norm of the single-step update, reducing abrupt parameter changes caused by abnormal batches. It still needs to be used in conjunction with learning rate, gradient monitoring, and data checks.

### QK-clip and Attention Stability

QK-clip directly restricts the range of $QK^\top$ in attention:

```python
def attention_with_qk_clip(Q, K, V, clip_value=30.0):
    # Standard attention
    scores = Q @ K.T / sqrt(d)

    # QK-clip: Prevent attention scores from becoming too large
    scores = torch.clamp(scores, min=-clip_value, max=clip_value)

    # Softmax + weighted sum
    attn = softmax(scores)
    output = attn @ V

    return output
```

In long-context scenarios, attention scores may continuously increase, causing the Softmax to concentrate excessively on a few tokens and amplify numerical errors. QK-clip limits the score range before entering the Softmax, thereby reducing outliers at the attention computation layer.

### Training Performance

Public results report the effectiveness of this combination in terms of stability, speed, and final performance:

- **Training Stability**: Loss spikes decrease from occurring on average once every 1T tokens to once every 10T tokens.
- **Training Speed**: Compared to Adam, there is an improvement of about 15%.
- **Final Performance**: Kimi K2 achieves high results on multiple public benchmarks.

### Engineering Insights from MuonClip

This case illustrates that stability tools need to be aligned with specific failure layers:

- **Parameter Update Layer**: MuonClip restricts the norm of abnormal updates.
- **Attention Computation Layer**: QK-clip limits extreme attention scores in long contexts.
- **Engineering Implementation Layer**: Training logs must separately record update norms and attention statistics to determine which layer is effective.

## 10. Putting Public Methods Back into the Four Layers

The previous four cases demonstrate that "training stability methods" do not all operate at the same location. The following table adds other teams' public methods, indicating which layer they primarily address:

| Team              | Representative Model | Public Method               | Main Target Layer                                   |
| ----------------- | -------------------- | --------------------------- | --------------------------------------------------- |
| **DeepSeek**      | R1, V3.2             | GRPO and variants           | Policy variation                                    |
| **Ali Qwen**      | Qwen3 series         | GSPO                        | Policy variation, MoE system                        |
| **Byte Seed**     | Doupan Pro, Seedance | DAPO, VAPO                  | Data and reward, policy variation                   |
| **Moonshot Kimi** | K2, K2.5             | GRPO, MuonClip              | Policy variation, numerical update                  |
| **GLM**           | GLM-4.6              | GSPO-style                  | Policy variation, MoE system                        |
| **MiniMax**       | M1, M2               | CISPO                       | Policy variation, low-precision numeric computation |
| **StepFun**       | Step3                | Multimodal training methods | Data and reward, multimodal evaluation system       |

These names correspond to different issues. GRPO, GSPO, DAPO, CISPO, and VAPO primarily adjust policy objectives, advantage estimation, or clipping methods, while MuonClip addresses optimizer updates. When comparing methods, one should first confirm which layer they target and then determine whether they can be combined.

## 11. What to Check After Scaling Model Size

### Super Large Models and Long Context

- The total number of parameters and the activated parameters continue to increase, making MoE routing and cross-device communication part of the stability metrics.
- The requirement of 10M+ token context necessitates controlling memory usage, attention values, and the cost of long trajectories during training.
- While optimizer clipping can handle parameter updates, it still needs to be combined with data, reward, and system monitoring.

### Native Multimodal RL

- Text, images, and environment actions enter the same training trajectory, and rewards must also check cross-modal evidence.
- Llama 4's Early Fusion demonstrates a way to unify modalities from the pre-training stage.
- Multimodal RL requires replayable images, videos, and interactive environments.

### Industrialization of Agentic RL

- Training tasks expand from software engineering to customer service, research, and computer operations.
- Agent trajectories need to record tool parameters, environment returns, and intermediate states.
- Trajectories of varying lengths increase the cost of environment scheduling, fault recovery, and asynchronous training.

### Training Cost and Efficiency

- Training cost is composed of pre-training, data generation, Rollout, model updates, and evaluation.
- Higher generation throughput, smaller activated parameters, and more accurate data screening can reduce unnecessary computation.
- Small teams can first reproduce a complete closed-loop on a verifiable small task before deciding whether to scale the model and cluster size.

## 12. How the Four Cases Connect Back to the Troubleshooting Order

The four cases fall into different levels:

- **GLM-4.6 and Llama 4**: MoE, multimodal, and long context change the constraints of the training system.
- **Seed-Thinking**: Data organization, strategy optimization, self-validation, and curriculum learning form the reasoning training process.
- **MuonClip and QK-clip**: Parameter updates and attention computation require separate control of outliers.
- **Public Methods from Different Teams**: Algorithm goals, optimizers, and system frameworks need to be compared according to their hierarchical roles.

These cases illustrate how model architecture, reasoning training, and numerical stability affect large-scale RL training.

Related Chapters:

- [Chapter 16: Reasoning Models](../chapter19_reasoning/r1-zero-pure-rl-reasoning) — Detailed discussion of reasoning models
- [Chapter 17: PRM](../chapter20_prm_search/outcome-vs-process) — Industrial practice of process rewards
- [Chapter 20: RL-based SWE](../chapter23_rl_based_swe/swe-bench-and-rlvr) — Training of code agents

## 13. Summary of This Section

Training stability requires simultaneous observation of data and rewards, policy changes, numerical updates, and the training system. When troubleshooting, follow this order: first confirm that the training objective is trustworthy, then check whether KL divergence and entropy are abnormal, followed by examining the loss, gradients, and optimizer, and finally verifying the model versions and probabilities at both the generation and training ends. Optimizers and clipping only control parameter updates and cannot fix issues related to data, rewards, or weight synchronization.

[18.4](./distributed-sync) will continue to explore the last layer of issues: when generation, rewards, and training are distributed across multiple GPUs, how does the system ensure that data and model versions flow correctly.
