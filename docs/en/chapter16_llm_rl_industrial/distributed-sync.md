# 18.4 Distributed RL Training

[18.3](./modern-industrial-practice) has already explained that the training system must ensure consistency in sampling strategies, model versions, and numerical computations. Now, let's consider distributing a training session across multiple GPUs and examine how data flows.

Suppose the rollout GPU generates a batch of responses. The reward process reads these responses and computes scores, and the training GPU then uses these responses to update the Actor. After the update is complete, the new parameters must be synchronized back to the rollout GPU so that the next batch of responses can come from the latest policy:

```text
Rollout GPUs generate responses
        ↓
Reward workers compute scores
        ↓
Training GPUs update the actor
        ↓
Synchronize new weights to rollout GPUs
        └──────────────► next response batch
```

In a single-machine code, the four functions become data transmission and waiting between different processes in a multi-machine system. The system design primarily addresses three issues:

1. **Where to place the model.** The Actor, Reference, Reward Model, and rollout engine may not all fit into a group of GPUs simultaneously.

2. **Who is waiting for whom.** The time required for generating responses, computing rewards, and updating parameters varies, and the slowest step can lead to idle time on other devices.

3. **When to exchange data and parameters.** Frequent synchronization increases communication overhead, while long intervals may result in the rollout using outdated policies.

## Three System Decisions in Distributed RL

| System Question                                    | Common Choices                                 | Main Trade-offs                                                                                                |
| -------------------------------------------------- | ---------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| Is training and generation shared on GPU?          | Shared Deployment / Separated Deployment       | Shared deployment saves GPU resources; separated deployment reduces switching and improves parallel throughput |
| Should we wait for the entire rollout to complete? | Synchronous / Asynchronous                     | Synchronous training keeps data fresh; asynchronous training reduces waiting but may produce stale experiences |
| How to split a model across multiple GPUs?         | FSDP, Tensor Parallelism, Pipeline Parallelism | Allocates memory, but increases communication and scheduling complexity                                        |

Make choices based on bottlenecks: first address model partitioning when memory is insufficient; then consider separating generation and training or using asynchronous methods when GPUs are idling; when using MoE, also handle token routing and communication between experts.

::: tip Read this section on first pass
Remember the data order: **Generation → Reward → Update → Synchronize Parameters**. veRL, slime, and OpenRLHF differ in how they assign these four steps to GPUs and when they exchange data.
:::

## 1. Ensuring the Generation Side and Training Side Use the Same Policy

Parameter synchronization addresses the issue of model version mismatches. Once the rollout GPU and training GPU obtain the same set of weights, they must go through their respective computation engines to derive the token probabilities. The inference side typically uses vLLM or SGLang, and employs KV Cache and low-precision computation. The training side commonly uses FSDP or Megatron, retaining the computational graph required for backpropagation. These two computational paths differ, and the resulting probabilities may also differ.

Let $\pi_{\text{rollout}}$ denote the policy executed by the rollout engine and $\pi_{\text{old}}$ the policy recorded by the trainer. They should match. Different model versions, numerical precision, MoE routing, or incorrect log-probability recomputation can make them diverge; this is a **training-inference mismatch**.

### 1.1 Sources of Discrepancy Between Generation and Training Strategies

- On the rollout side, vLLM or SGLang are typically used to generate responses with FP8/BF16 precision, and KV Cache optimization is enabled.
- On the training side, FSDP or Megatron are typically used to compute log-probabilities and gradients in BF16/FP32 precision, and activation recomputation may be enabled.

Although the model weights are the same, the two sides can only ensure they start from the same set of parameters. The calculation precision, operator implementation, and MoE expert routing will still alter the log-probability of tokens. Small errors in high-probability tokens typically have limited impact; however, the large number of low-probability tokens may lead to significant changes in gradient estimation after error accumulation.

### 1.2 Applicability Boundaries of PPO Clipping

PPO uses the following constraint on the importance sampling ratio to limit the magnitude of a single update:

$$
\mathcal{L}^{\text{CLIP}} = \mathbb{E}\left[\min\left( r_t(\theta) \hat{A}_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right)\right],
$$

where

$$
r_t(\theta) = \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\text{old}}(a_t\mid s_t)}.
$$

$\hat A_t$ represents how much better the action $a_t$ is compared to the current average, and $\epsilon$ determines the range of deviation allowed from 1 for the ratio. Suppose the old policy generates a token with probability $0.20$, and the new policy increases it to $0.24$, then $r_t = 0.24 / 0.20 = 1.2$. When $\epsilon = 0.2$, this change just reaches the upper bound of clipping; further increasing the probability will not result in additional optimization benefits for this term.

This calculation relies on a premise: the denominator $\pi_{\text{old}}$ must be the actual policy used to generate the actions. If the response comes from $\pi_{\text{rollout}}$, but the training side recalculates $\pi_{\text{old}}$ using a different path, then $r_t$ becomes inaccurate before the update begins. Clipping can only restrict the parameter update, but cannot correct the discrepancy between the probabilities calculated by the two engines.

### 1.3 Order of Diagnosing Training-Generation Bias

When diagnosing, align the generation and training paths step by step:

1. Verify the model version used for rollout, and confirm that parameter synchronization has been completed.
2. Record the log-probability separately for the rollout side and the training side using the same batch of tokens, and observe where the error is concentrated.
3. Align the floating-point precision and operator implementations on both sides, and compare whether the error has been reduced.
4. For MoE models, additionally record the expert routing, and confirm whether the training side reproduces the routing used during generation.

### 1.4 Methods for Correcting Training-Deployment Discrepancy

- **Unify Computational Precision.** First, replace FP8 on the rollout side with FP16/BF16 to determine whether low-precision computation is a major source of error. If FP8 must be used, bias monitoring and importance sampling correction should be applied simultaneously.
- **Record the True Behavior Policy.** Directly save the log-probability during rollout to prevent the training side from treating recomputed results as the true behavior probabilities.
- **Recalculate and Validate.** Recompute the log-probability using the training engine before training, and compare it token by token with the recorded rollout values. While recalculation itself cannot recover the true behavior policy, it can expose the location and magnitude of discrepancies.
- **Limit Extreme Ratios.** Techniques such as Truncated Importance Sampling (TIS) truncate excessively large importance sampling ratios, thereby reducing the impact of a few abnormal tokens on the gradient.
- **Handle Long-Tail Tokens.** Methods such as dynamic vocabulary pruning filter out the low-probability regions with the largest bias, reducing the accumulation of error in long sequences.
- **Playback MoE Routing.** R3 (Rollout Routing Replay) reproduces the expert selection during rollout in training, reducing the probability bias caused by changes in routing.

### 1.5 Related Work

The following categories of work address this issue from the perspectives of numerical precision, distribution correction, and system scheduling:

- _When Speed Kills Stability: Demystifying RL Collapse from the Training-Inference Mismatch_ (Liu et al., 2025) focuses on analyzing the relationship between training-inference mismatch and training collapse.
- _Defeating the Training-Inference Mismatch via FP16_ (Qi et al., 2025) examines the impact of floating-point precision on the discrepancy of log-probabilities between the two sides.
- _Taming the Tail: Stable LLM Reinforcement Learning via Dynamic Vocabulary Pruning_ (arXiv:2512.23087) addresses the more pronounced bias on low-probability tokens.
- _Stabilizing Reinforcement Learning with LLMs: Formulation and Practices_ (Zheng et al., arXiv:2512.01374) discusses training-inference consistency, policy timeliness, and MoE routing replay.
- FP8-RL (Qiu et al., arXiv:2601.18150) combines W8A8 low-precision training with importance sampling correction in veRL.
- TIS (Yao et al., NeurIPS 2025) and MinPRO (Lei et al., arXiv:2601.22718) restrict the extreme importance sampling ratios that arise after policy shift.
- Dynamic optimization methods (Zhang et al., arXiv:2602.01826) adjust the optimization process based on training signals such as response length.

In a distributed system, the degree of on-policy training depends on the distance between $\pi_{\text{rollout}}$ and the current training policy. Weight synchronization controls the model version, while numerical alignment, recorded probabilities, and importance-sampling corrections control deviations introduced by different execution paths. [Chapter 4: Algorithm Taxonomy](../chapter03_mdp/algorithm-taxonomy) introduces the algorithmic distinction between on-policy and off-policy learning; here we study how that distinction is measured and corrected in a distributed system.

## 2. Arranging Models and GPUs

### 2.1 How veRL Organizes Five Roles

Training and inference consistency solves the question of "whether data can correctly update the model." The next step is to enable Actor, Critic, Reference Model, Reward Model, and rollout engine to collaborate across different GPUs. veRL (Volcano Engine Reinforcement Learning) breaks this problem into three levels: algorithm main loop, model computation, and resource allocation, corresponding to the paper [HybridFlow](https://arxiv.org/abs/2409.19256).

#### Design Goals of HybridFlow

HybridFlow abstracts RLHF/GRPO/PPO training into a **single-controller multi-model orchestration**:

```
┌─────────────────────────────────────────────────────────┐
│              Single Controller (Driver)                  │
│  - Algorithm loop (PPO or GRPO)                          │
│  - Resource scheduling (which model runs on each GPU)    │
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
              ResourcePool (GPU group)
```

#### Three Core Abstractions of HybridFlow

##### ResourcePool

Group GPUs, with each group capable of holding one or more models:

```python
# Simplified veRL resource configuration
resource_pools = {
    "actor_pool": num_gpus=8,    # 8 GPUs for the actor
    "critic_pool": num_gpus=4,   # 4 GPUs for the critic
    "rollout_pool": num_gpus=8,  # 8 GPUs for rollout
    "ref_pool": num_gpus=2,      # 2 GPUs for the reference model
}
```

Different models can **share a GPU** (colocate) or **exclusive GPU** (disaggregated):

```python
# Colocation: actor and rollout share the same GPU pool
mapping = {
    "actor": "actor_rollout_pool",
    "rollout": "actor_rollout_pool",  # shared pool
    "critic": "critic_pool",
    "ref": "ref_pool",
}
```

##### Worker

Each Worker is an independent model instance that encapsulates the specific training/inference logic:

```python
class ActorWorker:
    def __init__(self, model_config):
        self.model = FSDPActor(model_config)

    def update(self, batch):
        # Compute the PPO/GRPO loss and backpropagate.
        loss = compute_ppo_loss(batch, self.model)
        loss.backward()
        self.optimizer.step()

    def get_weights(self):
        # Expose weights to the rollout engine.
        return self.model.state_dict()

class RolloutWorker:
    def __init__(self, model_config):
        self.engine = vLLMEngine(model_config)

    def generate(self, prompts):
        return self.engine.generate(prompts)

    def sync_weights(self, new_weights):
        self.engine.load_weights(new_weights)
```

##### Driver (Single Controller)

The Driver is the main loop of the RL algorithm, orchestrating all Workers:

```python
class PPODriver:
    def train(self, num_epochs):
        for epoch in range(num_epochs):
            # 1. Synchronize the actor weights to rollout.
            weights = self.actor_worker.get_weights()
            self.rollout_worker.sync_weights(weights)

            # 2. Sample from the current policy.
            prompts = sample_prompts(self.dataset)
            responses = self.rollout_worker.generate(prompts)

            # 3. Score the responses with the reward model.
            rewards = self.reward_worker.score(prompts, responses)

            # 4. Estimate values with the critic.
            values = self.critic_worker.value(prompts, responses)

            # 5. Compute advantages and update the actor.
            advantages = compute_gae(rewards, values)
            self.actor_worker.update(prompts, responses, advantages)

            # 6. Update the critic.
            self.critic_worker.update(prompts, responses, rewards)
```

#### HybridFlow's Hybrid Control Mechanism

**Hybrid** refers to the **unified hybrid parallelism strategy** — within the same framework, the following can be combined:

- **3D Parallelism**: TP (Tensor Parallelism) × PP (Pipeline Parallelism) × DP (Data Parallelism)
- **Colocate vs Disaggregated**: Models can be shared or exclusive to GPUs
- **Multiple Training Backends**: FSDP, Megatron, DeepSpeed ZeRO
- **Multiple Inference Backends**: vLLM, SGLang, HuggingFace generate

These configurations determine whether the Actor, Critic, Reference Model, Reward Model, and rollout engine can share resources. The main differences between frameworks also lie here: some allow flexible resource pool combinations, while others require each model to use independent processes or fixed backends.

#### Framework Architecture Comparison

| Dimension               | veRL (HybridFlow) | OpenRLHF          | NeMo-Aligner     | TRL            |
| ----------------------- | ----------------- | ----------------- | ---------------- | -------------- |
| **Orchestration**       | Single-controller | Single-controller | Multi-controller | Single-process |
| **Resource Allocation** | Any combination   | Strict separation | NVIDIA stack     | Single GPU     |
| **Training Backend**    | FSDP + Megatron   | FSDP/DeepSpeed    | Megatron         | Accelerate     |
| **Inference Backend**   | vLLM/SGLang       | vLLM              | TRT-LLM          | HF generate    |
| **Typical Scale**       | 8-1024 GPUs       | 8-256 GPUs        | 8-512 GPUs       | 1-8 GPUs       |

[Chapter 15 GRPO Practice](../chapter18_grpo/grpo-practice-and-mechanism) uses veRL.

### 2.2 Other Implementations of the Same Training Process

veRL uses a Driver to orchestrate multiple roles, but this is not the only way to implement it. OpenRLHF emphasizes the separation of roles among Ray processes, NeMo-Aligner is built around NVIDIA's Megatron and TRT-LLM, and TRL compresses the complexity into a Trainer interface suitable for single-machine experiments. When comparing them, we still focus on the same three questions: how to place the models, what backend to use for generation, and how to synchronize the parameters.

#### OpenRLHF

[OpenRLHF, arXiv:2405.11143](https://arxiv.org/abs/2405.11143) is maintained by the OpenLLMAI team and is one of the earliest open-source RLHF frameworks.

OpenRLHF adopts the following structure:

- Based on **Ray** for distributed scheduling
- Strict **Actor/Critic/Ref/RM separation** — each model runs in an independent Ray Actor process
- Uses a relatively straightforward configuration interface to organize these Ray Actors

```python
# Simplified OpenRLHF PPO pseudocode
from openrlhf import PPOTrainer, ModelGroup

actor = ModelGroup(num_gpus=8, backend="deepspeed")
critic = ModelGroup(num_gpus=8, backend="deepspeed")
ref = ModelGroup(num_gpus=4)
reward = ModelGroup(num_gpus=4)
vllm = VLLMRollout(num_gpus=8)

trainer = PPOTrainer(actor, critic, ref, reward, vllm)
trainer.train(dataset, num_epochs=100)
```

It is suitable for research purposes and medium-scale training (8–256 GPUs). The model roles are strictly separated, making it easy to independently scale up or down, while also increasing data transmission between roles.

#### NeMo-Aligner

[NeMo-Aligner](https://github.com/NVIDIA/NeMo-Aligner) is the official stack from NVIDIA, deeply integrated with Megatron-LM and TRT-LLM.

NeMo-Aligner adopts the following architecture:

- **Megatron** as the training backend, responsible for tensor, pipeline, and data parallelism.
- **TRT-LLM** as the inference backend, responsible for generation optimization on NVIDIA GPUs.
- Training, inference, and communication are all configured around the NVIDIA software stack.

It is suitable for clusters already using NVIDIA NeMo with Megatron, especially for models larger than 70B parameters. Teams using other training backends need to evaluate the cost of migration.

#### TRL (Transformer Reinforcement Learning)

[TRL](https://github.com/huggingface/trl) is a lightweight framework developed by HuggingFace.

TRL adopts the following architecture:

- Based on **Accelerate** (HuggingFace's distributed abstraction)
- Single-process models, with Accelerate automatically splitting them
- Reduces the configuration cost for small-scale experiments through the Trainer interface

```python
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

model = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")
config = PPOConfig(batch_size=8)
trainer = PPOTrainer(config, model)
trainer.train(dataset)
```

It is suitable for learning, prototyping, and small-scale experiments with 1–8 GPUs. Once the number of models, generation throughput, and cross-node scheduling become bottlenecks, it is necessary to switch to a dedicated distributed RL framework.

#### Applicable Scenarios for Four Frameworks

| Framework        | Ease of Use | Performance | Scale Limit | Industrial Adoption                 |
| ---------------- | ----------- | ----------- | ----------- | ----------------------------------- |
| **veRL**         | Medium      | High        | 1024+ GPUs  | Qwen, DeepSeek, ByteDance internal  |
| **OpenRLHF**     | High        | Medium      | 256 GPUs    | SimpleRL, some open-source projects |
| **NeMo-Aligner** | Low         | Very High   | 512+ GPUs   | NVIDIA customers, Nemotron          |
| **TRL**          | Very High   | Low         | 8 GPUs      | Research, teaching                  |

Selection should be based on the current scale and existing technology stack:

- Learning and prototyping: TRL
- Research and medium-scale: OpenRLHF or veRL
- Large-scale production: veRL or NeMo-Aligner (depending on the hardware stack)

## 3. Improving Generation Throughput and Controlling VRAM

### 3.1 How the Rollout Engine Enhances Generation Throughput

In many LLM RL tasks, the time required to generate a response exceeds the time needed for a single parameter update, with specific costs detailed in [Appendix A.2](../appendix_industrial_training/rl-infrastructure). The Rollout Engine thus directly determines whether the training process can continuously obtain new data. Below, we use vLLM as an example to explain three optimizations at the generation end.

#### Three Core Optimizations of vLLM

##### PagedAttention

Traditional KV cache is allocated in a contiguous manner, leading to severe VRAM fragmentation. vLLM draws inspiration from the paging mechanism in the OS, dividing the KV cache into fixed-size blocks:

```python
# Conventional contiguous KV-cache allocation
seq_len = 2048
kv_cache = torch.empty(batch_size, seq_len, num_heads, head_dim)
# VRAM utilization: 50-70%

# vLLM PagedAttention: block allocation
block_size = 16
blocks = allocate_blocks(num_blocks)
# VRAM utilization: over 95%
```

The VRAM utilization has been improved from 50-70% to 95%+, and the batch size has been increased by 2-4 times.

##### Continuous Batching

Traditional batching is "wait for one batch to be fully generated before switching". vLLM employs **dynamic batching** — once a sequence is generated, it is immediately replaced by a new sequence:

```
time:       ──────────────────────────────────────►
sequence A: [tok][tok][tok][tok][EOS]
sequence B: [tok][tok][tok][tok][tok][tok][EOS]
sequence C:           [tok][tok][tok][tok][EOS]  ← starts when A ends
sequence D:                    [tok][tok][tok][EOS]  ← starts when C ends
```

**Throughput Improvement of 5-10x Compared to Static Batching**

##### Speculative Decoding

Use a small model to draft several tokens first, and have the large model verify them in parallel:

```python
def speculative_decode(prompt, draft_model, target_model, num_draft=4):
    while not done:
        # 1. The draft model generates num_draft tokens.
        draft_tokens = draft_model.generate(prompt, max_tokens=num_draft)

        # 2. The target model verifies them in parallel.
        target_logits = target_model.forward(prompt + draft_tokens)

        # 3. Accept matching tokens; regenerate after a rejection.
        for i, token in enumerate(draft_tokens):
            if target_logits[i].argmax() == token:
                prompt.append(token)
            else:
                prompt.append(target_logits[i].argmax())
                break
```

Throughput improved by 2-3 times (typical LLM inference).

#### Position of vLLM in the RL Data Flow

In veRL, vLLM serves as the RolloutWorker:

```python
class VLLMRolloutWorker:
    def __init__(self, model_path, tensor_parallel_size=8):
        from vllm import LLM
        self.engine = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            enable_prefix_caching=True,  # Reuse KV states across a GRPO group.
            gpu_memory_utilization=0.9,
        )

    def generate(self, prompts, sampling_params):
        # Batched generation
        return self.engine.generate(prompts, sampling_params)

    def sync_weights(self, new_weights):
        # vLLM 0.5+ supports online weight updates.
        self.engine.load_weights(new_weights)
```

**Prefix Caching** is particularly important for GRPO — generating $G=8$ responses for the same prompt, the KV cache of the prefix (prompt part) can be reused, saving 70–80% of memory and time.

#### SGLang's Generation and Scheduling Mechanism

[SGLang](https://github.com/sgl-project/sglang) is developed by the LMSYS team and is faster than vLLM in agentic scenarios:

- **RadixAttention**: Manages KV cache using a radix tree, enabling cross-request reuse
- **Programmatic Frontend**: Supports complex control flow (multi-turn calls, branches, loops)
- **Constrained Decoding**: Built-in JSON and regex constraints for generation

In industrial practice:

- **vLLM**: General rollout, single-turn generation
- **SGLang**: Agentic rollout, multi-turn, structured output
- **TRT-LLM**: Inference optimization for NVIDIA GPUs

### 3.2 How to Allocate VRAM Across Multiple GPUs

After addressing the issue of generation speed, the training side still needs to accommodate weights, gradients, optimizer states, and activations. Full-parameter training of a 70B model in BF16 far exceeds the capacity of a single 80GB H100 GPU, so these states must be partitioned or recomputed.

#### Composition of Training VRAM

Training VRAM includes weights, gradients, optimizer states, and activations. Taking a common example of BF16 weights and gradients, FP32 main weights, and Adam first- and second-moment estimates, each parameter requires approximately:

$$
\begin{aligned}
M \approx {}& \underbrace{2N}_{\text{BF16 weights}} \\
&+ \underbrace{2N}_{\text{BF16 gradients}} \\
&+ \underbrace{4N}_{\text{FP32 main weights}} \\
&+ \underbrace{8N}_{\text{Adam's } m \text{ and } v} \\
&+ M_{\text{act}},
\end{aligned}
$$

where $N$ is the number of parameters, the first four terms are in bytes, and $M_{\text{act}}$ represents the activation memory. That is, without considering activations, this configuration requires approximately 16 bytes per parameter. Different optimizers and precision configurations will change this number. For example, omitting the FP32 main weights will reduce the memory by 4N bytes.

For a 70B model:

- Weights: 140 GB
- Gradients: 140 GB
- FP32 Main Weights: 280 GB
- Adam's $m$, $v$: 560 GB
- Activations: ~100 GB (depending on batch size and sequence length)
- **Total**: Approximately 1.22 TB

This estimation serves to determine the scale, not to make an exact prediction of memory usage. Activation checkpointing, optimizer implementation, sequence length, and batch size all affect the actual memory consumption; however, it is already sufficient to illustrate that 70B full-parameter training cannot fit on a single 80GB GPU.

#### ZeRO (Zero Redundancy Optimizer)

[DeepSpeed ZeRO, arXiv:1910.02054](https://arxiv.org/abs/1910.02054) splits the training states across multiple GPUs:

| Stage      | Split Content                 | Memory Saving Factor      | Communication Overhead |
| ---------- | ----------------------------- | ------------------------- | ---------------------- |
| **ZeRO-1** | Optimizer state               | 4×                        | Low                    |
| **ZeRO-2** | Optimizer + Gradient          | 8×                        | Medium                 |
| **ZeRO-3** | Optimizer + Gradient + Weight | $N$× (N = number of GPUs) | High                   |

ZeRO-3 splits the weights as well, such that each GPU stores only $1/N$ of the weights. However, during forward and backward passes, all-gather is required to reconstruct the full weights.

```python
# DeepSpeed ZeRO-3 configuration
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

#### FSDP (Fully Sharded Data Parallel)

The native ZeRO-3 equivalent in PyTorch, which is easier to use than DeepSpeed:

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = LlamaForCausalLM(config)
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # Equivalent to ZeRO-3
    mixed_precision=MixedPrecision(param_dtype=torch.bfloat16),
    cpu_offload=CPUOffload(offload_params=False),  # Optional CPU offload
)
```

veRL defaults to using FSDP — it is more stable than DeepSpeed and more compatible with the PyTorch ecosystem.

#### Gradient Checkpointing

Instead of splitting the model, it trades computation for memory by not saving intermediate activations during the forward pass and recomputing them during the backward pass:

```python
from torch.utils.checkpoint import checkpoint

class CheckpointedBlock(nn.Module):
    def forward(self, x):
        # Wrap the transformer block with activation checkpointing.
        return checkpoint(self._forward, x, use_reentrant=False)

    def _forward(self, x):
        return self.transformer_block(x)
```

The memory usage is reduced from $O(L)$ to $O(\sqrt{L})$ (where $L$ is the number of layers), at the cost of performing forward computation twice — training is 20–30% slower.

#### Memory Estimation When Combining Techniques

For a 70B model (8 H100 80GB GPUs):

| Configuration                          | Per-GPU Memory          | Training Speed |
| -------------------------------------- | ----------------------- | -------------- |
| Full Parameters + Adam (baseline)      | 940 GB (exceeds memory) | -              |
| ZeRO-3                                 | 118 GB (exceeds memory) | -              |
| ZeRO-3 + Gradient Checkpointing        | 30 GB                   | 1×             |
| ZeRO-3 + Gradient Checkpointing + LoRA | 8 GB                    | 1.2×           |

LoRA (see [Chapter 18](./industrial-post-training)) trains only a small number of parameters, significantly reducing memory requirements. Industrial-scale 70B RL training typically uses LoRA + FSDP.

## 4. Keeping the Multi-Machine Pipeline Running

### 4.1 How Asynchronous Scheduling Reduces Waiting

Synchronous training requires waiting for all rollouts to complete before updating, which can lead to long waiting times for training GPUs. Asynchronous training allows generation and updates to proceed separately, with trajectories exchanged through a queue. Below is a comparison of the scheduling methods used in LlamaRL, AReaL, and AgentRL.

#### LlamaRL

[LlamaRL, Meta arXiv:2505.24034](https://arxiv.org/abs/2505.24034) is a distributed RL framework released by Meta in May 2025:

LlamaRL uses a decentralized design, without a centralized master node. Each worker continuously takes tasks based on its role and submits results.

```python
# Simplified LlamaRL architecture
class LlamaRLWorker:
    def run(self):
        while True:
            # Each worker runs the loop for its assigned role.
            if self.role == "rollout":
                prompts = self.fetch_from_queue()
                responses = self.generate(prompts)
                self.push_to_train_queue(responses)

            elif self.role == "train":
                batch = self.fetch_from_rollout_queue()
                self.update(batch)
                self.broadcast_weights()  # Asynchronous broadcast
```

This design brings three system characteristics:

- No single point of failure
- Easy to horizontally scale (adding workers is sufficient)
- Suitable for super-large scale (10k+ GPUs)

**Measured Result**: Training Llama-3-70B with GRPO on 4096 GPUs is **10.4× faster** than synchronous training.

#### AReaL (Asynchronous RL)

[AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning, arXiv:2505.24298](https://arxiv.org/abs/2505.24298) is a large-scale asynchronous LLM RL system open-sourced by Ant Group and Tsinghua University in 2025:

AReaL adopts fully asynchronous rollouts and explicitly handles policy staleness in PPO updates. Rollout workers continuously generate samples, and training workers immediately update upon receiving a batch; each trajectory records the policy version and probability at the time of generation, allowing the training side to correct old policy data accordingly.

```python
# Simplified AReaL update
def staleness_aware_update(batch, current_weights):
    # The batch stores the policy version and log probabilities at rollout time.
    gen_log_probs = batch["gen_log_probs"]
    current_log_probs = compute_log_probs(batch, current_weights)
    importance_weights = torch.exp(current_log_probs - gen_log_probs)

    # Clip importance weights so stale samples cannot produce extreme gradients.
    clipped_weights = torch.clamp(importance_weights, 0.8, 1.2)
    loss = -(clipped_weights * advantages).mean()

    return loss
```

This design allows:

- Training with old data is permitted, without requiring strict on-policy constraints
- The buffer can accumulate a large amount of data
- Training and generation are completely decoupled

**Empirical Results**: Running 671B MoE GRPO on 1024 GPUs is **2.77× faster** than synchronous training.

#### AgentRL

[AgentRL: Scaling Agentic Reinforcement Learning with a Multi-Turn, Multi-Task Framework, arXiv:2510.04206](https://arxiv.org/abs/2510.04206) is a multi-turn, multi-task agentic reinforcement learning framework released in October 2025, with code available at [THUDM/AgentRL](https://github.com/THUDM/AgentRL):

AgentRL combines the asynchronous generation training pipeline with a unified environment interface. On the training side, it uses three types of worker pools: rollout, Actor, and Reference. On the environment side, it manages heterogeneous tasks through function call interfaces, containers, Controller, and Task Workers. Cross-Policy Sampling increases multi-turn exploration, while Task Advantage Normalization aligns the advantage scales across different tasks.

```python
# Simplified asynchronous AgentRL structure
rollout_workers.stream_trajectories(task_manager)
actor_workers.update_policy(buffer.sample())
reference_workers.compute_kl(buffer.sample())
controller.route_function_calls(task_workers)
```

It primarily addresses the following requirements:

- Support for multi-turn, multi-task agentic reinforcement learning
- Asynchronous decoupling of trajectory collection and policy update
- Management of environment deployment through controller / task worker / transport layer
- Used for building AutoGLM

This architecture is suitable for multi-turn environment tasks such as SWE-Agent, Computer Use, and Deep Research Agent.

#### Comparison of Asynchronous Frameworks

| Framework   | Main Contributors      | Core Mechanism                                         | Speedup                | Applicable Scenarios        |
| ----------- | ---------------------- | ------------------------------------------------------ | ---------------------- | --------------------------- |
| **LlamaRL** | Meta                   | Fully decentralized                                    | 10.4×                  | Extremely large-scale Dense |
| **AReaL**   | Ant Group and Tsinghua | Fully asynchronous rollout + staleness-aware PPO       | 2.77×                  | Large-scale LLM RL          |
| **AgentRL** | THUDM / Zhipu          | Multi-turn, multi-task + unified environment interface | Not specified in paper | Agent training              |

### 4.2 Why MoE Increases System Complexity

DeepSeek V3, Qwen3, and GLM-4.5 all adopt the MoE architecture. Each token activates only a small number of experts, allowing parameters to be distributed across more GPUs; at the same time, the distribution of samples in reinforcement learning will change the load on experts, and the training system must also record routing and communication states.

#### Data Flow in MoE Training

The parameters of MoE models are unevenly distributed — most parameters reside in the experts, and each sample only activates a small number of experts:

```
MoE model structure (DeepSeek V3):
┌─────────────────────────────────────┐
│ Dense layers (attention, etc.): 20B │
├─────────────────────────────────────┤
│ MoE layers:                         │
│  - 256 experts x 5B = 1.28T         │
│  - 8 experts active per token       │
│  - 40B active MoE parameters        │
└─────────────────────────────────────┘
Total parameters: 1.3T; active parameters: 60B
```

#### Three System Issues in MoE RL

##### Uneven Expert Load

Some experts are frequently activated, while others remain idle. This leads to:

- Uneven computational load (some GPUs are overloaded)
- Biased training data distribution (some experts are under-trained)

**Solution**: **expert-balancing loss**:

```python
def expert_balancing_loss(router_logits, num_experts):
    # Compute the activation frequency of each expert.
    router_probs = torch.softmax(router_logits, dim=-1)
    expert_freq = router_probs.mean(dim=0)  # [num_experts]

    # Encourage a balanced distribution.
    target_freq = 1.0 / num_experts
    balance_loss = ((expert_freq - target_freq) ** 2).mean()

    return balance_loss
```

##### Communication Overhead

Experts in MoE are distributed across multiple GPUs (Expert Parallelism), and each sample requires all-to-all communication:

```
GPU 0: expert 0,1,2     ──┐
GPU 1: expert 3,4,5     ──┼── all-to-all ── return with another all-to-all
GPU 2: expert 6,7,8     ──┤
GPU 3: expert 9,10,11   ──┘
```

**Solution**: **DeepEP** (DeepSeek Expert Parallelism), optimizing the all-to-all communication pattern.

##### Variance of Token-Level Importance Sampling

[GRPO Family](../chapter18_grpo/grpo-family) mentioned that in the MoE setting, different tokens are routed to different experts, leading to significant fluctuations in the importance sampling ratio at the token level, which results in high gradient variance.

**Solution**: **GSPO (Group Sequence Policy Optimization)** — changing the importance sampling ratio from the token level to the sequence level:

```python
# PPO/GRPO: token-level importance sampling
token_ratio = exp(log_prob_new - log_prob_old)  # one ratio per token

# GSPO: sequence-level importance sampling
sequence_log_prob_new = sum(log_prob_new_per_token)
sequence_log_prob_old = sum(log_prob_old_per_token)
sequence_ratio = exp(sequence_log_prob_new - sequence_log_prob_old)
# Use one ratio for the entire sequence.
```

Qwen3 full series (including 235B-A22B) are all trained based on GSPO.

#### DeepSeek V3's MoE RL Scheme

RL training practice for DeepSeek V3 (671B MoE, 37B activation):

- **DualPipe**: Pipeline parallelism optimization (see 36.7)
- **FP8 Training**: Use FP8 to reduce memory and computation ([arXiv:2412.19437](https://arxiv.org/abs/2412.19437))
- **MTP (Multi-Token Prediction)**: Predict multiple tokens at once to improve training signal density

#### Step Flash's MoE Scheme

Step Flash is the MoE RL optimization introduced by Step Star in 2025:

- **Dynamic Expert Allocation**: Dynamically adjusts the number of experts based on the token distribution within a batch
- **Sparse Gradient Sync**: Only synchronizes gradients of the activated experts
- **Cache-aware Routing**: Considers the locality of KV cache during routing

#### GLM-4.5's MoE Scheme

GLM-4.5 is trained using the **slime** framework ([THUDM/slime](https://github.com/THUDM/slime)):

- Megatron training backend
- SGLang inference backend
- Native MoE optimization (DeepEP communication, fp8 rollout)

### 4.3 How to Reduce Pipeline Idle Time

After splitting the model across multiple GPUs, each GPU does not automatically remain busy. Pipeline stages may wait for the previous stage, and sequences of varying lengths within the same batch can leave idle slots. **DualPipe** reduces the waiting time between forward and backward passes, while **Best-Fit Packing** groups sequences of similar lengths into the same batch. They respectively address idle time caused by computational timing and sample shape mismatches.

#### DualPipe

The paper [DeepSeek V3 arXiv:2412.19437](https://arxiv.org/abs/2412.19437) proposes **DualPipe**—bidirectional pipeline parallelism.

Traditional pipeline parallelism (PP) suffers from the **bubble problem**:

- **Bubbles** occur when a stage has to wait for the previous stage to finish before it can proceed, leading to idle time.
- This is especially problematic when the model is split across multiple GPUs, as the pipeline stages may not be perfectly aligned in terms of computation time.

DualPipe addresses this by introducing a **bidirectional pipeline**, where the model is split into two parts, and the forward and backward passes are executed in parallel. This reduces the idle time between the forward and backward passes by overlapping the computation of different stages.

```
GPU 0: [F0][F1][F2][F3]              [B3][B2][B1][B0]
GPU 1:       [F0][F1][F2][F3]   [B3][B2][B1][B0]
GPU 2:             [F0][F1][F2][F3][B3][B2][B1][B0]
                   ↑                ↑
                   forward           backward
                   large pipeline bubble
```

DualPipe allows the forward and backward passes to **run simultaneously** — the forward stage N and the backward stage N-1 overlap on the same GPU:

```
GPU 0: [F0|B0][F1|B1][F2|B2][F3|B3]  ← forward and backward overlap
GPU 1:       [F0|B0][F1|B1][F2|B2][F3|B3]
GPU 2:             [F0|B0][F1|B1][F2|B2][F3|B3]
                                    little idle time
```

The bubble ratio has been reduced from the traditional $\frac{P-1}{M}$ (where $P$ is the number of PP stages and $M$ is the micro-batch count) to $\frac{P-1}{2M}$.

```python
# DualPipe pseudocode
class DualPipeScheduler:
    def schedule(self, num_stages, num_micro_batches):
        schedule = []
        for step in range(num_micro_batches + num_stages - 1):
            for stage in range(num_stages):
                # Schedule a forward and backward microbatch on the same stage.
                fwd_mb = step - stage
                bwd_mb = step - (num_stages - 1 - stage)
                if fwd_mb >= 0 and fwd_mb < num_micro_batches:
                    schedule.append(("forward", stage, fwd_mb))
                if bwd_mb >= 0 and bwd_mb < num_micro_batches:
                    schedule.append(("backward", stage, bwd_mb))
        return schedule
```

#### Best-Fit Packing

Traditional micro-batch allocation is uniform — each GPU receives the same number. However, under MoE, different experts have varying loads, and uniform allocation leads to imbalance.

**Best-Fit Packing**: Use bin packing algorithms to allocate micro-batches of different sizes to GPUs:

```python
def best_fit_pack(items, bin_capacity):
    """Pack variable-size microbatches into bins representing GPU capacity."""
    bins = [[]]
    for item in sorted(items, reverse=True):  # Largest first
        # Find the fullest bin that still has enough remaining capacity.
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

DeepSeek V3 Uses Best-Fit Packing to Boost GPU Utilization from 70% to 95%.

### 4.4 How to Identify Performance Bottlenecks

Each of the previous techniques may push the bottleneck to the next stage: after generation speeds up, weight synchronization may become slower; after model partitioning, cross-GPU communication may dominate the time; after sample packing improves, data reading may again become the limiting factor. Performance analysis first records the actual time consumed at each step, then decides whether to add more GPUs, modify the parallelism strategy, or adjust the batch size.

#### Performance Analysis Tools

##### PyTorch Profiler

```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
) as prof:
    trainer.train_step()

# Print the ten operations with the highest CUDA time.
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

##### NVIDIA Nsight Systems

```bash
# Profile the training process with nsys.
nsys profile -o rl_train_profile python train.py

# Inspect the timeline in the Nsight Systems GUI.
nsys-ui rl_train_profile.qdrep
```

Visualizing Execution Time of Each CUDA Kernel, CPU-GPU Synchronization, and Communication Overhead

##### veRL Built-in Profiler

veRL provides profiling specific to reinforcement learning:

```python
from verl.utils.profiler import RLProfiler

with RLProfiler() as p:
    trainer.train()
    p.print_summary()
# Example output:
#   rollout time: 3500s (85%)
#   actor update time: 120s (3%)
#   critic update time: 80s (2%)
#   weight sync time: 30s (0.7%)
#   communication: 400s (10%)
```

#### Common Bottlenecks and Optimization Directions

| Bottleneck                 | Symptoms                                | Optimization                                   |
| -------------------------- | --------------------------------------- | ---------------------------------------------- |
| **Slow Rollout**           | Rollout accounts for 80%+ of time       | Add more rollout GPUs, use vLLM prefix caching |
| **Slow Weight Sync**       | Sync accounts for 5%+ of time           | Use LoRA, NCCL batched transmission            |
| **Communication Overhead** | All-reduce accounts for 10%+ of time    | Increase batch size, use gradient accumulation |
| **Memory Explosion**       | OOM (Out of Memory)                     | Gradient checkpointing                         |
| **Uneven Expert Load**     | Some GPUs at 90%+, others at 30%        | Expert balancing loss, dynamic routing         |
| **Slow Human Problem**     | Longest sequence in batch dictates time | Length bucketing, Seer divided rollout         |

#### MFU (Model FLOPs Utilization)

MFU is calculated by dividing the actual number of floating-point operations executed by the peak operations that the hardware can provide within the same time period:

$$\text{MFU} = \frac{\text{Actual FLOPs}}{\text{Peak FLOPs} \times \text{Time}}$$

For example, if 8 GPUs each have a theoretical peak of 1000 TFLOPS, and they run continuously for 10 seconds, they can complete a maximum of $8 \times 1000 \times 10 = 80{,}000$ TFLOP of computation. If the model actually completes 32,000 TFLOP, then the MFU is $32{,}000 / 80{,}000 = 40\%$. The remaining time may be spent on communication, waiting for data, or generating trajectories.

The peak performance of H100 in bf16 is approximately 1000 TFLOPS. Typical MFU for LLM RL training:

| Configuration                                    | MFU                                       |
| ------------------------------------------------ | ----------------------------------------- |
| Dense + FSDP + checkpointing                     | 35-45%                                    |
| MoE + EP + DualPipe                              | 50-60%                                    |
| Asynchronous RL (generation/training separation) | 70-80% (rollout part accelerated by vLLM) |

When the MFU is below 30%, it is advisable to combine time decomposition with further checks on communication, data loading, and rollout waiting. The MFU itself cannot indicate where the specific bottleneck lies.

### 4.5 Putting Various Techniques into Large-Scale Clusters

Model parallelism, asynchronous rollout, memory optimization, and fault recovery ultimately need to collaborate within the same cluster configuration. Below is a large-scale MoE training configuration that illustrates the relationships between these parameters.

#### Typical Configuration

Take the GRPO training of Qwen3-235B-A22B (235B total parameters, 22B active MoE) as an example:

```yaml
# Cluster configuration
total_gpus: 12288 # 12k H100
intra_node_bandwidth: 900 GB/s # NVLink
inter_node_bandwidth: 50 GB/s # InfiniBand

# Model parallelism
tensor_parallel: 8 # TP=8 within a node
pipeline_parallel: 4 # PP=4 across nodes
expert_parallel: 16 # EP=16
data_parallel: 24 # DP=24

# Training configuration
algorithm: GSPO # A GRPO variant designed for sequence-level optimization
batch_size_per_gpu: 1
gradient_accumulation: 32
seq_len: 32768
group_size: 8 # Eight responses per prompt

# Asynchronous configuration
async_mode: disaggregated
rollout_buffer_size: 100000
weight_sync: lora # Synchronize only the LoRA adapter
weight_sync_method: nccl_packed
```

#### Performance Metrics

In the field of reinforcement learning, performance metrics are essential for evaluating the effectiveness of an agent's learning process and its ability to achieve the desired objectives within a given environment. These metrics provide a quantitative measure of how well an agent is performing, allowing researchers and practitioners to compare different algorithms, tune hyperparameters, and assess the generalization capabilities of the learned policies.

Common performance metrics in reinforcement learning include:

- **Reward Accumulation**: The total reward accumulated by the agent over a series of episodes or a single episode. This metric reflects the agent's ability to maximize cumulative rewards, which is typically the primary objective in most RL tasks.

- **Episode Return**: The average reward per episode, which helps in understanding the agent's performance across multiple episodes. It is particularly useful when the environment has a fixed number of steps per episode.

- **Learning Curve**: A plot that shows the agent's performance over time, typically measured as the average reward per episode or the success rate. Learning curves are valuable for identifying convergence, overfitting, or underfitting issues.

- **Success Rate**: The percentage of episodes in which the agent successfully completes a task or reaches a specific goal. This metric is especially relevant in tasks with clear success or failure conditions.

- **Policy Stability**: The consistency of the agent's policy across different episodes or environments. A stable policy indicates that the agent has learned a robust strategy that generalizes well to new situations.

- **Sample Efficiency**: The number of interactions (environment steps) required for the agent to achieve a certain level of performance. High sample efficiency is

```text
Training one epoch (10B tokens):
  Total time: 24 hours
  GPU hours: 294912

Time breakdown:
  Rollout: 18 hours (75%)
  Actor update: 3 hours (12.5%)
  Critic update: 2 hours (8%)
  Weight sync: 0.5 hours (2%)
  Other: 0.5 hours (2.5%)

MFU: 52% (MoE + DualPipe + FP8)
```

#### Failures and Bottlenecks in Large-Scale Training

##### Failure Recovery

With 12,288 cards, 5–10 cards fail on average per day. It is essential to:

- **Checkpoint Frequency**: Save a checkpoint every 30 minutes, allowing rollback in case of failure
- **Redundancy Design**: 8 backup cards are allocated for every 1,024 cards
- **Auto Restart**: Automatically recover from the most recent checkpoint after detecting a failure

##### Communication Bottlenecks

Slow cross-node communication in a large-scale cluster with 10,000+ GPUs. Network design for such a cluster includes:

- **Topology-aware**: Prioritize adjacent GPUs to form tensor parallel groups
- **Overlap Communication and Computation**: Initiate gradient all-reduce during backward propagation concurrently with computation
- **Gradient Bucketing**: Merge small gradients to reduce the number of communication steps

##### MoE Routing Stability

In MoE training, expert routing may suddenly collapse — all tokens are routed to a small subset of experts. Monitoring is required to detect such instability.

```python
# Monitor expert load during training.
def monitor_expert_balance(model):
    while training:
        for layer in model.moe_layers:
            router_probs = layer.router.get_recent_probs()
            entropy = -torch.sum(router_probs * torch.log(router_probs + 1e-10))
            if entropy < threshold:  # Routing entropy is too low.
                alert(f"Layer {layer.id}: expert routing collapse!")
        time.sleep(60)
```

##### Data Pipeline Bottlenecks

A multi-chip cluster can consume millions of tokens per second, making data loading itself potentially a bottleneck:

- **Prefetching**: Preparing data for the next 10 batches in advance
- **Data Compression**: Storing data in more compact formats
- **Distributed Storage**: Distributing data across multiple SSDs to avoid single-point I/O bottlenecks

## Summary of This Section

- Multi-agent RL still proceeds along the four steps of generation, reward, update, and parameter synchronization.
- Model partitioning addresses memory issues, resource scheduling for training and generation resolves waiting issues, and weight synchronization ensures that rollout uses the correct policy version.
- Synchronous training is easier to guarantee data freshness; asynchronous training reduces waiting time but requires handling of stale experience and importance sampling corrections.
- MoE requires additional handling of expert load balancing and routing communication beyond standard parallelism.

Multi-machine systems address how to coordinate computational power, but training still requires a continuous supply of executable and verifiable data. [18.5 Large-Scale RL Data Engineering](./data-engineering) will follow the lifecycle of a trajectory, explaining how tasks, environments, rewards, and failure samples enter the next round of training.

## Further Reading

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
