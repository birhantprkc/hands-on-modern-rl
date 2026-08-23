# 24.2 From Audio Rewards to Real-Time Agents: A Minimal Training Loop

[24.1](./reward-design) separated audio rewards into content, prosody, and real-time behavior. Knowing what to reward still does not explain how that reward changes the model. A trainer must sample several responses to the same audio, send text and synthesized speech to different evaluators, and propagate one response-level score back to generated tokens. If any interface is wired incorrectly, the signal updates a behavior other than the intended one.

This section follows one sample through audio encoding, grouped sampling, speech synthesis, reward calculation, within-group comparison, and policy update. The code explains the data flow rather than replacing an industrial training framework. We then place the single-turn model in a continuing environment and examine interruptions, tool calls, and multi-turn state.

```mermaid
flowchart LR
    A["Audio question"] --> B["Encode acoustic features"]
    B --> C["Sample G responses"]
    C --> D["Judge text correctness"]
    C --> E["Synthesize and judge prosody"]
    D --> F["Combine reward"]
    E --> F
    F --> G["Normalize within group"]
    G --> H["Clipped policy update"]
```

## Hands-On: Minimal Audio GRPO Training

Industrial training requires distributed rollout, inference and reward services, and checkpoint management. The minimal workflow below keeps only the interface between reward design and policy updates.

### Experimental Setup

```python
# requirements: torch, transformers, librosa, soundfile
import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioDialogueConfig:
    # Audio encoder (pseudo-code: actual use Qwen2-Audio encoder)
    audio_encoder_dim = 1280
    audio_frame_rate = 12.5  # Hz, after downsampling
    # LLM decoder (actual use Qwen2.5-32B, simplified here)
    llm_hidden = 4096
    vocab_size = 152000
    # RL configuration
    group_size = 16         # GRPO samples per group
    max_response_len = 1024
    clip_eps = 0.2          # PPO clip
    beta_kl = 0.0           # Step-Audio sets to 0, allowing free exploration
```

### Model Structure

```python
class AudioDialoguePolicy(nn.Module):
    """Audio Dialogue Policy: Audio encoding → LLM reasoning → Text + codec generation"""
    def __init__(self, config):
        super().__init__()
        # Audio encoder (frozen)
        self.audio_encoder = AudioEncoder(config.audio_encoder_dim)
        for p in self.audio_encoder.parameters():
            p.requires_grad = False
        # Adaptor: 25 Hz → 12.5 Hz
        self.adaptor = nn.Conv1d(config.audio_encoder_dim, config.llm_hidden,
                                  kernel_size=2, stride=2)
        # LLM decoder
        self.llm = TransformerDecoder(config.llm_hidden, config.vocab_size)

    def forward(self, audio, question, response_tokens):
        # 1. Encode audio
        audio_feat = self.audio_encoder(audio)         # (B, T, D)
        audio_feat = self.adaptor(audio_feat.transpose(1,2)).transpose(1,2)

        # 2. Concatenate [audio, question, response] sequences
        inputs = concat_modalities(audio_feat, question, response_tokens)

        # 3. Autoregressively predict response logits
        logits = self.llm(inputs)
        return logits
```

### Reward Function

Implement the three types of rewards described in Section 24.1:

```python
class AudioReward:
    def __init__(self, grm_model, prosody_ref_dist):
        self.grm = grm_model                # Generative Reward Model
        self.prosody_ref = prosody_ref_dist # Human Prosody Distribution

    def content_reward(self, response_text, ground_truth):
        """Content Accuracy"""
        # Use LLM-as-judge to assess semantic equivalence
        prompt = f"Judge whether the answer is equivalent: \nReference: {ground_truth}\nAnswer: {response_text}\nReturn 1 if equivalent, else 0"
        return float(self.grm(prompt))

    def prosody_reward(self, response_audio):
        """Prosody Naturalness"""
        f0 = librosa.pyin(response_audio)         # Fundamental frequency
        f0_var = np.std(f0)
        # Wasserstein distance to human distribution
        f0_w = wasserstein_distance(
            np.histogram(f0, bins=50)[0] / len(f0),
            self.prosody_ref['f0_hist']
        )
        # Penalize flatness (a common failure mode in RLVR)
        flat_penalty = -max(0, 0.3 - f0_var)
        return -f0_w + 0.5 * flat_penalty

    def format_reward(self, response_text):
        """Check for 🤔...✅ format (a key trick in MGRD)"""
        has_think = '🤔' in response_text and '✅' in response_text
        return 1.0 if has_think else 0.0

    def total(self, response_text, response_audio, ground_truth, weights=(0.7, 0.2, 0.1)):
        w_c, w_p, w_f = weights
        return (w_c * self.content_reward(response_text, ground_truth)
              + w_p * self.prosody_reward(response_audio)
              + w_f * self.format_reward(response_text))
```

::: tip The Role of Format Reward
The Step-Audio-R1 paper found that removing the format reward ($w_f=0$) reduced the number of reasoning tokens from 2,800 to 1,500 and lowered the MMAU score by 1.2 percentage points. The optimizer had learned a shorter strategy: answer immediately and omit the `<think>...</think>` segment.

A format-reward weight of $0.2$ was sufficient to stabilize the reasoning behavior in this experiment. Audio RL therefore uses an explicit signal to preserve the required response structure while the content and prosody rewards evaluate the answer itself.
:::

### GRPO Training Loop

We use [GRPO](../chapter18_grpo/grpo-family) (Group Relative Policy Optimization) for training—no critic is needed, making it more suitable for large models:

```python
def grpo_train_step(policy, ref_policy, reward_fn, batch, config):
    """Single-step GRPO Training"""
    advantages = []
    log_probs_all = []

    for prompt, audio in batch:
        # 1. Sample G responses for each prompt
        responses = []
        for _ in range(config.group_size):
            with torch.no_grad():
                resp = policy.sample(audio, prompt, config.max_response_len)
            responses.append(resp)

        # 2. Compute reward for each response
        rewards = torch.tensor([
            reward_fn.total(r.text, r.audio, r.gt) for r in responses
        ])

        # 3. Normalize within group to get advantage (core of GRPO)
        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        advantages.extend(adv.tolist())

        # 4. Compute new policy log π(a|s)
        for resp, a in zip(responses, adv):
            log_probs = policy.log_prob(audio, prompt, resp.tokens)
            log_probs_all.append(log_probs)

    # 5. PPO clip objective (Step-Audio sets β_kl = 0)
    advantages = torch.tensor(advantages).unsqueeze(1)
    policy_loss = 0
    for logp_new, resp in zip(log_probs_all, [r for b in batch for r in [None]]):
        # Simplified: actual implementation should compute ratio per token
        pass

    # Complete PPO clip (refer to Chapter 8)
    # ratio = exp(logp_new - logp_old)
    # clipped = clip(ratio, 1-eps, 1+eps)
    # loss = -min(ratio * adv, clipped * adv).mean()

    return policy_loss

# Main loop
for epoch in range(num_epochs):
    for batch in dataloader:
        loss = grpo_train_step(policy, ref_policy, reward_fn, batch, config)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

::: details Why Use GRPO Instead of PPO
Industrial audio LLMs almost all use GRPO (e.g., [DeepSeek-R1](https://arxiv.org/abs/2501.12948)) or its variants, rather than classical PPO. Reasons:

1. **Save critic memory**: Training a critic for a 32B model requires additional VRAM; GRPO replaces the critic with intra-group normalization
2. **Better for discrete rewards**: Audio rewards are mostly 0/1 binary, making it hard for critics to learn
3. **Training stability**: Intra-group baseline naturally adapts to different difficulty levels

The RL implementation of Step-Audio-R1 is on-policy PPO, but samples 16 prompts per request and applies intra-group normalization — essentially an engineering interpretation of GRPO.

:::

### Self-Cognition Correction

There's a non-mainstream but critical issue in industrial audio RL: **the model forgets it is an audio model**. Pretraining data is mostly text-based, so models often respond with "I can't hear anything" or "I am a text model." The self-cognition correction process for Step-Audio-R1:

```python
def self_cognition_correction(policy):
    """Three-stage correction for self-cognition errors"""
    # Stage 1: Iterative self-distillation + LLM judge filtering
    for t in range(T):
        responses = policy.sample(audio_perception_queries)
        # Judge only retains responses with correct self-cognition
        correct = [r for r in responses if judge_acknowledges_audio(r)]
        policy.sft(correct)

    # Stage 2: DPO fine-tuning
    # 8000 preference pairs: correct cognition (w) vs. incorrect cognition (l)
    pref_pairs = build_preference_pairs(correct_cog=positive, text_only=negative)
    policy.dpo(pref_pairs, beta=0.1)
```

Results:

| Training Stage                    | Self-Cognition Error Rate |
| --------------------------------- | ------------------------- |
| Base model                        | 6.76%                     |
| Iterative self-distillation       | 2.63%                     |
| Iterative self-distillation + DPO | **0.02%**                 |

The precise alignment of DPO brings the error rate close to zero. This step may seem trivial, but it is crucial during deployment — users expect the model to confidently handle audio inputs, rather than apologetically saying "I can't hear."

## Summary of Audio Direction

Audio reinforcement learning extends reasoning and preference optimization to continuous acoustic signals. This section, together with 24.1, highlights three core advancements:

1. **MGRD in Step-Audio-R1**: Addresses the inverted scaling issue in the audio domain — the root cause is text-based reasoning, and the solution is iterative distillation, transferring the reasoning base from text to acoustics. R1 is the first to enable audio models to benefit from test-time compute scaling.
2. analogously, **RLHF Paradigm Migration in Step-Audio-R1.5**: Identifies and breaks the "verifiable reward trap" — RLVR optimizes "what to say," while users care about "how to say," necessitating the use of RLHF's multi-dimensional preference modeling to complete prosody, emotion, and coherence.
3. **Audio Reward Design**: A weighted combination of content, prosody, and real-time performance, with rubric-based generative RM replacing scalar RM, which is the core engineering difference between audio RL and text RL.

At the methodological level, this chapter reveals three universal lessons:

- **Modality Grounding Determines Reasoning Quality**: Reasoning capability can be transferred across modalities, but it must be explicitly anchored to the features of the correct modality.
- **Data Quality > Data Quantity**: A carefully selected set of 5K samples with pass@8 ∈ [3, 6] outperforms 200K unfiltered samples.
- **Reward Design is the Soul of RL**: A single verifiable reward collapses model behavior, while multi-dimensional rubrics are key to aligning with real-world experience.

The next section [24.3 VLA Models](../chapter28_vla/embodied-intelligence/) will connect multimodal perception to physical actions: strategies must not only understand sound and image but also act in continuous control, real-world cost, and physical constraints. Training methods for multi-agent collaborative RL are discussed in [Chapter 19 on Multi-Agent Collaboration](../chapter22_agentic/multi-agent-swarm).

## Further Reading

- [Step-Audio-R1 Technical Report (StepFun, 2025.11, arXiv:2511.15848)](https://arxiv.org/abs/2511.15848) — Original paper of the MGRD framework, foundational work on audio reasoning
- [Step-Audio-R1.5 Technical Report (StepFun, 2026.04, arXiv:2604.25719)](https://arxiv.org/abs/2604.25719) — Paradigm transfer of RLHF, breaking the verifiable reward trap
- [Step-Audio 2 Technical Report](https://arxiv.org/abs/2507.16632) — Foundation models of the Step-Audio series
- [EnCodec: High Fidelity Neural Audio Compression (Meta, 2022)](https://arxiv.org/abs/2210.13438) — Classic work on RVQ encoder-decoder
- [SoundStream: An End-to-End Neural Audio Codec (Google, 2021)](https://arxiv.org/abs/2107.03312) — Original paper of SoundStream
- [SpeechTokenizer: Unified Speech Tokenizer for Speech LLMs (2023)](https://arxiv.org/abs/2308.16692) — Semantic/acoustic hierarchical tokenization
- [WavTokenizer: An Efficient Acoustic Discrete Codec Tokenizer (ICLR 2025)](https://arxiv.org/abs/2408.16532) — Extreme compression (40-75 tokens/s)
- [Moshi: A Speech-Text Foundation Model for Real-Time Dialogue (Kyutai, 2024)](https://arxiv.org/abs/2410.00037) — Full-duplex real-time dialogue, Mimi encoder-decoder
- [GPT-4o System Card (OpenAI, 2024)](https://arxiv.org/abs/2410.21276) — Milestone of industrial-grade real-time speech interaction
- [DeepSeek-R1: Incentivizing Reasoning Capability via RL (2025)](https://arxiv.org/abs/2501.12948) — RLVR + GRPO training paradigm, foundational work of Step-Audio-R1
