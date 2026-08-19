# 24.5 Video RLHF and Physical Perception Generation

[Section 24.4 on Visual Generation with RL](./visual-generation-dancegrpo) discussed the foundations of diffusion RL — algorithms such as DDPO and DPOK. That section took an **algorithmic perspective**: how to model diffusion training as an MDP, and how to use policy gradient optimization.

This section takes a different perspective — the **industrial level**: how video generation models such as Seedance, LongCat-Video, Hailuo, Wan, and Kling were trained using RL in 2025–2026. These works represent the current state-of-the-art in video generation with RL.

## 24.5.1 From Images to Video: New Challenges for RL

RL for image generation has matured ([DDPO](./visual-generation-dancegrpo), DPOK). However, video generation brings new challenges:

### Long Sequences

- **Image**: 1 image (1024×1024 pixels)
- **Video**: 30–300 frames (each 1024×1024), with a total data volume 30–300 times that of an image

The explosion in sequence length makes credit assignment in RL extremely difficult — in a 100-frame video, which frame or which pixel is problematic?

### Temporal Consistency

A video must not only look good in individual frames, but also be **temporally consistent** — the same person, the same scene, and continuous actions.

```text
Image reward: single-frame quality (clarity, aesthetics, prompt alignment)
Video reward: single-frame quality + temporal consistency + motion smoothness + physical plausibility
```

Video reward is significantly more complex than image reward.

### Computational Costs

- Image generation (diffusion): 50 denoising steps × single frame = several seconds
- Video generation: 50 denoising steps × 100 frames = several minutes

RL training requires a large number of rollouts — each rollout takes several minutes, making the training cost of video RL 100+ times that of image RL.

### Scarcity of Reward Models

Image reward models include [LAION-Aesthetics](https://laion.ai/blog/laion-aesthetics/), [PickScore](https://arxiv.org/abs/2305.01569), and other open-source models. Video reward models are almost nonexistent — the cost of labeling video preference data is more than 10 times that of image data.

These challenges have slowed progress in video generation RL in 2024. The major industrial breakthroughs in 2025 come from two directions:

- **DanceGRPO**: Applying the GRPO idea to diffusion (image + video)
- **Seedance / LongCat**: Using RLHF-style training + engineering optimization

## 24.5.2 DanceGRPO: GRPO for Diffusion

[DanceGRPO](https://arxiv.org/abs/2505.07818) (ByteDance Seed, 2025.05) is a significant breakthrough in diffusion RL. Its core contribution is: **applying the GRPO idea directly to diffusion training**.

### Core Idea of DanceGRPO

Reviewing [Chapter 15 GRPO](../chapter18_grpo/grpo-practice-and-mechanism):

- Generate G rollouts for the same prompt
- Compute the reward for each rollout
- Use intra-group normalization to obtain advantage
- No need for a critic

DanceGRPO applies this idea to diffusion:

```text
┌─────────────────────────────────────────────────────────┐
│ 1. For the same prompt, let diffusion generate G videos │
│    (G is typically 4-8)                                 │
├─────────────────────────────────────────────────────────┤
│ 2. Use the video reward model to score each video       │
├─────────────────────────────────────────────────────────┤
│ 3. Intra-group normalization (subtract mean, optionally divide by std) to obtain advantage │
├─────────────────────────────────────────────────────────┤
│ 4. Use policy gradient to update the parameters of diffusion │
└─────────────────────────────────────────────────────────┘
```

This process is almost identical to GRPO for LLMs — the only difference is:

- Rollouts for LLMs are token sequences
- Rollouts for diffusion are denoising trajectories

### Comparison between DanceGRPO and DDPO

| Dimension            | DDPO                    | DanceGRPO                                            |
| -------------------- | ----------------------- | ---------------------------------------------------- |
| Advantage Estimation | Single rollout + reward | Normalization within group                           |
| Requires Critic      | No                      | No                                                   |
| Training Stability   | Moderate                | Significant improvement                              |
| Training Efficiency  | Medium                  | High (group normalization strengthens reward signal) |
| Applicable Model     | Early diffusion         | Modern video diffusion                               |

Key advantages of DanceGRPO:

1. **A clearer reward signal** — comparing multiple videos generated from the same prompt reveals "which video is truly better," not just the absolute score.
2. **No critic needed** — it avoids a value model, consistent with GRPO for LLMs.
3. **More stable training** — group normalization avoids update jitter from inconsistent reward scales across prompts.

### Experiments with DanceGRPO

Byte Seed trained multiple video generation models using DanceGRPO:

- **Image Generation** (FLUX, SD3): Aesthetic score improved by 15–20%
- **Video Generation** (Wan, Seedance): Dynamic quality improved by 10–15%

DanceGRPO has already become the default choice in industry for diffusion reinforcement learning — this aligns with GRPO's status in the LLM field.

## 24.5.3 Seedance: ByteDance's Video Generation Flagship

[Seedance](https://seed.bytedance.com/) (ByteDance, released in March 2025, upgraded to 1.0 Pro in October 2025) is one of the leading video generation models in China. It has ranked first multiple times on VBench (video generation benchmark).

### Seedance's Training Process

```text
┌──────────────────────────────────────────────────────────┐
│ Phase 1: Large-scale video pre-training                   │
│   - Billions of video-text pairs                         │
│   - Learning the basic distribution of videos            │
├──────────────────────────────────────────────────────────┤
│ Phase 2: High-quality data SFT                           │
│   - Filtering high-quality videos (4K, professional shot)│
│   - Teaching the model what "high quality" means         │
├──────────────────────────────────────────────────────────┤
│ Phase 3: DanceGRPO RL                                    │
│   - Using video reward model for reinforcement learning │
│   - Optimizing prompt following, dynamic quality, and temporal consistency │
├──────────────────────────────────────────────────────────┤
│ Phase 4: Expert Iteration                                │
│   - RL → Collect new data → SFT → RL → ...               │
│   - Data flywheel                                        │
└──────────────────────────────────────────────────────────┘
```

### Reward Design in Seedance

The reward in Seedance is composed of multiple components:

**Component 1: Prompt Following**

Does the video content align with the prompt description? Scored using a video-text alignment model.

**Component 2: Aesthetic Quality**

Video aesthetics — composition, color, lighting. Scored using an aesthetic model.

**Component 3: Motion Quality**

Naturalness of motion — are the human actions and object movements physically plausible? Scored using a motion model.

**Component 4: Temporal Consistency**

Temporal consistency — are the frames in the video coherent across time? Scored using frame-to-frame similarity.

**Component 5: Human Preference**

Human preference — a reward model trained on RLHF preference data.

The final reward is computed as:

$$r_{\text{total}} = w_1 \cdot r_{\text{prompt}} + w_2 \cdot r_{\text{aesthetic}} + w_3 \cdot r_{\text{motion}} + w_4 \cdot r_{\text{temporal}} + w_5 \cdot r_{\text{human}}$$

The weights $w_1, \ldots, w_5$ are optimized through grid search.

### Engineering Optimization of Seedance

**Optimization 1: Latent Diffusion**

Instead of training in the pixel space, training is conducted in the latent space (compressed using a VAE) — significantly reducing computational costs.

**Optimization 2: 3D Attention**

Rather than using attention on single frames, 3D attention (time × space) is employed — capturing temporal dependencies.

**Optimization 3: Classifier-free Guidance**

During training, prompts are randomly dropped (10–20%) so that the model learns unconditional generation. During inference, the guidance scale controls the strength of the conditional generation.

**Optimization 4: Flow Matching**

As an alternative to traditional diffusion, flow matching is used (which is more stable and efficient). This has become a popular diffusion alternative since 2024.

### Performance of Seedance 1.0 Pro

VBench 2025.10 Ranking:

| Model            | VBench Total |
| ---------------- | ------------ |
| Seedance 1.0 Pro | 86.7%        |
| Wan 2.5          | 84.2%        |
| Kling 2.0        | 83.1%        |
| Hailuo 02        | 81.5%        |
| Sora 2 (OpenAI)  | 80.8%        |
| Veo 3 (Google)   | 79.5%        |

Seedance is the state-of-the-art video generation model in China, surpassing Sora 2 and Veo 3.

## 24.5.4 LongCat-Video: Efficient Long-Video Generation

[LongCat-Video](https://arxiv.org/abs/2510.22200) (Meituan, 2025.10) is another important work — focused on **long-video generation**.

### Challenges of Long-Video Generation

Standard video generation lasts 5–10 seconds. LongCat-Video aims for **more than 30 seconds**, bringing new challenges:

- **Context Explosion**: The latent representation of a 30-second video is massive.
- **Story Coherence**: Long videos need to tell a complete story, not just fragments.
- **Computational Cost**: Generating a 30-second video takes more than six times longer than a 5-second video.

### Design of LongCat-Video

**Design 1: Chunked Generation**

The long video is divided into multiple 5-second chunks, each generated independently, but with **overlap regions** to maintain coherence:

```text
Chunk 1: [0-5s]
Chunk 2: [4-9s]  ← Overlaps with Chunk 1 in [4-5s]
Chunk 3: [8-13s] ← Overlaps with Chunk 2 in [8-9s]
...
```

The generated results in the overlap region are averaged to ensure smooth transitions.

**Design 2: Story-level Reward**

It is not only frame-level reward, but also **story-level reward** — using an LLM to evaluate whether the video tells a coherent story.

```python
def story_reward(video, prompt):
    # Use LLM to evaluate the narrative quality of the video
    frames = sample_frames(video, n=10)
    description = vlm.describe(frames)
    story_quality = llm.judge_story(description, prompt)
    return story_quality
```

**Design 3: Hierarchical Diffusion**

Two-level diffusion:

- **High-level**: Generate the "skeleton" (key frames) of the video.
- **Low-level**: Interpolate to generate intermediate frames based on the skeleton.

This hierarchical structure is consistent with the hierarchical RL approach in [DeepSWE's hierarchical RL](../chapter23_rl_based_swe/world-model-and-deep-swe).

### Performance of LongCat-Video

LongCat-Video achieves state-of-the-art results in long video generation:

| Model             | 30-Second Video Consistency | Story Coherence |
| ----------------- | --------------------------- | --------------- |
| Sora 2            | 65%                         | 60%             |
| Veo 3             | 68%                         | 65%             |
| Wan 2.5 Long      | 70%                         | 68%             |
| **LongCat-Video** | **78%**                     | **75%**         |

## 24.5.5 Hailuo: MiniMax Video Generation

[Hailuo](https://hailuoai.video/) (MiniMax, released in September 2024, upgraded in July 2025, version 02) is another Chinese video generation SOTA.

### Features of Hailuo

- **Strong Motion Capture**: Excels in scenarios involving human actions, dance, and sports
- **Physics Simulation**: Relatively accurate simulation of gravity, collisions, and fluids
- **Open Source Ecosystem**: Some models are open-sourced (MiniMax-VL-01)

### Training Method of Hailuo

Hailuo uses a training process similar to Seedance:

- Large-scale pre-training
- High-quality SFT
- DanceGRPO-style RL
- Expert iteration

Internal research at MiniMax (e.g., [CISPO](../chapter18_grpo/grpo-family)) also contributes to the training of Hailuo — the stability of CISPO in low-precision training makes large-scale video RL feasible.

## 24.5.6 Other Mainstream Video Generation Models

### Wan (Alibaba)

[Wan](https://github.com/Wan-Video/Wan2.1) (Alibaba, 2025.02) is an open-source video generation SOTA. Wan 2.1 is open-sourced on HuggingFace and is widely used in the community.

### Kling (Kuaishou)

[Kling](https://klingai.com/) (Kuaishou) — strong in action and physics simulation. Competes with Seedance on multiple benchmarks.

### Sora 2 (OpenAI)

[Sora 2](https://openai.com/sora/) (2025.10) — OpenAI's flagship video generation model. Features include long videos and strong physics simulation.

### Veo 3 (Google)

[Veo 3](https://deepmind.google/models/veo/) (2025.05) — Google's video generation model. Features include audio-synchronized generation (video + audio joint generation).

## 24.5.7 Industrial Landscape of Video Generation with Reinforcement Learning

As of mid-2026, the industrial landscape of video generation with reinforcement learning:

| Vendor    | Representative Model  | Algorithm       | Features                    |
| --------- | --------------------- | --------------- | --------------------------- |
| Byte Seed | Seedance, LongCat     | DanceGRPO       | Chinese SOTA, parallelism   |
| MiniMax   | Hailuo                | CISPO + GRPO    | Strong actions, open source |
| Alibaba   | Wan                   | DanceGRPO       | Open source ecosystem       |
| Kuaishou  | Kling                 | Internal method | Strong physics              |
| OpenAI    | Sora 2                | Not disclosed   | Long video                  |
| Google    | Veo 3                 | Not disclosed   | Audio-video joint           |
| Anthropic | (No video generation) | -               | Focused on text             |

Observations:

- **Chinese vendors lead video generation with reinforcement learning research** — the most open-source papers
- **DanceGRPO is the mainstream algorithm** — an extension of GRPO
- **Data and engineering matter more than algorithmic innovation** — most improvements come from data quality and engineering optimization

## 24.5.8 Future Directions of Video Generation with Reinforcement Learning

### Longer Videos

- **Current SOTA**: 30–60 seconds
- **Future Goal**: 5–10 minutes (short film level)
- **Challenges**: context, coherence, cost

### Audio-Video Joint Generation

- **Current**: Audio and video are generated separately, then composited in post-production
- **Future**: Joint generation with natural synchronization
- **Challenges**: Multimodal reinforcement learning, cross-modal consistency

### Interactive Video Generation

- **Current**: One-time generation of complete video
- **Future**: Users can intervene, modify, and guide the generation process
- **Challenges**: Real-time reinforcement learning, user reward modeling

### Controllable Generation

- **Current**: Only controlled by text prompts
- **Future**: Fine-grained control over pose, motion, camera, lighting, etc.
- **Challenges**: Multi-condition reward modeling, control reinforcement learning

### Physical Plausibility

- **Current**: Physics is mostly "hallucination" — models draw based on memory
- **Future**: True physics simulation
- **Challenges**: Integration with physics engines, physics-based reward modeling

## Summary

Video generation with reinforcement learning has achieved major breakthroughs in 2025:

- **DanceGRPO** applies the GRPO idea to diffusion models, becoming the mainstream algorithm
- **Seedance / LongCat** achieve state-of-the-art performance in industrial applications
- **Hailuo / Wan / Kling** collectively push Chinese research in video generation to the forefront

The core challenges of video generation with reinforcement learning — long sequences, temporal consistency, and computational cost — are being gradually addressed by industrial practice. Future directions include 5–10 minute videos, audio-video joint generation, and interactive generation.

This section, together with [24.4 Visual Generation with Reinforcement Learning](./visual-generation-dancegrpo), forms a complete system:

- **24.4**: Algorithm foundations (DDPO, DPOK)
- **24.5**: Industrial practice (DanceGRPO, Seedance, LongCat)

Together, they cover the full landscape of visual generation with reinforcement learning.

From audio rewards, speech agents, and VLA, Chapter 24 has progressed to image and video generation. The next chapter, [Chapter 25: Reward Hacking and RL Evaluation](../chapter30_alignment_failures/classical-failures), will discuss the reward vulnerabilities, alignment failures, and evaluation issues that must be addressed when extending to multimodal capabilities.
