# 24.5 Temporal Consistency in Video

Imagine four sharp, naturally lit frames of a red ball hitting toy blocks. A frame-by-frame inspection passes. The prompt also passes: it asks for a collision followed by the ball stopping, and all of those objects and events appear. The timeline reveals a different failure. The ball's path breaks between seconds three and four, and the blocks fall before contact occurs.

This example needs four separate checks:

1. **Single-frame quality** checks anatomy, edges, lighting, and composition independently in every frame.
2. **Text and event alignment** checks whether the requested objects, actions, and event order appear.
3. **Temporal consistency** checks whether identity, position, appearance, and background structure remain explainable from one frame to the next.
4. **Physical and causal consistency** checks whether contact precedes its effect and whether motion follows a continuous path.

A single total score can hide the third and fourth failures behind high image quality. Training and evaluation must preserve these component signals.

For a video $x=(x_1,\ldots,x_F)$, $F$ is the frame count. Five seconds at 16 fps produces 80 frames; at 24 fps it produces 120. A generator must satisfy spatial constraints within each frame and temporal constraints across adjacent frames.

## Why One Terminal Reward Is Not Enough

Video generation can be treated as a sampling trajectory $\tau$. The simplest objective gives one reward after the final video:

$$
R(\tau)=R_{\text{quality}}+R_{\text{alignment}}+R_{\text{temporal}}+R_{\text{physics}}.
$$

A five-second video may require roughly 50 denoising steps, each making decisions over latent representations for 80 frames—on the order of four thousand model evaluations. A teleportation defect at second three may arise from only a small part of that decision chain, while the terminal evaluator returns one scalar.

Suppose candidate A scores $(0.9,0.8,0.3,0.35)$ on the four components and candidate B scores $(0.6,0.55,0.6,0.65)$. Equal weighting produces 0.59 for A and 0.60 for B. The totals say only that B is slightly better. The components reveal that A has a localized temporal and causal failure, while B is uniformly mediocre. Those diagnoses require different updates.

The formula above is a teaching template, not a shared objective disclosed by VADER, DanceGRPO, and Seedance. Every component can itself be exploited. Adjacent-frame similarity rewards nearly static videos; unconstrained motion rewards meaningless camera shake. Each reward therefore needs an independent evaluation that was not used for training.

## Three Ways to Apply Video Rewards

### VADER: Differentiate Through the Denoising Chain

[VADER](https://arxiv.org/abs/2407.08737) differentiates a reward through a video diffusion model:

$$
J(\theta)=\mathbb{E}_{c,\,x_0\sim p_\theta(x_0\mid c)}[R(x_0,c)],
$$

$$
\nabla_\theta R(x_0,c)=\sum_{t=0}^{T}
\frac{\partial R(x_0,c)}{\partial x_t}
\frac{\partial x_t}{\partial\theta}.
$$

Backpropagating through the entire chain is memory intensive. VADER keeps gradients only through the last $K$ denoising steps and stops gradients earlier. It combines frame-level HPSv2 and PickScore alignment, a VideoMAE action score, and V-JEPA temporal predictability. Smooth representation dynamics improve temporal stability but do not by themselves prove Newtonian physics.

VADER also extends Stable Video Diffusion autoregressively by feeding the last generated frame into the next segment. Direct extension fails because the model has not trained on its own outputs; a V-JEPA consistency reward reduces the resulting distribution shift. The route is sample efficient because a differentiable evaluator supplies dense pixel-directed gradients, but it requires a stable differentiable reward whose bias becomes the optimization direction.

### DanceGRPO: Normalize Each Black-Box Reward

[DanceGRPO](https://arxiv.org/abs/2505.07818) constructs stochastic transitions for diffusion and rectified-flow sampling, generates a group under one condition, and updates from relative advantages. When rewards have different scales, it normalizes each reward within the group before summing:

$$
A_i=\sum_{k=1}^{K}\frac{r_i^k-\mu^k}{\sigma^k}.
$$

The paper reports that HPS-v2.1 alone can produce an unnatural glossy appearance; adding CLIP constrains that failure. In image-to-video generation, the input image already fixes much of content and alignment, leaving motion quality as the principal degree of freedom. Using VideoAlign's motion score produced a 118% relative improvement on that dimension. DanceGRPO can also learn from thresholded binary feedback, but each update must generate a complete group of videos.

### Industrial Reports: Separate Disclosed Facts from Unknown Recipes

Seedance uses a DiT backbone with visual tokens from a VAE and text encoded by a tuned decoder-only language model. Its report describes pretraining, continued training, supervised fine-tuning, and RLHF, plus a separately trained super-resolution refiner. Continued training filters for aesthetics and optical-flow motion; short captions omit static information already supplied by the first frame, forcing alignment toward motion. SFT trains specialized models on curated categories and then merges them.

The report describes direct maximization of multiple rewards during simulated inference and alternating updates of the diffusion model and reward models. It applies RLHF to the refiner as well. Distillation reduces sampling steps, a redesigned VAE decoder gives about a 2× decoding speedup, and the public measurement reports 41.4 seconds for a five-second 1080p video on an NVIDIA L20—about a 10× end-to-end speedup.

LongCat-Video unifies text-to-video, image-to-video, and continuation by varying the number of conditioning frames. It first generates 480p at 15 fps, then uses a LoRA refinement expert for 720p at 30 fps. Block-sparse attention selects the most relevant three-dimensional key blocks and retains under 10% of dense attention computation in the reported setting.

Its GRPO post-training uses LoRA and stabilizes reward normalization with the maximum standard deviation across groups:

$$
\hat A_{k,t}^{i}=\frac{R_k(x_0^i,c_j)-\mu_k}{\sigma_{\max}}.
$$

The four disclosed signals cover visual quality, motion, and text-video alignment. A gray-scale VideoAlign reward isolates motion, while a color model checks alignment. Training only HPSv3 drives the model toward static video; the motion reward counteracts that exploit. For a 720p, 93-frame sample, the report reduces 1429.5 seconds for dense 50-step generation to 244.6 seconds after 16-step distillation and 116.5 seconds after coarse-to-fine generation and block-sparse attention.

Product pages for Sora and Veo demonstrate capabilities but disclose only parts of their data, reward composition, and optimizer. Undisclosed industrial recipes should remain labeled as unknown.

Consider a five-second clip of a red ball rolling behind a box and emerging on the other side. Every individual frame can look sharp while the clip still fails: the ball changes size, disappears too early, or emerges before it reaches the box. An image scorer sees several attractive frames. A viewer sees one broken event.

Video therefore changes the object we must evaluate. The model must preserve identity, motion, contact, and causal order across a timeline. A single terminal score can still train the model, but it cannot tell us which moment caused the failure, and a visually strong reward can hide temporal mistakes.

This section follows that problem from simple frame checks to video-specific rewards, [VADER](https://arxiv.org/abs/2407.08737), [DanceGRPO](https://arxiv.org/abs/2505.07818), Seedance, and LongCat-Video. We will finish with a small evaluation harness that can test physical consistency without first training a video model.

<img src="../../chapter29_visual_generation/images/seedance-stage-comparison.png" alt="Video samples from different Seedance training stages" style="width: auto; max-width: 100%; max-height: 620px;" />

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em>Figure 1: Video frames produced after pretraining, continued training, supervised fine-tuning, and RLHF. Read each row along time: check whether the action and identity remain continuous instead of judging one attractive frame. Source: <a href="https://arxiv.org/abs/2506.09113" target="_blank" rel="noopener noreferrer">Seedance 1.0 technical report</a>.</em>
</div>

## 24.5.1 What a Timeline Adds

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

RL training requires many rollouts, so longer clips, higher resolution, and more denoising steps make each update substantially more expensive. The factor depends on architecture, compression, clip length, and parallelism; it should be measured rather than treated as one universal multiplier.

### Scarcity of Reward Models

Image reward resources include [LAION-Aesthetics](https://laion.ai/blog/laion-aesthetics/) and [PickScore](https://arxiv.org/abs/2305.01569). Video evaluation has fewer standardized public reward models because annotators must judge both frame quality and temporal behavior. Collection cost depends strongly on clip length and labeling protocol.

These challenges have slowed progress in video generation RL in 2024. The major industrial breakthroughs in 2025 come from two directions:

- **DanceGRPO**: Applying the GRPO idea to diffusion (image + video)
- **Seedance / LongCat**: Using RLHF-style training + engineering optimization

## 24.5.2 DanceGRPO and GRPO for Diffusion

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

- **Dimension — Advantage Estimation**
  - DDPO: Single rollout + reward
  - DanceGRPO: Normalization within group
- **Dimension — Requires Critic**
  - DDPO: No
  - DanceGRPO: No
- **Dimension — Training Stability**
  - DDPO: Moderate
  - DanceGRPO: Significant improvement
- **Dimension — Training Efficiency**
  - DDPO: Medium
  - DanceGRPO: High (group normalization strengthens reward signal)
- **Dimension — Applicable Model**
  - DDPO: Early diffusion
  - DanceGRPO: Modern video diffusion

Key advantages of DanceGRPO:

1. **A clearer reward signal** — comparing multiple videos generated from the same prompt reveals "which video is truly better," not just the absolute score.
2. **No critic needed** — it avoids a value model, consistent with GRPO for LLMs.
3. **More stable training** — group normalization avoids update jitter from inconsistent reward scales across prompts.

### Experiments with DanceGRPO

Byte Seed trained multiple video generation models using DanceGRPO:

The DanceGRPO paper reports experiments on image and video generation backbones under several reward signals. The useful evidence is the per-reward comparison and ablation, not a universal percentage that applies across models. DanceGRPO is one representative critic-free route for visual generation RL; differentiable-reward methods and preference optimization remain separate alternatives.

## 24.5.3 What the Seedance Technical Report Actually Shows

The [Seedance 1.0 technical report](https://arxiv.org/abs/2506.09113) describes a staged system: large-scale pretraining establishes basic generation, continued training and fine-grained supervised fine-tuning improve data quality and instruction following, and video-specific RLHF further adjusts motion, temporal coherence, and preference. This evidence is more precise than inferring a complete training recipe from a product demo.

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
│ Phase 3: Video-specific RLHF                              │
│   - Compare or score generated clips                     │
│   - Improve instruction following, motion, and coherence │
├──────────────────────────────────────────────────────────┤
│ Phase 4: Expert Iteration                                │
│   - RL → Collect new data → SFT → RL → ...               │
│   - Data flywheel                                        │
└──────────────────────────────────────────────────────────┘
```

### Reward Design in Seedance

The report motivates monitoring several dimensions separately. A practical video reward system may include the following signals:

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

For teaching purposes, we can write a weighted reward template as

$$r_{\text{total}} = w_1 \cdot r_{\text{prompt}} + w_2 \cdot r_{\text{aesthetic}} + w_3 \cdot r_{\text{motion}} + w_4 \cdot r_{\text{temporal}} + w_5 \cdot r_{\text{human}}$$

Here the $w_i$ values express engineering trade-offs. This equation is a teaching template, not a claim that the report fixes one universal set of weights. A single total reward should always be accompanied by the separate curves, because one component can improve while another declines.

<img src="../../chapter29_visual_generation/images/seedance-reward-curves.png" alt="Multiple reward curves reported for Seedance" style="width: 100%; max-width: 760px; max-height: none;" />

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em>Figure 2: Multiple reward signals reported by Seedance. Base quality, motion, and aesthetics must be monitored separately; one aggregate curve cannot show whether they improve together. Source: <a href="https://arxiv.org/abs/2506.09113" target="_blank" rel="noopener noreferrer">Seedance 1.0 technical report</a>.</em>
</div>

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

- **Model — Seedance 1.0 Pro:** 86.7%
- **Model — Wan 2.5:** 84.2%
- **Model — Kling 2.0:** 83.1%
- **Model — Hailuo 02:** 81.5%
- **Model — Sora 2 (OpenAI):** 80.8%
- **Model — Veo 3 (Google):** 79.5%

Seedance is the state-of-the-art video generation model in China, surpassing Sora 2 and Veo 3.

## 24.5.4 LongCat-Video and Efficient Long-Video Generation

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

- **Model — Sora 2**
  - 30-Second Video Consistency: 65%
  - Story Coherence: 60%
- **Model — Veo 3**
  - 30-Second Video Consistency: 68%
  - Story Coherence: 65%
- **Model — Wan 2.5 Long**
  - 30-Second Video Consistency: 70%
  - Story Coherence: 68%
- **Model — LongCat-Video**
  - 30-Second Video Consistency: **78%**
  - Story Coherence: **75%**

## 24.5.5 Hailuo and MiniMax Video Generation

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

- **Vendor — Byte Seed**
  - Representative Model: Seedance, LongCat
  - Algorithm: DanceGRPO
  - Features: Chinese SOTA, parallelism
- **Vendor — MiniMax**
  - Representative Model: Hailuo
  - Algorithm: CISPO + GRPO
  - Features: Strong actions, open source
- **Vendor — Alibaba**
  - Representative Model: Wan
  - Algorithm: DanceGRPO
  - Features: Open source ecosystem
- **Vendor — Kuaishou**
  - Representative Model: Kling
  - Algorithm: Internal method
  - Features: Strong physics
- **Vendor — OpenAI**
  - Representative Model: Sora 2
  - Algorithm: Not disclosed
  - Features: Long video
- **Vendor — Google**
  - Representative Model: Veo 3
  - Algorithm: Not disclosed
  - Features: Audio-video joint
- **Vendor — Anthropic**
  - Representative Model: (No video generation)
  - Algorithm: -
  - Features: Focused on text

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

## 24.5.8 A Minimal Harness for Physical Consistency

We can test the central failure modes without training a video model. Start with 20 prompts, each describing one observable event: an object passes behind an occluder, a cup receives water, a ball bounces after contact, or a person picks up one named object. Generate several clips per prompt and keep the seed, sampler, resolution, frame rate, and model version fixed.

For each clip, save four kinds of evidence:

- **Identity:** does the same object keep its color, shape, and count?
- **Trajectory:** does its position change continuously rather than teleport?
- **Contact order:** does contact occur before the resulting motion or deformation?
- **Task completion:** does the requested event actually finish?

Suppose the ball is at horizontal positions 8, 12, and 15 in three consecutive frames. The changes are 4 and 3, which is plausible under smooth motion. Positions 10, 12, and 2 would contain a sudden jump. A simple trajectory diagnostic is

$$
e_{\text{motion}}=\frac{1}{T-2}\sum_{t=2}^{T-1}
\left\|p_{t+1}-2p_t+p_{t-1}\right\|_2,
$$

where $p_t$ is the tracked object position in frame $t$. This second difference measures abrupt changes in velocity. It is only a diagnostic: camera cuts and intentional impacts can create large values, so the metric must be read together with the prompt and annotated event boundaries.

Run the same harness before and after post-training. Report each physical dimension separately, include failure clips rather than only successful examples, and use human review on ambiguous cases. This makes it possible to distinguish “prettier frames” from “more coherent events.”

## Summary

Moving from images to video adds a timeline. Reward design must therefore preserve frame quality while measuring identity, motion, event order, and physical consistency. VADER propagates differentiable reward gradients, DanceGRPO compares groups under the same condition, and video RLHF learns from preference signals; each route pays a different compute and evaluator cost.

Technical reports such as Seedance and LongCat explain parts of modern training systems, while product demonstrations from Hailuo, Wan, Kling, Sora, and Veo show capabilities without revealing every training detail. A reliable reading keeps those two kinds of evidence separate.

The practical deliverable is the evaluation harness: fixed prompts and seeds, separate physical dimensions, repeated samples, ambiguous-case review, and retained failure clips. The next chapter, [Reward Hacking and RL Evaluation](../chapter30_alignment_failures/classical-failures), studies what happens when the evaluator itself becomes the object that optimization learns to exploit.

Official resources: [Seedance project page](https://seed.bytedance.com/) and [Seedance 1.0 technical report](https://arxiv.org/abs/2506.09113).
