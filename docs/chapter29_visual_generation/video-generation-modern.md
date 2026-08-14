# 24.5 视频 RLHF 与物理感知生成

[24.4 节视觉生成 RL](./visual-generation-dancegrpo) 讨论了 diffusion RL 的基础——DDPO、DPOK 等算法。那一节的视角是**算法层面**：怎么把 diffusion 训练建模为 MDP，怎么用策略梯度优化。

这一节我们换视角——**工业层面**：2025-2026 年的视频生成模型（Seedance、LongCat-Video、Hailuo、Wan、Kling）是怎么用 RL 训练的？这些工作代表了视频生成 RL 的工业 SOTA。

## 24.5.1 从图像到视频 与 RL 的新挑战

图像生成的 RL 已经成熟（[DDPO](./visual-generation-dancegrpo)、DPOK）。但视频生成带来新挑战：

### 长序列

- 图像：1 张图（1024×1024 像素）
- 视频：30-300 帧（每帧 1024×1024），总数据量是图像的 30-300 倍

序列长度爆炸让 RL 的 credit assignment 变得极其困难——一个 100 帧的视频，哪一帧、哪一像素出了问题？

### 时序一致性

视频不仅要单帧好看，还要**前后帧一致**——同一人物、同一场景、连续动作。

```text
图像 reward：单帧质量（清晰度、美感、prompt 匹配）
视频 reward：单帧质量 + 时序一致性 + 动作流畅性 + 物理合理性
```

视频 reward 比图像复杂得多。

### 计算成本

- 图像生成（diffusion）：50 步去噪 × 单帧 = 几秒
- 视频生成：50 步去噪 × 100 帧 = 几分钟

RL 训练需要大量 rollout——每次 rollout 几分钟，让视频 RL 的训练成本是图像 RL 的 100+ 倍。

### reward model 的稀缺

图像 reward model 有 [LAION-Aesthetics](https://laion.ai/blog/laion-aesthetics/)、[PickScore](https://arxiv.org/abs/2305.01569) 等开源模型。视频 reward model 几乎没有——视频偏好数据标注成本是图像的 10 倍以上。

这些挑战让视频生成 RL 在 2024 年进展缓慢。2025 年的工业突破主要来自两个方向：

- **DanceGRPO**：把 GRPO 思想用到 diffusion（图像 + 视频）
- **Seedance / LongCat**：用 RLHF-style 训练 + 工程优化

## 24.5.2 DanceGRPO 与 Diffusion 的 GRPO

[DanceGRPO](https://arxiv.org/abs/2505.07818)（字节 Seed, 2025.05）是 diffusion RL 的重要突破。它的核心贡献是：**把 GRPO 思想直接用到 diffusion 训练**。

### DanceGRPO 的核心思想

回顾 [第 15 章 GRPO](../chapter18_grpo/grpo-practice-and-mechanism)：

- 对同一 prompt 生成 G 个 rollout
- 计算每个 rollout 的 reward
- 用组内归一化得到 advantage
- 不需要 critic

DanceGRPO 把这个思路用到了 diffusion：

```text
┌─────────────────────────────────────────────────────────┐
│ 1. 对同一 prompt，让 diffusion 生成 G 个视频            │
│    （G 通常 4-8）                                       │
├─────────────────────────────────────────────────────────┤
│ 2. 用 video reward model 给每个视频打分                 │
├─────────────────────────────────────────────────────────┤
│ 3. 组内归一化（减均值，可选除 std）得到 advantage       │
├─────────────────────────────────────────────────────────┤
│ 4. 用策略梯度更新 diffusion 的参数                      │
└─────────────────────────────────────────────────────────┘
```

这个流程与 LLM 的 GRPO 几乎完全一样——区别只是：

- LLM 的 rollout 是 token 序列
- Diffusion 的 rollout 是去噪轨迹

### DanceGRPO 与 DDPO 的对比

| 维度           | DDPO                | DanceGRPO                             |
| -------------- | ------------------- | ------------------------------------- |
| Advantage 估计 | 单 rollout + reward | 组内归一化                            |
| 需要 Critic    | 否                  | 否                                    |
| 训练稳定性     | 一般                | 显著提升                              |
| 训练效率       | 中                  | 高（组内归一化让 reward signal 更强） |
| 适用模型       | 早期 diffusion      | 现代 video diffusion                  |

DanceGRPO 的核心优势：

1. **组内归一化让 reward signal 更清晰**——同一 prompt 的多个视频比较，能识别"哪个视频真的更好"
2. **不需要 critic**——节省一个 value model，与 GRPO 一样
3. **训练稳定**——组内归一化让 advantage 估计更稳定

### DanceGRPO 的实验

字节 Seed 用 DanceGRPO 训练了多个视频生成模型：

- **图像生成**（FLUX、SD3）：美感分数提升 15-20%
- **视频生成**（Wan、Seedance）：动态质量提升 10-15%

DanceGRPO 在工业上已经替代 DDPO/DPOK 成为 diffusion RL 的默认选择——这与 GRPO 在 LLM 领域的地位一致。

## 24.5.3 Seedance 与 字节跳动的视频生成旗舰

[Seedance](https://seed.bytedance.com/)（字节跳动，2025.03 发布，2025.10 升级 1.0 Pro）是中国视频生成 SOTA 之一。它在 VBench（视频生成 benchmark）上多次排名第一。

### Seedance 的训练流程

```text
┌──────────────────────────────────────────────────────────┐
│ Phase 1: 大规模视频预训练                                │
│   - 数亿视频-文本对                                      │
│   - 学习视频的基本分布                                   │
├──────────────────────────────────────────────────────────┤
│ Phase 2: 高质量数据 SFT                                  │
│   - 筛选高质量视频（4K、专业拍摄）                       │
│   - 让模型学会"高质量"是什么样的                         │
├──────────────────────────────────────────────────────────┤
│ Phase 3: DanceGRPO RL                                    │
│   - 用 video reward model 做 RL                          │
│   - 优化 prompt 跟随、动态质量、时序一致性               │
├──────────────────────────────────────────────────────────┤
│ Phase 4: Expert Iteration                                │
│   - RL → 收集新数据 → SFT → RL → ...                    │
│   - 数据 flywheel                                        │
└──────────────────────────────────────────────────────────┘
```

### Seedance 的 Reward 设计

Seedance 的 reward 由多个组件组成：

**组件 1：Prompt Following**

视频内容是否符合 prompt 描述？用 video-text alignment model 评分。

**组件 2：Aesthetic Quality**

视频美感——构图、色彩、光线。用 aesthetic model 评分。

**组件 3：Motion Quality**

动作自然度——人物动作、物体运动是否符合物理？用 motion model 评分。

**组件 4：Temporal Consistency**

时序一致性——前后帧是否连贯？用 frame-to-frame similarity 评分。

**组件 5：Human Preference**

人类偏好——通过 RLHF 偏好数据训练的 reward model。

最终 reward：

$$r_{\text{total}} = w_1 \cdot r_{\text{prompt}} + w_2 \cdot r_{\text{aesthetic}} + w_3 \cdot r_{\text{motion}} + w_4 \cdot r_{\text{temporal}} + w_5 \cdot r_{\text{human}}$$

权重 $w_1, \ldots, w_5$ 通过 grid search 调优。

### Seedance 的工程优化

**优化 1：Latent diffusion**

不在 pixel space 训练，而在 latent space（用 VAE 压缩）——大幅减少计算量。

**优化 2：3D Attention**

不是单帧 attention，而是 3D attention（时间 × 空间）——捕获时序依赖。

**优化 3：Classifier-free guidance**

训练时随机 drop prompt（10-20%），让模型学会无条件生成。推理时用 guidance scale 控制条件强度。

**优化 4：Flow matching**

替代传统 diffusion，用 flow matching（更稳定、更高效）。这是 2024 年开始流行的 diffusion 替代方案。

### Seedance 1.0 Pro 的成绩

VBench 2025.10 排行：

| 模型             | VBench Total |
| ---------------- | ------------ |
| Seedance 1.0 Pro | 86.7%        |
| Wan 2.5          | 84.2%        |
| Kling 2.0        | 83.1%        |
| Hailuo 02        | 81.5%        |
| Sora 2（OpenAI） | 80.8%        |
| Veo 3（Google）  | 79.5%        |

Seedance 是中国视频生成 SOTA，超越 Sora 2 和 Veo 3。

## 24.5.4 LongCat-Video 与 高效长视频生成

[LongCat-Video](https://arxiv.org/abs/2510.22200)（美团, 2025.10）是另一个重要工作——专注**长视频生成**。

### 长视频的挑战

标准视频生成 5-10 秒。LongCat-Video 目标是 **30 秒以上**，带来新挑战：

- **Context 爆炸**：30 秒视频的 latent 表征巨大
- **故事连贯性**：长视频需要讲一个完整故事，不只是片段
- **计算成本**：30 秒视频生成时间是 5 秒的 6 倍以上

### LongCat-Video 的设计

**设计 1：分块生成（Chunked Generation）**

把长视频分成多个 5 秒 chunk，每个 chunk 独立生成，但通过 **overlap region** 保持连贯：

```text
Chunk 1: [0-5s]
Chunk 2: [4-9s]  ← 与 Chunk 1 在 [4-5s] overlap
Chunk 3: [8-13s] ← 与 Chunk 2 在 [8-9s] overlap
...
```

Overlap 区域的生成结果被平均，保证平滑过渡。

**设计 2：Story-level Reward**

不只是 frame-level reward，还有 **story-level reward**——用 LLM 评估视频是否讲了一个连贯的故事。

```python
def story_reward(video, prompt):
    # 用 LLM 评估视频叙事质量
    frames = sample_frames(video, n=10)
    description = vlm.describe(frames)
    story_quality = llm.judge_story(description, prompt)
    return story_quality
```

**设计 3：Hierarchical Diffusion**

两级 diffusion：

- **High-level**：生成视频的"骨架"（关键帧）
- **Low-level**：在骨架基础上插值生成中间帧

这种分层结构与 [DeepSWE 的分层 RL](../chapter23_rl_based_swe/world-model-and-deep-swe) 思路一致。

### LongCat-Video 的成绩

LongCat-Video 在长视频生成上达到 SOTA：

| 模型              | 30 秒视频一致性 | 故事连贯性 |
| ----------------- | --------------- | ---------- |
| Sora 2            | 65%             | 60%        |
| Veo 3             | 68%             | 65%        |
| Wan 2.5 Long      | 70%             | 68%        |
| **LongCat-Video** | **78%**         | **75%**    |

## 24.5.5 Hailuo 与 MiniMax 的视频生成

[Hailuo](https://hailuoai.video/)（MiniMax，2024.09 发布，2025.07 升级 02）是另一个中国视频生成 SOTA。

### Hailuo 的特点

- **强动作捕捉**：在人物动作、舞蹈、运动等场景表现突出
- **物理模拟**：相对准确地模拟重力、碰撞、流体
- **开源生态**：部分模型开源（MiniMax-VL-01）

### Hailuo 的训练方法

Hailuo 用了类似 Seedance 的训练流程：

- 大规模预训练
- 高质量 SFT
- DanceGRPO-style RL
- Expert iteration

MiniMax 内部的研究（如 [CISPO](../chapter18_grpo/grpo-family)）也对 Hailuo 的训练有贡献——CISPO 在低精度训练中的稳定性让大规模 video RL 成为可能。

## 24.5.6 其他主流视频生成模型

### Wan（阿里）

[Wan](https://github.com/Wan-Video/Wan2.1)（阿里，2025.02）是开源视频生成 SOTA。Wan 2.1 在 HuggingFace 开源，社区广泛使用。

### Kling（快手）

[Kling](https://klingai.com/)（快手）—— 强动作、强物理模拟。在多项 benchmark 上与 Seedance 竞争。

### Sora 2（OpenAI）

[Sora 2](https://openai.com/sora/)（2025.10）—— OpenAI 视频生成旗舰。特点是长视频、强物理模拟。

### Veo 3（Google）

[Veo 3](https://deepmind.google/models/veo/)（2025.05）—— Google 视频生成。特点是音频同步生成（视频 + 音频联合）。

## 24.5.7 视频生成 RL 的工业格局

到 2026 年中，视频生成 RL 的工业格局：

| 厂商      | 代表模型          | 算法         | 特点                |
| --------- | ----------------- | ------------ | ------------------- |
| 字节 Seed | Seedance, LongCat | DanceGRPO    | 中国 SOTA，多线并行 |
| MiniMax   | Hailuo            | CISPO + GRPO | 强动作，开源        |
| 阿里      | Wan               | DanceGRPO    | 开源生态            |
| 快手      | Kling             | 内部方法     | 强物理              |
| OpenAI    | Sora 2            | 未公开       | 长视频              |
| Google    | Veo 3             | 未公开       | 音视频联合          |
| Anthropic | （不做视频生成）  | -            | 专注文本            |

可以看到：

- **中国厂商主导视频生成 RL 研究**——开源论文最多
- **DanceGRPO 是主流算法**——基于 GRPO 的扩展
- **数据 + 工程 > 算法创新**——大部分提升来自数据质量和工程优化

## 24.5.8 视频生成 RL 的未来方向

### 更长的视频

- 当前 SOTA：30-60 秒
- 未来目标：5-10 分钟（短片级别）
- 挑战：context、coherence、cost

### 音视频联合生成

- 当前：音频和视频分别生成，后期合成
- 未来：联合生成，自然同步
- 挑战：多模态 RL，跨模态 consistency

### 交互式视频生成

- 当前：一次性生成完整视频
- 未来：用户可以干预、修改、引导生成
- 挑战：实时 RL、用户 reward

### 可控生成

- 当前：只能用文本 prompt 控制
- 未来：pose、motion、camera、lighting 等精细控制
- 挑战：多条件 reward，control RL

### 物理合理性

- 当前：物理基本是"幻觉"——模型凭记忆画
- 未来：真正的物理模拟
- 挑战：与 physics engine 集成，physics reward

## 小结

视频生成 RL 在 2025 年取得了重大突破：

- **DanceGRPO** 把 GRPO 思想用到 diffusion，成为主流算法
- **Seedance / LongCat** 在工业上达到视频生成 SOTA
- **Hailuo / Wan / Kling** 共同推动中国视频生成研究领先

视频生成 RL 的核心挑战——长序列、时序一致性、计算成本——正在被工业实践逐步解决。未来 5-10 分钟视频、音视频联合、交互式生成是主要方向。

本节与 [24.4 视觉生成 RL](./visual-generation-dancegrpo) 形成完整体系：

- 24.4：算法基础（DDPO、DPOK）
- 24.5：工业实践（DanceGRPO、Seedance、LongCat）

两者一起覆盖了视觉生成 RL 的全貌。

至此，第 24 章从音频奖励、语音 Agent 和 VLA 一路推进到图像与视频生成。下一章进入[第 25 章奖励黑客与 RL 评估](../chapter30_alignment_failures/classical-failures)，讨论多模态能力扩展后必须面对的奖励漏洞、对齐失败与评估问题。
