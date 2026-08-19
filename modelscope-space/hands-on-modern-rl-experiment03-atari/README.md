---
title: Atari CPU 在线训练街机厅
emoji: 🕹️
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 6.17.3
app_file: app.py
pinned: false
license: apache-2.0
---

# WalkingLab × Hands-On Modern RL · Atari CPU Training Arcade

**WalkingLab** 与开源课程 **Hands-On Modern RL（《动手学现代强化学习》）** 的 Atari/ALE 配套实验。每次运行都会在 ModelScope CPU 容器中训练 DQN、评估 checkpoint，并从当前策略生成真实模拟器回放。

Preview 与当前选择的游戏及其最后保存的 DQN 模型严格对应。默认回放种子与训练种子相同；修改 `Preview seed` 后点击 `Run rollout`，只会让同一模型从新的初始状态重新运行一次，不会重新训练。相同模型和相同种子会得到可复现的确定性动作序列。

课程项目：<https://github.com/walkinglabs/hands-on-modern-rl>

WalkingLab：<https://modelscope.cn/organization/walkinglab>

## 配套实验 Notebook

[直接在 ModelScope Notebook 中运行 Atari 实验](https://modelscope.cn/notebook/share/github/walkinglabs/hands-on-modern-rl/blob/main/code/online-experiments/hands-on-modern-rl-experiment03-atari.ipynb)。Notebook 与当前创空间复用同一份 ALE/DQN 训练运行时，并显示完整日志、评估曲线和本次策略回放。
