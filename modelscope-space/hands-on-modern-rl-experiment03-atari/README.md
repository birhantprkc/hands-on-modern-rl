---
title: Atari xGPU 在线训练街机厅
emoji: 🕹️
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 6.17.3
app_file: app.py
pinned: false
license: apache-2.0
---

# WalkingLab × Hands-On Modern RL · Atari xGPU Training Arcade

**WalkingLab** 与开源课程 **Hands-On Modern RL（《动手学现代强化学习》）** 的 Atari/ALE 配套实验。每次运行都会在 ModelScope xGPU 容器中使用 CUDA 训练 DQN、评估 checkpoint，并从当前策略生成真实模拟器回放。训练入口会先验证 CUDA；xGPU 未被正确调度时会明确停止，不会静默退回 CPU。

`budget` 表示一次运行的总环境步数，`Saved policy checkpoints` 表示这次运行要保留的策略数量（2–12）。页面会实时列出精确保存步数。默认 baseline 保存 6 个策略；因此一次 1,000,000 步训练会在约 166,667、333,333、500,000、666,667、833,333 和 1,000,000 步生成策略快照。即使训练中途停止，已经完成的检查点仍会保留。

Preview 区域的“策略检查点”选择框按运行 ID 和训练步数列出模型。训练结束时只立即录制本次评估最佳检查点的 GIF；选择其他检查点时，页面会从该模型权重生成一次真实 ALE 回放并缓存，避免训练过程被六次录屏拖慢。每个检查点只保存网络与优化器状态，不包含体积很大的经验回放缓冲区。

配置区默认载入每个游戏各自的 **Atari DQN xGPU baseline v3**。该配方使用 4 帧堆叠、训练奖励裁剪、原始奖励评估、经验回放预热、`ε=1.0→0.01` 探索调度，并在全部检查点中保存评估最好的策略用于最终 GIF。页面为实时日志而分段调用训练器，但学习率和探索率始终共享同一个完整 `budget` 调度，不会在第一段提前衰减到末值。低于推荐预算的设置仍可用于 smoke test，但页面会明确提示它通常不足以产生可辨认的学习行为。Freeway 的 300,000 步 baseline 是最快的首次验证入口；其余视觉任务默认需要 1,000,000–2,000,000 步。

低于推荐预算的 smoke run 每个检查点只运行 1 个评估回合；正式 baseline 每次运行 3 个评估回合。在网页中做快速工程验证时，建议把检查点数调为 2；Notebook 或 Python API 没有显式传入该参数时，短流程自动使用 2 个，正式 baseline 自动使用 6 个。
训练预算滑块最低支持 1,000 步，便于快速验证环境、日志、模型保存和 GIF 流程；该档位只验证工程链路，不代表策略已经学会游戏。

课程项目：<https://github.com/walkinglabs/hands-on-modern-rl>

WalkingLab：<https://modelscope.cn/organization/walkinglab>

## 配套实验 Notebook

[直接在 ModelScope Notebook 中运行 Atari 实验](https://modelscope.cn/notebook/share/github/walkinglabs/hands-on-modern-rl/blob/main/code/online-experiments/hands-on-modern-rl-experiment03-atari.ipynb)。Notebook 与当前创空间复用同一份 ALE/DQN 训练运行时，并显示完整日志、评估曲线和本次策略回放；运行时需要选择 xGPU Notebook。
