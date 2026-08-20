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

每次成功训练都会保存为一个独立的 DQN 模型，不再覆盖上一次结果。Preview 区域的“已训练模型”选择框会列出当前游戏的全部成功运行；选择某个模型后，页面显示该模型在训练完成时生成的策略回放。新启动且尚未训练的创空间会显示空选择框。

配置区默认载入每个游戏各自的 **Atari DQN CPU baseline v1**。该配方使用 4 帧堆叠、训练奖励裁剪、原始奖励评估、经验回放预热、`ε=1.0→0.01` 探索调度，并在全部检查点中保存评估最好的策略用于最终 GIF。低于推荐预算的设置仍可用于 smoke test，但页面会明确提示它通常不足以产生可辨认的学习行为。Freeway 的 300,000 步 baseline 是最快的首次验证入口；其余视觉任务默认需要 1,000,000–2,000,000 步。

为避免短流程测试把大部分时间耗在评估上，低于推荐预算的 smoke run 自动使用 2 个检查点、每个检查点 1 个评估回合；达到推荐预算的正式 baseline 仍使用 6 个检查点、每次 3 个评估回合。

课程项目：<https://github.com/walkinglabs/hands-on-modern-rl>

WalkingLab：<https://modelscope.cn/organization/walkinglab>

## 配套实验 Notebook

[直接在 ModelScope Notebook 中运行 Atari 实验](https://modelscope.cn/notebook/share/github/walkinglabs/hands-on-modern-rl/blob/main/code/online-experiments/hands-on-modern-rl-experiment03-atari.ipynb)。Notebook 与当前创空间复用同一份 ALE/DQN 训练运行时，并显示完整日志、评估曲线和本次策略回放。
