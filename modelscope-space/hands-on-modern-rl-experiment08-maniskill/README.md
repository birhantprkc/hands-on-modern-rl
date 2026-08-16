---
title: ManiSkill xGPU Robot Lab
emoji: 🧪
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.17.3
app_file: app.py
pinned: false
license: apache-2.0
---

# WalkingLab × Hands-On Modern RL · ManiSkill xGPU Robot Lab

**WalkingLab** 与开源课程 **Hands-On Modern RL（《动手学现代强化学习》）** 的实验 08。它使用 ManiSkill 3 的 GPU PhysX 并行环境训练机器人 PPO 策略，并从本次学习到的策略生成 GIF 回放。

- Project: <https://github.com/walkinglabs/hands-on-modern-rl>
- WalkingLab: <https://modelscope.cn/organization/walkinglab>
- Companion chapter: <https://walkinglabs.github.io/hands-on-modern-rl>
- Live Studio: <https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment08-maniskill>
- Training source: <https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment08-maniskill/file/view/master/space_runtime.py>

The live Studio may bundle the official SAPIEN `linux-so.zip` GPU PhysX runtime to avoid a slow first-run GitHub download. The binary remains under NVIDIA's BSD-3-Clause terms; the required notice is included at `assets/physx-BSD-3-Clause.txt`.

State-based training keeps ManiSkill's renderer disabled, so GPU PhysX training does not depend on the container's host Vulkan ICD. The final replay uses Mesa's CPU Vulkan driver when available and otherwise generates a GIF from the real learned-policy rollout telemetry.
