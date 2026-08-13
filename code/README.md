# 代码索引

本目录包含课程各章的配套代码。建议先进入 `code/` 目录，再按章节安装依赖并运行脚本。

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 全量依赖，适合通读课程时使用
pip install -r requirements.txt

# 或者只安装某一章依赖
pip install -r chapter01_cartpole/requirements.txt
```

## 章节代码一览

| 章节                    | 目录                            | 主要代码                             | 说明                                                                                |
| ----------------------- | ------------------------------- | ------------------------------------ | ----------------------------------------------------------------------------------- |
| Ch01 CartPole           | `chapter01_cartpole/`           | `1-ppo_cartpole.py`                  | SB3 PPO 训练 CartPole，记录 SwanLab 指标                                            |
|                         |                                 | `2-pytorch_ppo.py`                   | 纯 PyTorch PPO：Actor-Critic、GAE、clip、指标记录                                   |
|                         |                                 | `plot_curves.py`                     | 从训练导出的 CSV 绘制可复核指标曲线                                                 |
|                         |                                 | `capture_frames.py`                  | 加载模型并捕获 Gymnasium 真实渲染帧                                                 |
| Ch02 DPO                | `chapter17_dpo/`                | `0-download_model.py`                | 下载 Qwen2.5-0.5B-Instruct                                                          |
|                         |                                 | `1-generate_data.py`                 | 生成过度顺从纠正偏好数据                                                            |
|                         |                                 | `2-test_before.py`                   | 微调前测试                                                                          |
|                         |                                 | `3-train_dpo.py`                     | TRL DPO 训练                                                                        |
|                         |                                 | `4-test_after.py`                    | 微调后测试                                                                          |
| Ch03 MDP                | `chapter03_mdp/`                | `two_armed_bandit.py`                | 两臂老虎机策略对比                                                                  |
|                         |                                 | `bellman_equation_verify.py`         | 贝尔曼方程数值验证                                                                  |
|                         |                                 | `gridworld_q_learning.py`            | GridWorld Q-Learning 与路径可视化                                                   |
| Ch04 DQN                | `chapter04_dqn/`                | `dqn_cartpole.py`                    | 从零实现 DQN 训练 CartPole                                                          |
|                         |                                 | `double_dqn_cartpole.py`             | DQN 与 Double DQN 对比                                                              |
|                         |                                 | `dqn_gym_sb3.py`                     | SB3 DQN 训练 CartPole、MountainCar、LunarLander 等离散环境，记录 SwanLab 和评估曲线 |
|                         |                                 | `dqn_atari_sb3.py`                   | SB3 DQN 真实训练 Atari，含 wrapper、SwanLab、评估和日志                             |
|                         |                                 | `export_dqn_curves.py`               | 从第 4 章 DQN eval CSV 导出讲义图片                                                 |
|                         |                                 | `dqn_pokemon_red_pyboy.py`           | PyBoy + SB3 DQN 训练宝可梦早期探索任务                                              |
| Ch05 Policy Gradient    | `chapter05_policy_gradient/`    | `reinforce_cartpole.py`              | REINFORCE 训练 CartPole                                                             |
|                         |                                 | `reinforce_with_baseline.py`         | REINFORCE 与 baseline 对比                                                          |
|                         |                                 | `actor_critic_cartpole.py`           | Actor-Critic 与 TD Error                                                            |
| Ch07 PPO                | `chapter07_ppo/`                | `ppo_lunar_lander.py`                | SB3 PPO 训练 LunarLander-v3                                                         |
|                         |                                 | `ppo_from_scratch.py`                | 纯 PyTorch PPO                                                                      |
|                         |                                 | `gae_visualization.py`               | GAE 参数可视化                                                                      |
| Ch08 RLHF               | `chapter08_rlhf/`               | `sft_pipeline.py`                    | SFT 管线                                                                            |
|                         |                                 | `reward_model_training.py`           | 奖励模型训练                                                                        |
|                         |                                 | `rlhf_ppo_train.py`                  | 简化 PPO-RLHF 训练循环                                                              |
|                         | `chapter08_rlhf/verl_gsm8k/`    | `run_qwen2_5_0_5b_ppo_single_gpu.sh` | 8.7 veRL + GSM8K 外部框架适配脚本                                                   |
| Ch09 Alignment          | `chapter09_alignment/`          | `dpo_hands_on.py`                    | DPO 对齐与 beta 对比                                                                |
|                         |                                 | `dpo_math_reward.py`                 | 数学偏好数据上的 DPO 实验                                                           |
| Ch09 GRPO/RLVR          | `chapter09_grpo_rlvr/`          | `grpo_mechanism.py`                  | GRPO 机制演示                                                                       |
|                         |                                 | `grpo_math_reasoning.py`             | 数学推理 GRPO 小实验                                                                |
|                         |                                 | `rule_based_reward.py`               | 规则奖励函数                                                                        |
| Ch09 Continuous Control | `chapter09_continuous_control/` | `sac_halfcheetah.py`                 | SAC 训练 HalfCheetah-v4                                                             |
|                         |                                 | `ppo_td3_sac_comparison.py`          | PPO、TD3、SAC 对比                                                                  |
| Ch10 Agentic RL         | `chapter10_agentic_rl/`         | `tool_use_agent.py`                  | 工具选择策略训练                                                                    |
|                         |                                 | `multi_turn_rl.py`                   | 多轮交互信用分配                                                                    |
|                         |                                 | `generate_synthetic_data.py`         | 合成轨迹数据                                                                        |
|                         |                                 | `mini_deep_research_grpo.py`         | Mini Deep Research GRPO 示例                                                        |
| Ch11 VLM RL             | `chapter11_vlm_rl/`             | `geometry_counting_dataset.py`       | 几何计数数据集                                                                      |
|                         |                                 | `multi_modal_reward.py`              | 多模态规则奖励                                                                      |
|                         |                                 | `vlm_grpo_train.py`                  | VLM GRPO 训练示例                                                                   |
| Ch12 Future Trends      | `chapter12_future_trends/`      | `tree_of_thought.py`                 | Tree of Thought 搜索演示                                                            |
|                         |                                 | `multi_agent_marl.py`                | 多智能体 GridWorld                                                                  |
| Appendix Pitfalls       | `appendix_common_pitfalls/`     | `debug_reward_hacking.py`            | 奖励作弊复现                                                                        |
|                         |                                 | `debug_training_collapse.py`         | 训练崩溃诊断                                                                        |

## 说明

- 每个章节目录下的 `requirements.txt` 是该章的最小依赖。
- LLM 相关章节默认使用小模型，但仍建议在有 GPU 的环境中运行。
- 部分脚本会在当前工作目录下生成 `output/`、模型权重或图像文件。
- 第 4 章 DQN 脚本默认使用 SwanLab 本地模式；运行后可用 `swanlab watch swanlog` 查看曲线。
