# 8. PPO：稳定的 Actor-Critic

## 为什么需要 PPO

上一章我们搭建了 [Actor-Critic 架构](../chapter09_actor_critic/actor-critic)——Actor 负责选择动作，Critic 负责评估动作的好坏，两者通过[优势函数](../chapter09_actor_critic/advantage-function) $A(s,a)$ 协作。在简单环境中，Actor-Critic 表现得相当不错。但当你把同样的架构搬到更复杂的环境（比如 BipedalWalker 双足行走）或者更大的模型（比如数十亿参数的语言模型）时，一个严重的问题会浮出水面：**训练不稳定**。

[策略梯度方法](../chapter08_policy_gradient/reinforce)有一个臭名昭著的弱点——一步更新太大就会"策略崩溃"。想象你在学骑自行车，如果你一次调整太大的重心，结果不是骑得更好，而是直接摔车。Actor-Critic 的 [TD Error](../chapter09_actor_critic/critic-training) 信号虽然降低了方差，但并没有从根本上解决这个问题。我们需要一种机制来约束每一步更新的幅度，让策略"小步快跑"而不是"一步登天"。

PPO（Proximal Policy Optimization，近端策略优化）就是为解决这个问题而生的。它不是一个全新的架构，而是 Actor-Critic 的**稳定化改进**——保留 Actor 和 Critic 的分工，只在"怎么更新 Actor"这一步加了一套更稳的规则：通过裁剪目标函数，确保新策略不会离旧策略太远。PPO 因其实现简单、训练稳定，成为了深度强化学习（以及后来的 LLM 对齐）最常用的算法之一。

## 前置知识

::: tip 前置知识回顾
本章会频繁用到以下概念：

- [策略梯度 $\nabla_\theta J$](../chapter08_policy_gradient/reinforce)——PPO 在策略梯度的基础上加约束
- [REINFORCE 的高方差问题](../chapter08_policy_gradient/cartpole)——为什么需要各种改进
- [优势函数 $A(s,a)$](../chapter09_actor_critic/advantage-function)——PPO 的策略更新依赖优势信号
- [TD Error 与 Critic 训练](../chapter09_actor_critic/critic-training)——PPO 中 Critic 的训练方式
- [Actor-Critic 架构](../chapter09_actor_critic/actor-critic)——PPO 是 Actor-Critic 的实用变体
  :::

## 本章学习路线

本章沿着"动手 → 约束 → 估计 → 拓展"的路径展开：先在 BipedalWalker 上跑一个连续控制实验，亲眼看到训练曲线、策略熵、裁剪比例和 KL 散度；再拆解背后的约束机制和优势估计方法；最后给出数学推导、游戏项目和长程任务等拓展内容。

| 小节                                                 | 你会回答的问题                                                                                |
| ---------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| [8.1 动手：PPO 控制 BipedalWalker](./ppo-bipedal-walker) | PPO 的训练过程长什么样？Reward、Entropy、Clip Fraction 和 KL 怎么看？它如何处理连续动作空间？ |
| [8.2 信任域约束与 PPO-Clip](./trust-region-clipping)        | 为什么一步更新太大会崩溃？TRPO 的 KL 约束和 PPO 的裁剪各是怎么做的？                          |
| [8.3 GAE 优势估计](./gae-reward-model)             | GAE 如何在偏差和方差之间插值？LLM 对齐中的 PPO 需要几个模型同时跑？                           |

**加餐内容**（选读）：

| 加餐                                                 | 内容                                                                                |
| ---------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| [PPO 数学推导](./ppo-math)                           | PPO 的公式怎么来的？从策略梯度到裁剪代理目标的完整推导链条是什么？完整损失函数包含哪几项？    |
| [PPO 游戏项目](./ppo-game-benchmark)                 | 哪些游戏已经有人用 PPO 跑过？玩家入口、训练环境和复现证据分别在哪里？                         |
| [长程任务中的 RL 探索](./rl-long-horizon-planning)   | 传统 RL 如何应对长程任务？分层 RL、HER、世界模型、奖励塑形各自怎么做？                        |

让我们先跑起来看看效果——[8.1 动手：PPO 控制 BipedalWalker](./ppo-bipedal-walker)。
