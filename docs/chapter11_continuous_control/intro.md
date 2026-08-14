# 9.1 确定性策略梯度与 DDPG

## 本节导读

**核心内容**

- 理解为什么离散动作的 Q-learning 方法（如 DQN）无法直接处理连续动作——argmax 在连续空间上不可计算。
- 理解确定性策略梯度（DPG）定理怎样把策略梯度从随机策略扩展到确定性策略，使 off-policy 训练成为可能。
- 掌握 DDPG 如何把 DQN 的经验回放、目标网络和 Actor-Critic 框架结合，解决连续控制问题。
- 看到 DDPG 的三大缺陷：Q 值过估计、超参数敏感、训练不稳定——这些是下一节 TD3 和 SAC 要解决的问题。

前几章我们学了很多算法：Q-learning、DQN 处理离散动作，REINFORCE、PPO 用随机策略处理连续动作。你可能已经注意到一个问题：DQN 虽然效果好、样本效率高（因为有经验回放，可以 off-policy 复用数据），但它只能处理离散动作；PPO 虽然能处理连续动作，但它是 on-policy 的——每次更新后必须重新采样数据，**样本效率极低**。

在真实世界里，连续动作任务太多了：机器人关节的力矩、汽车方向盘的角度、机械臂末端的位置——这些都是连续值，没法枚举。能不能把 DQN 的 off-policy 优势"搬"到连续动作上来？

这就是本章要解决的核心问题。我们会看到：

- 9.1 先解决"连续动作怎么 off-policy 学"这个问题——答案是 DDPG
- 9.2 看到 DDPG 不稳定，然后用 TD3（工程补丁）和 SAC（重新设计目标函数）来修复
- 9.3 再往前走一步：不只是复用历史数据，而是学一个环境模型来"想象"数据，进一步提升样本效率
- 9.4 走到 model-based RL 的极致：AlphaGo/MuZero/Dreamer 用搜索和世界模型达到超人类水平

先从最基础的问题开始：**为什么 DQN 不能直接用在连续动作上？**

## 连续动作带来的根本困难

CartPole 里你只有两个选择：左推或右推。这是离散动作——动作空间是有限集合，可以枚举。Atari 游戏也是，你只有几个按键。

但机器人控制不一样。比如一个机械臂有 7 个关节，每个关节的力矩可以是 $[-1, 1]$ 之间的任意实数——这是一个 7 维连续空间。方向盘转角可以是 $[-\pi, \pi]$ 之间的任意值。油门开度是 $[0, 1]$ 之间的任意值。这些动作没法枚举。

DQN 的核心步骤是什么？回忆一下：

$$
a^* = \arg\max_a Q(s, a)
$$

在离散动作中，这个 $\arg\max$ 很简单——把所有动作的 Q 值算一遍，挑最大的那个就行。比如 CartPole 算 Q(s,左) 和 Q(s,右)，哪个大选哪个。

但在连续动作空间里，$a \in \mathbb{R}^n$，你没法枚举所有可能的动作。你不可能对每一个可能的关节角度都算一遍 Q 值——那是无穷多个。

这就是第一个困难：**Q 函数无法直接 argmax**。

那 PPO 是怎么处理连续动作的？它不用 argmax Q。PPO 学一个随机策略 $\pi_\theta(a \mid s)$——通常是高斯分布，输出均值和标准差——然后从这个分布里采样动作。策略梯度定理告诉它怎么更新参数。但这个方法要求你**用当前策略采样数据**——这是 on-policy 的，样本效率低。

那能不能把两者结合起来：既像 DQN 一样 off-policy 复用数据，又像 PPO 一样处理连续动作？

Silver 等人在 2014 年给出了答案：**确定性策略梯度（Deterministic Policy Gradient, DPG）**。

## 确定性策略梯度定理

我们先回忆一下第 6 章的随机策略梯度定理：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{s \sim d^\pi, a \sim \pi_\theta}\left[\nabla_\theta \log \pi_\theta(a\mid s) \cdot Q^\pi(s, a)\right]
$$

这个式子的意思是：策略 $\pi_\theta$ 是一个概率分布（随机策略），在状态 $s$ 选动作 $a$ 有一个概率 $\pi_\theta(a \mid s)$。我们更新参数，提高能带来高 Q 值的动作的概率。

但是这里有一个对动作 $a$ 的期望 $\mathbb{E}_{a \sim \pi_\theta}[\cdot]$——我们要对所有可能的动作求期望，每个动作按策略概率加权。这在连续空间里需要采样很多动作才能估计准确。

Silver 等人证明了一个惊人的结论：**如果策略是确定性的 $a = \mu_\theta(s)$——也就是在状态 $s$ 直接输出一个确定的动作，不采样——那么策略梯度定理仍然成立，而且形式更简单：**

$$
\nabla_\theta J(\theta) = \mathbb{E}_{s \sim d^\mu}\left[\nabla_\theta \mu_\theta(s) \cdot \nabla_a Q^\mu(s, a)\bigg|_{a=\mu_\theta(s)}\right]
$$

等一下，这个式子看起来有点复杂，我们把它拆成两部分用链式法则来理解：

- **第一部分** $\nabla_\theta \mu_\theta(s)$：参数 $\theta$ 变化一点点，策略输出的动作 $a$ 会怎么变？
- **第二部分** $\nabla_a Q^\mu(s, a)\big|_{a=\mu_\theta(s)}$：动作 $a$ 变化一点点，Q 值会怎么变？

乘起来的意思是：如果把动作往"能让 Q 值变大"的方向调，参数就往那个方向更新。这和我们的直觉完全一致——调整策略，让它输出的动作能拿到更高的 Q 值。

和随机版本比，确定性版本好在哪里？

- **不需要对动作 $a$ 积分**：随机策略梯度要对所有可能动作求期望，确定性版本不用——策略直接输出一个动作，只需要对状态求期望。这大大降低了估计方差。
- **天然适合 off-policy**：因为动作是确定的，我们可以用任何行为策略（比如加了噪声的旧策略）采集的数据来训练，不需要用当前策略采样。这就是样本效率高的原因。

等等，你可能发现一个问题：**确定性策略不探索啊！** 如果 $\mu_\theta(s)$ 每次都精确返回同一个 $a$，智能体永远只尝试那一个动作，不会试别的，那怎么发现更好的动作？

这是个关键问题。DDPG 的解决方案很直接：**训练和执行的时候，给动作加噪声**。

## DDPG：深度确定性策略梯度

Lillicrap 等人在 2015 年把 DPG 定理和 DQN 的深度网络技巧结合起来，提出了 DDPG（Deep Deterministic Policy Gradient）。它把我们已经熟悉的几个组件拼在了一起：

- **Actor（策略网络）**：$\mu_\theta(s)$ 直接输出连续动作——这是确定性策略，输入状态，输出动作向量。
- **Critic（价值网络）**：$Q_\phi(s, a)$ 评估动作价值——输入状态和动作，输出一个 Q 值。这和 DQN 不一样：DQN 只输入状态，输出所有动作的 Q 值；DDPG 的 Critic 同时输入状态和动作，输出一个标量。
- **目标网络**：Actor 和 Critic 都有自己的目标网络，用软更新来稳定训练——这是从 DQN 继承来的。
- **经验回放**：用 Replay Buffer 存储所有交互数据，每次随机采样一批更新——也是 DQN 的技巧，实现 off-policy。

我们来看代码理解整个流程：

```python
class DDPG:
    def __init__(self, state_dim, action_dim, action_max):
        # 主网络
        self.actor = Actor(state_dim, action_dim, action_max)
        self.critic = Critic(state_dim, action_dim)
        # 目标网络（软更新）
        self.actor_target = copy(self.actor)
        self.critic_target = copy(self.critic)
        self.replay_buffer = ReplayBuffer(capacity=1_000_000)
        self.gamma = 0.99
        self.tau = 0.005  # 软更新系数

    def select_action(self, state, explore=True):
        with torch.no_grad():
            action = self.actor(state)
        if explore:
            # 训练时加高斯噪声探索
            action += np.random.normal(0, 0.1, size=action.shape)
        return np.clip(action, -self.action_max, self.action_max)

    def update(self, batch_size=256):
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(batch_size)

        # === Critic 更新 ===
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + self.gamma * (1 - dones) * target_q
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)
        self.critic_optim.zero_grad(); critic_loss.backward()
        self.critic_optim.step()

        # === Actor 更新：最大化 Q(s, μ(s)) ===
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optim.zero_grad(); actor_loss.backward()
        self.actor_optim.step()

        # === 软更新目标网络 ===
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)
```

让我们一步步看这个 update 函数在做什么。

**第一步：更新 Critic**。这和 DQN 很像：

- 用目标 Actor 选择下一状态的动作：$a' = \mu_{\theta'}(s')$
- 用目标 Critic 计算目标 Q 值：$y = r + \gamma Q_{\phi'}(s', a')$
- 拟合当前 Critic：让 $Q_\phi(s, a)$ 接近 $y$

**第二步：更新 Actor**。这是 DDPG 最核心的部分，对应确定性策略梯度定理：

- Actor 想要最大化 $Q(s, \mu_\theta(s))$——也就是 Critic 对 Actor 选的动作的评分。
- 所以 loss 是 $-Q(s, \mu_\theta(s))$（负号因为我们做梯度下降，最小化负 Q 等于最大化 Q）。
- 注意梯度是怎么流的：Critic 给出 Q 值对动作 $a$ 的梯度，然后通过 $a = \mu_\theta(s)$ 传到 Actor 的参数 $\theta$。这正好对应 DPG 公式里的 $\nabla_a Q \cdot \nabla_\theta \mu$。

**第三步：软更新目标网络**。DDPG 不用 DQN 那种"每隔 N 步硬拷贝"的方式，而是每一步都缓慢更新：

$$
\theta' \leftarrow \tau \theta + (1 - \tau)\theta'
$$

$\tau = 0.005$ 是个很小的数，意味着目标网络每次只向主网络靠近 0.5%。这让目标值变化非常缓慢，训练更稳定。

**探索的处理**：注意 `select_action` 函数里，训练时给 Actor 输出的动作加了高斯噪声 $\mathcal{N}(0, 0.1^2)$。这就解决了"确定性策略不探索"的问题——我们用一个带噪声的行为策略收集数据，然后训练确定性的目标策略。这就是 off-policy 的好处：行为策略和目标策略可以不一样。

DDPG 在 MuJoCo 物理环境（HalfCheetah、Hopper、Walker2d）上首次让深度 RL 打败了基于线性特征的 TRPO 等传统方法。听起来很好，对吧？

但是——如果你真的去用 DDPG，你会发现它非常难调，经常训练着训练着就崩了。

## DDPG 的三大缺陷

DDPG 虽然开了连续控制 off-policy 的先河，但它有三个广受诟病的问题，让它在实践中几乎不可用：

**问题一：Q 值过估计**。

回忆 DQN 里也有过估计问题——因为 target 里用了 $\max_a Q$，如果 Q 值有噪声，max 总是选那个被噪声高估的动作，导致 Q 值系统性偏高。DDPG 里更严重：Critic 不仅要对动作取 max（通过 Actor），而且 Actor 在每个状态都直接输出那个被认为能最大化 Q 的动作——如果 Critic 对某个动作估计偏高，Actor 就会直接朝那个方向更新，然后 Critic 再被更新后的 Actor 带得更高，形成正反馈循环。

**问题二：超参数敏感**。

学习率、噪声尺度、网络结构、软更新系数 $\tau$——随便改一个，DDPG 可能就从"学得很好"变成"完全发散"。你需要花大量时间调参才能让它在一个环境上工作，换个环境又要重新调。

**问题三：训练不稳定，容易崩溃**。

这是前两个问题叠加的结果：Critic 只要学坏一点点，给了错误的梯度，Actor 就会被更新到一个不好的方向；然后新的 Actor 生成更差的数据，让 Critic 学的更差；恶性循环，策略彻底崩溃——而且一旦崩溃就救不回来。

这三个问题不是小毛病——它们让 DDPG 只能在精心调参的情况下工作，很难作为通用算法。下一节我们就来看两套解决方案：TD3 用三个工程补丁给 DDPG 打稳定性补丁；SAC 则从根本上改变目标函数，用最大熵 RL 彻底解决这些问题。

::: details 加餐：Ornstein-Uhlenbeck 噪声 vs 高斯噪声
原始 DDPG 论文里用的是 Ornstein-Uhlenbeck（OU）过程来生成噪声，而不是简单的高斯噪声。OU 过程生成的噪声是时间相关的——$x_{t+1} = x_t + \theta(\mu - x_t) + \sigma \epsilon_t$——它会让噪声有惯性，适合控制任务中那种"动量相关"的探索。

但后续研究发现（包括 TD3 论文），简单的不相关高斯噪声效果一样好，甚至更好。OU 噪声的额外复杂度并没有带来实际收益。所以现在实践中大家都直接用高斯噪声。
:::

## 本节总结

确定性策略梯度（DPG）定理把策略梯度从随机策略扩展到确定性策略：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{s \sim d^\mu}\left[\nabla_\theta \mu_\theta(s) \cdot \nabla_a Q^\mu(s, a)\bigg|_{a=\mu_\theta(s)}\right]
$$

DDPG 把 DPG 和 DQN 的深度网络技巧（经验回放、目标网络、软更新）结合起来，构建了第一个能稳定工作的深度连续控制 off-policy 算法：Actor 输出确定动作，Critic 评估 Q 值，训练时加噪声探索。

但 DDPG 有三大缺陷：Q 值过估计、超参数敏感、训练不稳定。下一节 [9.2 TD3 与 SAC](./td3-sac) 给出两套互补的修补方案——TD3 用工程 trick 稳定 DDPG，SAC 用最大熵 RL 从根本上重构目标函数。
