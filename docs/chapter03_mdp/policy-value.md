# 2.3 策略、价值与回报

## 本节导读

**核心内容**

- 理解策略 $\pi$ 如何与环境交互并产生一条轨迹。
- 区分一条实际轨迹的回报 $G_t$ 与多次可能结果的期望价值。
- 理解状态价值 $V^\pi(s)$、动作价值 $Q^\pi(s,a)$ 和优势函数 $A^\pi(s,a)$ 分别回答什么问题。

上一节用 MDP 五元组描述了序列决策问题：$\mathcal{S}$ 和 $\mathcal{A}$ 规定智能体面对什么、能够做什么，$P$ 和 $R$ 规定动作之后环境怎样变化、给出多少奖励，$\gamma$ 规定未来奖励的权重。策略 $\pi$ 决定每一步怎样选，折扣回报 $G_t$ 则把一条轨迹中的奖励合成长期总收益。

有了这些符号，我们已经能完整记录一次 CartPole 交互。智能体根据当前状态选择向左或向右推，环境返回奖励并进入下一状态；不断重复，就得到一条轨迹。等杆子倒下以后，可以沿着这条轨迹算出每个时刻的回报 $G_t$。

真正做决策时，智能体不能等到轨迹结束再判断。它站在某个中间状态，需要提前回答两个问题：从这个状态继续走，未来平均能得到多少回报；如果现在先选某个动作，结果又会怎样。单次回报 $G_t$ 只描述已经发生的一条轨迹，而同一个状态在不同尝试中可能产生不同的动作、转移和回报。

因此，本节沿着上一节留下的符号继续前进：先看策略怎样产生轨迹，再用回报衡量轨迹，最后对所有可能回报取期望，得到状态价值 $V^\pi(s)$ 和动作价值 $Q^\pi(s,a)$。这一步把“描述一次交互”推进到“在行动之前评价状态和动作”，并为下一章的贝尔曼方程做好准备。

## 策略与决策规则

**策略**（policy）是 agent 从状态到动作的映射，分两类：

- **确定性策略**：$\pi: \mathcal{S} \to \mathcal{A}$，给定状态直接输出动作 $a = \pi(s)$
- **随机策略**：$\pi: \mathcal{S} \to \Delta(\mathcal{A})$，给定状态输出动作上的分布 $a \sim \pi(\cdot \mid s)$

随机策略更一般，确定性策略是它的特例（分布坍缩到单点）。策略梯度方法通常直接学习随机策略；DQN 等价值型方法则常从动作价值中导出确定性的贪心策略。

```python
# CartPole 的一个简单随机策略示例
import torch
import torch.nn as nn

class CartPolePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 32), nn.Tanh(),
            nn.Linear(32, 2)  # 2 个动作的 logits
        )

    def forward(self, state):
        logits = self.net(state)
        return torch.distributions.Categorical(logits=logits)

    def act(self, state):
        dist = self.forward(state)
        action = dist.sample()
        return action.item(), dist.log_prob(action)
```

### 最优策略

RL 的目标是找到**最优策略** $\pi^*$，使长期累积奖励最大化：

$$\pi^* = \arg\max_\pi \mathbb{E}_{\pi}\left[\sum_{t=0}^{\infty} \gamma^t R(s_t, a_t)\right]$$

DQN、PPO、SAC 采用不同的策略表示和更新方法，但最终都希望提高策略产生的期望回报。

## 回报与轨迹衡量

智能体在一个 episode 中经历一条轨迹 $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots)$。**回报**（return）是从 $t$ 时刻起的累积奖励：

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

### 折扣因子 γ 的作用

$\gamma \in [0, 1]$ 是**折扣因子**，它规定远期奖励在当前回报中的权重。$\gamma$ 越小，权重衰减越快；$\gamma$ 越接近 1，远期奖励保留的权重越大。

对于奖励有界的无限时域任务，$\gamma<1$ 还能使折扣回报保持有限。对于有明确终点的有限时域任务，可以根据目标采用 $\gamma=1$，让各时刻的奖励等权相加。$\gamma$ 是任务目标的一部分，不能仅根据算法名称决定。

### CartPole 中的回报

CartPole 每步 reward = 1（杆子还立着）。episode 终止时杆子倒了或 cart 出界。回报：

$$G_0 = 1 + \gamma + \gamma^2 + \cdots + \gamma^{T-1} = \frac{1 - \gamma^T}{1 - \gamma}$$

其中 $T$ 是 episode 长度。当 $\gamma = 0.99, T = 500$ 时，$G_0 \approx 99$。

## 价值函数与长期收益

**价值函数**衡量"在某个状态/动作下，按当前策略 $\pi$ 行动，期望能拿到多少回报"。分两种：

### 状态价值 V(s)

$$V^\pi(s) = \mathbb{E}_\pi\left[G_t \mid s_t = s\right] = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \mid s_t = s\right]$$

含义：从状态 $s$ 出发，按策略 $\pi$ 走，期望累积回报。

### 动作价值 Q(s, a)

$$Q^\pi(s, a) = \mathbb{E}_\pi\left[G_t \mid s_t = s, a_t = a\right]$$

含义：从状态 $s$ 出发，**先采取动作 $a$，之后按 $\pi$**，期望累积回报。

### V 与 Q 的关系

$$V^\pi(s) = \sum_a \pi(a \mid s) Q^\pi(s, a)$$

即 $V^\pi(s)$ 是 $Q^\pi(s, a)$ 在策略 $\pi$ 下的期望。

### GridWorld 数值示例

考虑一条只有三个非终止状态的走廊。策略始终向右走，进入终点时获得 $+1$，其他转移奖励为 $0$，并取 $\gamma=0.9$：

```text
S0 ──→ S1 ──→ S2 ──→ 终点
0.81   0.90   1.00     0
```

从 $S_2$ 出发，下一步就得到 $+1$，所以 $V^\pi(S_2)=1$。从 $S_1$ 出发，奖励晚一步到达，因此 $V^\pi(S_1)=0.9$；同理，$V^\pi(S_0)=0.9^2=0.81$。这里距离终点越近，价值越高，是因为同一份终点奖励经历的折扣次数更少。

## 优势函数与动作评价

**优势函数**（advantage function）衡量动作 $a$ 相对于当前策略在状态 $s$ 下平均选择的动作好多少：

$$A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$$

- $A > 0$：动作 $a$ 比平均好
- $A < 0$：动作 $a$ 比平均差
- $A = 0$：动作 $a$ 是平均水平

优势函数在策略梯度（[第 6 章](../chapter08_policy_gradient/policy-gradient)）和 Actor-Critic（[第 7 章](../chapter09_actor_critic/actor-critic)）中是核心概念。

## 贝尔曼方程的预告

价值函数满足**贝尔曼方程**——一个递归关系，把 $V(s)$ 写成 $V(s')$ 的函数：

$$V^\pi(s) = \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s, a) \left[R(s, a, s') + \gamma V^\pi(s')\right]$$

这个方程是价值估计和控制方法的重要基础。[第 3 章 价值函数与贝尔曼方程](./value-bellman) 会从具体例子出发继续推导。

## 本节总结

第 2.1 节从老虎机出发，说明智能体为什么需要在探索与利用之间取舍。第 2.2 节加入会随动作变化的状态，用 MDP 描述环境规则。本节再加入策略与回报，说明智能体怎样产生一条轨迹，以及怎样衡量这条轨迹的长期收益。

到这里，第 2 章已经建立了描述强化学习问题所需的基本对象：

1. **策略 $\pi$**：agent 的决策规则，随机策略 $a \sim \pi(\cdot \mid s)$ 是最一般形式
2. **回报 $G_t$**：从 $t$ 起的折扣累积奖励 $\sum \gamma^k r$
3. **价值函数**：$V^\pi(s)$ 状态价值，$Q^\pi(s, a)$ 动作价值；优势 $A = Q - V$ 衡量相对好坏

这些对象已经能描述“发生了什么”，但还没有解决一个计算问题：回报 $G_t$ 要等轨迹实际发生后才能得到；智能体在中间状态作出选择时，需要提前估计从这里出发的长期收益。第 3 章将用价值函数回答这个问题。

下面先集中整理第 2 至 4 章会用到的核心公式，作为后续学习的查阅索引。

## 2.4 折扣、轨迹与 POMDP

### 内容概述

第 2 至 4 章围绕序列决策建模、回报定义、价值函数、贝尔曼方程、价值估计、Q-Learning、策略优化和奖励设计展开。作为这一组基础章节的总结，本节汇总第 2.1 至 4.3 节的核心公式，并给出它们在理论结构中的位置。

这组章节的主要结论可以概括为八点：

1. 强化学习问题可以用 MDP 五元组形式化描述。
2. 智能体优化的是折扣累积回报，而不是单步即时奖励。
3. 状态价值函数和动作价值函数分别评估状态与动作的长期回报。
4. 贝尔曼方程给出了价值函数的递归结构。
5. DP、MC、TD 是三类基本的价值估计方法。
6. 参数化策略可以通过策略目标函数直接优化。
7. 算法根据数据来源分为 On-policy/Off-policy 和 Online/Offline。
8. 奖励函数决定学习问题本身，奖励设计会影响算法最终行为。

这些内容构成后续深度 Q 网络、策略梯度、Actor-Critic、PPO 以及大模型强化学习方法的共同理论基础。

### 核心公式索引

下面集中列出第 2.1 至 4.3 节的核心公式。每条公式都标注名称、作用和对应讲解位置。

#### 2.4.1 两台老虎机

$$
\mathbb{E}[R_a] = p_a \cdot (+1) + (1-p_a)\cdot(-1) = 2p_a - 1
\quad \text{（单臂期望奖励；作用：比较单个动作的平均收益；详见 2.1）}
$$

$$
\mathbb{E}[R_T] = \mathbb{E}[R_{a_1}] + \mathbb{E}[R_{a_2}] + \cdots + \mathbb{E}[R_{a_T}] = \sum_{t=1}^{T} \mathbb{E}[R_{a_t}]
\quad \text{（T 轮期望总回报；作用：衡量一整套策略的累计表现；详见 2.1）}
$$

$$
\mathrm{Regret}(T) = T\mu^* - \sum_{t=1}^{T}\mu_{a_t}, \qquad \mu^*=\max_a \mu_a
\quad \text{（Regret；作用：衡量探索相对最优策略损失了多少；详见 2.1）}
$$

#### 2.4.2 MDP

$$
\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle
\quad \text{（MDP 五元组；作用：描述序列决策问题的完整规则；详见 2.2）}
$$

$$
P(s' \mid s,a), \qquad R(s,a), \qquad \gamma \in [0,1]
\quad \text{（转移、奖励与折扣；作用：描述状态转移、即时奖励与未来奖励权重；详见 2.2）}
$$

$$
G_t = \sum_{k=0}^{\infty}\gamma^k r_{t+k} = r_t + \gamma G_{t+1}
\quad \text{（折扣累积回报；作用：定义从时刻 t 开始的长期优化目标；详见 2.2）}
$$

$$
a = \pi(s), \qquad \pi(a\mid s)=P(a\mid s)
\quad \text{（确定性策略与随机性策略；作用：描述智能体如何选动作；详见 2.3）}
$$

#### 2.4.3 V(s) 与贝尔曼方程

$$
V^\pi(s)=\mathbb{E}_\pi\left[\sum_{k=0}^{\infty}\gamma^k r_{t+k}\mid s_t=s\right]
\quad \text{（状态价值函数；作用：评估一个状态的长期平均回报；详见 3.1）}
$$

$$
V^\pi(s)=\sum_{a\in\mathcal{A}}\pi(a\mid s)\left[R(s,a)+\gamma\sum_{s'\in\mathcal{S}}P(s'\mid s,a)V^\pi(s')\right]
\quad \text{（贝尔曼期望方程；作用：在固定策略下递归计算价值；详见 3.1）}
$$

$$
V^*(s)=\max_a\left[R(s,a)+\gamma\sum_{s'\in\mathcal{S}}P(s'\mid s,a)V^*(s')\right]
\quad \text{（贝尔曼最优方程；作用：定义最优状态价值；详见 3.2）}
$$

$$
\text{Target}=r+\gamma V(s'), \qquad \delta=\text{Target}-V(s)
\quad \text{（Bellman Target 与 TD Error 雏形；作用：将贝尔曼递推转化为采样学习信号；详见 4.1）}
$$

#### 2.4.4 DP、MC、TD

$$
V(s) \leftarrow \sum_a \pi(a\mid s)\left[R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V(s')\right]
\quad \text{（DP 策略评估更新；作用：已知模型时迭代价值；详见 4.1）}
$$

$$
\pi'(s)=\arg\max_a\left[R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V^\pi(s')\right]
\quad \text{（策略改进；作用：用当前价值构造更好的贪心策略；详见 4.1）}
$$

$$
V(s) \leftarrow V(s)+\alpha\left[G_t-V(s)\right]
\quad \text{（MC 价值更新；作用：用完整回报修正价值估计；详见 4.1）}
$$

$$
V(s) \leftarrow V(s)+\alpha\left[r+\gamma V(s')-V(s)\right]
\quad \text{（TD(0) 价值更新；作用：走一步就用自举目标更新价值；详见 4.1）}
$$

$$
\delta = r+\gamma V(s')-V(s)
\quad \text{（TD Error；作用：衡量当前价值估计违反贝尔曼关系的程度；详见 4.1）}
$$

#### 2.4.5 Q(s,a)

$$
Q^\pi(s,a)=\mathbb{E}_\pi\left[G_t\mid s_t=s,a_t=a\right]
\quad \text{（动作价值函数；作用：评估在状态 s 先做动作 a 的长期回报；详见 3.2）}
$$

$$
V^\pi(s)=\sum_a\pi(a\mid s)Q^\pi(s,a)
\quad \text{（V-Q 关系式；作用：用动作价值的策略加权平均得到状态价值；详见 3.2）}
$$

$$
Q^\pi(s,a)=R(s,a)+\gamma\sum_{s'\in\mathcal{S}}P(s'\mid s,a)\sum_{a'\in\mathcal{A}}\pi(a'\mid s')Q^\pi(s',a')
\quad \text{（Q 的贝尔曼期望方程；作用：固定策略下递归计算动作价值；详见 3.2）}
$$

$$
Q^*(s,a)=R(s,a)+\gamma\sum_{s'\in\mathcal{S}}P(s'\mid s,a)\max_{a'}Q^*(s',a')
\quad \text{（Q 的贝尔曼最优方程；作用：递归定义最优动作价值；详见 3.2）}
$$

$$
\pi^*(s)=\arg\max_a Q^*(s,a)
\quad \text{（贪心最优策略；作用：由最优动作价值诱导最优策略；详见 3.2）}
$$

#### 2.4.6 Q-Learning

$$
\text{TD Target}=r+\gamma\max_{a'}Q(s',a')
\quad \text{（Q-Learning TD 目标；作用：用一步经验构造当前动作价值的学习目标；详见 3.2）}
$$

$$
\delta=r+\gamma\max_{a'}Q(s',a')-Q(s,a)
\quad \text{（Q-Learning TD Error；作用：衡量当前动作价值估计和 TD 目标的差距；详见 3.2）}
$$

$$
Q(s,a)\leftarrow Q(s,a)+\alpha\left[r+\gamma\max_{a'}Q(s',a')-Q(s,a)\right]
\quad \text{（Q-Learning 更新；作用：在线修正状态-动作价值表；详见 3.2）}
$$

$$
a_t=
\begin{cases}
\text{随机动作} & \text{概率 }\varepsilon\\
\arg\max_a Q(s_t,a) & \text{概率 }1-\varepsilon
\end{cases}
\quad \text{（}\varepsilon\text{-贪婪策略；作用：在探索和利用之间折中；详见 3.2）}
$$

#### 2.4.7 策略目标

$$
\pi_\theta(a\mid s)=P_\theta(a\mid s)
\quad \text{（参数化随机策略；作用：用参数 theta 表示动作分布；详见 2.3）}
$$

$$
J(\theta)=\mathbb{E}_{\pi_\theta}\left[G_t\right]
=\mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^{\infty}\gamma^t r_t\right]
\quad \text{（策略目标函数；作用：衡量参数化策略的平均长期回报；详见 2.3）}
$$

$$
\theta^*=\arg\max_\theta J(\theta)
\quad \text{（最优策略参数；作用：将策略学习表述为最大化问题；详见 2.3）}
$$

$$
\nabla_\theta J(\theta)\propto
\mathbb{E}_{\pi_\theta}\left[\nabla_\theta\log\pi_\theta(a\mid s)\cdot G_t\right]
\quad \text{（策略梯度估计式；作用：提高高回报动作的概率；详见 2.3）}
$$

#### 2.4.8 奖励设计

$$
R(s,a)=
\begin{cases}
+1 & \text{达到目标}\\
0 & \text{其他}\\
-1 & \text{失败}
\end{cases}
\quad \text{（稀疏奖励函数；作用：只在成败时给学习信号；详见 4.3）}
$$

$$
R_{\text{shaping}}(s,a,s')=-\left(\text{dist}(s',\text{goal})-\text{dist}(s,\text{goal})\right)
\quad \text{（距离奖励塑形；作用：根据到目标距离的变化提供中间奖励；详见 4.3）}
$$

$$
F(s,a,s')=\gamma\Phi(s')-\Phi(s)
\quad \text{（势函数奖励塑形；作用：增强中间信号且不改变最优策略；详见 4.3）}
$$

$$
r_t^{\text{intrinsic}}=\left\|f(s_t,a_t)-s_{t+1}\right\|^2
\quad \text{（预测误差内在奖励；作用：鼓励探索模型还预测不准的状态；详见 4.3）}
$$

$$
r_t^{\text{RND}}=\left\|\hat{\phi}(s_t)-\phi(s_t)\right\|^2
\quad \text{（RND 内在奖励；作用：用随机网络蒸馏衡量状态新颖性；详见 4.3）}
$$

$$
r_t^{\text{total}}=r_t^{\text{extrinsic}}+\beta r_t^{\text{intrinsic}}
\quad \text{（总奖励组合式；作用：合并任务奖励和探索奖励；详见 4.3）}
$$

### 标量与矩阵形式

以上公式均采用逐状态（标量）形式。将所有状态排成向量、转移关系写成矩阵后，$n$ 个标量方程可压缩为一行矩阵方程。

#### 符号约定

为避免维度表达过长，下面用 $n=|\mathcal{S}|$ 表示状态数量，用
$n_A=|\mathcal{A}|$ 表示动作数量。

| 符号                 | 维度            | 含义                                                                   |
| -------------------- | --------------- | ---------------------------------------------------------------------- |
| $\boldsymbol{v}_\pi$ | $n \times 1$    | 所有状态的价值                                                         |
| $\boldsymbol{r}_\pi$ | $n \times 1$    | 每个状态的期望即时奖励                                                 |
| $P_\pi$              | $n \times n$    | 策略诱导的转移矩阵，$P_\pi[i,j]=\sum_a \pi(a\mid s_i)p(s_j\mid s_i,a)$ |
| $\boldsymbol{q}_\pi$ | $nn_A \times 1$ | 所有 $(s,a)$ 对的 Q 值                                                 |
| $P$                  | $nn_A \times n$ | 转移矩阵，$P[(s,a),s']=P(s'\mid s,a)$                                  |
| $\Pi_\pi$            | $n \times nn_A$ | 策略矩阵，$\Pi_\pi[s,(s,a)]=\pi(a\mid s)$                              |

#### 对照总表

**贝尔曼期望方程**

逐状态形式：

$$
V^\pi(s)=\sum_a\pi(a\mid s)\left[R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V^\pi(s')\right]
$$

矩阵形式：

$$
\boldsymbol{v}_\pi
=
\boldsymbol{r}_\pi+\gamma P_\pi\boldsymbol{v}_\pi
$$

**贝尔曼最优方程**

逐状态形式：

$$
V^*(s)=\max_a\left[R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V^*(s')\right]
$$

矩阵形式：

$$
\boldsymbol{v}_*
=
\boldsymbol{r}_*+\gamma P_*\boldsymbol{v}_*
\quad\text{（逐行取 max）}
$$

**闭式解**

矩阵形式：

$$
\boldsymbol{v}=(I-\gamma P)^{-1}\boldsymbol{r}
$$

**V-Q 关系**

逐状态形式：

$$
V^\pi(s)=\sum_a\pi(a\mid s)Q^\pi(s,a)
$$

矩阵形式：

$$
\boldsymbol{v}_\pi=\Pi_\pi\boldsymbol{q}_\pi
$$

**Q 贝尔曼期望方程**

逐状态形式：

$$
Q^\pi(s,a)
=
R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)\sum_{a'}\pi(a'\mid s')Q^\pi(s',a')
$$

矩阵形式：

$$
\boldsymbol{q}_\pi
=
\boldsymbol{r}+\gamma P\Pi_\pi\boldsymbol{q}_\pi
$$

**Q 贝尔曼最优方程**

逐状态形式：

$$
Q^*(s,a)
=
R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)\max_{a'}Q^*(s',a')
$$

矩阵形式：

$$
\boldsymbol{q}_*
=
\boldsymbol{r}+\gamma P\cdot\mathrm{rowmax}(\boldsymbol{q}_*)
$$

**DP 策略评估**

逐状态形式：

$$
V(s)\leftarrow
\sum_a\pi(a\mid s)
\left[
R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V(s')
\right]
$$

矩阵形式：

$$
\boldsymbol{v}_{k+1}
=
\boldsymbol{r}_\pi+\gamma P_\pi\boldsymbol{v}_k
$$

MC 和 TD 基于采样更新单个状态，没有对应的矩阵形式。

#### 从 Q 到 V

将 $\boldsymbol{v}_\pi = \Pi_\pi \boldsymbol{q}_\pi$ 代入 $\boldsymbol{q}_\pi = \boldsymbol{r} + \gamma P \boldsymbol{v}_\pi$，两边左乘 $\Pi_\pi$：

$$
\Pi_\pi \boldsymbol{q}_\pi = \Pi_\pi \boldsymbol{r} + \gamma \Pi_\pi P \boldsymbol{v}_\pi
\quad\Longrightarrow\quad
\boldsymbol{v}_\pi = \underbrace{\Pi_\pi \boldsymbol{r}}_{\boldsymbol{r}_\pi} + \gamma \underbrace{\Pi_\pi P}_{P_\pi} \boldsymbol{v}_\pi
$$

Q 的矩阵形式保留了动作维度（策略平均由 $\Pi_\pi$ 单独完成），V 的矩阵形式已把策略平均融进了 $\boldsymbol{r}_\pi$ 和 $P_\pi$——这正是"$Q$ 比 $V$ 携带更细粒度信息"的矩阵语言表达。

### 公式依赖关系

这些公式并非相互独立，而是构成一组逐层递进的定义和推论。

| 层级     | 核心问题                                     | 关键对象                                                       |
| -------- | -------------------------------------------- | -------------------------------------------------------------- |
| 问题建模 | 环境、动作、反馈和未来权重是什么？           | $\mathcal{M}=\langle\mathcal{S},\mathcal{A},P,R,\gamma\rangle$ |
| 优化目标 | 如何度量一条轨迹从某时刻开始的长期回报？     | $G_t=\sum_{k=0}^{\infty}\gamma^k r_{t+k}$                      |
| 行为规则 | 智能体在状态中如何选择动作？                 | $\pi(s)$、$\pi(a\mid s)$、$\pi_\theta(a\mid s)$                |
| 状态评估 | 当前状态在长期意义下值多少？                 | $V^\pi(s)=\mathbb{E}_\pi[G_t\mid s_t=s]$                       |
| 递归结构 | 长期回报如何拆成一步奖励和未来价值？         | 贝尔曼期望方程、贝尔曼最优方程                                 |
| 数据学习 | 在环境模型未知或难以完全枚举时如何估计价值？ | DP、MC、TD、$\delta$                                           |
| 动作评估 | 固定第一步动作后如何评估长期回报？           | $Q^\pi(s,a)$、$Q^*(s,a)$                                       |
| 表格控制 | 如何用采样经验学习动作价值并导出策略？       | Q-Learning、TD Target、$\varepsilon$-greedy                    |
| 策略优化 | 如何直接优化一个参数化策略？                 | $J(\theta)$、$\nabla_\theta J(\theta)$                         |
| 目标设计 | 算法所最大化的奖励信号是什么？               | $R(s,a)$、$F(s,a,s')$、$r_t^{\text{total}}$                    |

上述层级体现了第 2 至 4 章的逻辑顺序：环境定义先于回报定义，回报定义先于价值定义；价值递推是 DP、MC、TD 的基础；状态价值和动作价值为策略改进提供依据；奖励信号则决定所有优化目标的具体含义。

### 基础章节主线

#### 从回报到贝尔曼递推

第 2 至 4 章最重要的数学结构是递归性。折扣回报可写为无限求和：

$$
G_t=\sum_{k=0}^{\infty}\gamma^k r_{t+k}
$$

该表达式也可等价写成一步递推形式：

$$
G_t=r_t+\gamma G_{t+1}
$$

该递推形式将长期回报分解为当前即时奖励与下一时刻回报。贝尔曼方程正是将这一轨迹层面的递推关系推广到期望价值 $V^\pi(s)$。

#### 从状态价值到采样学习

若环境模型 $P$ 和 $R$ 已知，可以直接使用贝尔曼期望方程进行 DP 更新。若模型未知，则需要基于采样轨迹进行价值估计：

- MC 使用完整回报 $G_t$ 作为目标，估计无偏但方差高。
- TD 使用 $r+\gamma V(s')$ 作为自举目标，方差低且可以在线更新。
- TD Error $\delta=r+\gamma V(s')-V(s)$ 衡量当前价值估计与一步贝尔曼目标之间的差距。

这一思想构成后续 Critic、DQN target、GAE 等技术的基础。

#### 从状态价值到动作价值

$V^\pi(s)$ 评估状态价值，但不直接给出状态 $s$ 下各动作的相对优劣。为刻画动作层面的长期回报，第 3.2 节引入动作价值：

$$
Q^\pi(s,a)=\mathbb{E}_\pi[G_t\mid s_t=s,a_t=a]
$$

该定义固定第一步动作，并评估随后按照策略 $\pi$ 行动所得的长期回报。因此，$Q$ 函数比 $V$ 函数包含更直接的动作选择信息。当最优动作价值 $Q^*(s,a)$ 已知时，最优策略可由 $\arg\max_a Q^*(s,a)$ 诱导得到。

#### 从动作价值到 Q-Learning

第 3.2 节将 TD 思想应用到动作价值表上。每一步经验 $(s,a,r,s')$ 都可以构造 TD 目标：

$$
r+\gamma\max_{a'}Q(s',a')
$$

并用它修正当前的 $Q(s,a)$。这使智能体不需要完整环境模型，也不必等一条轨迹结束，就能逐步学习一张可用于决策的 Q 表。表格 Q-Learning 适合小规模离散状态空间；一旦状态空间巨大或连续，就需要第 5 章讨论的函数近似方法。

#### 从策略表示到策略优化

第 2.3 节给出另一种策略学习表述：不必先显式学习每个动作的价值，也可以将策略表示为带参数的分布 $\pi_\theta(a\mid s)$，并最大化：

$$
J(\theta)=\mathbb{E}_{\pi_\theta}[G_t]
$$

策略梯度公式表明，参数更新方向由两部分组成：$\nabla_\theta\log\pi_\theta(a\mid s)$ 描述提高当前动作概率的参数方向，$G_t$ 则为该方向提供回报权重。第 6 章将对这一结果作进一步推导。

#### 奖励函数决定优化目标

所有价值函数、策略目标和更新规则最终都依赖于奖励的累积和。奖励过于稀疏会导致学习信号不足；奖励设计不当则可能使智能体优化偏离任务意图。第 4.3 节讨论的奖励塑形与内在奖励，均用于在增强学习信号的同时尽量保持任务目标不变。

### 本章复习问题

完成第 2 至 4 章学习后，应能够回答以下问题：

1. 给定一个任务，如何写出它的 MDP 五元组？

::: details 参考答案
将任务表示为 $\mathcal{M}=\langle\mathcal{S},\mathcal{A},P,R,\gamma\rangle$。其中，$\mathcal{S}$ 描述智能体可能处于的状态集合，$\mathcal{A}$ 描述可选动作集合，$P(s'\mid s,a)$ 描述执行动作后的状态转移规律，$R(s,a)$ 或 $R(s,a,s')$ 描述即时奖励，$\gamma$ 描述未来奖励的折扣权重。写 MDP 时，应说明每个要素在具体任务中的含义，而不仅是列出符号。
:::

2. 为什么 RL 优化的是折扣累积回报，而不只是即时奖励？

::: details 参考答案
强化学习研究的是序列决策。某一步动作不仅影响当前奖励，也会改变后续状态，从而影响未来可获得的奖励。因此，优化即时奖励可能导致短视策略。折扣累积回报

$$
G_t=\sum_{k=0}^{\infty}\gamma^k r_{t+k}
$$

将当前及未来奖励统一为一个长期目标，并通过 $\gamma$ 控制未来奖励的重要程度。在无限期任务中，$\gamma<1$ 还可保证回报有限。
:::

3. $G_t$、$V^\pi(s)$、$Q^\pi(s,a)$、$J(\theta)$ 分别在评估什么？

::: details 参考答案
$G_t$ 是从时刻 $t$ 开始沿某一条具体轨迹得到的折扣累积回报。$V^\pi(s)$ 是从状态 $s$ 出发并按照策略 $\pi$ 行动时，$G_t$ 的期望，用于评估状态价值。$Q^\pi(s,a)$ 是在状态 $s$ 先执行动作 $a$，之后按照策略 $\pi$ 行动时，$G_t$ 的期望，用于评估动作价值。$J(\theta)$ 是参数化策略 $\pi_\theta$ 的总体期望回报，用于衡量并优化整个策略。
:::

4. 贝尔曼期望方程和贝尔曼最优方程有什么区别？

::: details 参考答案
贝尔曼期望方程用于评估给定策略 $\pi$，其动作选择由 $\pi(a\mid s)$ 加权平均：

$$
V^\pi(s)=\sum_a\pi(a\mid s)\left[R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V^\pi(s')\right].
$$

贝尔曼最优方程用于定义最优价值，它不再固定某个策略，而是在所有动作中取最大值：

$$
V^*(s)=\max_a\left[R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V^*(s')\right].
$$

前者回答“按照该策略行动时价值是多少”，后者回答“在最优行动下价值最高是多少”。
:::

5. DP、MC、TD 分别需要什么信息，什么时候更新，误差来源是什么？

::: details 参考答案
DP 需要已知环境模型 $P$ 和 $R$，通过对所有可能动作和下一状态求期望进行更新，误差主要来自迭代尚未收敛或函数近似误差。MC 不需要环境模型，但需要等一条 episode 结束后用完整回报 $G_t$ 更新；它以真实回报为目标，估计无偏，但方差较高。TD 不需要环境模型，也不必等 episode 结束，走一步即可用 $r+\gamma V(s')$ 更新；它方差较低，但由于使用估计值 $V(s')$ 自举，会引入偏差。
:::

6. TD Error 为什么会成为后续 Critic、深度 Q 网络和 GAE 的共同学习信号？

::: details 参考答案
TD Error

$$
\delta=r+\gamma V(s')-V(s)
$$

衡量当前价值估计与一步贝尔曼目标之间的差距。Critic 可以用它更新状态价值函数；深度 Q 网络使用同样的自举思想构造 Q 函数的训练目标；GAE 则将多个时间步的 TD Error 加权累积，用于估计优势函数。因此，TD Error 是将贝尔曼递推转化为可采样训练信号的基本形式。
:::

7. 为什么 $Q(s,a)$ 能直接诱导动作选择？

::: details 参考答案
$Q(s,a)$ 表示在状态 $s$ 选择动作 $a$ 后的长期期望回报。若已知各动作的动作价值，则可直接比较同一状态下不同动作的 $Q$ 值。最优动作价值 $Q^*(s,a)$ 已知时，最优策略可写为

$$
\pi^*(s)=\arg\max_a Q^*(s,a).
$$

因此，$Q$ 函数不仅评估动作，还可通过最大化动作价值直接给出动作选择规则。
:::

8. 为什么参数化策略需要目标函数 $J(\theta)$？

::: details 参考答案
参数化策略 $\pi_\theta(a\mid s)$ 的学习对象是参数 $\theta$。为了优化这些参数，需要定义一个以 $\theta$ 为自变量的目标函数：

$$
J(\theta)=\mathbb{E}_{\pi_\theta}[G_t].
$$

$J(\theta)$ 衡量当前策略的期望长期回报。策略梯度方法通过估计 $\nabla_\theta J(\theta)$ 来调整参数，使高回报轨迹中的动作概率增大，从而改进策略。
:::

9. 奖励塑形为什么可能加速学习，又为什么可能带来目标偏移？

::: details 参考答案
奖励塑形通过增加中间奖励，使智能体在尚未到达最终目标前也能获得学习信号，因此可缓解稀疏奖励导致的学习困难。例如，距离目标更近时给予额外奖励，可以引导探索方向。但若塑形奖励设计不当，智能体可能优化塑形信号而非原始任务目标，产生目标偏移。势函数塑形

$$
F(s,a,s')=\gamma\Phi(s')-\Phi(s)
$$

在理论上可保持最优策略不变，因此是一类较安全的奖励塑形形式。
:::

上述问题强调公式背后的概念角色。掌握这些基础内容的关键，不仅在于记忆公式形式，还在于理解每个对象在强化学习问题中的功能。

### 后续章节如何使用

| 后续章节             | 主要使用的本章对象                                                   | 用法                                                    |
| -------------------- | -------------------------------------------------------------------- | ------------------------------------------------------- |
| 第 5 章深度 Q 网络    | $Q(s,a)$、$Q^*(s,a)$、$\arg\max_a Q(s,a)$、TD Target                 | 用神经网络近似动作价值，用自举目标更新 Q 函数           |
| 第 6 章策略梯度       | $\pi_\theta(a\mid s)$、$J(\theta)$、$\nabla_\theta J(\theta)$、$G_t$ | 直接优化参数化策略，提高高回报动作概率                  |
| 第 7 章 Actor-Critic  | $V(s)$、TD Error、$J(\theta)$                                        | 用价值函数作为 Critic，为策略更新提供低方差信号         |
| 第 8 章 PPO           | $V(s)$、优势函数、TD Error、策略目标                                 | 用 Critic 估计优势，并约束策略更新幅度                  |
| 第 13 章以后大模型 RL | 策略、奖励、回报、目标函数                                           | 把 token 生成看作序列决策，把偏好或验证信号转成优化目标 |

因此，第 2 至 4 章中的公式不仅用于当前练习，也将在后续算法中反复出现。随着模型从表格表示过渡到函数近似，这些对象将以神经网络、损失函数、训练目标、优势函数和 KL 约束等形式重新出现。

### 小结

第 2 至 4 章建立了强化学习理论基础的基本结构：

1. 用 MDP 五元组定义序列决策问题。
2. 用折扣累积回报 $G_t$ 定义长期优化目标。
3. 用 $V^\pi(s)$ 和 $Q^\pi(s,a)$ 评估状态与动作。
4. 用贝尔曼方程揭示价值的递归结构。
5. 用 DP、MC、TD 说明价值可以如何被计算或估计。
6. 用 $J(\theta)$ 将参数化策略学习表述为优化问题。
7. 掌握从数据获取维度区分算法（On/Off-policy, Online/Offline）。
8. 用奖励设计说明优化目标从何而来，以及为什么目标定义本身会影响学习结果。

这份索引给出了后续基础章节的全貌。正式的推导仍从第 2 章留下的问题开始：轨迹尚未发生时，怎样估计一个状态的长期收益。下一章进入[第 3 章：价值函数与贝尔曼方程](./value-bellman)，先定义状态价值，再推导贝尔曼期望方程、贝尔曼最优方程与最优策略。
