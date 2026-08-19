# 24.4 视觉生成与 DanceGRPO

第 23 章讨论了 VLM 的**理解**侧，前面的 24.1～24.3 又把多模态强化学习扩展到音频交互和物理动作。现在转到视觉 AI 的另一侧：**生成**。模型接收文字、图像或控制条件，经过完整的去噪轨迹生成一张图或一段视频，策略优化的对象也从答案 token 变成了生成过程。

这件事表面上像“让模型画得更好看”。但在真实应用里，用户很少只要“好看”。用户真正想要的是：主体要对，数量要对，位置关系要对，细节要对，整体风格还要自然。

例如 prompt 写的是：

> 玻璃走廊里有三把红色雨伞，右侧墙上有一块蓝色指示牌。

模型生成了一张很漂亮的玻璃走廊，但只有两把伞，指示牌也不是蓝色。这张图应该给高分还是低分？如果只看审美，它可能很好；如果看指令遵循，它明显失败。

因此，视觉生成 RL 的核心问题是：

> **能不能把“生成得好”拆成可以学习、可以比较、可以优化的反馈信号？**

本节沿着一次完整的生成轨迹来讲：先看视觉生成为什么比视觉问答更难写 reward，再把 Diffusion 的去噪过程翻译成 MDP，最后落到 DDPO 的策略梯度、训练步骤和 reward model 设计。

![DDPO Training Teaser](./images/ref-ddpo-teaser.jpg)

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em>图 1：DDPO 论文/项目展示的 RL 后训练效果。不同 reward 会把 Diffusion 模型推向不同的生成偏好，直观说明视觉生成 RL 的关键：reward 设计会直接塑造最终图像分布。来源：<a href="https://github.com/kvablack/ddpo-pytorch" target="_blank" rel="noopener noreferrer">DDPO GitHub</a>，对应论文 Black et al., 2024</em>
</div>

这张图对应的算法主线来自 DDPO 论文；后文把 Diffusion 写成 MDP、再用策略梯度更新去噪轨迹，也以这篇论文为核心参照[^ddpo]。

## 从 LLM 到 VLM 再到视觉生成：RL 迁移时到底变了什么？

与其只问“VLM 能不能直接迁移到生成”，不如先看一条更长的路线：

> **LLM 文本 RL → VLM 理解 RL → 视觉生成 RL**

这三者都在使用 RL 的同一套基本语言：模型是 policy，模型输出形成 trajectory，reward 评价这条 trajectory，训练时再用 KL、clipping 或 advantage 让更新更稳定。但每往前走一步，被优化的对象都在变化。

先看 LLM。LLM 的输入是文本上下文，输出也是文本。一次回答可以看成一条 token 轨迹：

$$
y=(y_1,y_2,\ldots,y_T)
$$

每一步 action 是“选下一个 token”。reward 可以来自人类偏好模型、规则检查、数学验证器、代码运行结果，或者格式约束。PPO、DPO、GRPO 这些方法虽然细节不同，但大多围绕“如何让文本回答更符合 reward”展开。

到了 VLM 理解侧，输入多了图像：

$$
c=(\text{image}, \text{text prompt})
$$

但很多任务的输出仍然是文本、选项、坐标或 bounding box。也就是说，模型多了视觉证据，action 却仍然经常落在 token 或结构化答案上。reward 也相对容易写：答案是否正确、框是否对齐、IoU 是否够高、推理格式是否满足要求。这就是前几节 VLM-R1 / VISTA-Gym 这类工作的核心：让模型学会利用视觉信息，而不是只靠语言先验猜答案。

再到视觉生成，事情才真正换了一个层级。模型的目标不再是“看图后回答”，而是“根据 prompt 创造一个新的视觉结果”。输出不再是一串答案 token，而是一张图、一段视频，或者更准确地说，是一条 latent / denoising 轨迹。此时 reward 也不再主要问“答案是否等于标准答案”，而是问：

- 图像是否符合 prompt？
- 数量、颜色、空间关系有没有错？
- 人类是否更偏好这个结果？
- 画面是否自然、清晰、风格一致？
- 视频里前后帧是否连贯？

可以把这三站放在一张表里：

| 阶段           | 输入            | 输出                    | RL 里的 action              | reward 更像什么              |
| -------------- | --------------- | ----------------------- | --------------------------- | ---------------------------- |
| LLM 文本后训练 | 文本 prompt     | 文本回答                | 下一个 token                | 偏好、规则、验证器           |
| VLM 理解后训练 | 图像 + 文本问题 | 文本、选项、框、坐标    | 主要还是 token 或结构化答案 | 答案正确性、IoU、工具验证    |
| 视觉生成后训练 | 文本 / 图像条件 | 图像、视频、latent 轨迹 | 每一步去噪转移              | 偏好、对齐、质量、细粒度约束 |

所以，视觉生成 RL 不是把前面的东西推翻，而是把同一套 RL 语言换到一个更难的对象上。

能继承的部分包括：policy gradient、advantage、KL 正则、PPO-style clipping、reward model、judge model。真正要重写的部分是 state、action、trajectory 和 reward。

这也是为什么 DDPO 这类工作会先做一件看似朴素、但非常重要的事：把 Diffusion 的去噪过程翻译成状态、动作、轨迹和奖励[^ddpo]。只有这一步翻译清楚了，我们才知道策略梯度到底在更新什么。

## 先从 Diffusion 的采样过程看起

Diffusion 模型的生成过程可以理解为“从噪声出发，逐步去噪”。

最开始，模型有一个接近随机噪声的 latent，记作 $x_T$。然后模型一步一步生成：

$$
x_T \rightarrow x_{T-1} \rightarrow \cdots \rightarrow x_1 \rightarrow x_0
$$

这里 $x_0$ 是最终图像对应的 latent。再经过 decoder，就得到用户看到的图像。

每一步去噪时，模型会看三个东西：

| 符号  | 含义                  |
| ----- | --------------------- |
| $x_t$ | 当前还带噪声的 latent |
| $t$   | 当前去噪步数          |
| $c$   | prompt 或条件信息     |

模型要决定的是下一步 latent：

$$
x_{t-1}\sim p_\theta(x_{t-1}\mid x_t,t,c)
$$

这行公式的意思是：给定当前噪声状态 $x_t$、时间步 $t$ 和 prompt $c$，模型用参数 $\theta$ 定义一个概率分布，并从中采样下一步 $x_{t-1}$。

为什么说它像 policy？因为在 RL 里，policy 的定义就是：

$$
\pi_\theta(a\mid s)
$$

也就是“给定当前状态 $s$，选择动作 $a$ 的概率分布”。

LLM 里我们很熟悉这个形式：

$$
\pi_\theta(y_t\mid y_{<t},c)
$$

给定前面的 token $y_{<t}$ 和上下文 $c$，模型选择下一个 token $y_t$。所以 token 是 action，文本上下文是 state。

Diffusion 的去噪分布也有同样的形状：

$$
p_\theta(x_{t-1}\mid x_t,t,c)
$$

给定当前 noisy latent、时间步和 prompt，模型选择下一步 latent。于是可以把 $(x_t,t,c)$ 看成 state，把 $x_{t-1}$ 或等价的去噪方向看成 action。

当然，这句话只是在说“形式上可以看成 policy”。它还不等于已经完成 RL。只有再定义最终图像的 reward，并用这个 reward 更新 $p_\theta$，这个采样过程才真正变成强化学习问题。

## 把 Diffusion 翻译成 MDP 语言

DDPO（Denoising Diffusion Policy Optimization）的关键观察是：Diffusion 的采样过程可以被看成一个有限长度的 MDP。Black 等人的 DDPO 论文明确把 denoising 视为 multi-step decision-making problem，然后用 policy gradient 直接优化下游 reward[^ddpo]。

这个翻译非常重要。我们逐项来看：

| RL 里的概念       | Diffusion 里的对应                        |
| ----------------- | ----------------------------------------- |
| 状态 $s_t$        | 当前 latent、时间步和 prompt：$(x_t,t,c)$ |
| 动作 $a_t$        | 采样下一步 latent，或预测去噪方向         |
| 轨迹 $\tau$       | 一整条去噪链路：$x_T,\ldots,x_0$          |
| 奖励 $R$          | 最终图像由 reward model 给出的分数        |
| 策略 $\pi_\theta$ | Diffusion 模型的去噪分布 $p_\theta$       |

因此，一次生成就像一个 episode：

$$
\tau=(x_T,x_{T-1},\ldots,x_0)
$$

在 RL 里，episode 指的是一次完整交互：从初始状态开始，智能体连续选择动作，环境连续给出下一个状态，直到任务终止为止。比如 CartPole 里，从小车和杆子被初始化，到杆子倒下或达到最大步数，是一个 episode。文本生成里，从开始 token 生成到结束 token，也可以看成一个 episode。

episode 的意义，是给“结果”划出边界。它告诉我们：哪些状态和动作属于同一次尝试，最终的好坏应该回看哪一串决策。对图像生成来说，单独看某一个中间 latent 很难判断它是不是“好图”。真正能被人类偏好模型、CLIP score、审美模型或任务 reward 打分的，通常是最后得到的 $x_0$。所以我们把从纯噪声 $x_T$ 一步步去噪到 $x_0$ 的整条链路当成一个 episode，终止状态就是最终图像。

episode 结束以后，reward model 才看到最终图像，并给出分数：

$$
R=r_\phi(x_0,c)
$$

注意这里的 $r_\phi$ 不是生成模型本身，而是另一个打分模型。它的参数是 $\phi$，生成模型的参数是 $\theta$。

这样一来，生成模型的目标可以写成：

$$
J(\theta)=\mathbb{E}_{\tau\sim p_\theta}\left[r_\phi(x_0,c)\right]
$$

这句话读作：我们希望在模型自己采样出来的轨迹上，最终图像的平均 reward 尽可能高。

## 用策略梯度更新去噪策略

有了上面的 MDP 翻译，DDPO 就不神秘了。它本质上是在 Diffusion 采样轨迹上做策略梯度。

先把我们要做的事情，和它对应的经典论文线索摆在一起。表里左边是我们马上要做的事，右边是它来自的经典线索：

| 我们要做的事情                                               | 对应论文线索                           |
| ------------------------------------------------------------ | -------------------------------------- |
| 把一次去噪生成看成一个 episode / MDP                         | DDPO：Black et al., 2024[^ddpo]        |
| 高分样本提高概率，低分样本降低概率；数学上叫策略梯度         | REINFORCE：Williams, 1992[^reinforce]  |
| 用 old/new logprob ratio 和 clipping，让每次更新不要迈太大步 | PPO：Schulman et al., 2017[^ppo]       |
| 用 KL 约束限制模型不要偏离参考模型太远                       | DPOK：Fan et al., 2023[^dpok]          |
| 用人类偏好或审美偏好训练 reward model                        | Pick-a-Pic / HPS v2[^pickapic][^hpsv2] |

其中最容易被术语吓到的是第二行。用大白话说就是：

> 如果一条去噪轨迹最后生成了高分图像，就让模型以后更容易采到这条轨迹里的那些步骤；如果最后得分低，就让这些步骤以后不那么容易被采到。

问题在于，训练模型不能只说“让它更容易发生”。我们需要一个可以计算的梯度方向。REINFORCE 里的 log-derivative trick，就是把这句话变成可训练公式的那一步。

先把接下来会出现的符号对齐一下：

| 符号             | 可以先怎么理解                                           |
| ---------------- | -------------------------------------------------------- |
| $\theta$         | Diffusion 模型的参数，也就是训练时要改的东西             |
| $c$              | prompt                                                   |
| $\tau$           | 一整条生成轨迹，从 $x_T$ 去噪到 $x_0$                    |
| $p_\theta(\tau)$ | 当前模型采到这条轨迹的概率                               |
| $R(\tau,c)$      | 这条轨迹最后生成图像的分数                               |
| $J(\theta)$      | 当前模型的平均分数；训练目标就是让它变大                 |
| $\nabla_\theta$  | “参数往哪个方向改，$J(\theta)$ 会变大”的方向，中文叫梯度 |

我们先把一条去噪轨迹的概率写出来。为了简化记号，下面默认 prompt $c$ 是给定的：

$$
p_\theta(\tau\mid c)
=
p(x_T)\prod_{t=1}^{T}
p_\theta(x_{t-1}\mid x_t,t,c)
$$

这行公式有两个意思。

第一，初始噪声 $x_T$ 通常从标准高斯分布采样，它不依赖模型参数 $\theta$。第二，真正由模型控制的是每一步去噪分布 $p_\theta(x_{t-1}\mid x_t,t,c)$。

这个连乘也很直观：一整条轨迹要发生，必须第 $T$ 步采到 $x_{T-1}$，第 $T-1$ 步采到 $x_{T-2}$，一直到最后采到 $x_0$。所以整条轨迹的概率，就是每一步概率乘在一起。

生成模型想最大化最终 reward：

$$
J(\theta)
=
\mathbb{E}_{\tau\sim p_\theta(\tau\mid c)}
\left[R(\tau,c)\right]
$$

其中 $R(\tau,c)=r_\phi(x_0,c)$，也就是 reward model 对最终图像的打分。

先用一个离散版的小例子理解它。假设同一个 prompt 下，模型只可能采出三条去噪轨迹：

| 轨迹     | 模型采到它的概率 | 最终 reward |
| -------- | ---------------- | ----------- |
| $\tau_1$ | $p_1$            | $R_1$       |
| $\tau_2$ | $p_2$            | $R_2$       |
| $\tau_3$ | $p_3$            | $R_3$       |

那平均 reward 就是：

$$
J=p_1R_1+p_2R_2+p_3R_3
$$

如果 $\tau_2$ 的 reward 很高，我们当然希望 $p_2$ 变大。也就是说，RL 更新的直觉不是“直接把图片像素往某个方向推”，而是“改变模型的采样概率”：高分轨迹的概率往上调，低分轨迹的概率往下调。

真实的 Diffusion 不只有三条轨迹，而是有连续、巨量的可能轨迹。把上面这个加权平均写成积分，就是：

$$
J(\theta)
=
\int p_\theta(\tau\mid c)R(\tau,c)\,d\tau
$$

这里的积分可以先不用想得太可怕。它就是“把所有可能轨迹的概率乘以分数，再全部加起来”。离散时是 $p_1R_1+p_2R_2+p_3R_3$；连续时就写成积分。

现在对 $\theta$ 求梯度，也就是问：模型参数往哪里改，平均 reward 会变高？

$$
\nabla_\theta J(\theta)
=
\int \nabla_\theta p_\theta(\tau\mid c)R(\tau,c)\,d\tau
$$

现在问题来了：这个式子里有 $\nabla_\theta p_\theta(\tau\mid c)$，意思是“模型参数变化时，这条完整轨迹的概率怎么变化”。但训练时我们拿到的是模型采样出来的一批轨迹，不可能枚举所有轨迹。我们希望把梯度改写成一个“对采样轨迹求平均”的形式，这样就能用实际采样来估计。

这里用到一个很小的等式，叫 **log-derivative trick**，也叫 **score-function trick**。它正是 REINFORCE 这类策略梯度方法的核心技巧[^reinforce]：

$$
\nabla_\theta p_\theta(\tau\mid c)
=
p_\theta(\tau\mid c)\nabla_\theta\log p_\theta(\tau\mid c)
$$

这个等式只是把 $\nabla p$ 改写成了 $p\nabla\log p$。原因是：

$$
\nabla_\theta\log p_\theta
=
\frac{1}{p_\theta}\nabla_\theta p_\theta
$$

两边同时乘上 $p_\theta$，就得到：

$$
p_\theta\nabla_\theta\log p_\theta
=
\nabla_\theta p_\theta
$$

它听起来像技巧，本质上只是一次代数改写。它的好处是：公式里重新出现了 $p_\theta(\tau\mid c)$，而这正好表示“从当前模型里采样轨迹”。于是我们就能用实际采样出来的轨迹估计梯度。

代回去：

$$
\nabla_\theta J(\theta)
=
\int p_\theta(\tau\mid c)
\nabla_\theta\log p_\theta(\tau\mid c)
R(\tau,c)\,d\tau
$$

也就是：

$$
\nabla_\theta J(\theta)
=
\mathbb{E}_{\tau\sim p_\theta}
\left[
\nabla_\theta\log p_\theta(\tau\mid c)R(\tau,c)
\right]
$$

这一步很关键，它把一个无法直接计算的期望，变成了可以靠采样估计的形式。训练时只要做三件事：

1. 用当前 Diffusion 模型采样一条轨迹 $\tau$；
2. 用 reward model 给最终图像打分，得到 $R(\tau,c)$；
3. 看这条轨迹在模型下的 log probability，也就是 $\log p_\theta(\tau\mid c)$，然后按 reward 的高低调大或调小它。

所以，策略梯度不需要对 reward 本身求导。reward model 可以是不可微的，也可以是一个黑盒打分器；我们只需要知道“这条轨迹得了多少分”。DDPO 利用的正是这个性质：reward 可以来自美学模型、压缩率、VLM feedback 或其他不可直接反传的目标[^ddpo]。

接下来展开轨迹的 log probability：

$$
\log p_\theta(\tau\mid c)
=
\log p(x_T)
+
\sum_{t=1}^{T}
\log p_\theta(x_{t-1}\mid x_t,t,c)
$$

为什么要取 log？因为原来的轨迹概率是一串概率的乘积。乘积很长时不好处理；取 log 以后，乘法会变成加法：

$$
\log(ab)=\log a+\log b
$$

所以整条轨迹的 log probability，就等于每一步 log probability 加起来。

由于 $\log p(x_T)$ 不依赖 $\theta$，求梯度时它会消失：

$$
\nabla_\theta\log p_\theta(\tau\mid c)
=
\sum_{t=1}^{T}
\nabla_\theta
\log p_\theta(x_{t-1}\mid x_t,t,c)
$$

所以最朴素的策略梯度就是：

$$
\nabla_\theta J
=
\mathbb{E}\left[
\sum_{t=1}^{T}
\nabla_\theta \log p_\theta(x_{t-1}\mid x_t,t,c)
\cdot R(\tau,c)
\right]
$$

这就是 REINFORCE 在 Diffusion 轨迹上的形式[^reinforce]：如果某条去噪轨迹最后拿到高 reward，就提高这条轨迹中每一步采样动作的概率；如果 reward 低，就降低它们的概率。Black 等人的 DDPO 论文就是把这条思路搬到 Diffusion 的去噪轨迹上[^ddpo]。

### Baseline 和 advantage 为什么可以减？

直接用 $R(\tau,c)$ 更新会有很大方差。一个 prompt 可能天然更容易生成高分图，另一个 prompt 可能天然更难。我们更关心的是：这次采样结果是否比同类样本更好。

因此可以减去一个 baseline $b(c)$：

$$
\hat{A}=R(\tau,c)-b(c)
$$

这里的 $\hat{A}$ 叫 advantage。它不是问“这张图绝对分数是多少”，而是问“它比参考水平好多少”。如果 reward 是 8 分，baseline 是 6 分，advantage 就是 +2，说明这次生成比预期好；如果 reward 是 5 分，baseline 是 6 分，advantage 就是 -1，说明这次生成比预期差。

那为什么可以减 baseline？直觉是：如果给同一组样本的分数都减去同一个常数，谁比谁好这件事没有变。训练真正需要的是“相对更好”还是“相对更差”。

数学上也可以验证它不会改变期望梯度。我们只需要证明：被减掉的 baseline 那一项，平均起来等于 0。

$$
\mathbb{E}_{\tau\sim p_\theta}
\left[
\nabla_\theta\log p_\theta(\tau\mid c)b(c)
\right]
=
b(c)\int p_\theta(\tau\mid c)
\nabla_\theta\log p_\theta(\tau\mid c)d\tau
$$

这里把 $b(c)$ 提到外面，是因为同一个 prompt 下它是固定数，不依赖具体采样动作。再用同一个 log-derivative trick：

$$
=
b(c)\int \nabla_\theta p_\theta(\tau\mid c)d\tau
=
b(c)\nabla_\theta \int p_\theta(\tau\mid c)d\tau
=
b(c)\nabla_\theta 1
=0
$$

最后一行为什么是 1？因为 $\int p_\theta(\tau\mid c)d\tau$ 表示“所有可能轨迹的概率加起来”，概率总和必然是 1。1 对参数求梯度还是 0。所以，减掉不依赖具体动作的 baseline，不会改变平均更新方向，只会让更新更稳定。

实际训练里，$\hat{A}$ 可以有几种常见做法：

| Advantage 做法        | 含义                                   |
| --------------------- | -------------------------------------- |
| $R-\bar{R}$           | 减去同一 batch 的平均 reward           |
| $R-b(c)$              | 减去 prompt 级别的历史平均 reward      |
| $R-V_\psi(x_t,t,c)$   | 减去 value model 对当前状态的预测      |
| normalize 后的 reward | 对 batch reward 做标准化，让尺度更稳定 |

加入 advantage 后，DDPO 常用的策略梯度可以写成：

$$
\nabla_\theta J
=
\mathbb{E}\left[
\sum_{t=1}^{T}
\nabla_\theta \log p_\theta(x_{t-1}\mid x_t,t,c)
\cdot \hat{A}_t
\right]
$$

如果只用终局 reward，那么每一步可以共享同一个 $\hat{A}$。如果训练了 value model，也可以给不同时间步不同的 $\hat{A}_t$。

### 这和 Diffusion 的 log probability 怎么对应？

在很多 Diffusion 实现中，每一步反向转移可以写成一个高斯分布：

$$
p_\theta(x_{t-1}\mid x_t,t,c)
=
\mathcal{N}\left(
\mu_\theta(x_t,t,c),
\sigma_t^2 I
\right)
$$

这里 $\mu_\theta$ 是模型预测出来的去噪均值，$\sigma_t$ 是这一步的噪声尺度。DDPO 的实现需要记录每一步动作的 log probability，本质上就是在这个反向转移分布上取 logprob[^ddpo]。于是这一动作的 log probability 近似是：

$$
\log p_\theta(x_{t-1}\mid x_t,t,c)
=
-
\frac{1}{2\sigma_t^2}
\left\|
x_{t-1}-\mu_\theta(x_t,t,c)
\right\|_2^2
+ \text{const}
$$

这行公式的意思也很朴素：如果实际采样出来的 $x_{t-1}$ 离模型预测的均值 $\mu_\theta(x_t,t,c)$ 很近，平方距离就小，log probability 就高；如果离得很远，平方距离就大，log probability 就低。

这解释了伪代码里的 `step.logprob` 是什么：它不是一句抽象的 RL 符号，而是当前模型在第 $t$ 步采样出这个 $x_{t-1}$ 的对数概率。

### 从最大化目标到最小化 loss

深度学习框架通常最小化 loss，而策略梯度是在最大化 $J(\theta)$。所以实现时会写成负号：

$$
\mathcal{L}_{\text{pg}}
=
-
\mathbb{E}\left[
\sum_{t=1}^{T}
\log p_\theta(x_{t-1}\mid x_t,t,c)
\cdot \hat{A}_t
\right]
$$

最小化这个 loss 等价于最大化策略梯度目标。直观地看：

| 情况                 | loss 会推动什么                      |
| -------------------- | ------------------------------------ |
| $\hat{A}_t>0$        | 提高这一步采样动作的 log probability |
| $\hat{A}_t<0$        | 降低这一步采样动作的 log probability |
| $\hat{A}_t\approx 0$ | 基本不更新这一步                     |

这里和第 6 章 REINFORCE 的思想完全一致，只是动作从“选择 token”变成了“选择下一步 latent”。

### 为什么还需要 KL 约束？

如果只最大化 reward，模型很容易走偏。原因很简单：reward model 本身并不完美。模型可能找到一些 reward model 喜欢、但人类并不真正喜欢的模式。

所以实际训练常常保留一个参考模型 $p_{\text{ref}}$，并惩罚当前模型偏离它太远。DPOK 也把“policy optimization + KL regularization”作为 text-to-image diffusion RL 微调的核心结构[^dpok]：

$$
\mathcal{L}_{\text{DDPO}}
=
\mathcal{L}_{\text{pg}}
+
\beta\,
\mathbb{E}\left[
\sum_{t=1}^{T}
\mathrm{KL}\left(
p_\theta(\cdot\mid x_t,t,c)
\|p_{\text{ref}}(\cdot\mid x_t,t,c)
\right)
\right]
$$

这个式子可以分成两部分理解：

| 项         | 作用                                     |
| ---------- | ---------------------------------------- |
| 策略梯度项 | 让高 reward 的采样轨迹更可能出现         |
| KL 项      | 限制模型不要为了追 reward 而远离原始模型 |

这和 RLHF、DPO、GRPO 里的思想相同：让模型改进，同时别偏离参考模型太远。

### DDPO 的最小训练流程

上面的推导说明了“为什么可以更新”。现在再把训练过程拆开看清楚：一次 DDPO update 里，数据是怎么从 prompt 走到 loss 的。

可以先记住一句话：

> DDPO 不是拿已有图片做监督学习，而是让当前模型自己生成图片，再用 reward 判断这些生成结果好不好，最后把好坏信号回传到采样轨迹上[^ddpo]。

这也是它和普通 diffusion fine-tuning 的核心区别。普通监督微调给模型看“应该生成什么”，DDPO 给模型看“你自己生成的这些结果里，哪些更值得变得更可能”。

#### Step 1：取一批 prompt

第一步不是取图片，而是取 prompt：

$$
\mathcal{B}=\{c_i\}_{i=1}^{B}
$$

其中 $B$ 是 batch size，$c_i$ 是第 $i$ 个 prompt。

prompt 数据的质量会直接影响训练方向。如果 prompt 太简单，模型可能只学会提高通用美学分；如果 prompt 里有数量、颜色、位置、关系等细粒度约束，reward model 才有机会训练模型的指令遵循能力。

实践中，一个好的 prompt batch 往往会混合几类样本：

| Prompt 类型       | 训练作用                       |
| ----------------- | ------------------------------ |
| 简单场景 prompt   | 稳定基础生成质量               |
| 多属性 prompt     | 训练颜色、材质、数量等细节     |
| 空间关系 prompt   | 训练左右、上下、遮挡、相对位置 |
| 长指令 prompt     | 训练复杂条件下的指令遵循       |
| 评测集风格 prompt | 让训练目标和最终评测更一致     |

这一步看似普通，但很关键：RL 只能优化模型在这些 prompt 分布上的行为。如果 prompt 分布太窄，模型很容易只在窄场景里变好。

#### Step 2：用当前模型做 rollout

第二步是用当前 Diffusion 模型生成图片。RL 里通常把这一步叫 **rollout**，意思是让策略自己跑一条轨迹。

对每个 prompt $c_i$，模型从噪声 $x_T$ 开始，采样一整条去噪链：

$$
\tau_i=(x_T^{(i)},x_{T-1}^{(i)},\ldots,x_0^{(i)})
$$

这里有一个容易被忽略的细节：训练时不能只保存最终图像，还要保存每一步去噪的关键信息。

| 要保存的内容                                        | 为什么要保存                                |
| --------------------------------------------------- | ------------------------------------------- |
| $x_t$                                               | 之后要重新计算这一步的 log probability      |
| $x_{t-1}$                                           | 这是第 $t$ 步实际采样出来的 action          |
| $\log p_{\theta_{\text{old}}}(x_{t-1}\mid x_t,t,c)$ | 如果后面做 PPO-style 更新，需要 old logprob |
| 最终图像 $x_0$ 或 decoded image                     | reward model 要对最终结果打分               |

为什么会出现 $\theta_{\text{old}}$？因为采样图片时用的是更新前的模型。等我们做梯度更新时，模型参数已经准备变化了。为了知道“新模型相对旧模型把这一步动作概率改了多少”，常常要保存 old logprob。

如果只做一次最朴素的 REINFORCE 更新，可以直接用采样时的 logprob。但在真实训练中，为了提高样本利用率，通常会对同一批 rollout 做多个 update epoch，这时 old logprob 就很重要。这个 old/new policy ratio 的思想来自 PPO[^ppo]，DDPO 里的 importance-sampling 变体也沿用了这种“固定 rollout、再用概率比修正更新”的思路[^ddpo]。

#### Step 3：用 reward model 给最终结果打分

第三步是把生成图像交给 reward model：

$$
R_i=r_\phi(x_0^{(i)},c_i)
$$

这里要特别注意：reward model 只负责打分，不一定参与反向传播。策略梯度需要的是“这条轨迹得了多少分”，而不是 reward 对像素或 latent 的梯度。

这也是 DDPO 相比可微 reward backprop 的一个优势：reward 可以来自很复杂的系统，比如 VLM judge、人类偏好模型、规则检查器，甚至多个模型的组合。只要最后能给出一个标量分数，就能作为策略梯度的信号。相对地，DRaFT 和 VADER 这类工作会利用可微 reward 的梯度直接反传到图像或视频扩散模型中[^draft][^vader]。

一个常见的 reward 计算流程是：

1. 把 latent $x_0$ decode 成图像。
2. 用文本-图像对齐模型检查是否符合 prompt。
3. 用偏好模型或美学模型给视觉质量打分。
4. 用规则或 VLM 检查数量、颜色、空间关系等硬约束。
5. 合并得到最终 reward $R_i$。

这一步最怕 reward 尺度不稳定。比如有的 reward 在 $[0,1]$，有的 reward 在 $[-10,10]$，直接相加会让某一项主导训练。因此实际训练常常会做 clipping、归一化或分层过滤。

#### Step 4：把 reward 变成 advantage

第四步是从 reward 计算 advantage。最简单的做法是 batch 内中心化：

$$
\hat{A}_i=R_i-\frac{1}{B}\sum_{j=1}^{B}R_j
$$

如果还想让尺度更稳定，可以再除以标准差：

$$
\hat{A}_i=
\frac{R_i-\mathrm{mean}(R)}
{\mathrm{std}(R)+\epsilon}
$$

这样做以后，$\hat{A}_i>0$ 表示第 $i$ 张图比本 batch 平均更好，$\hat{A}_i<0$ 表示它比平均更差。

为什么不直接用 $R_i$？因为绝对分数经常不好解释。某个 prompt 本身很难，生成 0.6 分已经不错；另一个 prompt 很简单，0.8 分可能只是平均水平。advantage 更关心“相对表现”，所以训练更稳。

在更完整的实现里，也可以训练一个 value model：

$$
V_\psi(x_t,t,c)\approx
\mathbb{E}[R\mid x_t,t,c]
$$

然后用：

$$
\hat{A}_{i,t}=R_i-V_\psi(x_t^{(i)},t,c_i)
$$

这样不同时间步可以有不同的 advantage。入门阶段理解 DDPO，用 batch mean baseline 就足够抓住核心。

#### Step 5：计算策略梯度 loss

第五步才是真正更新 Diffusion 模型。

先看最小版 REINFORCE loss。它只做一件事：把“这条轨迹的 log probability”和“这条轨迹好不好”乘在一起。

$$
\mathcal{L}_{\text{pg}}
=
-
\frac{1}{B}
\sum_{i=1}^{B}
\sum_{t=1}^{T}
\log p_\theta(x_{t-1}^{(i)}\mid x_t^{(i)},t,c_i)
\cdot \hat{A}_i
$$

这个公式可以分三层来读：

| 公式部分                                           | 含义                                                |
| -------------------------------------------------- | --------------------------------------------------- |
| $\log p_\theta(x_{t-1}^{(i)}\mid x_t^{(i)},t,c_i)$ | 模型在第 $t$ 步采到这个去噪动作的 log 概率          |
| $\hat{A}_i$                                        | 第 $i$ 张图比平均水平好多少                         |
| 前面的负号                                         | 因为优化器默认最小化 loss，而我们想最大化好轨迹概率 |

如果 $\hat{A}_i>0$，说明这张图比平均好，最小化 loss 会提高这条轨迹里每一步动作的 log probability。如果 $\hat{A}_i<0$，说明这张图比平均差，最小化 loss 会降低这些动作的 log probability。

很多实现还会使用 PPO-style 的重要性比率。这个 ratio 和后面的 clip objective 对应 PPO 论文的核心稳定化设计[^ppo]：

$$
\rho_{i,t}(\theta)
=
\frac{
p_\theta(x_{t-1}^{(i)}\mid x_t^{(i)},t,c_i)
}{
p_{\theta_{\text{old}}}(x_{t-1}^{(i)}\mid x_t^{(i)},t,c_i)
}
=
\exp\left(
\log p_\theta(x_{t-1}^{(i)}\mid x_t^{(i)},t,c_i)
-
\log p_{\theta_{\text{old}}}(x_{t-1}^{(i)}\mid x_t^{(i)},t,c_i)
\right)
$$

它表示：新模型相对旧模型，给同一个去噪动作的概率提高了多少。比如 $\rho=1.2$，表示新模型让这个动作大约更容易发生 20%；$\rho=0.7$，表示新模型让它更不容易发生。实现里常用 logprob 相减再 `exp`，只是因为 logprob 更稳定、也更容易从采样过程里保存下来。

然后可以写出 clipped objective：

$$
\mathcal{L}_{\text{clip}}
=
-
\frac{1}{B}
\sum_{i=1}^{B}
\sum_{t=1}^{T}
\min\left(
\rho_{i,t}\hat{A}_i,
\mathrm{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)\hat{A}_i
\right)
$$

clip 的作用是限制一步更新太猛。假设 $\epsilon=0.2$，那么 ratio 通常会被限制在 $[0.8,1.2]$ 附近。也就是说，就算某张图 reward 很高，也不允许新模型一次性把某个动作的概率放大太多。

这个公式里的 `min` 也可以这样读：当更新方向有利时，只允许它带来有限收益；超过 clip 范围以后，继续放大 ratio 不会让目标函数变得更好。这样模型就不会因为一小批高分样本而突然偏移。放到 Diffusion 里理解，就是不要让一次 reward 更新把去噪分布从原始模型附近推得太远；KL 正则和 ratio clipping 都是在控制这件事[^ppo][^dpok]。

#### Step 6：加 KL 正则并更新参数

最后一步是把策略梯度 loss、KL 正则和其他稳定项合在一起：

$$
\mathcal{L}
=
\mathcal{L}_{\text{clip}}
+
\beta\mathcal{L}_{\text{KL}}
$$

其中：

$$
\mathcal{L}_{\text{KL}}
=
\frac{1}{B}
\sum_{i=1}^{B}
\sum_{t=1}^{T}
\mathrm{KL}\left(
p_\theta(\cdot\mid x_t^{(i)},t,c_i)
\|p_{\text{ref}}(\cdot\mid x_t^{(i)},t,c_i)
\right)
$$

$p_{\text{ref}}$ 通常是 RL 开始前的基础 Diffusion 模型，用来防止模型为了追求 reward model 的偏好而偏离太远。

KL 这一项可以先理解成“两个概率分布的距离”。如果当前模型在某一步给出的去噪分布，和参考模型很接近，KL 就小；如果当前模型为了追 reward，给出了很不一样的分布，KL 就大。$\beta$ 控制这个惩罚有多重：$\beta$ 大，模型更保守；$\beta$ 小，模型更敢追 reward。

到这一步，才执行标准的反向传播：

1. 计算总 loss。
2. `loss.backward()` 得到梯度。
3. 对梯度做 clipping，避免爆炸。
4. `optimizer.step()` 更新 Diffusion 模型。
5. 进入下一批 prompt，重复 rollout 和 update。

把上面六步合起来，可以得到一个更接近真实训练的伪代码。它不是某个仓库的逐行复刻，而是把 DDPO 的 rollout/reward 更新[^ddpo]、PPO 的 clipped objective[^ppo] 和 DPOK 强调的 KL 约束[^dpok] 放在同一个最小训练框架里：

```python
for prompts in prompt_loader:
    # Step 1-2: rollout with the current policy
    with torch.no_grad():
        trajectories = diffusion.sample_trajectories(
            prompts,
            return_states=True,
            return_actions=True,
            return_logprobs=True,
        )
        old_logprobs = trajectories.logprobs
        images = decoder(trajectories.final_latents)

    # Step 3: score final images
    with torch.no_grad():
        rewards = reward_model(prompts, images)

    # Step 4: turn rewards into advantages
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-6)

    # Step 5-6: update the diffusion policy
    for _ in range(update_epochs):
        logprobs = diffusion.logprob(
            states=trajectories.states,
            actions=trajectories.actions,
            prompts=prompts,
        )

        ratio = torch.exp(logprobs - old_logprobs)
        unclipped = ratio * advantages[:, None]
        clipped = ratio.clamp(1 - eps, 1 + eps) * advantages[:, None]
        policy_loss = -torch.minimum(unclipped, clipped).mean()

        kl_loss = diffusion.kl_to(reference_model, trajectories, prompts)
        loss = policy_loss + beta * kl_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(diffusion.parameters(), max_norm)
        optimizer.step()
```

这段代码比前面的数学公式多了一个工程细节：**采样和更新是分开的**。采样时用的是旧模型，所以要保存 `old_logprobs`；更新时用当前模型重新计算 `logprobs`，再通过 ratio 判断新模型相对旧模型改了多少。

如果把 DDPO 压缩成一句工程直觉，就是：

> 对同一批 prompt，让模型自己生成一批样本；把生成结果按 reward 分出好坏；提高好样本去噪轨迹的概率，降低差样本去噪轨迹的概率，同时用 KL 和 clipping 防止模型偏移过猛。

## Reward Model：生成 RL 的真正瓶颈

到这里，算法已经有了。但生成 RL 的困难往往不在“能不能写出策略梯度”，而在“reward 到底可信吗”。

如果 reward model 太弱，它给不出有效方向；如果 reward model 有偏，生成模型就会学到偏差；如果 reward 太复杂，不同目标之间还会互相拉扯。

视觉生成的 reward 通常来自三类信号。

### 人类偏好

人类偏好数据最常见的形式是成对比较。给定同一个 prompt，让用户在两张候选图之间选择更喜欢的一张。Pick-a-Pic 就是公开收集 text-to-image 用户偏好的代表数据集，HPS v2 则进一步提供了面向人类偏好的评测 benchmark 和现成的 reward model[^pickapic][^hpsv2]。

![Pick-a-Pic Preference UI](./images/ref-pick-a-pic-ui.png)

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em>图 2：Pick-a-Pic 的人类偏好采集界面。用户对同一 prompt 下的两张候选图进行偏好选择，这类数据可以训练 PickScore 等 reward model。来源：<a href="https://stability.ai/research/pick-a-pic" target="_blank" rel="noopener noreferrer">Stability AI Research</a>，对应论文 Kirstain et al., 2023</em>
</div>

Pick-a-Pic 的贡献不只是提供界面截图，而是把大规模 text-to-image 成对偏好数据整理成可训练、可评测的公开资源[^pickapic]。

这样的数据可以写成：

$$
\mathcal{D}_{\text{pref}}=\{(c,x^+,x^-)\}
$$

其中 $x^+$ 是用户更喜欢的图，$x^-$ 是被比较后落选的图。reward model 的训练目标通常是让 $x^+$ 的分数高于 $x^-$：

$$
\mathcal{L}_{\text{rm}}
=
-\mathbb{E}_{\mathcal{D}_{\text{pref}}}
\log\sigma\left(r_\phi(c,x^+)-r_\phi(c,x^-)\right)
$$

这就是 Bradley-Terry 风格的偏好建模。它不要求人类给绝对分数，只要求比较两张图哪张更好。Pick-a-Pic 这类数据集正是用这种成对偏好来训练或评估图像偏好模型[^pickapic]。

这种信号的优点是接近真实用户偏好。缺点是成本高，而且偏好数据会继承标注人群的审美、文化和任务分布。

### 文本-图像对齐

文本-图像对齐检查的是：图像是否真的符合 prompt。

这可以从粗到细拆成几层：

| 层次     | 例子                               | 可能的检查方法               |
| -------- | ---------------------------------- | ---------------------------- |
| 全局语义 | 是否大体生成了指定场景             | CLIP Score、VLM 判断         |
| 对象存在 | prompt 中的关键对象是否出现        | 检测器、VLM 问答             |
| 属性匹配 | 颜色、材质、大小是否正确           | 细粒度 caption 后逐项比对    |
| 关系匹配 | 左右、上下、遮挡、交互关系是否正确 | 关系抽取、VLM judge          |
| 数量匹配 | 指定数量是否正确                   | 计数模型、目标检测、VLM 检查 |

这一层和前几节的 VLM RL 直接接上了。一个训练得更会看图的 VLM，可以被拿来当 captioner、judge 或 reward model，帮助生成模型判断“有没有画对”。

### 视觉质量

视觉质量检查的是图像本身是否自然、清晰、有良好的构图和光影。常见信号包括 aesthetic score、无参考图像质量评估、人工排序等。HPS v2 这类 benchmark 试图把“人类更偏好哪种生成结果”做成可复现的评测和模型信号[^hpsv2]。

它很有用，但不能单独使用。因为视觉质量模型通常更容易奖励“看起来高级”的图，而不是“严格遵守 prompt”的图。生成模型如果只追这个分数，可能会变得更漂亮，但更不听话。

### Reward 不是一个越复杂越好的公式

把所有 reward 加权求和很自然：

$$
R_{\text{total}}
=
w_1R_{\text{align}}
+w_2R_{\text{quality}}
+w_3R_{\text{instruction}}
$$

但这个公式只是起点，不是答案。多组件 reward 最大的问题是：每个分量都可能被模型投机利用，而且分量之间可能互相冲突。

一个更稳的工程做法是分层使用 reward：

1. 先用规则或 VLM 检查硬约束，例如数量、颜色、对象是否满足。
2. 再用偏好模型对合格样本排序。
3. 最后用人工抽查或离线 benchmark 找 reward model 的盲区。

这样 reward 就不再是一个万能分数，而是一套筛选和校准流程。

![PickScore Ranking Examples](./images/ref-pickscore-ranking.png)

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em>图 3：PickScore 用偏好模型重新排序候选生成结果。它说明视觉 reward 不只是离线评测数字，也可以直接改变采样或排序阶段展示给用户的结果。来源：<a href="https://stability.ai/research/pick-a-pic" target="_blank" rel="noopener noreferrer">Stability AI Research</a></em>
</div>

## 两种用 reward 的方式：训练时用，还是推理时用？

有了 reward model 以后，不一定马上做 RL 微调。它有两种常见用法。

第一种是**推理时使用**，也叫 reward-guided sampling 或 reranking。比如同一个 prompt 生成 $N$ 张图，用 reward model 排序，选择分数最高的图。这个方法简单、安全，适合先验证 reward model 是否靠谱。

第二种是**训练时使用**，也就是 DDPO、DPOK 这类 RL fine-tuning[^ddpo][^dpok]。模型不只是被筛选，而是真的更新参数，把偏好内化到生成策略里。

| 方法                    | 它做什么                         | 优点                  | 缺点                     |
| ----------------------- | -------------------------------- | --------------------- | ------------------------ |
| Best-of-$N$ / reranking | 多生成几张，再用 reward model 选 | 实现简单，不改模型    | 推理成本高，能力不固化   |
| Reward-guided sampling  | 采样过程中用 reward 引导方向     | 比纯 reranking 更主动 | 每次生成仍然要额外评估   |
| RL fine-tuning          | 用 reward 更新模型参数           | 可以内化偏好          | 训练更贵，也更容易不稳定 |

实践中，经常先做 reranking。如果 reward model 连排序都排不好，就不应该直接拿它做 RL。

## 同一个问题，多了一条时间轴

视频生成可以看成图像生成的扩展，但不能简单理解为“多生成几张图”。视频多了一条时间轴，所以 reward 也要多一层。Emu Video 这类工作把图像条件和视频生成拆开建模[^emu]；后续视频对齐工作则开始探索用 reward gradient 或 MLLM feedback 来优化视频生成结果[^vader][^t2vfeedback]。

一段视频要同时满足三件事：

1. 每一帧都要清晰、自然、符合 prompt。
2. 相邻帧之间要连贯，主体不能突然变化。
3. 整段视频要表达 prompt 中的事件顺序。

因此，视频 reward 常常写成下面这种分层形式。它不是某篇论文的固定公式，而是把单帧质量、时序一致性和整体事件对齐这三类常见评估信号抽象到同一个 reward 里：

$$
R_{\text{video}}
=
\alpha \cdot \frac{1}{T}\sum_t R_{\text{frame}}(x_t,c)
+ \beta \cdot \frac{1}{T-1}\sum_t R_{\text{temporal}}(x_t,x_{t+1})
+ \gamma \cdot R_{\text{overall}}(\{x_t\}_{t=1}^T,c)
$$

这三个分量分别对应：

| 分量                  | 检查什么                       |
| --------------------- | ------------------------------ |
| $R_{\text{frame}}$    | 单帧质量和单帧文本对齐         |
| $R_{\text{temporal}}$ | 帧间一致性和运动自然性         |
| $R_{\text{overall}}$  | 整段视频是否完成 prompt 的事件 |

视频 RL 的困难也随之增加：

| 挑战          | 为什么更难                      | 常见缓解思路                          |
| ------------- | ------------------------------- | ------------------------------------- |
| 时序一致性    | 单帧都好，不代表连起来合理      | 光流一致性、轨迹一致性、视频 VLM 评估 |
| 长 horizon    | 视频 token 和 latent 数远超图像 | 分段优化、短片段 reward shaping       |
| 计算成本      | 每次采样和打分都更贵            | latent 空间训练、低帧率评估、候选重排 |
| 文本-视频对齐 | prompt 可能包含先后顺序         | 分段 caption、事件级 reward           |

直觉上，图像生成的错误常常是“某个地方画错了”；视频生成的错误常常是“前后关系断了”。这就是为什么视频 reward 更依赖片段级和整体级评估。

## On-Policy 蒸馏：把 RL 得到的能力固化下来

RL 微调后的模型可能更符合偏好，但它也可能更慢、更贵，或者只适合某个特定采样设置。On-policy 蒸馏的目标，是把 RL 后模型在当前分布上产生的高质量样本，重新变成更便宜的监督学习信号。

可以把它理解成三步：

1. 用 RL 后的 teacher 模型在线生成样本。
2. 用 reward model 或规则过滤留下高质量样本。
3. 让 student 模型学习这些样本，用更低成本复现 teacher 的行为。

这和前文的蒸馏思想一致：强模型负责探索和筛选，弱模型负责把能力压缩成更便宜的推理路径。区别在于，视觉生成蒸馏通常发生在 latent、去噪轨迹或视频 token 空间里，而不是普通文本 token 空间里。

## 与前面章节的联系

视觉生成 RL 看起来和 VLM 问答离得很远，但它复用了本书前面几条主线。

| 前面章节                | 在视觉生成 RL 中的对应                                      |
| ----------------------- | ----------------------------------------------------------- |
| 第 6 章 REINFORCE       | DDPO 把去噪链路当成策略轨迹，用终局 reward 更新每一步采样   |
| 第 25 章 Reward Hacking | 生成模型可能讨好 reward model，却牺牲真实用户意图           |
| 第 15 章 RLVR           | 细粒度属性、数量、关系可以变成局部可验证信号                |
| 第 19 章 Agentic RL     | 长 horizon 信用分配、多组件 reward 和 KL 约束都再次出现     |
| 第 23.1-23.4 节 VLM RL  | VLM 可以反过来做生成模型的 judge、captioner 和 reward model |

最后这一点尤其重要。理解模型和生成模型彼此连通。VLM 学会看图以后，可以检查生成图是否符合 prompt；生成模型可以合成更丰富的数据，反过来训练 VLM。到了多模态后训练阶段，“看”和“生成”会相互促进。

## 小结

视觉生成 RL 的目标不是简单地让模型“画得更漂亮”，而是把用户意图拆成可学习的反馈信号，让生成模型在偏好、规则和多模态评估下持续改进。

本节最重要的结论有四个：

1. **Diffusion 可以被看成 MDP**：去噪轨迹就是 episode，最终图像得到 reward，策略梯度把 reward 分配回每一步。
2. **DDPO 的核心是翻译问题**：把去噪概率看成 policy，把最终图像分数看成 reward，就能使用策略梯度。
3. **Reward model 是生成 RL 的瓶颈**：人类偏好、文本对齐和视觉质量都重要，但必须防止 reward hacking。
4. **训练和推理都可以用 reward**：reranking 更安全，RL fine-tuning 更能固化能力，视频生成则进一步放大了时序和计算问题。

下一节 [24.5 视频 RLHF 与物理感知生成](./video-generation-modern) 把这套方法扩展到长视频，重点讨论时序一致性、多奖励组合和工业训练系统。

## 参考资料

[^reinforce]: Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. _Machine Learning_. <https://doi.org/10.1007/BF00992696>

[^ppo]: Schulman, J. et al. (2017). Proximal Policy Optimization Algorithms. <https://arxiv.org/abs/1707.06347>

[^ddpo]: Black, K., Janner, M., Du, Y., et al. (2024). Training Diffusion Models with Reinforcement Learning. _ICLR_. <https://arxiv.org/abs/2305.13301>

[^dpok]: Fan, Y., Watkins, O., Du, Y., et al. (2023). DPOK: Reinforcement Learning for Fine-tuning Text-to-Image Diffusion Models. _NeurIPS_. <https://arxiv.org/abs/2305.16381>

[^draft]: Clark, K. et al. (2024). Directly Fine-Tuning Diffusion Models on Differentiable Rewards. _ICLR_. <https://arxiv.org/abs/2309.17400>

[^vader]: Prabhudesai, M. et al. (2024). Video Diffusion Alignment via Reward Gradients. <https://arxiv.org/abs/2407.08737>

[^pickapic]: Kirstain, S. et al. (2023). Pick-a-Pic: Open Dataset of Human Preferences for Text-to-Image Generation. _NeurIPS_. <https://arxiv.org/abs/2305.01569>

[^hpsv2]: Wu, X. et al. (2023). Human Preference Score v2: A Benchmark for Evaluating Human Preferences of Text-to-Image Synthesis. _NeurIPS_. <https://arxiv.org/abs/2306.09341>

[^emu]: Girdhar, R. et al. (2024). Emu Video: Factorizing Text-to-Video Generation by Explicit Image Conditioning. _ECCV_. <https://arxiv.org/abs/2311.10709>

[^t2vfeedback]: Wu, X. et al. (2024). Boosting Text-to-Video Generative Model with MLLMs Feedback. _NeurIPS_. <https://neurips.cc/virtual/2024/poster/96722>
