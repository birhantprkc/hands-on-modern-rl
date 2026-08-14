# 9.1 Deterministic Policy Gradients and DDPG

> [Chapter 8: PPO](../chapter10_ppo/ppo-clip-objective) addressed policy learning in continuous action spaces by using a Gaussian policy to output continuous actions and clipping updates to maintain stability. PPO, however, is on-policy: after every policy update, it must collect new samples, resulting in **extremely low sample efficiency**. This chapter addresses two questions: (1) How can continuous actions be handled off-policy with DDPG, TD3, and SAC? (2) How can an environment model further improve sample efficiency through model-based RL, AlphaZero, and Dreamer?

## Deterministic Policy Gradients and DDPG

Problems such as CartPole and Atari have discrete actions—left or right, or movement in the four cardinal directions—which Q-Learning or a softmax policy can handle directly. In robot control, autonomous driving, and robotic manipulation, however, actions are **continuous**: joint angles $\theta \in \mathbb{R}^n$, throttle values in $[0, 1]$, or steering angles in $[-\pi, \pi]$.

Continuous actions introduce two challenges:

1. **The Q-function cannot be maximized by enumeration**: in the discrete case, $a^* = \arg\max_a Q(s, a)$ can be computed by checking every action; continuous actions cannot be enumerated
2. **The policy output changes form**: instead of softmax probabilities, the policy outputs distribution parameters such as a mean and variance

### The Continuous Version of the Policy-Gradient Theorem

[Chapter 6: Policy Gradients](../chapter08_policy_gradient/reinforce) gave

$$\nabla_\theta J(\theta) = \mathbb{E}_{s \sim d^\pi, a \sim \pi_\theta}\left[\nabla_\theta \log \pi_\theta(a\mid s) \cdot Q^\pi(s, a)\right]$$

This expression requires the policy $\pi_\theta(a \mid s)$ to be **stochastic**, that is, a probability distribution. Silver et al. (2014), however, proved that a similar gradient theorem holds when the policy is **deterministic**, with $a = \mu_\theta(s)$:

$$\nabla_\theta J(\theta) = \mathbb{E}_{s \sim d^\mu}\left[\nabla_\theta \mu_\theta(s) \cdot \nabla_a Q^\mu(s, a)\big|_{a=\mu_\theta(s)}\right]$$

This is the **Deterministic Policy Gradient (DPG)** theorem. It is more sample-efficient than the stochastic version:

- **No integration over $a$ is required**: stochastic policy gradients take an expectation over every possible action, while deterministic policy gradients require an expectation only over states
- **It is compatible with off-policy learning**: a deterministic policy can be trained with data collected by any behavior policy

A deterministic policy does not explore on its own. If $\mu_\theta(s)$ always returns the same $a$, the agent will never try other actions. DDPG solves this problem by **adding noise to actions during training**.

### Deep Deterministic Policy Gradient

Deep Deterministic Policy Gradient (Lillicrap et al., 2015) combines DPG with the deep-network techniques used in DQN:

- **Actor**: $\mu_\theta(s)$ outputs a continuous action directly through neural-network regression
- **Critic**: $Q_\phi(s, a)$ evaluates the action value
- **Target networks**: stabilize training, inherited from DQN
- **Experience replay**: reuses data off-policy, inherited from DQN

### Main Algorithm Loop

```python
class DDPG:
    def __init__(self, state_dim, action_dim, action_max):
        # Online networks
        self.actor = Actor(state_dim, action_dim, action_max)
        self.critic = Critic(state_dim, action_dim)
        # Target networks (soft updates)
        self.actor_target = copy(self.actor)
        self.critic_target = copy(self.critic)
        self.replay_buffer = ReplayBuffer(capacity=1_000_000)
        self.gamma = 0.99
        self.tau = 0.005  # Soft-update coefficient

    def select_action(self, state, explore=True):
        with torch.no_grad():
            action = self.actor(state)
        if explore:
            # Exploration with Ornstein-Uhlenbeck or Gaussian noise
            action += np.random.normal(0, 0.1, size=action.shape)
        return np.clip(action, -self.action_max, self.action_max)

    def update(self, batch_size=256):
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(batch_size)

        # === Critic update ===
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + self.gamma * (1 - dones) * target_q
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)
        self.critic_optim.zero_grad(); critic_loss.backward()
        self.critic_optim.step()

        # === Actor update: maximize Q(s, μ(s)) ===
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optim.zero_grad(); actor_loss.backward()
        self.actor_optim.step()

        # === Soft-update the target networks ===
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)
```

In MuJoCo physics environments such as HalfCheetah, Hopper, and Walker2d, DDPG was the first deep RL method to outperform methods such as TRPO and CES that used linear features. DDPG nevertheless has several widely criticized weaknesses:

- **Q-value overestimation**: the target Q-value uses a maximum and is easily inflated by noise
- **Hyperparameter sensitivity**: small changes to the learning rate, noise scale, or network architecture can cause divergence
- **Training instability**: when the critic learns an inaccurate value function, the actor follows it, producing a positive feedback loop

## Section Summary

The Deterministic Policy Gradient (DPG) theorem extends policy gradients from stochastic to deterministic policies, allowing continuous action spaces to be trained off-policy. DDPG combines DPG with the deep-network techniques of DQN and was the first deep RL method to outperform classical methods on MuJoCo.

DDPG nevertheless suffers from Q-value overestimation, hyperparameter sensitivity, and training instability. The next section, [9.2 TD3 and SAC](./td3-sac), presents two complementary remedies: TD3 stabilizes DDPG with three engineering techniques, while SAC reformulates the objective through maximum-entropy RL.
