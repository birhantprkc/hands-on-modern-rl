"""
第1章：撕开黑盒 —— 用纯 PyTorch 实现 PPO 训练 CartPole
展示 SB3 的 model.learn() 背后的核心逻辑

训练过程通过 SwanLab 记录指标（奖励曲线、损失等），
训练结束后可选弹出 GUI 窗口展示学习成果。

运行方式：
    # 默认：训练 + SwanLab 曲线（不开 GUI，速度快）
    python 2-pytorch_ppo.py

    # 打开 GUI 演示（训练完弹出小车动画窗口）
    python 2-pytorch_ppo.py --gui

关于 --gui 参数：
    训练阶段始终是 headless（无渲染），速度不受 GUI 影响。
    --gui 只控制训练结束后的演示环节是否弹出 CartPole 动画窗口。
    开启 GUI 时，演示环节每帧需要等待屏幕刷新（~16ms），会明显变慢；
    关闭 GUI 时，演示环节纯计算，几秒内跑完。
"""

import argparse
import csv
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import swanlab


# ==========================================
# 第一部分：Actor-Critic 网络（独立头 + 正交初始化）
# ==========================================
class ActorCritic(nn.Module):
    """
    独立 Actor-Critic 网络（与 SB3 MlpPolicy 对齐）：
    - Actor 和 Critic 使用各自的隐藏层，避免梯度冲突
    - 正交初始化：actor 输出层 gain=0.01 保证初始策略接近均匀分布
    """

    def __init__(self, obs_dim=4, act_dim=2, hidden=64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self._init_weights()

    def _init_weights(self):
        """正交初始化，与 SB3 默认一致"""
        for module in self.actor:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
        for module in self.critic:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
        # actor 输出层用小 gain → 初始策略接近均匀
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.constant_(self.actor[-1].bias, 0)
        # critic 输出层 gain=1
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
        nn.init.constant_(self.critic[-1].bias, 0)

    def forward(self, x):
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value.squeeze(-1)

    def get_action(self, obs, deterministic=False):
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value


# ==========================================
# 第二部分：收集轨迹（Rollout）
# ==========================================
def collect_rollout(
    model,
    env,
    obs,
    episode_reward=0.0,
    episode_length=0,
    num_steps=2048,
):
    """
    收集轨迹，正确处理 terminated vs truncated：
    - terminated（杆子倒了）：V(s')=0
    - truncated（达到步数上限）：V(s')需要 bootstrap
    - rollout 末尾未结束：也用 V(s') bootstrap
    """
    transitions = []
    completed_rewards = []
    completed_lengths = []

    for _ in range(num_steps):
        obs_tensor = torch.FloatTensor(obs)
        with torch.no_grad():
            action, log_prob, value = model.get_action(obs_tensor)

        next_obs, reward, terminated, truncated, _ = env.step(action.item())
        with torch.no_grad():
            if terminated:
                next_value = 0.0
            else:
                _, next_value_tensor = model(torch.FloatTensor(next_obs))
                next_value = next_value_tensor.item()

        # 保存该步的 V(s')，使终止、截断和 rollout 末尾共用同一条 GAE 公式。
        transitions.append({
            "obs": obs,
            "action": action.item(),
            "log_prob": log_prob.item(),
            "value": value.item(),
            "reward": float(reward),
            "terminated": terminated,
            "truncated": truncated,
            "next_value": next_value,
        })

        episode_reward += float(reward)
        episode_length += 1

        obs = next_obs
        if terminated or truncated:
            completed_rewards.append(episode_reward)
            completed_lengths.append(episode_length)
            episode_reward = 0.0
            episode_length = 0
            obs, _ = env.reset()

    return (
        transitions,
        obs,
        completed_rewards,
        completed_lengths,
        episode_reward,
        episode_length,
    )


# ==========================================
# 第三部分：计算 GAE 优势
# ==========================================
def compute_gae(transitions, gamma=0.99, lam=0.95):
    """
    广义优势估计，正确处理：
    - terminated（真正结束）：不传播 GAE，V(s')=0
    - truncated（时间截断）：用 V(s') bootstrap，但不跨越 reset 传播 GAE
    - rollout 末尾：用已保存的 V(s') bootstrap
    """
    raw_advantages = []
    gae = 0

    for step in reversed(range(len(transitions))):
        t = transitions[step]
        episode_end = t["terminated"] or t["truncated"]
        delta = t["reward"] + gamma * t["next_value"] - t["value"]
        gae = delta + gamma * lam * (1.0 - float(episode_end)) * gae
        raw_advantages.insert(0, gae)

    raw_advantages = torch.tensor(raw_advantages, dtype=torch.float32)
    values = torch.tensor([t["value"] for t in transitions], dtype=torch.float32)
    # Critic 学习未归一化的回报目标；归一化只用于策略损失。
    returns = raw_advantages + values
    advantages = (raw_advantages - raw_advantages.mean()) / (
        raw_advantages.std(unbiased=False) + 1e-8
    )

    return advantages, returns


# ==========================================
# 第四部分：PPO 更新
# ==========================================
def ppo_update(model, optimizer, transitions, advantages, returns,
               clip_eps=0.2, epochs=10, batch_size=64):
    """PPO 裁剪目标函数更新"""
    obs = np.array([t["obs"] for t in transitions])
    actions = np.array([t["action"] for t in transitions])
    old_log_probs = np.array([t["log_prob"] for t in transitions])

    obs = torch.FloatTensor(obs)
    actions = torch.LongTensor(actions)
    old_log_probs = torch.FloatTensor(old_log_probs)

    total_policy_loss = 0
    total_value_loss = 0
    total_entropy = 0
    total_kl = 0
    total_clip_frac = 0
    n_updates = 0

    for _ in range(epochs):
        indices = np.random.permutation(len(transitions))

        for start in range(0, len(transitions), batch_size):
            idx = indices[start:start + batch_size]

            batch_obs = obs[idx]
            batch_actions = actions[idx]
            batch_old_log_probs = old_log_probs[idx]
            batch_advantages = advantages[idx]
            batch_returns = returns[idx]

            logits, values = model(batch_obs)
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(batch_actions)

            # PPO 裁剪目标
            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # 价值函数损失
            value_loss = ((values - batch_returns) ** 2).mean()

            # 熵奖励（鼓励探索）
            entropy = dist.entropy().mean()

            loss = policy_loss + 0.5 * value_loss - 0.0 * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            # 统计指标
            with torch.no_grad():
                log_ratio = new_log_probs - batch_old_log_probs
                # 非负 KL 近似，与 SB3 的 approx_kl 计算一致。
                total_kl += ((log_ratio.exp() - 1) - log_ratio).mean().item()
                total_clip_frac += ((ratio - 1.0).abs() > clip_eps).float().mean().item()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            n_updates += 1

    return {
        "policy_loss": total_policy_loss / n_updates,
        "value_loss": total_value_loss / n_updates,
        "entropy": total_entropy / n_updates,
        "approx_kl": total_kl / n_updates,
        "clip_fraction": total_clip_frac / n_updates,
    }


# ==========================================
# 第五部分：训练循环
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="纯 PyTorch PPO CartPole 训练")
    parser.add_argument(
        "--gui", action="store_true",
        help="训练结束后弹出 GUI 窗口演示智能体（默认关闭，仅输出得分）",
    )
    parser.add_argument("--seed", type=int, default=42, help="训练随机种子")
    parser.add_argument("--iterations", type=int, default=40, help="PPO 迭代轮数")
    parser.add_argument("--steps-per-rollout", type=int, default=2048, help="每轮采样步数")
    parser.add_argument(
        "--log-csv", default="output/training_metrics.csv",
        help="原始训练指标 CSV 的保存位置",
    )
    parser.add_argument(
        "--swanlab-mode",
        choices=["local", "cloud", "disabled"],
        default="local",
        help="SwanLab 记录模式；复现实验但不需要看板时可设为 disabled",
    )
    return parser.parse_args()


def train():
    args = parse_args()
    model_path = os.path.join(
        os.path.dirname(os.path.abspath(args.log_csv)),
        "pytorch_ppo_cartpole.pth",
    )
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    env = gym.make("CartPole-v1")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.action_space.seed(args.seed)
    obs, _ = env.reset(seed=args.seed)

    # 打印环境信息（状态空间、动作空间、边界阈值）
    print("=" * 50)
    print("CartPole-v1 环境信息")
    print("=" * 50)
    print(f"  观测空间:  {env.observation_space}")
    print(f"  动作空间:  {env.action_space}")
    print(f"  观测上限:  {env.observation_space.high}")
    print(f"  观测下限:  {env.observation_space.low}")
    print(f"  终止条件:  位置 > ±{env.unwrapped.x_threshold}, "
          f"角度 > ±{env.unwrapped.theta_threshold_radians:.4f} rad "
          f"(≈ ±{np.degrees(env.unwrapped.theta_threshold_radians):.0f}°)")
    print("=" * 50)

    model = ActorCritic()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    total_iterations = args.iterations
    steps_per_rollout = args.steps_per_rollout

    # 初始化 SwanLab
    swanlab.init(
        project="cartpole-pytorch",
        experiment_name="PPO-PyTorch-CartPole-v1",
        mode=args.swanlab_mode,
        config={
            "algorithm": "PPO",
            "lr": 3e-4,
            "total_iterations": total_iterations,
            "steps_per_rollout": steps_per_rollout,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_eps": 0.2,
            "epochs": 10,
            "batch_size": 64,
            "seed": args.seed,
        },
    )

    print("开始训练（纯 PyTorch PPO + SwanLab）...")
    print("-" * 60)

    total_timesteps = 0

    csv_dir = os.path.dirname(args.log_csv)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
    metric_rows = []
    ongoing_episode_reward = 0.0
    ongoing_episode_length = 0

    for iteration in range(total_iterations):
        # 收集数据
        (
            transitions,
            obs,
            ep_rewards,
            ep_lengths,
            ongoing_episode_reward,
            ongoing_episode_length,
        ) = collect_rollout(
            model,
            env,
            obs,
            ongoing_episode_reward,
            ongoing_episode_length,
            steps_per_rollout,
        )

        total_timesteps += len(transitions)

        # 计算优势和 Critic 的未归一化回报目标
        advantages, returns = compute_gae(transitions)

        # 在本轮更新前设置学习率。
        frac = 1.0 - iteration / total_iterations
        lr = 3e-4 * frac
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # PPO 更新
        metrics = ppo_update(
            model, optimizer, transitions, advantages, returns
        )

        # 解释方差要对比收集 rollout 时的价值预测与回报目标。
        return_values = returns.numpy()
        rollout_values = np.array([t["value"] for t in transitions])
        var_returns = np.var(return_values)
        if var_returns < 1e-6:
            # 所有回报相同（如全部 500 分），EV 无意义，置为 0
            explained_variance = 0.0
        else:
            explained_variance = 1 - np.var(return_values - rollout_values) / var_returns

        mean_reward = np.mean(ep_rewards) if ep_rewards else 0
        mean_ep_len = np.mean(ep_lengths) if ep_lengths else 0

        # 记录到 SwanLab（与 SB3 指标对齐）
        swanlab.log({
            "rollout/ep_rew_mean": mean_reward,
            "rollout/ep_len_mean": mean_ep_len,
            "train/policy_gradient_loss": metrics["policy_loss"],
            "train/value_loss": metrics["value_loss"],
            "train/entropy_loss": -metrics["entropy"],
            "train/approx_kl": metrics["approx_kl"],
            "train/clip_fraction": metrics["clip_fraction"],
            "train/clip_range": 0.2,
            "train/explained_variance": explained_variance,
            "train/learning_rate": lr,
            "train/n_updates": (iteration + 1) * 10 * (steps_per_rollout // 64),
            "time/total_timesteps": total_timesteps,
            "time/iterations": iteration + 1,
        }, step=iteration)

        metric_rows.append({
            "seed": args.seed,
            "iteration": iteration + 1,
            "total_timesteps": total_timesteps,
            "completed_episodes": len(ep_rewards),
            "mean_episode_reward": mean_reward,
            "mean_episode_length": mean_ep_len,
            "policy_loss": metrics["policy_loss"],
            "value_loss": metrics["value_loss"],
            "entropy": metrics["entropy"],
            "approx_kl": metrics["approx_kl"],
            "clip_fraction": metrics["clip_fraction"],
            "explained_variance": explained_variance,
            "learning_rate": lr,
        })

        print(
            f"  迭代 {iteration + 1:2d}/{total_iterations} | "
            f"回合数: {len(ep_rewards):3d} | "
            f"平均奖励: {mean_reward:6.1f} | "
            f"KL: {metrics['approx_kl']:.4f} | "
            f"clip%: {metrics['clip_fraction']:.1%}"
        )

    print("-" * 60)

    fieldnames = list(metric_rows[0].keys())
    temporary_csv = f"{args.log_csv}.tmp"
    with open(temporary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metric_rows)
    os.replace(temporary_csv, args.log_csv)
    print(f"原始训练指标已保存到 {args.log_csv}")

    # 最终评估
    eval_rewards = []
    for _ in range(20):
        obs, _ = env.reset(seed=args.seed + 10_000 + len(eval_rewards))
        done, truncated, score = False, False, 0
        while not (done or truncated):
            obs_tensor = torch.FloatTensor(obs)
            with torch.no_grad():
                action, _, _ = model.get_action(obs_tensor, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action.item())
            score += reward
        eval_rewards.append(score)

    mean_reward = np.mean(eval_rewards)
    std_reward = np.std(eval_rewards)
    print(f"\n训练完成！20 回合评估: {mean_reward:.1f} +/- {std_reward:.1f}")

    swanlab.log({
        "eval/mean_reward": mean_reward,
        "eval/std_reward": std_reward,
    })

    # 保存模型
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存到 {model_path}")

    # GUI 演示
    if args.gui:
        try:
            vis_env = gym.make("CartPole-v1", render_mode="human")
            print("\n正在演示学习成果（5 个回合）...")
            for ep in range(5):
                obs, _ = vis_env.reset(seed=args.seed + 20_000 + ep)
                done, truncated, score = False, False, 0
                while not (done or truncated):
                    obs_tensor = torch.FloatTensor(obs)
                    with torch.no_grad():
                        action, _, _ = model.get_action(obs_tensor, deterministic=True)
                    obs, reward, done, truncated, _ = vis_env.step(action.item())
                    score += reward
                print(f"  演示回合 {ep + 1} 得分: {score}")
            vis_env.close()
            print("\nGUI 演示结束。")
        except Exception:
            print("(跳过 GUI 演示，无图形界面)")
    else:
        print("\n提示: 加 --gui 可弹出小车动画窗口查看演示效果。")

    env.close()
    swanlab.finish()

    print("SwanLab 实验看板: swanlab watch swanlog")


if __name__ == "__main__":
    train()
