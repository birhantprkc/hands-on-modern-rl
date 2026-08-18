from __future__ import annotations

import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


ROOT = Path(__file__).resolve().parent

# Pygame's RGB-array renderer does not need a display server.  Explicit headless
# settings keep SDL from probing XDG/Wayland on the ModelScope CPU container.
RUNTIME_DIR = Path("/tmp/hands-on-modern-rl-runtime")
RUNTIME_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
RUNTIME_DIR.chmod(0o700)
os.environ.setdefault("XDG_RUNTIME_DIR", str(RUNTIME_DIR))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

SPACE = {
    "title": {"en": "Multi-Agent CPU Game Arena", "zh": "多智能体 CPU 游戏训练场"},
    "description": {
        "en": "Train shared policies in cooperative and competitive PettingZoo games, inspect per-checkpoint team rewards, and replay every agent together.",
        "zh": "在 PettingZoo 合作与竞争游戏中训练共享策略，观察团队奖励，并在同一回放中展示所有智能体。",
    },
    "badge": "EXPERIMENT 05 · MULTI-AGENT",
    "training_guide": {
        "success": {"en": "Team return or win rate should improve over early checkpoints, and the replay should show coordinated behavior rather than one agent acting alone.", "zh": "团队回报或胜率应高于早期检查点，回放中还应出现多智能体协作，而不是只有单个智能体行动。"},
        "preview": {"en": "The final Preview replays all agents from this run together. Watch spacing, collision avoidance, pursuit, and role coordination for the selected task.", "zh": "最终 Preview 会同时回放本次运行中的所有智能体，请根据任务观察站位、避碰、追逐和角色协作。"},
        "time": {"en": "Default CPU recipes usually take 30 seconds–3 minutes; pixel-based multi-agent tasks are slower than vector tasks.", "zh": "默认 CPU 配方通常需要 30 秒到 3 分钟；像素观测的多智能体任务会比向量观测任务更慢。"},
    },
    "course_url": "https://walkinglabs.github.io/hands-on-modern-rl/chapter14_exploration_marl_hierarchical/marl",
    "source_url": "https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment05-multiagent-games/file/view/master/space_runtime.py",
    "notebook_url": "https://modelscope.cn/notebook/share/github/walkinglabs/hands-on-modern-rl/blob/main/"
    "code/online-experiments/hands-on-modern-rl-experiment05-multiagent-games.ipynb",
}


def task(key, title, environment, description, observation, action, preview, policy, budget):
    return {
        "key": key,
        "title": {"en": title, "zh": title},
        "environment": environment,
        "description": description,
        "observation": observation,
        "action": action,
        "algorithm": "Parameter-sharing PPO",
        "policy": policy,
        "preview": preview,
        "budget": budget,
        "learning_rate": (1e-5, 0.003, 0.0003, 1e-5),
        "gamma": (0.8, 1.0, 0.99, 0.005),
        "epsilon": (0.0, 0.2, 0.01, 0.005),
        "checkpoints": 6,
    }


TASKS = [
    task("simple-spread", "Cooperative Navigation", "MPE/simple_spread_v3", {"en": "Three agents coordinate to cover landmarks while avoiding collisions.", "zh": "三个智能体协调覆盖地标，同时避免彼此碰撞。"}, {"en": "Local positions and velocities", "zh": "局部位置和速度"}, {"en": "Five discrete movement actions", "zh": "五个离散移动动作"}, "assets/simple-spread.png", "MlpPolicy", (2_000, 200_000, 20_000, 2_000)),
    task("simple-tag", "Predator & Prey", "MPE/simple_tag_v3", {"en": "Predators cooperate to tag a faster adversary around fixed obstacles.", "zh": "多个捕食者协作，在障碍物周围追捕速度更快的对手。"}, {"en": "Relative positions and velocities", "zh": "相对位置和速度"}, {"en": "Five discrete movement actions", "zh": "五个离散移动动作"}, "assets/simple-tag.png", "MlpPolicy", (2_000, 200_000, 20_000, 2_000)),
    task("pistonball", "Pistonball", "Butterfly/pistonball_v6", {"en": "A line of pistons learns coordinated timing to push one ball toward the goal.", "zh": "一排活塞学习协同时机，把球推向目标。"}, {"en": "Local RGB crop", "zh": "局部 RGB 画面"}, {"en": "Move piston up or down", "zh": "控制活塞上下移动"}, "assets/pistonball.png", "CnnPolicy", (2_000, 100_000, 10_000, 2_000)),
    task("cooperative-pong", "Cooperative Pong", "Butterfly/cooperative_pong_v6", {"en": "Two paddles cooperate to keep the ball in play for as long as possible.", "zh": "两个球拍协作，让球尽可能长时间保持运动。"}, {"en": "RGB game frame", "zh": "RGB 游戏画面"}, {"en": "Move paddle up or down", "zh": "控制球拍上下移动"}, "assets/cooperative-pong.png", "CnnPolicy", (2_000, 100_000, 10_000, 2_000)),
]


def runtime_status():
    try:
        import pettingzoo
        from mpe2 import simple_spread_v3

        env = simple_spread_v3.parallel_env(render_mode="rgb_array", max_cycles=10, continuous_actions=False)
        env.reset(seed=0)
        env.close()
        return f"PettingZoo {pettingzoo.__version__} · READY"
    except Exception as exc:
        return f"startup check pending · {type(exc).__name__}"


def _raw_env(key: str, max_cycles: int, render_mode="rgb_array"):
    if key == "simple-spread":
        from mpe2 import simple_spread_v3
        return simple_spread_v3.parallel_env(render_mode=render_mode, max_cycles=max_cycles, continuous_actions=False, N=3)
    if key == "simple-tag":
        from mpe2 import simple_tag_v3
        return simple_tag_v3.parallel_env(render_mode=render_mode, max_cycles=max_cycles, continuous_actions=False, num_good=1, num_adversaries=3, num_obstacles=2)
    if key == "pistonball":
        from pettingzoo.butterfly import pistonball_v6
        return pistonball_v6.parallel_env(render_mode=render_mode, n_pistons=10, continuous=False, max_cycles=max_cycles)
    if key == "cooperative-pong":
        from pettingzoo.butterfly import cooperative_pong_v6
        return cooperative_pong_v6.parallel_env(render_mode=render_mode, max_cycles=max_cycles)
    raise KeyError(key)


def _wrapped_env(task, max_cycles: int, vector: bool):
    import supersuit as ss

    env = _raw_env(task["key"], max_cycles=max_cycles)
    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)
    if task["policy"] == "CnnPolicy":
        env = ss.color_reduction_v0(env, mode="B")
        env = ss.resize_v1(env, x_size=84, y_size=84)
        env = ss.frame_stack_v2(env, stack_size=4)
    if vector:
        env = ss.pettingzoo_env_to_vec_env_v1(env)
        env = ss.concat_vec_envs_v1(env, 1, num_cpus=0, base_class="stable_baselines3")
        if hasattr(env, "venv") and not hasattr(env.venv, "seed"):
            env.venv.seed = lambda seed=None: [seed] * env.num_envs
    return env


def _rollout(model, task, seed: int, render: bool):
    env = _wrapped_env(task, max_cycles=300 if render else 160, vector=False)
    observations, _ = env.reset(seed=seed)
    totals = {agent: 0.0 for agent in env.possible_agents}
    frames: list[np.ndarray] = []
    steps = 0
    while env.agents:
        if render:
            frame = env.render()
            if frame is not None and (steps % 2 == 0 or steps < 10):
                frames.append(np.asarray(frame))
        actions = {}
        for agent in env.agents:
            action, _ = model.predict(observations[agent], deterministic=True)
            actions[agent] = int(np.asarray(action).reshape(-1)[0])
        observations, rewards, terminations, truncations, _ = env.step(actions)
        for agent, reward in rewards.items():
            totals[agent] = totals.get(agent, 0.0) + float(reward)
        steps += 1
        if all(terminations.get(agent, False) or truncations.get(agent, False) for agent in set(terminations) | set(truncations)):
            break
    env.close()
    return float(np.mean(list(totals.values()))) if totals else 0.0, frames


def run(key: str, budget: int, learning_rate: float, gamma: float, epsilon: float, seed: int):
    from stable_baselines3 import PPO

    task = next(item for item in TASKS if item["key"] == key)
    train_env = _wrapped_env(task, max_cycles=160, vector=True)
    model = PPO(
        task["policy"], train_env, learning_rate=learning_rate, gamma=gamma,
        ent_coef=epsilon, n_steps=128, batch_size=64, n_epochs=4,
        seed=seed, verbose=0, device="cpu",
    )
    checkpoints = int(task["checkpoints"])
    chunk = max(1, budget // checkpoints)
    x: list[float] = []
    y: list[float] = []
    completed = 0
    yield {"step": 0, "x": x, "y": y, "log": f"Initialized shared PPO policy for {train_env.num_envs} vector-agent slots"}
    try:
        while completed < budget:
            current = min(chunk, budget - completed)
            model.learn(total_timesteps=current, reset_num_timesteps=False, progress_bar=False)
            completed += current
            score, _ = _rollout(model, task, seed + completed, render=False)
            x.append(float(completed)); y.append(score)
            yield {"step": completed, "score": score, "x": x, "y": y, "metric_detail": "mean reward per agent", "log": f"PPO update step={completed:,} mean_agent_reward={score:.4f}"}
        score, frames = _rollout(model, task, seed + 10_000, render=True)
        artifacts = ROOT / "artifacts"
        artifacts.mkdir(parents=True, exist_ok=True)
        model.save(str(artifacts / f"{key}-model"))
        preview = artifacts / f"{key}-learned-policy.gif"
        if not frames:
            raise RuntimeError("PettingZoo returned no RGB frames for the learned-policy replay")
        imageio.mimsave(preview, frames, duration=.07, loop=0)
        yield {"phase": "complete", "step": completed, "score": score, "x": x, "y": y, "preview": str(preview), "log": f"Rendered {len(frames)} synchronized multi-agent frames"}
    finally:
        train_env.close()
