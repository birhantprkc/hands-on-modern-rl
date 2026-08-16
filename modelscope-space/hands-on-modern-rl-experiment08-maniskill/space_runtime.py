from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from sb3_tools import save_gif


ROOT = Path(__file__).resolve().parent
SPACE = {
    "title": {"en": "ManiSkill xGPU Robot Lab", "zh": "ManiSkill xGPU 机器人训练场"},
    "description": {
        "en": "Run dozens of robot simulations in parallel on the GPU, train PPO from state observations, and render the learned policy.",
        "zh": "在 GPU 上并行运行数十个机器人仿真环境，用状态观察训练 PPO，并渲染学习后的策略。",
    },
    "badge": "EXPERIMENT 08 · MANISKILL",
    "device": "xGPU · CUDA + Vulkan",
    "organization_url": "https://modelscope.cn/organization/walkinglab",
    "project_url": "https://github.com/walkinglabs/hands-on-modern-rl",
    "course_url": "https://walkinglabs.github.io/hands-on-modern-rl/",
    "source_url": "https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment08-maniskill/file/view/master/space_runtime.py",
}


def _task(key: str, env_id: str, title: str, zh: str, description: str, description_zh: str,
          action: str, preview: str, default_budget: int, gamma: float) -> dict[str, Any]:
    return {
        "key": key, "title": {"en": title, "zh": zh}, "environment": env_id,
        "description": {"en": description, "zh": description_zh},
        "observation": {"en": "Robot joint state + object pose", "zh": "机器人关节状态与物体位姿"},
        "action": {"en": action, "zh": action}, "algorithm": "GPU PPO", "policy": "MlpPolicy",
        "device": "cuda", "preview": preview, "budget": (10_000, 1_000_000, default_budget, 10_000),
        "learning_rate": (1e-5, 0.001, 0.0003, 1e-5), "gamma": (0.8, 1.0, gamma, 0.005),
        "epsilon": (0.0, 0.2, 0.02, 0.005), "checkpoints": 6,
    }


TASKS = [
    _task("push-cube", "PushCube-v1", "PushCube · Panda", "PushCube · 熊猫机械臂", "Push the cube to the marked goal position.", "将方块推到标记的目标位置。", "7D end-effector delta pose", "assets/maniskill-push.svg", 100_000, 0.8),
    _task("pick-cube", "PickCube-v1", "PickCube · Panda", "PickCube · 熊猫机械臂", "Grasp a cube and lift it to a target pose.", "抓住方块并将其抬升到目标位姿。", "7D end-effector delta pose + gripper", "assets/maniskill-pick.svg", 150_000, 0.9),
    _task("stack-cube", "StackCube-v1", "StackCube · Panda", "StackCube · 熊猫机械臂", "Pick up one cube and stack it stably on another.", "拾取一个方块，并将它稳定叠放在另一个方块上。", "7D end-effector delta pose + gripper", "assets/maniskill-stack.svg", 250_000, 0.95),
    _task("peg-insertion", "PegInsertionSide-v1", "PegInsertionSide · Panda", "PegInsertionSide · 熊猫机械臂", "Align a peg and insert it into a horizontal socket.", "对齐插销，并把它插入水平插座。", "7D end-effector delta pose + gripper", "assets/maniskill-peg.svg", 300_000, 0.95),
]


def runtime_status() -> str:
    try:
        import mani_skill
        import torch
        if torch.cuda.is_available():
            return f"ManiSkill {mani_skill.__version__} · {torch.cuda.get_device_name(0)} · GPU PHYSX"
        return f"ManiSkill {mani_skill.__version__} · waiting for xGPU"
    except Exception as exc:
        return f"installing ManiSkill runtime · {type(exc).__name__}"


def _make_vec_env(task: dict[str, Any], num_envs: int, render_mode: str | None = None):
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    from mani_skill.vector.wrappers.sb3 import ManiSkillSB3VectorEnv

    raw = gym.make(task["environment"], num_envs=num_envs, obs_mode="state", reward_mode="dense",
                   render_mode=render_mode, reconfiguration_freq=1)
    return raw, ManiSkillSB3VectorEnv(raw)


def _evaluate(model: Any, task: dict[str, Any], seed: int, episodes: int = 16) -> tuple[float, float]:
    raw, env = _make_vec_env(task, episodes)
    try:
        observation = env.reset()
        returns = np.zeros(episodes, dtype=np.float64)
        for _ in range(200):
            actions, _ = model.predict(observation, deterministic=True)
            observation, rewards, dones, infos = env.step(actions)
            returns += np.asarray(rewards, dtype=np.float64).reshape(-1)[:episodes]
        return float(returns.mean()), float(returns.std())
    finally:
        env.close()
        del raw


def _record(model: Any, task: dict[str, Any], seed: int) -> str:
    raw, env = _make_vec_env(task, 1, render_mode="rgb_array")
    frames: list[np.ndarray] = []
    try:
        observation = env.reset()
        for index in range(200):
            if index % 2 == 0:
                frame = raw.render()
                if hasattr(frame, "detach"):
                    frame = frame.detach().cpu().numpy()
                frames.append(np.asarray(frame))
            actions, _ = model.predict(observation, deterministic=True)
            observation, rewards, dones, infos = env.step(actions)
    finally:
        env.close()
        del raw
    return save_gif(frames, ROOT / "artifacts" / f"{task['key']}-learned-policy.gif", fps=24)


def run(key: str, budget: int, learning_rate: float, gamma: float, epsilon: float, seed: int) -> Iterator[dict[str, Any]]:
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback

    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires a scheduled ModelScope xGPU; CUDA is not currently visible")
    task = next(item for item in TASKS if item["key"] == key)
    parallel_envs = max(16, min(128, int(budget) // 1_000))
    yield {"phase": "initializing", "step": 0, "log": f"Creating {parallel_envs} parallel {task['environment']} simulations on {torch.cuda.get_device_name(0)}"}
    raw, train_env = _make_vec_env(task, parallel_envs)

    class MetricsCallback(BaseCallback):
        latest: dict[str, Any]
        def __init__(self) -> None:
            super().__init__(verbose=0); self.latest = {}
        def _on_step(self) -> bool:
            self.latest = dict(self.logger.name_to_value); return True

    n_steps = 50
    rollout = parallel_envs * n_steps
    batch_size = next(size for size in (512, 400, 320, 256, 200, 160, 128, 100, 80, 64, 50, 40, 32, 25, 20, 16) if size <= rollout and rollout % size == 0)
    callback = MetricsCallback()
    model = PPO("MlpPolicy", train_env, learning_rate=float(learning_rate), gamma=float(gamma),
                gae_lambda=0.9, ent_coef=max(0.0, float(epsilon)), n_steps=n_steps,
                batch_size=batch_size, n_epochs=6, device="cuda", seed=int(seed), verbose=0)
    checkpoints = 6
    chunk = max(rollout, (int(budget) // checkpoints // rollout) * rollout)
    x: list[float] = []
    y: list[float] = []
    completed = 0
    try:
        while completed < int(budget):
            current = min(chunk, int(budget) - completed)
            model.learn(total_timesteps=current, reset_num_timesteps=False, callback=callback, progress_bar=False)
            completed += current
            score, spread = _evaluate(model, task, int(seed) + completed)
            x.append(float(completed)); y.append(score)
            metrics = callback.latest
            log = (f"PPO update · step={completed:,}\n"
                   f"parallel_envs={parallel_envs}  rollout={rollout}  device={model.device}\n"
                   f"policy_loss={float(metrics.get('train/policy_gradient_loss') or float('nan')):.6g}  "
                   f"value_loss={float(metrics.get('train/value_loss') or float('nan')):.6g}\n"
                   f"EVAL mean_dense_return={score:.3f} std={spread:.3f}")
            yield {"phase": "training", "step": completed, "score": score, "x": x, "y": y,
                   "detail": f"{completed:,}/{int(budget):,} environment steps",
                   "metric_detail": f"mean dense return ± {spread:.2f}", "log": log}
        artifact_dir = ROOT / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(artifact_dir / f"{key}-ppo"))
        preview = _record(model, task, int(seed) + 10_000)
        (artifact_dir / f"{key}-model.json").write_text(json.dumps({"environment": task["environment"], "algorithm": "PPO", "budget": int(budget), "parallel_envs": parallel_envs, "seed": int(seed)}, indent=2), encoding="utf-8")
        yield {"phase": "complete", "step": completed, "score": y[-1], "x": x, "y": y, "preview": preview,
               "log": f"Saved the GPU policy and rendered {Path(preview).name} from the learned policy"}
    finally:
        train_env.close()
        del raw
