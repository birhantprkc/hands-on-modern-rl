from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterator

import imageio.v2 as imageio
import numpy as np


def stable_baselines_runtime():
    from stable_baselines3 import DQN, PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.evaluation import evaluate_policy

    try:
        from stable_baselines3 import A2C
    except ImportError:
        A2C = None
    return {"DQN": DQN, "PPO": PPO, "A2C": A2C}, BaseCallback, evaluate_policy


def format_metrics(metrics: dict[str, Any], step: int) -> str:
    rows: list[tuple[str, str]] = []
    preferred = (
        "rollout/ep_rew_mean",
        "rollout/ep_len_mean",
        "time/fps",
        "train/loss",
        "train/value_loss",
        "train/policy_gradient_loss",
        "train/entropy_loss",
        "train/learning_rate",
        "train/n_updates",
    )
    for key in preferred:
        value = metrics.get(key)
        if value is None:
            continue
        if isinstance(value, (float, np.floating)):
            rendered = f"{float(value):.6g}"
        else:
            rendered = str(value)
        rows.append((key, rendered))
    if not rows:
        return f"PPO/DQN update · total_timesteps={step:,}"
    width = max(len(name) for name, _ in rows)
    bar = "-" * (width + 22)
    body = "\n".join(f"| {name:<{width}} | {value:>15} |" for name, value in rows)
    return f"update · step {step:,}\n{bar}\n{body}\n{bar}"


def train_sb3(
    *,
    root: Path,
    task: Any,
    make_train_env: Callable[[], Any],
    make_eval_env: Callable[[], Any],
    make_record_env: Callable[[], Any],
    budget: int,
    learning_rate: float,
    gamma: float,
    epsilon: float,
    seed: int,
    record_episode: Callable[[Any, Any, Path, int], str],
) -> Iterator[dict[str, Any]]:
    algorithms, BaseCallback, evaluate_policy = stable_baselines_runtime()
    algorithm_name = str(getattr(task, "algorithm", task.get("algorithm") if isinstance(task, dict) else "PPO"))
    algorithm_cls = algorithms.get(algorithm_name)
    if algorithm_cls is None:
        raise RuntimeError(f"Unsupported Stable-Baselines3 algorithm: {algorithm_name}")

    class MetricsCallback(BaseCallback):
        def __init__(self):
            super().__init__(verbose=0)
            self.latest: dict[str, Any] = {}

        def _on_step(self) -> bool:
            self.latest = dict(self.logger.name_to_value)
            return True

    train_env = make_train_env()
    eval_env = make_eval_env()
    kwargs = {
        "learning_rate": learning_rate,
        "gamma": gamma,
        "seed": seed,
        "verbose": 0,
        "device": "cpu",
    }
    policy = getattr(task, "policy", task.get("policy", "MlpPolicy") if isinstance(task, dict) else "MlpPolicy")
    if algorithm_name == "DQN":
        kwargs.update(
            buffer_size=max(10_000, min(100_000, budget * 2)),
            learning_starts=max(100, min(2_000, budget // 10)),
            exploration_initial_eps=max(.05, epsilon),
            exploration_final_eps=max(.01, epsilon * .1),
            train_freq=4,
            gradient_steps=1,
            target_update_interval=max(250, min(2_000, budget // 8)),
        )
    elif algorithm_name in {"PPO", "A2C"}:
        kwargs.update(n_steps=max(64, min(512, budget // 8)), ent_coef=max(0.0, epsilon * .01))
        if algorithm_name == "PPO":
            kwargs.update(batch_size=64, n_epochs=4)

    callback = MetricsCallback()
    model = algorithm_cls(policy, train_env, **kwargs)
    checkpoints = int(getattr(task, "checkpoints", task.get("checkpoints", 6) if isinstance(task, dict) else 6))
    checkpoints = max(2, min(12, checkpoints))
    chunk = max(1, budget // checkpoints)
    x: list[float] = []
    y: list[float] = []
    yield {"phase": "training", "step": 0, "x": x, "y": y, "log": f"Initialized {algorithm_name} with {policy} on CPU"}
    completed = 0
    try:
        while completed < budget:
            current = min(chunk, budget - completed)
            model.learn(total_timesteps=current, reset_num_timesteps=False, callback=callback, progress_bar=False)
            completed += current
            rewards, lengths = evaluate_policy(model, eval_env, n_eval_episodes=3, deterministic=True, return_episode_rewards=True, warn=False)
            score = float(np.mean(rewards))
            spread = float(np.std(rewards))
            x.append(float(completed)); y.append(score)
            details = format_metrics(callback.latest, completed)
            details += f"\nEVAL step={completed:,} mean_reward={score:.2f} std={spread:.2f} mean_length={np.mean(lengths):.1f}"
            yield {
                "phase": "training",
                "step": completed,
                "score": score,
                "x": x,
                "y": y,
                "detail": f"{completed:,}/{budget:,} environment steps",
                "metric_detail": f"mean reward ± {spread:.2f}",
                "log": details,
            }

        artifacts = root / "artifacts"
        artifacts.mkdir(parents=True, exist_ok=True)
        model_path = artifacts / f"{getattr(task, 'key', task.get('key'))}-model"
        model.save(str(model_path))
        record_env = make_record_env()
        preview = record_episode(model, record_env, artifacts, seed + 10_000)
        metadata = artifacts / f"{getattr(task, 'key', task.get('key'))}-model.json"
        metadata.write_text(json.dumps({"algorithm": algorithm_name, "policy": policy, "budget": budget, "seed": seed}, indent=2), encoding="utf-8")
        yield {
            "phase": "complete",
            "step": completed,
            "score": y[-1] if y else None,
            "x": x,
            "y": y,
            "preview": preview,
            "log": f"Saved model and generated learned-policy replay: {Path(preview).name}",
        }
    finally:
        for env in (train_env, eval_env):
            try:
                env.close()
            except Exception:
                pass


def save_gif(frames: list[np.ndarray], path: Path, fps: int = 20) -> str:
    if not frames:
        raise RuntimeError("The environment returned no RGB frames for the learned-policy replay")
    normalized: list[np.ndarray] = []
    for frame in frames:
        array = np.asarray(frame)
        if array.ndim == 4:
            array = array[0]
        if array.shape[-1] == 4:
            array = array[..., :3]
        normalized.append(array.astype(np.uint8))
    imageio.mimsave(path, normalized, duration=1 / max(1, fps), loop=0)
    return str(path)
