from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Iterator
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sb3_tools import save_gif


ROOT = Path(__file__).resolve().parent
BUNDLED_PHYSX = ROOT / "assets" / "physx-105.1-physx-5.3.1.patch0-linux-so.zip"
PERSISTENT_CACHE = Path(
    os.environ.get("HOMRL_PERSISTENT_CACHE", "/mnt/workspace/hands-on-modern-rl")
) / "maniskill"
_WARMUP_LOCK = threading.Lock()
_WARMUP_DONE = threading.Event()
_WARMUP_THREAD: threading.Thread | None = None
_WARMUP_STATE: dict[str, Any] = {
    "phase": "pending",
    "detail": "GPU PhysX cache is queued for background preparation",
    "progress": 0.0,
    "error": None,
}
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


def _set_warmup_state(phase: str, detail: str, progress: float = 0.0, error: str | None = None) -> None:
    with _WARMUP_LOCK:
        _WARMUP_STATE.update(phase=phase, detail=detail, progress=float(progress), error=error)


def _warmup_snapshot() -> dict[str, Any]:
    with _WARMUP_LOCK:
        return dict(_WARMUP_STATE)


def _prepare_persistent_sapien_home() -> Path:
    """Map SAPIEN's hard-coded ~/.sapien cache onto ModelScope persistent storage."""
    persistent_home = PERSISTENT_CACHE / "sapien-home"
    persistent_home.mkdir(parents=True, exist_ok=True)
    sapien_home = Path.home() / ".sapien"
    if sapien_home.is_symlink():
        return sapien_home
    if not sapien_home.exists():
        sapien_home.parent.mkdir(parents=True, exist_ok=True)
        sapien_home.symlink_to(persistent_home, target_is_directory=True)
        return sapien_home

    # A base image may have created ~/.sapien already. In that case persist the
    # large PhysX subtree without removing any image-owned files.
    persistent_physx = persistent_home / "physx"
    persistent_physx.mkdir(parents=True, exist_ok=True)
    physx_home = sapien_home / "physx"
    if not physx_home.exists():
        physx_home.symlink_to(persistent_physx, target_is_directory=True)
    return sapien_home


def _download_physx(url: str, archive: Path, target: Path, dll: Path) -> None:
    if shutil.which("aria2c") is None:
        raise RuntimeError("aria2c is required to prepare the GPU PhysX cache efficiently")
    process = subprocess.Popen(
        [
            "aria2c", "--allow-overwrite=true", "--auto-file-renaming=false", "--continue=true",
            "--file-allocation=none", "--max-connection-per-server=16", "--split=16",
            "--min-split-size=1M", "--summary-interval=2", "--console-log-level=notice",
            "--enable-color=false",
            "--dir", str(archive.parent), "--out", archive.name, url,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    assert process.stdout is not None
    last_update = 0.0
    for line in process.stdout:
        clean = line.strip()
        now = time.monotonic()
        if clean and ("Download complete" in clean or ("[#" in clean and now - last_update >= 4.0)):
            _set_warmup_state("downloading", f"GPU PhysX · {clean}", 0.5)
            last_update = now
    if process.wait() != 0:
        raise RuntimeError(f"GPU PhysX parallel download exited with code {process.returncode}")
    _set_warmup_state("extracting", "Extracting the GPU PhysX runtime into persistent storage", 0.98)
    with ZipFile(archive) as bundle:
        bundle.extractall(target)
    archive.unlink(missing_ok=True)
    if not dll.exists():
        raise RuntimeError(f"PhysX archive extracted without the expected library: {dll.name}")


def _warmup_runtime() -> None:
    try:
        _set_warmup_state("preparing", "Connecting SAPIEN to persistent ModelScope storage", 0.01)
        _prepare_persistent_sapien_home()
        import sapien

        physx_version = sapien.physx.version()
        target = Path.home() / ".sapien" / "physx" / physx_version
        target.mkdir(parents=True, exist_ok=True)
        dll = target / "libPhysXGpu_64.so"
        if not dll.exists():
            if BUNDLED_PHYSX.exists() and BUNDLED_PHYSX.stat().st_size > 1_000_000:
                _set_warmup_state("extracting", "Loading the bundled BSD-licensed GPU PhysX runtime", 0.96)
                with ZipFile(BUNDLED_PHYSX) as bundle:
                    bundle.extractall(target)
                if not dll.exists():
                    raise RuntimeError("Bundled PhysX archive is missing libPhysXGpu_64.so")
            else:
                url = (
                    "https://github.com/sapien-sim/physx-precompiled/releases/download/"
                    f"{physx_version}/linux-so.zip"
                )
                _set_warmup_state("downloading", "Downloading the GPU PhysX runtime", 0.02)
                _download_physx(url, target / "linux-so.zip.part", target, dll)
        _set_warmup_state("loading", "Loading the cached GPU PhysX runtime", 0.99)
        sapien.physx.enable_gpu()
        _set_warmup_state("ready", f"GPU PhysX {physx_version} is cached and ready", 1.0)
    except Exception as exc:
        _set_warmup_state(
            "error",
            f"GPU PhysX preparation failed: {type(exc).__name__}: {exc}",
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        _WARMUP_DONE.set()


def start_runtime_warmup() -> None:
    global _WARMUP_THREAD
    with _WARMUP_LOCK:
        if _WARMUP_THREAD is not None:
            return
        _WARMUP_THREAD = threading.Thread(target=_warmup_runtime, name="physx-warmup", daemon=True)
        _WARMUP_THREAD.start()


def _wait_for_runtime() -> Iterator[str]:
    start_runtime_warmup()
    last_detail = ""
    while not _WARMUP_DONE.wait(timeout=1.0):
        detail = str(_warmup_snapshot()["detail"])
        if detail != last_detail:
            yield detail
            last_detail = detail
    state = _warmup_snapshot()
    if state["error"]:
        raise RuntimeError(str(state["detail"]))
    if str(state["detail"]) != last_detail:
        yield str(state["detail"])


def runtime_status() -> str:
    try:
        import mani_skill
        import torch
        warmup = _warmup_snapshot()
        if torch.cuda.is_available():
            if warmup["phase"] == "ready":
                return f"ManiSkill {mani_skill.__version__} · {torch.cuda.get_device_name(0)} · GPU PHYSX READY"
            return f"ManiSkill {mani_skill.__version__} · {torch.cuda.get_device_name(0)} · {warmup['detail']}"
        return f"ManiSkill {mani_skill.__version__} · waiting for xGPU"
    except Exception as exc:
        return f"installing ManiSkill runtime · {type(exc).__name__}"


def _make_vec_env(task: dict[str, Any], num_envs: int, render_mode: str | None = None):
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    from mani_skill.vector.wrappers.sb3 import ManiSkillSB3VectorEnv

    # ModelScope xGPU images expose CUDA/PhysX but do not always expose the
    # host NVIDIA Vulkan ICD. State-based training does not need rendering, so
    # keep the renderer disabled there. A single CPU renderer is requested only
    # while producing the final replay.
    raw = gym.make(
        task["environment"],
        num_envs=num_envs,
        obs_mode="state",
        reward_mode="dense",
        render_mode=render_mode,
        render_backend="none" if render_mode is None else "cpu",
        sim_backend="physx_cuda" if num_envs > 1 else "physx_cpu",
        reconfiguration_freq=1,
    )
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


def _telemetry_frame(
    task: dict[str, Any],
    step: int,
    rewards: list[float],
    action_norms: list[float],
    width: int = 720,
    height: int = 420,
) -> np.ndarray:
    """Render a real policy rollout as a lightweight GIF without Vulkan."""
    image = Image.new("RGB", (width, height), "#f5f7ff")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.rounded_rectangle((18, 18, width - 18, height - 18), radius=24, fill="#ffffff", outline="#d9def4", width=2)
    draw.text((42, 38), "LEARNED POLICY TELEMETRY REPLAY", fill="#5661e9", font=font)
    draw.text((42, 70), f"{task['environment']}  ·  GPU PPO", fill="#151a38", font=font)
    draw.text((42, 96), f"Policy step {step + 1:03d} / 200", fill="#59617a", font=font)

    left, top, right, bottom = 56, 145, width - 42, height - 66
    draw.rounded_rectangle((left, top, right, bottom), radius=12, fill="#f7f8fc", outline="#e0e4f2")
    for fraction in (0.25, 0.5, 0.75):
        y = int(top + (bottom - top) * fraction)
        draw.line((left, y, right, y), fill="#e5e8f3", width=1)
    values = rewards[: step + 1]
    if values:
        low, high = min(values), max(values)
        span = max(1e-6, high - low)
        points = []
        for index, value in enumerate(values):
            x = left + int((right - left) * index / 199)
            y = bottom - int((bottom - top) * (value - low) / span)
            points.append((x, y))
        if len(points) > 1:
            draw.line(points, fill="#5661e9", width=4)
        draw.ellipse((points[-1][0] - 5, points[-1][1] - 5, points[-1][0] + 5, points[-1][1] + 5), fill="#16a673")
    cumulative = float(np.sum(values)) if values else 0.0
    action_norm = action_norms[step] if step < len(action_norms) else 0.0
    draw.text((56, height - 48), f"Cumulative dense reward  {cumulative:8.2f}", fill="#252b45", font=font)
    draw.text((width - 250, height - 48), f"Action norm  {action_norm:6.3f}", fill="#252b45", font=font)
    return np.asarray(image)


def _record_telemetry(model: Any, task: dict[str, Any], seed: int) -> str:
    raw, env = _make_vec_env(task, 1)
    rewards: list[float] = []
    action_norms: list[float] = []
    try:
        observation = env.reset()
        for _ in range(200):
            actions, _ = model.predict(observation, deterministic=True)
            observation, reward, dones, infos = env.step(actions)
            rewards.append(float(np.asarray(reward).reshape(-1)[0]))
            action_norms.append(float(np.linalg.norm(np.asarray(actions).reshape(-1))))
    finally:
        env.close()
        del raw
    frames = [
        _telemetry_frame(task, index, rewards, action_norms)
        for index in range(0, len(rewards), 4)
    ]
    return save_gif(frames, ROOT / "artifacts" / f"{task['key']}-learned-policy.gif", fps=12)


def _record(model: Any, task: dict[str, Any], seed: int) -> tuple[str, str]:
    raw = None
    env = None
    try:
        raw, env = _make_vec_env(task, 1, render_mode="rgb_array")
        frames: list[np.ndarray] = []
        observation = env.reset()
        for index in range(200):
            if index % 2 == 0:
                frame = raw.render()
                if hasattr(frame, "detach"):
                    frame = frame.detach().cpu().numpy()
                frames.append(np.asarray(frame))
            actions, _ = model.predict(observation, deterministic=True)
            observation, rewards, dones, infos = env.step(actions)
        preview = save_gif(frames, ROOT / "artifacts" / f"{task['key']}-learned-policy.gif", fps=24)
        return preview, "Rendered the learned policy with the CPU Vulkan backend"
    except Exception as exc:
        if env is not None:
            env.close()
            env = None
        raw = None
        preview = _record_telemetry(model, task, seed)
        return preview, f"CPU Vulkan replay unavailable ({type(exc).__name__}); generated a real policy telemetry GIF"
    finally:
        if env is not None:
            env.close()
        del raw


def run(key: str, budget: int, learning_rate: float, gamma: float, epsilon: float, seed: int) -> Iterator[dict[str, Any]]:
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback

    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires a scheduled ModelScope xGPU; CUDA is not currently visible")
    for detail in _wait_for_runtime():
        yield {"phase": "initializing", "step": 0, "log": detail}
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
        preview, preview_detail = _record(model, task, int(seed) + 10_000)
        (artifact_dir / f"{key}-model.json").write_text(json.dumps({"environment": task["environment"], "algorithm": "PPO", "budget": int(budget), "parallel_envs": parallel_envs, "seed": int(seed)}, indent=2), encoding="utf-8")
        yield {"phase": "complete", "step": completed, "score": y[-1], "x": x, "y": y, "preview": preview,
               "log": f"Saved the GPU policy and generated {Path(preview).name}. {preview_detail}"}
    finally:
        train_env.close()
        del raw
