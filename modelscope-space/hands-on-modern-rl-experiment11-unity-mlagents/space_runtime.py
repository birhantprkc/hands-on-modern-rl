from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Iterator

import yaml


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
UNITY_BUNDLE_URL = "https://storage.googleapis.com/mlagents-test-environments/1.1.0/linux/Startup.zip"
UNITY_CACHE = Path(os.environ.get("UNITY_MLAGENTS_CACHE", "/mnt/workspace/hands-on-modern-rl/unity-mlagents-1.1.0"))

SPACE = {
    "title": {"en": "Unity ML-Agents xGPU Arena", "zh": "Unity ML-Agents xGPU 训练场"},
    "description": {
        "en": "Train PPO inside official Unity ML-Agents Linux scenes, follow the native trainer console, and replay the rendered run.",
        "zh": "在 Unity ML-Agents 官方 Linux 场景中训练 PPO，查看原生训练日志，并回放本次训练过程。",
    },
    "badge": "EXPERIMENT 11 · UNITY",
    "device": "xGPU",
    "organization_url": "https://modelscope.cn/organization/walkinglab",
    "project_url": "https://github.com/walkinglabs/hands-on-modern-rl",
    "course_url": "https://walkinglabs.github.io/hands-on-modern-rl/",
    "source_url": "https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment11-unity-mlagents/file/view/master/space_runtime.py",
}


def _task(key: str, title: str, title_zh: str, scene: str, behavior: str, description: str,
          description_zh: str, observation: str, action: str, preview: str,
          budget: tuple[int, int, int, int], trainer: dict[str, Any]) -> dict[str, Any]:
    return {
        "key": key, "title": {"en": title, "zh": title_zh}, "environment": f"Unity/{scene}",
        "scene": scene, "behavior": behavior,
        "description": {"en": description, "zh": description_zh},
        "observation": {"en": observation, "zh": observation},
        "action": {"en": action, "zh": action}, "algorithm": "Unity PPO", "preview": preview,
        "budget": budget, "learning_rate": (1e-5, 0.001, 0.0003, 1e-5),
        "gamma": (0.8, 1.0, float(trainer["reward_signals"]["extrinsic"]["gamma"]), 0.005),
        "epsilon": (0.05, 0.35, float(trainer["hyperparameters"]["epsilon"]), 0.01),
        "checkpoints": 6, "trainer": trainer,
    }


TASKS = [
    _task("unity-basic", "Basic · Discrete PPO", "Basic · 离散 PPO", "Basic", "Basic",
          "Match the target value with a short sequence of discrete decisions.", "通过一小段离散决策使数值匹配目标。",
          "Vector target and current state", "Discrete left / stay / right", "assets/unity-basic.svg", (2_000, 120_000, 20_000, 2_000),
          {"trainer_type": "ppo", "hyperparameters": {"batch_size": 32, "buffer_size": 256, "learning_rate": 0.0003, "beta": 0.005, "epsilon": 0.2, "lambd": 0.95, "num_epoch": 3, "learning_rate_schedule": "linear"}, "network_settings": {"normalize": False, "hidden_units": 20, "num_layers": 1, "vis_encode_type": "simple"}, "reward_signals": {"extrinsic": {"gamma": 0.9, "strength": 1.0}}, "keep_checkpoints": 3, "time_horizon": 3}),
    _task("unity-3dball", "3D Ball · Continuous PPO", "3D Ball · 连续 PPO", "3DBall", "3DBall",
          "Tilt a platform in two axes and keep the ball from falling.", "控制平台在两个方向倾斜，使小球不掉落。",
          "Ball position/velocity and platform rotation", "Continuous platform tilt", "assets/unity-3dball.svg", (8_000, 500_000, 60_000, 4_000),
          {"trainer_type": "ppo", "hyperparameters": {"batch_size": 64, "buffer_size": 12000, "learning_rate": 0.0003, "beta": 0.001, "epsilon": 0.2, "lambd": 0.99, "num_epoch": 3, "learning_rate_schedule": "linear"}, "network_settings": {"normalize": True, "hidden_units": 128, "num_layers": 2, "vis_encode_type": "simple"}, "reward_signals": {"extrinsic": {"gamma": 0.99, "strength": 1.0}}, "keep_checkpoints": 3, "time_horizon": 1000}),
    _task("unity-food", "Food Collector · Visual PPO", "Food Collector · 视觉 PPO", "FoodCollector", "GridFoodCollector",
          "Collect green food while avoiding red food and competing agents.", "收集绿色食物，同时避开红色食物与其他智能体。",
          "Ray sensors and local visual state", "Move, rotate, and fire", "assets/unity-food.svg", (10_000, 500_000, 80_000, 5_000),
          {"trainer_type": "ppo", "hyperparameters": {"batch_size": 1024, "buffer_size": 10240, "learning_rate": 0.0003, "beta": 0.005, "epsilon": 0.2, "lambd": 0.95, "num_epoch": 3, "learning_rate_schedule": "linear"}, "network_settings": {"normalize": False, "hidden_units": 256, "num_layers": 1, "vis_encode_type": "simple"}, "reward_signals": {"extrinsic": {"gamma": 0.99, "strength": 1.0}}, "keep_checkpoints": 3, "time_horizon": 64}),
    _task("unity-walker", "Walker · Locomotion PPO", "Walker · 运动控制 PPO", "Walker", "Walker",
          "Coordinate a many-jointed body to move toward the target direction.", "协调多关节身体，沿目标方向稳定行走。",
          "Joint rotations, velocities, contacts, and target direction", "Continuous joint targets", "assets/unity-walker.svg", (20_000, 1_000_000, 120_000, 10_000),
          {"trainer_type": "ppo", "hyperparameters": {"batch_size": 2048, "buffer_size": 20480, "learning_rate": 0.0003, "beta": 0.005, "epsilon": 0.2, "lambd": 0.95, "num_epoch": 3, "learning_rate_schedule": "linear"}, "network_settings": {"normalize": True, "hidden_units": 256, "num_layers": 3, "vis_encode_type": "simple"}, "reward_signals": {"extrinsic": {"gamma": 0.995, "strength": 1.0}}, "keep_checkpoints": 3, "time_horizon": 1000}),
]


def runtime_status() -> str:
    try:
        import mlagents
        import torch
        accelerator = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU fallback"
        return f"ML-Agents {mlagents.__version__} · {accelerator} · OFFICIAL REGISTRY"
    except Exception as exc:
        return f"installing ML-Agents runtime · {type(exc).__name__}"


def _ensure_unity_bundle() -> Path:
    candidates = [path for path in UNITY_CACHE.rglob("Startup.x86_64") if path.is_file()]
    if not candidates:
        candidates = [path for path in UNITY_CACHE.rglob("*.x86_64") if path.is_file()]
    if candidates:
        candidates[0].chmod(0o755)
        return candidates[0]
    UNITY_CACHE.mkdir(parents=True, exist_ok=True)
    archive, partial = UNITY_CACHE / "Startup.zip", UNITY_CACHE / "Startup.zip.part"
    urllib.request.urlretrieve(UNITY_BUNDLE_URL, partial)
    partial.replace(archive)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(UNITY_CACHE)
    candidates = [path for path in UNITY_CACHE.rglob("Startup.x86_64") if path.is_file()]
    if not candidates:
        candidates = [path for path in UNITY_CACHE.rglob("*.x86_64") if path.is_file()]
    if not candidates:
        raise RuntimeError("The official Unity bundle did not contain a Linux executable")
    selected = candidates[0]
    selected.chmod(0o755)
    return selected


def _scaled_config(task: dict[str, Any], budget: int, learning_rate: float, gamma: float,
                   epsilon: float, seed: int, executable: Path, run_id: str) -> Path:
    import torch

    trainer = yaml.safe_load(yaml.safe_dump(task["trainer"]))
    hyper = trainer["hyperparameters"]
    hyper["learning_rate"], hyper["epsilon"] = learning_rate, epsilon
    trainer["reward_signals"]["extrinsic"]["gamma"] = gamma
    min_buffer = max(256, int(hyper["batch_size"]) * 4)
    hyper["buffer_size"] = max(min_buffer, min(int(hyper["buffer_size"]), max(min_buffer, budget // 4)))
    hyper["batch_size"] = max(16, min(int(hyper["batch_size"]), int(hyper["buffer_size"]) // 4))
    trainer.update(max_steps=budget, summary_freq=max(200, budget // 8), checkpoint_interval=max(1_000, budget))
    config = {
        "behaviors": {task["behavior"]: trainer},
        "env_settings": {"env_path": str(executable), "env_args": ["--mlagents-scene-name", f"Assets/ML-Agents/Examples/{task['scene']}/Scenes/{task['scene']}.unity"], "num_envs": 1, "seed": seed, "timeout_wait": 180},
        "engine_settings": {"width": 960, "height": 540, "quality_level": 2, "time_scale": 20, "target_frame_rate": -1, "capture_frame_rate": 60, "no_graphics": False},
        "checkpoint_settings": {"run_id": run_id, "results_dir": str(ARTIFACTS / "unity-results"), "force": True},
        "torch_settings": {"device": "cuda" if torch.cuda.is_available() else "cpu"},
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS / f"{run_id}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _start_xvfb() -> subprocess.Popen[str]:
    display = os.environ.setdefault("DISPLAY", ":99")
    if shutil.which("Xvfb") is None:
        raise RuntimeError("Xvfb is required to render the Unity replay")
    return subprocess.Popen(["Xvfb", display, "-screen", "0", "960x540x24", "-ac", "+extension", "GLX"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, text=True, start_new_session=True)


def _start_capture(target: Path) -> subprocess.Popen[str]:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to record the Unity replay")
    return subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "x11grab", "-framerate", "12", "-video_size", "960x540", "-i", os.environ["DISPLAY"], "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(target)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True, start_new_session=True)


def _stop_process(process: subprocess.Popen[Any] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=8)
    except Exception:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except Exception:
            pass


def _make_gif(video: Path, output: Path) -> str:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-sseof", "-10", "-i", str(video), "-vf", "fps=12,scale=640:-1:flags=lanczos", str(output)], check=True, timeout=90)
    if not output.exists() or output.stat().st_size < 1_000:
        raise RuntimeError("Unity completed, but the replay capture was empty")
    return str(output)


def run(key: str, budget: int, learning_rate: float, gamma: float, epsilon: float, seed: int) -> Iterator[dict[str, Any]]:
    task = next(item for item in TASKS if item["key"] == key)
    run_id = f"{key}-{int(time.time())}"
    yield {"phase": "initializing", "step": 0, "log": "Checking the official Unity ML-Agents 1.1.0 Linux environment cache"}
    executable = _ensure_unity_bundle()
    yield {"phase": "initializing", "step": 0, "log": f"Official Unity executable ready: {executable}\nScene: {task['scene']} · behavior: {task['behavior']}"}
    config = _scaled_config(task, int(budget), float(learning_rate), float(gamma), float(epsilon), int(seed), executable, run_id)
    video, replay = ARTIFACTS / f"{run_id}-training.mp4", ARTIFACTS / f"{key}-learned-policy.gif"
    xvfb = capture = trainer = None
    x: list[float] = []
    y: list[float] = []
    last_step = 0
    score_re = re.compile(r"Step:\s*([0-9,]+).*?Mean Reward:\s*(-?[0-9.]+)", re.IGNORECASE)
    try:
        xvfb = _start_xvfb()
        time.sleep(1.2)
        capture = _start_capture(video)
        trainer = subprocess.Popen(["mlagents-learn", str(config)], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env={**os.environ, "PYTHONUNBUFFERED": "1"}, start_new_session=True)
        assert trainer.stdout is not None
        for line in trainer.stdout:
            clean = line.rstrip()
            if not clean:
                continue
            match = score_re.search(clean)
            if match:
                last_step, score = int(match.group(1).replace(",", "")), float(match.group(2))
                x.append(float(last_step)); y.append(score)
                yield {"phase": "training", "step": last_step, "score": score, "x": x, "y": y, "detail": f"{last_step:,}/{int(budget):,} Unity steps", "metric_detail": "Unity trainer mean reward", "log": clean}
            else:
                yield {"phase": "training", "step": last_step, "x": x, "y": y, "detail": f"{last_step:,}/{int(budget):,} Unity steps", "log": clean}
        if trainer.wait() != 0:
            raise RuntimeError(f"Unity ML-Agents trainer exited with code {trainer.returncode}")
    finally:
        _stop_process(trainer)
        _stop_process(capture)
        _stop_process(xvfb)
    preview = _make_gif(video, replay)
    yield {"phase": "complete", "step": int(budget), "score": y[-1] if y else None, "x": x, "y": y, "preview": preview, "log": f"Unity PPO complete · recorded final rendered segment to {Path(preview).name}"}
