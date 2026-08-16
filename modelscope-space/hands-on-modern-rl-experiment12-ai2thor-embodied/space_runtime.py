from __future__ import annotations

import json
import hashlib
import math
import os
import shutil
import signal
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Iterator

import gymnasium as gym
import numpy as np
from PIL import Image

from sb3_tools import save_gif


ROOT = Path(__file__).resolve().parent
THOR_CACHE = Path("/mnt/workspace/hands-on-modern-rl/ai2thor")
os.environ.setdefault("DISPLAY", ":99")

SPACE = {
    "title": {"en": "AI2-THOR xGPU Embodied Home", "zh": "AI2-THOR xGPU 具身家庭环境"},
    "description": {
        "en": "Train a visual ObjectNav policy in interactive kitchens, living rooms, and bedrooms rendered by AI2-THOR.",
        "zh": "在 AI2-THOR 渲染的可交互厨房、客厅和卧室中训练视觉目标导航策略。",
    },
    "badge": "EXPERIMENT 12 · AI2-THOR",
    "device": "xGPU · Vulkan rendering",
    "organization_url": "https://modelscope.cn/organization/walkinglab",
    "project_url": "https://github.com/walkinglabs/hands-on-modern-rl",
    "course_url": "https://walkinglabs.github.io/hands-on-modern-rl/",
    "source_url": "https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment12-ai2thor-embodied/file/view/master/space_runtime.py",
}


def _task(key: str, title: str, zh: str, scene: str, target: str, room: str, room_zh: str, preview: str) -> dict[str, Any]:
    return {
        "key": key, "title": {"en": title, "zh": zh}, "environment": f"AI2-THOR/{scene}",
        "scene": scene, "target": target,
        "description": {"en": f"Navigate through the {room} until the {target.lower()} is close and visible.", "zh": f"在{room_zh}中移动，直到靠近并看见 {target}。"},
        "observation": {"en": "84×84 egocentric RGB + shaped target distance", "zh": "84×84 第一人称 RGB 与目标距离塑形"},
        "action": {"en": "Move, rotate, and look", "zh": "移动、旋转与抬头/低头"},
        "algorithm": "ObjectNav CNN PPO", "preview": preview,
        "budget": (1_024, 100_000, 8_192, 1_024), "learning_rate": (1e-5, 0.001, 0.00025, 1e-5),
        "gamma": (0.8, 1.0, 0.99, 0.005), "epsilon": (0.0, 0.2, 0.01, 0.005), "checkpoints": 6,
    }


TASKS = [
    _task("find-mug", "Find a Mug · Kitchen", "寻找杯子 · 厨房", "FloorPlan1", "Mug", "kitchen", "厨房", "assets/thor-kitchen.svg"),
    _task("find-apple", "Find an Apple · Kitchen", "寻找苹果 · 厨房", "FloorPlan2", "Apple", "kitchen", "厨房", "assets/thor-apple.svg"),
    _task("find-tv", "Find a Television · Living Room", "寻找电视 · 客厅", "FloorPlan201", "Television", "living room", "客厅", "assets/thor-living.svg"),
    _task("find-bed", "Find a Bed · Bedroom", "寻找床 · 卧室", "FloorPlan301", "Bed", "bedroom", "卧室", "assets/thor-bedroom.svg"),
]


def runtime_status() -> str:
    try:
        import ai2thor
        import torch
        device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "waiting for xGPU"
        return f"AI2-THOR {getattr(ai2thor, '__version__', '5.0.0')} · {device} · CLOUD RENDERING"
    except Exception as exc:
        return f"installing AI2-THOR runtime · {type(exc).__name__}"


def _prepare_cache() -> None:
    THOR_CACHE.mkdir(parents=True, exist_ok=True)
    default = Path.home() / ".ai2thor"
    if not default.exists():
        default.symlink_to(THOR_CACHE, target_is_directory=True)


def _safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"Unsafe path in AI2-THOR archive: {member.filename}")
        bundle.extractall(destination)


def _prepare_thor_build() -> Iterator[str]:
    """Download the official build in parallel, verify it, and persist the extraction."""
    import ai2thor.build
    from ai2thor.platform import CloudRendering

    releases = Path.home() / ".ai2thor" / "releases"
    build = ai2thor.build.Build(
        platform=CloudRendering,
        commit_id=ai2thor.build.COMMIT_ID,
        include_private_scenes=False,
        releases_dir=str(releases),
    )
    executable = Path(build.executable_path)
    if executable.exists():
        yield f"AI2-THOR CloudRendering cache ready: {executable}"
        return

    if shutil.which("aria2c") is None:
        raise RuntimeError("aria2c is required to prepare the 797 MB AI2-THOR build efficiently")
    downloads = THOR_CACHE / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    archive = downloads / f"{build.name}.zip"
    command = [
        "aria2c", "--allow-overwrite=true", "--auto-file-renaming=false", "--continue=true",
        "--file-allocation=none", "--max-connection-per-server=16", "--split=16",
        "--min-split-size=1M", "--summary-interval=4", "--console-log-level=notice",
        "--enable-color=false",
        "--dir", str(downloads), "--out", archive.name, build.url,
    ]
    yield "Downloading the official AI2-THOR CloudRendering build with 16 parallel ranges"
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    assert process.stdout is not None
    last_progress = 0.0
    latest_progress = ""
    for line in process.stdout:
        clean = line.strip()
        if clean and ("Download complete" in clean or "NOTICE" in clean):
            yield clean
        elif "[#" in clean:
            latest_progress = clean
            now = time.monotonic()
            if now - last_progress >= 4.0:
                yield latest_progress
                last_progress = now
    if latest_progress and time.monotonic() - last_progress < 4.0:
        yield latest_progress
    if process.wait() != 0:
        raise RuntimeError(f"AI2-THOR parallel download exited with code {process.returncode}")

    expected = subprocess.run(
        ["curl", "--fail", "--location", "--silent", "--show-error", build.sha256_url],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout.strip().split()[0].lower()
    digest = hashlib.sha256()
    with archive.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest().lower() != expected:
        raise RuntimeError("AI2-THOR build failed the official SHA-256 verification")

    yield "Official SHA-256 verified · extracting CloudRendering into persistent storage"
    releases.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{build.name}-", dir=str(THOR_CACHE)) as temporary:
        extracted = Path(temporary) / build.name
        _safe_extract(archive, extracted)
        base_dir = Path(build.base_dir)
        if base_dir.exists():
            raise RuntimeError(f"Incomplete AI2-THOR cache already exists: {base_dir}")
        os.replace(extracted, base_dir)
    executable.chmod(0o755)
    archive.unlink(missing_ok=True)
    yield f"AI2-THOR CloudRendering cache ready: {executable}"


def _start_xvfb() -> subprocess.Popen[str]:
    return subprocess.Popen(["Xvfb", os.environ["DISPLAY"], "-screen", "0", "1280x720x24", "-ac", "+extension", "GLX"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, text=True, start_new_session=True)


def _stop_process(process: subprocess.Popen[Any] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT); process.wait(timeout=6)
    except Exception:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except Exception:
            pass


class ThorObjectNavEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    ACTIONS = ("MoveAhead", "RotateLeft", "RotateRight", "LookUp", "LookDown", "MoveBack")

    def __init__(self, task: dict[str, Any], seed: int, max_steps: int = 256):
        from ai2thor.controller import Controller
        from ai2thor.platform import CloudRendering

        self.task, self.seed, self.max_steps, self.steps = task, seed, max_steps, 0
        self.controller = Controller(scene=task["scene"], platform=CloudRendering, gpu_device=0,
                                     width=320, height=240, fieldOfView=90, quality="Low",
                                     gridSize=0.25, rotateStepDegrees=30, visibilityDistance=1.5)
        self.action_space = gym.spaces.Discrete(len(self.ACTIONS))
        self.observation_space = gym.spaces.Box(0, 255, shape=(84, 84, 3), dtype=np.uint8)
        self.target_id = ""
        self.previous_distance = 0.0
        self.last_event = self.controller.last_event

    def _image(self) -> np.ndarray:
        return np.asarray(Image.fromarray(self.last_event.frame).resize((84, 84), Image.Resampling.BILINEAR), dtype=np.uint8)

    def _target(self) -> dict[str, Any]:
        targets = [obj for obj in self.last_event.metadata["objects"] if obj["objectType"] == self.task["target"]]
        if not targets:
            raise RuntimeError(f"{self.task['target']} is missing from {self.task['scene']}")
        agent = self.last_event.metadata["agent"]["position"]
        return min(targets, key=lambda obj: _distance(agent, obj["position"]))

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self.steps = 0
        self.last_event = self.controller.reset(scene=self.task["scene"])
        target = self._target()
        self.target_id = target["objectId"]
        self.previous_distance = _distance(self.last_event.metadata["agent"]["position"], target["position"])
        return self._image(), {"target": self.target_id, "distance": self.previous_distance}

    def step(self, action: int):
        self.last_event = self.controller.step(action=self.ACTIONS[int(action)], moveMagnitude=0.25)
        self.steps += 1
        objects = {obj["objectId"]: obj for obj in self.last_event.metadata["objects"]}
        target = objects[self.target_id]
        distance = _distance(self.last_event.metadata["agent"]["position"], target["position"])
        success = bool(target.get("visible") and distance <= 1.25)
        reward = (self.previous_distance - distance) * 0.3 - 0.005
        if not self.last_event.metadata.get("lastActionSuccess", True):
            reward -= 0.02
        if success:
            reward += 5.0
        self.previous_distance = distance
        terminated, truncated = success, self.steps >= self.max_steps
        return self._image(), float(reward), terminated, truncated, {"success": success, "distance": distance, "target": self.target_id}

    def render(self) -> np.ndarray:
        return np.asarray(self.last_event.frame, dtype=np.uint8)

    def close(self) -> None:
        self.controller.stop()


def _distance(left: dict[str, float], right: dict[str, float]) -> float:
    return math.sqrt(sum((float(left[axis]) - float(right[axis])) ** 2 for axis in ("x", "y", "z")))


def _make_env(task: dict[str, Any], seed: int):
    return gym.wrappers.RecordEpisodeStatistics(ThorObjectNavEnv(task, seed))


def _record(model: Any, task: dict[str, Any], seed: int) -> tuple[str, float, bool]:
    env = _make_env(task, seed)
    frames: list[np.ndarray] = []
    total, success = 0.0, False
    try:
        observation, _ = env.reset(seed=seed)
        for _ in range(256):
            frames.append(env.render())
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            total += float(reward); success = bool(info.get("success"))
            if terminated or truncated:
                break
    finally:
        env.close()
    path = save_gif(frames, ROOT / "artifacts" / f"{task['key']}-learned-policy.gif", fps=12)
    return path, total, success


def run(key: str, budget: int, learning_rate: float, gamma: float, epsilon: float, seed: int) -> Iterator[dict[str, Any]]:
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback

    if not torch.cuda.is_available():
        raise RuntimeError("This visual ObjectNav experiment requires a scheduled ModelScope xGPU")
    task = next(item for item in TASKS if item["key"] == key)
    _prepare_cache()
    yield {"phase": "initializing", "step": 0, "log": f"AI2-THOR persistent cache: {THOR_CACHE}"}
    for detail in _prepare_thor_build():
        yield {"phase": "initializing", "step": 0, "log": detail}
    yield {"phase": "initializing", "step": 0, "log": f"Preparing {task['scene']} with CloudRendering on {torch.cuda.get_device_name(0)}"}
    xvfb: subprocess.Popen[str] | None = None
    env = None
    try:
        xvfb = _start_xvfb(); time.sleep(1.0)
        env = _make_env(task, int(seed))

        class MetricsCallback(BaseCallback):
            def __init__(self) -> None:
                super().__init__(verbose=0); self.latest: dict[str, Any] = {}
            def _on_step(self) -> bool:
                self.latest = dict(self.logger.name_to_value); return True

        callback = MetricsCallback()
        model = PPO("CnnPolicy", env, learning_rate=float(learning_rate), gamma=float(gamma), ent_coef=float(epsilon),
                    n_steps=256, batch_size=64, n_epochs=4, device="cuda", seed=int(seed), verbose=0)
        x: list[float] = []
        y: list[float] = []
        completed = 0
        chunk = max(1_024, int(budget) // 6)
        while completed < int(budget):
            model.learn(total_timesteps=min(chunk, int(budget) - completed), reset_num_timesteps=False, callback=callback, progress_bar=False)
            completed = min(int(budget), int(model.num_timesteps))
            score = float(callback.latest.get("rollout/ep_rew_mean") or 0.0)
            x.append(float(completed)); y.append(score)
            log = (f"ObjectNav PPO update · step={completed:,}\n"
                   f"scene={task['scene']}  target={task['target']}  device={model.device}\n"
                   f"episode_reward_mean={score:.4f}  entropy_loss={float(callback.latest.get('train/entropy_loss') or 0):.5f}\n"
                   f"policy_gradient_loss={float(callback.latest.get('train/policy_gradient_loss') or 0):.5f}")
            yield {"phase": "training", "step": completed, "score": score, "x": x, "y": y,
                   "detail": f"{completed:,}/{int(budget):,} simulator steps", "metric_detail": "episode reward mean", "log": log}
        artifact_dir = ROOT / "artifacts"; artifact_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(artifact_dir / f"{key}-cnn-ppo"))
        env.close(); env = None
        preview, evaluation, success = _record(model, task, int(seed) + 10_000)
        (artifact_dir / f"{key}-model.json").write_text(json.dumps({"scene": task["scene"], "target": task["target"], "algorithm": "CNN PPO", "budget": int(budget), "seed": int(seed), "evaluation_return": evaluation, "success": success}, indent=2), encoding="utf-8")
        yield {"phase": "complete", "step": completed, "score": evaluation, "x": x, "y": y, "preview": preview,
               "metric_detail": "deterministic ObjectNav return", "log": f"Recorded learned ObjectNav policy · success={success} · {Path(preview).name}"}
    finally:
        if env is not None:
            env.close()
        _stop_process(xvfb)
