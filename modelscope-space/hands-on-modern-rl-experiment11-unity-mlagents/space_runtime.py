from __future__ import annotations

import hashlib
import os
import queue
import re
import shutil
import signal
import subprocess
import threading
import time
import zipfile
from pathlib import Path
from typing import Any, Iterator

import yaml
import imageio_ffmpeg
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
UNITY_DATASET_URL = "https://modelscope.cn/datasets/walkinglab/hands-on-modern-rl-unity-environments"
UNITY_DATASET_RESOLVE = f"{UNITY_DATASET_URL}/resolve/master"
UNITY_BUNDLE_URL = f"{UNITY_DATASET_RESOLVE}/linux/ml-agents-1.1.0/Startup.zip"
HUGGY_BUNDLE_URL = f"{UNITY_DATASET_RESOLVE}/linux/huggy/Huggy.zip"
UNITY_BUNDLE_SHA256 = "80e2322215fb7ff5c192e34bd67d63edc80d8cf24e66f8af858010b84a250a5d"
HUGGY_BUNDLE_SHA256 = "6b35692b1d867f74fdf8987a911700e06ff24d40b95b935460ccd175e3712d28"
UNITY_CACHE = Path(os.environ.get("UNITY_MLAGENTS_CACHE", "/mnt/workspace/hands-on-modern-rl/unity-mlagents-1.1.0"))

SPACE = {
    "title": {"en": "Unity ML-Agents xGPU Arena", "zh": "Unity ML-Agents xGPU 训练场"},
    "description": {
        "en": "Train PPO inside ready-to-run Unity ML-Agents Linux scenes, including Huggy the dog, and replay the rendered run.",
        "zh": "在可直接运行的 Unity ML-Agents Linux 场景中训练 PPO，包括小狗 Huggy，并回放本次训练画面。",
    },
    "badge": "EXPERIMENT 11 · UNITY",
    "device": "xGPU",
    "organization_url": "https://modelscope.cn/organization/walkinglab",
    "dataset_url": UNITY_DATASET_URL,
    "project_url": "https://github.com/walkinglabs/hands-on-modern-rl",
    "course_url": "https://walkinglabs.github.io/hands-on-modern-rl/",
    "source_url": "https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment11-unity-mlagents/file/view/master/space_runtime.py",
    "notebook_url": "https://modelscope.cn/notebook/share/github/walkinglabs/hands-on-modern-rl/blob/main/"
    "code/online-experiments/hands-on-modern-rl-experiment11-unity-mlagents.ipynb",
}


def _task(key: str, title: str, title_zh: str, scene: str, behavior: str, description: str,
          description_zh: str, observation: str, action: str, preview: str,
          budget: tuple[int, int, int, int], trainer: dict[str, Any], *,
          bundle_url: str = UNITY_BUNDLE_URL, cache_subdir: str | None = None,
          bundle_sha256: str = UNITY_BUNDLE_SHA256,
          executable: str | None = None, env_args: list[str] | None = None,
          reference_url: str | None = None) -> dict[str, Any]:
    task = {
        "key": key, "title": {"en": title, "zh": title_zh}, "environment": f"Unity/{scene}",
        "scene": scene, "behavior": behavior,
        "description": {"en": description, "zh": description_zh},
        "observation": {"en": observation, "zh": observation},
        "action": {"en": action, "zh": action}, "algorithm": "Unity PPO", "preview": preview,
        "budget": budget, "learning_rate": (1e-5, 0.001, 0.0003, 1e-5),
        "gamma": (0.8, 1.0, float(trainer["reward_signals"]["extrinsic"]["gamma"]), 0.005),
        "epsilon": (0.05, 0.35, float(trainer["hyperparameters"]["epsilon"]), 0.01),
        "checkpoints": 6, "trainer": trainer,
        "bundle_url": bundle_url,
        "bundle_sha256": bundle_sha256,
    }
    if cache_subdir:
        task["cache_subdir"] = cache_subdir
    if executable:
        task["executable"] = executable
    if env_args is not None:
        task["env_args"] = env_args
    if reference_url:
        task["reference_url"] = reference_url
    return task


TASKS = [
    _task("unity-huggy", "Huggy · Fetch the Stick", "Huggy · 小狗捡树枝", "Huggy", "Huggy",
          "Coordinate four articulated legs, run toward the randomly placed stick, and reach it without spinning out.",
          "协调四条腿的关节运动，跑向随机出现的树枝，并在不过度旋转的情况下抵达目标。",
          "Stick position, relative target direction, body state, and leg orientation",
          "Continuous joint-motor rotations for all four legs", "assets/unity-huggy.webp",
          (20_000, 2_000_000, 100_000, 20_000),
          {"trainer_type": "ppo", "hyperparameters": {"batch_size": 2048, "buffer_size": 20480, "learning_rate": 0.0003, "beta": 0.005, "epsilon": 0.2, "lambd": 0.95, "num_epoch": 3, "learning_rate_schedule": "linear"}, "network_settings": {"normalize": True, "hidden_units": 512, "num_layers": 3, "vis_encode_type": "simple"}, "reward_signals": {"extrinsic": {"gamma": 0.995, "strength": 1.0}}, "keep_checkpoints": 3, "time_horizon": 1000},
          bundle_url=HUGGY_BUNDLE_URL, bundle_sha256=HUGGY_BUNDLE_SHA256,
          cache_subdir="huggy", executable="Huggy/Huggy.x86_64", env_args=[],
          reference_url="https://huggingface.co/learn/deep-rl-course/unitbonus1/train"),
    _task("unity-basic", "Basic · Discrete PPO", "Basic · 离散 PPO", "Basic", "Basic",
          "Match the target value with a short sequence of discrete decisions.", "通过一小段离散决策使数值匹配目标。",
          "Vector target and current state", "Discrete left / stay / right", "assets/unity-basic.webp", (2_000, 120_000, 20_000, 2_000),
          {"trainer_type": "ppo", "hyperparameters": {"batch_size": 32, "buffer_size": 256, "learning_rate": 0.0003, "beta": 0.005, "epsilon": 0.2, "lambd": 0.95, "num_epoch": 3, "learning_rate_schedule": "linear"}, "network_settings": {"normalize": False, "hidden_units": 20, "num_layers": 1, "vis_encode_type": "simple"}, "reward_signals": {"extrinsic": {"gamma": 0.9, "strength": 1.0}}, "keep_checkpoints": 3, "time_horizon": 3}),
    _task("unity-3dball", "3D Ball · Continuous PPO", "3D Ball · 连续 PPO", "3DBall", "3DBall",
          "Tilt a platform in two axes and keep the ball from falling.", "控制平台在两个方向倾斜，使小球不掉落。",
          "Ball position/velocity and platform rotation", "Continuous platform tilt", "assets/unity-3dball.webp", (8_000, 500_000, 60_000, 4_000),
          {"trainer_type": "ppo", "hyperparameters": {"batch_size": 64, "buffer_size": 12000, "learning_rate": 0.0003, "beta": 0.001, "epsilon": 0.2, "lambd": 0.99, "num_epoch": 3, "learning_rate_schedule": "linear"}, "network_settings": {"normalize": True, "hidden_units": 128, "num_layers": 2, "vis_encode_type": "simple"}, "reward_signals": {"extrinsic": {"gamma": 0.99, "strength": 1.0}}, "keep_checkpoints": 3, "time_horizon": 1000}),
    _task("unity-food", "Food Collector · Visual PPO", "Food Collector · 视觉 PPO", "FoodCollector", "GridFoodCollector",
          "Collect green food while avoiding red food and competing agents.", "收集绿色食物，同时避开红色食物与其他智能体。",
          "Ray sensors and local visual state", "Move, rotate, and fire", "assets/unity-food.webp", (10_000, 500_000, 80_000, 5_000),
          {"trainer_type": "ppo", "hyperparameters": {"batch_size": 1024, "buffer_size": 10240, "learning_rate": 0.0003, "beta": 0.005, "epsilon": 0.2, "lambd": 0.95, "num_epoch": 3, "learning_rate_schedule": "linear"}, "network_settings": {"normalize": False, "hidden_units": 256, "num_layers": 1, "vis_encode_type": "simple"}, "reward_signals": {"extrinsic": {"gamma": 0.99, "strength": 1.0}}, "keep_checkpoints": 3, "time_horizon": 64}),
    _task("unity-walker", "Walker · Locomotion PPO", "Walker · 运动控制 PPO", "Walker", "Walker",
          "Coordinate a many-jointed body to move toward the target direction.", "协调多关节身体，沿目标方向稳定行走。",
          "Joint rotations, velocities, contacts, and target direction", "Continuous joint targets", "assets/unity-walker.webp", (20_000, 1_000_000, 120_000, 10_000),
          {"trainer_type": "ppo", "hyperparameters": {"batch_size": 2048, "buffer_size": 20480, "learning_rate": 0.0003, "beta": 0.005, "epsilon": 0.2, "lambd": 0.95, "num_epoch": 3, "learning_rate_schedule": "linear"}, "network_settings": {"normalize": True, "hidden_units": 256, "num_layers": 3, "vis_encode_type": "simple"}, "reward_signals": {"extrinsic": {"gamma": 0.995, "strength": 1.0}}, "keep_checkpoints": 3, "time_horizon": 1000}),
]


def runtime_status() -> str:
    try:
        from importlib.metadata import version

        import torch
        accelerator = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU fallback"
        renderer = "LIVE UNITY RENDER" if shutil.which("Xvfb") and shutil.which("ffmpeg") else "RENDERER MISSING"
        return f"ML-Agents {version('mlagents')} · {accelerator} · {renderer}"
    except Exception as exc:
        return f"installing ML-Agents runtime · {type(exc).__name__}"


def _find_unity_executable(cache: Path, relative_path: str | None) -> Path | None:
    if relative_path:
        exact = cache / relative_path
        if exact.is_file():
            return exact
    candidates = [path for path in cache.rglob("Startup.x86_64") if path.is_file()]
    if not candidates:
        candidates = [path for path in cache.rglob("*.x86_64") if path.is_file()]
    if candidates:
        candidates[0].chmod(0o755)
        return candidates[0]
    return None


def _verify_bundle(archive: Path, expected_sha256: str) -> None:
    digest = hashlib.sha256()
    with archive.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha256:
        raise RuntimeError(f"Unity scene bundle checksum failed: {archive.name}")


def _ensure_unity_bundle(task: dict[str, Any]) -> Path:
    cache = UNITY_CACHE / task["cache_subdir"] if task.get("cache_subdir") else UNITY_CACHE
    relative_path = task.get("executable")
    selected = _find_unity_executable(cache, relative_path)
    if selected:
        selected.chmod(0o755)
        return selected
    cache.mkdir(parents=True, exist_ok=True)
    bundle_url = str(task.get("bundle_url", UNITY_BUNDLE_URL))
    archive_name = bundle_url.rsplit("/", 1)[-1].split("?", 1)[0] or "UnityEnvironment.zip"
    bundled_archive = ROOT / "bundles" / archive_name
    archive, partial = cache / archive_name, cache / f"{archive_name}.part"
    if bundled_archive.is_file():
        archive = bundled_archive
    else:
        aria2 = shutil.which("aria2c")
        curl = shutil.which("curl")
        if aria2:
            command = [
                "aria2c", "--allow-overwrite=true", "--auto-file-renaming=false", "--continue=true",
                "--file-allocation=none", "--max-connection-per-server=16", "--split=16",
                "--min-split-size=1M", "--console-log-level=warn", "--enable-color=false",
                "--dir", str(partial.parent), "--out", partial.name, bundle_url,
            ]
        elif curl:
            command = [
                "curl", "--location", "--fail", "--retry", "5", "--retry-all-errors",
                "--connect-timeout", "20", "--continue-at", "-", "--output", str(partial),
                bundle_url,
            ]
        else:
            raise RuntimeError("The Unity scene bundle requires aria2c or curl")
        subprocess.run(command, check=True, timeout=1800)
        partial.replace(archive)
    try:
        _verify_bundle(archive, str(task["bundle_sha256"]))
    except RuntimeError:
        if archive != bundled_archive:
            archive.unlink(missing_ok=True)
        raise
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(cache)
    selected = _find_unity_executable(cache, relative_path)
    if not selected:
        raise RuntimeError(f"The Unity bundle for {task['scene']} did not contain the expected Linux executable")
    selected.chmod(0o755)
    return selected


def _scaled_config(task: dict[str, Any], budget: int, learning_rate: float, gamma: float,
                   epsilon: float, seed: int, executable: Path, run_id: str,
                   graphics_available: bool) -> Path:
    import torch

    trainer = yaml.safe_load(yaml.safe_dump(task["trainer"]))
    hyper = trainer["hyperparameters"]
    hyper["learning_rate"], hyper["epsilon"] = learning_rate, epsilon
    trainer["reward_signals"]["extrinsic"]["gamma"] = gamma
    min_buffer = max(256, int(hyper["batch_size"]) * 4)
    hyper["buffer_size"] = max(min_buffer, min(int(hyper["buffer_size"]), max(min_buffer, budget // 4)))
    hyper["batch_size"] = max(16, min(int(hyper["batch_size"]), int(hyper["buffer_size"]) // 4))
    trainer.update(max_steps=budget, summary_freq=max(200, budget // 8), checkpoint_interval=max(1_000, budget))
    env_args = task.get("env_args")
    if env_args is None:
        env_args = ["--mlagents-scene-name", f"Assets/ML-Agents/Examples/{task['scene']}/Scenes/{task['scene']}.unity"]
    env_args = list(env_args) + [
        "-force-glcore", "-screen-fullscreen", "0",
        "-screen-width", "960", "-screen-height", "540",
    ]
    config = {
        "behaviors": {task["behavior"]: trainer},
        "env_settings": {"env_path": str(executable), "env_args": env_args, "num_envs": 1, "seed": seed, "timeout_wait": 180},
        "engine_settings": {"width": 960, "height": 540, "quality_level": 2, "time_scale": 20, "target_frame_rate": -1, "capture_frame_rate": 60, "no_graphics": not graphics_available},
        "checkpoint_settings": {"run_id": run_id, "results_dir": str(ARTIFACTS / "unity-results"), "force": True},
        "torch_settings": {"device": "cuda" if torch.cuda.is_available() else "cpu"},
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS / f"{run_id}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _ffmpeg_executable() -> str:
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    bundled_ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    if bundled_ffmpeg and Path(bundled_ffmpeg).is_file():
        return bundled_ffmpeg
    raise RuntimeError("ffmpeg is required to record the Unity window")


def _start_xvfb() -> subprocess.Popen[str]:
    display = os.environ.get("UNITY_DISPLAY", ":99")
    os.environ["DISPLAY"] = display
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "4.5")
    xvfb = shutil.which("Xvfb")
    if xvfb is None:
        raise RuntimeError(
            "Xvfb is missing. This Studio must use the Docker runtime that installs the Unity rendering stack."
        )
    process = subprocess.Popen(
        [
            xvfb, display, "-screen", "0", "960x540x24", "-ac", "-nolisten", "tcp",
            "+extension", "GLX", "+render", "-noreset",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    socket_path = Path("/tmp/.X11-unix") / f"X{display.lstrip(':').split('.', 1)[0]}"
    deadline = time.monotonic() + 12.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            output = process.stdout.read().strip() if process.stdout else ""
            raise RuntimeError(f"Xvfb exited before Unity started: {output[-500:]}")
        if socket_path.exists():
            return process
        time.sleep(0.15)
    _stop_process(process)
    raise RuntimeError(f"Xvfb did not create display {display} within 12 seconds")


def _start_capture(target: Path, frames_dir: Path) -> subprocess.Popen[str]:
    ffmpeg = _ffmpeg_executable()
    frames_dir.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen([
        ffmpeg, "-y", "-loglevel", "error", "-f", "x11grab", "-framerate", "12",
        "-video_size", "960x540", "-i", os.environ["DISPLAY"],
        "-filter_complex", "[0:v]split=2[record][preview];[preview]fps=0.5,scale=640:-1:flags=lanczos[live]",
        "-map", "[record]", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(target),
        "-map", "[live]", "-q:v", "5", str(frames_dir / "frame-%06d.jpg"),
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True, start_new_session=True)


def _latest_preview_frame(frames_dir: Path) -> np.ndarray | None:
    candidates = sorted(frames_dir.glob("frame-*.jpg"))
    if not candidates:
        return None
    try:
        with Image.open(candidates[-1]) as image:
            frame = np.asarray(image.convert("RGB"))
        # Xvfb is black before the Unity window is mapped. Keep the task card
        # visible until a real rendered frame is available.
        return frame if float(frame.mean()) > 2.0 else None
    except (OSError, ValueError):
        return None


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
    subprocess.run([_ffmpeg_executable(), "-y", "-loglevel", "error", "-sseof", "-10", "-i", str(video), "-vf", "fps=12,scale=640:-1:flags=lanczos", str(output)], check=True, timeout=90)
    if not output.exists() or output.stat().st_size < 1_000:
        raise RuntimeError("Unity completed, but the replay capture was empty")
    with Image.open(output) as replay:
        frame_count = int(getattr(replay, "n_frames", 1))
        if frame_count < 12:
            raise RuntimeError(f"Unity replay contains only {frame_count} frame(s)")
        sample_indices = sorted({0, frame_count // 2, frame_count - 1})
        samples: list[np.ndarray] = []
        for frame_index in sample_indices:
            replay.seek(frame_index)
            samples.append(np.asarray(replay.convert("RGB"), dtype=np.float32))
        if max(float(frame.mean()) for frame in samples) <= 2.0:
            raise RuntimeError("Unity replay is black; the rendered window was not captured")
        if max(float(np.mean(np.abs(frame - samples[0]))) for frame in samples[1:]) < 0.35:
            raise RuntimeError("Unity replay is static; the rendered environment did not advance")
    return str(output)


def run(key: str, budget: int, learning_rate: float, gamma: float, epsilon: float, seed: int) -> Iterator[dict[str, Any]]:
    task = next(item for item in TASKS if item["key"] == key)
    run_id = f"{key}-{int(time.time())}"
    download_note = " · first run downloads a resumable 39 MB scene" if key == "unity-huggy" else ""
    yield {"phase": "initializing", "step": 0, "log": f"Checking the Unity ML-Agents 1.1.0 Linux environment cache{download_note}"}
    executable = _ensure_unity_bundle(task)
    yield {"phase": "initializing", "step": 0, "log": f"Unity executable ready: {executable}\nScene: {task['scene']} · behavior: {task['behavior']}"}
    xvfb_path = shutil.which("Xvfb")
    ffmpeg_path = _ffmpeg_executable()
    if not xvfb_path:
        raise RuntimeError(
            "Real Unity rendering is unavailable because Xvfb is missing. Redeploy this Studio with its Docker SDK."
        )
    yield {
        "phase": "initializing", "step": 0,
        "log": f"Replay mode: real Unity window · Xvfb={xvfb_path} · ffmpeg={ffmpeg_path}",
    }
    config = _scaled_config(task, int(budget), float(learning_rate), float(gamma), float(epsilon), int(seed), executable, run_id, True)
    video, replay = ARTIFACTS / f"{run_id}-training.mp4", ARTIFACTS / f"{key}-learned-policy.gif"
    frames_dir = ARTIFACTS / f"{run_id}-live"
    xvfb = capture = trainer = None
    x: list[float] = []
    y: list[float] = []
    last_step = 0
    number = r"-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    step_re = re.compile(r"Step:\s*([0-9,]+)", re.IGNORECASE)
    score_re = re.compile(rf"Step:\s*([0-9,]+).*?Mean Reward:\s*({number})", re.IGNORECASE)
    ansi_re = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
    try:
        xvfb = _start_xvfb()
        capture = _start_capture(video, frames_dir)
        trainer = subprocess.Popen(["mlagents-learn", str(config)], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env={**os.environ, "PYTHONUNBUFFERED": "1"}, start_new_session=True)
        assert trainer.stdout is not None
        output_queue: queue.Queue[str | None] = queue.Queue()

        def read_trainer_output() -> None:
            assert trainer is not None and trainer.stdout is not None
            for output_line in trainer.stdout:
                output_queue.put(output_line)
            output_queue.put(None)

        threading.Thread(target=read_trainer_output, daemon=True).start()
        pending: list[str] = []
        last_emit = time.monotonic()
        last_preview_emit = 0.0
        render_deadline = time.monotonic() + 45.0
        rendered_frame_seen = False
        stream_finished = False
        while not stream_finished:
            step_match = match = None
            try:
                line = output_queue.get(timeout=0.5)
            except queue.Empty:
                line = ""
            if line is None:
                stream_finished = True
            elif line:
                clean = ansi_re.sub("", line).rstrip()
                # ML-Agents prints a large Unicode logo one row at a time. It
                # has no diagnostic content and would force many UI redraws.
                if clean and re.search(r"[A-Za-z0-9]{3}", clean):
                    pending.append(clean)
                    step_match = step_re.search(clean)
                    if step_match:
                        last_step = int(step_match.group(1).replace(",", ""))
                    match = score_re.search(clean)
                    if match:
                        last_step, score = int(match.group(1).replace(",", "")), float(match.group(2))
                        x.append(float(last_step))
                        y.append(score)

            now = time.monotonic()
            if capture is not None and capture.poll() is not None:
                raise RuntimeError("ffmpeg stopped before it captured the Unity window")
            live_frame = None
            if now - last_preview_emit >= 2.0:
                live_frame = _latest_preview_frame(frames_dir)
                last_preview_emit = now
                rendered_frame_seen = rendered_frame_seen or live_frame is not None
            if not rendered_frame_seen and now >= render_deadline:
                raise RuntimeError(
                    "The Unity window did not produce a visible frame within 45 seconds. "
                    "Training was stopped instead of returning a fake replay."
                )
            should_emit = bool(
                step_match or match or live_frame is not None
                or (pending and now - last_emit >= 0.8)
                or now - last_emit >= 2.0
            )
            if should_emit:
                event = {
                    "phase": "training", "step": last_step, "x": x, "y": y,
                    "detail": f"{last_step:,}/{int(budget):,} Unity steps",
                    "log": "\n".join(pending),
                }
                if match:
                    event.update(score=score, metric_detail="Unity trainer mean reward")
                if live_frame is not None:
                    event["preview"] = live_frame
                yield event
                pending.clear()
                last_emit = now
        if pending:
            yield {"phase": "training", "step": last_step, "x": x, "y": y, "detail": f"{last_step:,}/{int(budget):,} Unity steps", "log": "\n".join(pending)}
        if trainer.wait() != 0:
            raise RuntimeError(f"Unity ML-Agents trainer exited with code {trainer.returncode}")
    finally:
        _stop_process(trainer)
        _stop_process(capture)
        _stop_process(xvfb)
    if not rendered_frame_seen or not video.exists() or video.stat().st_size <= 1_000:
        raise RuntimeError("Unity training completed without a valid rendered recording")
    preview = _make_gif(video, replay)
    yield {
        "phase": "complete", "step": int(budget), "score": y[-1] if y else None,
        "x": x, "y": y, "preview": preview,
        "log": f"Unity PPO complete · recorded a real animated Unity replay: {Path(preview).name}",
    }
