from __future__ import annotations

import copy
import json
import os
import signal
import subprocess
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import gymnasium as gym

from sb3_tools import save_gif


ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MINESTUDIO_DIR", "/mnt/workspace/hands-on-modern-rl/minestudio")
os.environ.setdefault("DISPLAY", ":99")
JAVA_CACHE = Path("/mnt/workspace/hands-on-modern-rl/temurin-jre8")
JAVA_URL = "https://api.adoptium.net/v3/binary/latest/8/ga/linux/x64/jre/hotspot/normal/eclipse"

SPACE = {
    "title": {"en": "MineStudio xGPU Minecraft Agent Lab", "zh": "MineStudio xGPU Minecraft 智能体训练场"},
    "description": {
        "en": "Start a real Minecraft simulator, train a compact visual PPO policy, inspect every update, and replay the learned run.",
        "zh": "启动真实 Minecraft 模拟器，训练紧凑的视觉 PPO 策略，查看每次更新，并回放学习后的运行过程。",
    },
    "badge": "EXPERIMENT 10 · MINESTUDIO",
    "device": "xGPU · visual PPO",
    "organization_url": "https://modelscope.cn/organization/walkinglab",
    "project_url": "https://github.com/walkinglabs/hands-on-modern-rl",
    "course_url": "https://walkinglabs.github.io/hands-on-modern-rl/",
    "source_url": "https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment10-minestudio/file/view/master/space_runtime.py",
}


def _task(key: str, title: str, zh: str, description: str, description_zh: str,
          commands: list[str], reward_event: str, objects: list[str], preview: str) -> dict[str, Any]:
    return {
        "key": key, "title": {"en": title, "zh": zh}, "environment": f"MineStudio/{key}",
        "description": {"en": description, "zh": description_zh},
        "observation": {"en": "84×84 RGB first-person frames", "zh": "84×84 RGB 第一人称画面"},
        "action": {"en": "10 reduced keyboard/mouse actions", "zh": "10 个简化键鼠动作"},
        "algorithm": "CNN PPO", "preview": preview, "commands": commands,
        "reward_event": reward_event, "reward_objects": objects,
        "budget": (1_024, 50_000, 8_192, 1_024), "learning_rate": (1e-5, 0.001, 0.00025, 1e-5),
        "gamma": (0.8, 1.0, 0.99, 0.005), "epsilon": (0.0, 0.2, 0.01, 0.005), "checkpoints": 6,
    }


TASKS = [
    _task("mine-dirt", "Mine Dirt · Visual PPO", "挖掘泥土 · 视觉 PPO", "Find the nearby dirt blocks, aim, and mine them.", "找到附近的泥土方块，瞄准并挖掘。", ["/give @s minecraft:stone_shovel", "/execute as @p at @s run fill ~1 ~ ~1 ~4 ~2 ~4 minecraft:dirt"], "mine_block", ["dirt"], "assets/minecraft-mine.svg"),
    _task("collect-wood", "Collect Wood · Visual PPO", "收集木材 · 视觉 PPO", "Approach the nearby oak logs and collect wood with an axe.", "走近附近的橡木原木，并使用斧头收集木材。", ["/give @s minecraft:wooden_axe", "/execute as @p at @s run fill ~1 ~ ~1 ~5 ~8 ~5 minecraft:oak_log"], "mine_block", ["oak_log", "log"], "assets/minecraft-wood.svg"),
    _task("hunt-sheep", "Hunt a Sheep · Visual PPO", "猎取绵羊 · 视觉 PPO", "Track a nearby sheep and attack with a wooden sword.", "追踪附近的绵羊，并使用木剑攻击。", ["/replaceitem entity @s weapon.mainhand minecraft:wooden_sword", "/summon minecraft:sheep ~2 ~ ~", "/give @p minecraft:bread 10"], "kill_entity", ["sheep"], "assets/minecraft-sheep.svg"),
    _task("combat-zombie", "Combat a Zombie · Visual PPO", "对抗僵尸 · 视觉 PPO", "Face one nearby zombie at night and learn a short combat policy.", "在夜间面对附近的一只僵尸，学习一段短程战斗策略。", ["/replaceitem entity @s armor.chest minecraft:diamond_chestplate", "/replaceitem entity @s weapon.mainhand minecraft:diamond_sword", "/summon minecraft:zombie ~3 ~ ~", "/time set night"], "kill_entity", ["zombie"], "assets/minecraft-zombie.svg"),
]


def runtime_status() -> str:
    try:
        import minestudio
        import torch
        device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "waiting for xGPU"
        return f"MineStudio {getattr(minestudio, '__version__', '1.1.6')} · {device} · ENGINE CACHE"
    except Exception as exc:
        return f"installing MineStudio runtime · {type(exc).__name__}"


def _start_xvfb() -> subprocess.Popen[str]:
    return subprocess.Popen(["Xvfb", os.environ["DISPLAY"], "-screen", "0", "1280x720x24", "-ac", "+extension", "GLX"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, text=True, start_new_session=True)


def _ensure_java8() -> Path:
    java_candidates = list(JAVA_CACHE.glob("*/bin/java"))
    if not java_candidates:
        JAVA_CACHE.mkdir(parents=True, exist_ok=True)
        archive, partial = JAVA_CACHE / "temurin8.tar.gz", JAVA_CACHE / "temurin8.tar.gz.part"
        urllib.request.urlretrieve(JAVA_URL, partial)
        partial.replace(archive)
        with tarfile.open(archive, "r:gz") as bundle:
            root = JAVA_CACHE.resolve()
            for member in bundle.getmembers():
                target = (JAVA_CACHE / member.name).resolve()
                if root not in target.parents and target != root:
                    raise RuntimeError(f"Unsafe path in Temurin archive: {member.name}")
            bundle.extractall(JAVA_CACHE)
        java_candidates = list(JAVA_CACHE.glob("*/bin/java"))
    if not java_candidates:
        raise RuntimeError("Temurin JRE 8 was downloaded but java was not found")
    java_home = java_candidates[0].parents[1]
    os.environ["JAVA_HOME"] = str(java_home)
    os.environ["PATH"] = f"{java_home / 'bin'}:{os.environ.get('PATH', '')}"
    return java_candidates[0]


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


class MinecraftDiscreteEnv(gym.Env):
    """A compact Gymnasium surface over MineStudio's full keyboard/mouse action dictionary."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, task: dict[str, Any], seed: int, max_steps: int = 256):
        from minestudio.simulator import MinecraftSim
        from minestudio.simulator.callbacks import CommandsCallback, RewardsCallback
        from minestudio.simulator.entry import check_engine

        check_engine(skip_confirmation=True)
        self.task, self.max_steps, self.steps = task, max_steps, 0
        callbacks = [CommandsCallback(commands=task["commands"]), RewardsCallback([{
            "event": task["reward_event"], "objects": task["reward_objects"], "reward": 1.0,
            "identity": task["key"], "max_reward_times": 8,
        }])]
        self.sim = MinecraftSim(action_type="env", obs_size=(84, 84), render_size=(640, 360), seed=seed,
                                num_empty_frames=10, callbacks=callbacks)
        self.action_space = gym.spaces.Discrete(10)
        self.observation_space = gym.spaces.Box(0, 255, shape=(84, 84, 3), dtype=np.uint8)
        self._last_position: np.ndarray | None = None

    def _position(self, info: dict[str, Any]) -> np.ndarray | None:
        stats = info.get("location_stats") or info.get("location") or {}
        try:
            return np.asarray([stats["xpos"], stats["ypos"], stats["zpos"]], dtype=np.float32)
        except Exception:
            return None

    def _action(self, index: int) -> dict[str, Any]:
        action = copy.deepcopy(self.sim.env.action_space.no_op())
        choices = {
            1: {"forward": 1}, 2: {"back": 1}, 3: {"left": 1}, 4: {"right": 1},
            5: {"jump": 1, "forward": 1}, 6: {"sprint": 1, "forward": 1},
            7: {"attack": 1}, 8: {"attack": 1, "forward": 1},
            9: {"camera": np.asarray([0.0, 12.0], dtype=np.float32)},
        }
        action.update(choices.get(int(index), {}))
        return action

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self.steps = 0
        obs, info = self.sim.reset()
        self._last_position = self._position(info)
        return np.asarray(obs["image"], dtype=np.uint8), info

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.sim.step(self._action(int(action)))
        self.steps += 1
        current = self._position(info)
        if current is not None and self._last_position is not None:
            reward += min(0.02, float(np.linalg.norm(current - self._last_position)) * 0.01)
        self._last_position = current
        truncated = bool(truncated or self.steps >= self.max_steps)
        return np.asarray(obs["image"], dtype=np.uint8), float(reward), bool(terminated), truncated, info

    def render(self) -> np.ndarray:
        return np.asarray(self.sim.render(), dtype=np.uint8)

    def close(self) -> None:
        self.sim.close()


def _make_env(task: dict[str, Any], seed: int):
    import gymnasium as gym
    return gym.wrappers.RecordEpisodeStatistics(MinecraftDiscreteEnv(task, seed))


def _record(model: Any, task: dict[str, Any], seed: int) -> tuple[str, float]:
    env = _make_env(task, seed)
    frames: list[np.ndarray] = []
    total = 0.0
    try:
        observation, _ = env.reset(seed=seed)
        for _ in range(256):
            frames.append(env.render())
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            if terminated or truncated:
                break
    finally:
        env.close()
    path = save_gif(frames, ROOT / "artifacts" / f"{task['key']}-learned-policy.gif", fps=20)
    return path, total


def run(key: str, budget: int, learning_rate: float, gamma: float, epsilon: float, seed: int) -> Iterator[dict[str, Any]]:
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback

    if not torch.cuda.is_available():
        raise RuntimeError("This visual experiment requires a scheduled ModelScope xGPU; CUDA is not currently visible")
    task = next(item for item in TASKS if item["key"] == key)
    xvfb: subprocess.Popen[str] | None = None
    env = None
    yield {"phase": "initializing", "step": 0, "log": f"Preparing persistent MineStudio engine cache at {os.environ['MINESTUDIO_DIR']}"}
    try:
        java = _ensure_java8()
        yield {"phase": "initializing", "step": 0, "log": f"Temurin Java 8 ready: {java}\nStarting the Minecraft renderer"}
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
            model.learn(total_timesteps=min(chunk, int(budget) - completed), reset_num_timesteps=False,
                        callback=callback, progress_bar=False)
            completed = min(int(budget), int(model.num_timesteps))
            score = float(callback.latest.get("rollout/ep_rew_mean") or 0.0)
            x.append(float(completed)); y.append(score)
            log = (f"Minecraft PPO update · step={completed:,}\n"
                   f"device={model.device}  fps={callback.latest.get('time/fps', '—')}\n"
                   f"episode_reward_mean={score:.4f}  value_loss={float(callback.latest.get('train/value_loss') or 0):.5f}\n"
                   f"policy_gradient_loss={float(callback.latest.get('train/policy_gradient_loss') or 0):.5f}")
            yield {"phase": "training", "step": completed, "score": score, "x": x, "y": y,
                   "detail": f"{completed:,}/{int(budget):,} Minecraft steps", "metric_detail": "episode reward mean", "log": log}
        artifact_dir = ROOT / "artifacts"; artifact_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(artifact_dir / f"{key}-cnn-ppo"))
        env.close(); env = None
        preview, evaluation = _record(model, task, int(seed) + 10_000)
        (artifact_dir / f"{key}-model.json").write_text(json.dumps({"environment": task["environment"], "algorithm": "CNN PPO", "budget": int(budget), "seed": int(seed), "evaluation_return": evaluation}, indent=2), encoding="utf-8")
        yield {"phase": "complete", "step": completed, "score": evaluation, "x": x, "y": y,
               "preview": preview, "metric_detail": "deterministic replay return",
               "log": f"Saved CNN PPO and recorded the learned Minecraft policy: {Path(preview).name}"}
    finally:
        if env is not None:
            env.close()
        _stop_process(xvfb)
