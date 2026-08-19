from __future__ import annotations

from pathlib import Path
import time

import numpy as np

from sb3_tools import save_gif, train_sb3


ROOT = Path(__file__).resolve().parent

SPACE = {
    "title": {"en": "Atari CPU Training Arcade", "zh": "Atari CPU 在线训练街机厅"},
    "description": {
        "en": "Train DQN agents from Atari pixels, inspect checkpoint rewards, and render this run's learned policy inside the original ALE emulator.",
        "zh": "让 DQN 从 Atari 像素画面中学习，观察检查点评估，并在 ALE 模拟器中生成本次策略回放。",
    },
    "badge": "EXPERIMENT 03 · ARCADE",
    "training_guide": {
        "success": {"en": "The evaluation reward should rise above early checkpoints, and the replay should sustain useful play or improve the game score. Training complete only confirms the run ended normally.", "zh": "评估奖励应高于早期检查点，回放中应能持续做出有效动作或提高游戏得分；“训练完成”只表示运行正常结束。"},
        "preview": {"en": "The first clip shows the selected Atari game. The completed run replaces it with a replay rendered by this run's learned DQN policy.", "zh": "初始画面展示所选 Atari 游戏；训练完成后会替换为本次 DQN 策略在模拟器中生成的回放。"},
        "time": {"en": "Default CPU runs usually take 1–5 minutes; harder games need a larger budget to show stable improvement.", "zh": "默认 CPU 训练通常需要 1–5 分钟；难度更高的游戏需要更大预算才能看到稳定提升。"},
    },
    "course_url": "https://walkinglabs.github.io/hands-on-modern-rl/chapter07_dqn/dqn-family",
    "source_url": "https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment03-atari/file/view/master/space_runtime.py",
    "notebook_url": "https://modelscope.cn/notebook/share/github/walkinglabs/hands-on-modern-rl/blob/main/"
    "code/online-experiments/hands-on-modern-rl-experiment03-atari.ipynb",
}


def task(key, title, environment, description, action, preview, budget=(5_000, 300_000, 30_000, 5_000)):
    return {
        "key": key,
        "title": {"en": title, "zh": title},
        "environment": environment,
        "description": description,
        "observation": {"en": "84×84 grayscale frame stack", "zh": "84×84 灰度帧堆叠"},
        "action": action,
        "algorithm": "DQN",
        "policy": "CnnPolicy",
        "preview": preview,
        "budget": budget,
        "learning_rate": (1e-5, 0.001, 0.0001, 1e-5),
        "gamma": (0.8, 1.0, 0.99, 0.005),
        "epsilon": (0.01, 1.0, 0.6, 0.01),
        "checkpoints": 6,
    }


TASKS = [
    task("pong", "Pong", "ALE/Pong-v5", {"en": "Track the ball and move the paddle to outscore the opponent.", "zh": "跟踪球的位置并移动球拍，以更高比分击败对手。"}, {"en": "Paddle and fire controls", "zh": "球拍移动与发球"}, "assets/pong.gif"),
    task("breakout", "Breakout", "ALE/Breakout-v5", {"en": "Bounce the ball, clear bricks, and preserve each life.", "zh": "反弹小球、清除砖块并尽量保住生命。"}, {"en": "Paddle and fire controls", "zh": "球拍移动与发球"}, "assets/breakout.gif"),
    task("space-invaders", "Space Invaders", "ALE/SpaceInvaders-v5", {"en": "Move, shoot invading rows, and avoid incoming fire.", "zh": "移动并射击入侵队列，同时躲避敌方火力。"}, {"en": "Move / fire", "zh": "移动、射击"}, "assets/space-invaders.gif"),
    task("freeway", "Freeway", "ALE/Freeway-v5", {"en": "Time vertical movements to cross lanes of traffic safely.", "zh": "掌握上下移动的时机，安全穿过多条车道。"}, {"en": "Up / down", "zh": "向上、向下"}, "assets/freeway.png", (2_000, 200_000, 20_000, 2_000)),
    task("seaquest", "Seaquest", "ALE/Seaquest-v5", {"en": "Rescue divers while managing oxygen, enemies, and ammunition.", "zh": "在管理氧气、敌人和弹药的同时营救潜水员。"}, {"en": "Move / fire", "zh": "移动、射击"}, "assets/seaquest.gif"),
    task("qbert", "Q*bert", "ALE/Qbert-v5", {"en": "Plan diagonal jumps to recolor the pyramid without colliding with enemies.", "zh": "规划斜向跳跃改变金字塔颜色，并避开敌人。"}, {"en": "Four diagonal jumps", "zh": "四个斜向跳跃动作"}, "assets/qbert.gif"),
    task("beam-rider", "Beam Rider", "ALE/BeamRider-v5", {"en": "Control horizontal movement and shooting in a fast scrolling arena.", "zh": "在快速滚动的竞技场中控制横向移动和射击。"}, {"en": "Move / fire", "zh": "移动、射击"}, "assets/beam-rider.gif"),
    task("enduro", "Enduro", "ALE/Enduro-v5", {"en": "Steer and accelerate through traffic over a long racing horizon.", "zh": "在长时间赛车过程中控制方向、加速并穿过车流。"}, {"en": "Steer / accelerate / brake", "zh": "转向、加速、刹车"}, "assets/enduro.gif"),
]


def runtime_status():
    try:
        import ale_py
        import gymnasium as gym

        gym.register_envs(ale_py)
        env = gym.make("ALE/Pong-v5", render_mode="rgb_array", frameskip=4)
        env.reset(seed=0)
        env.close()
        return f"ALE {ale_py.__version__} · ROM READY"
    except Exception as exc:
        return f"startup check pending · {type(exc).__name__}"


def _make_vec_env(environment: str, seed: int):
    import ale_py
    import gymnasium as gym
    from stable_baselines3.common.atari_wrappers import AtariWrapper
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

    gym.register_envs(ale_py)

    def factory():
        base = gym.make(environment, render_mode="rgb_array", frameskip=1, repeat_action_probability=0.0, full_action_space=False)
        return AtariWrapper(base, frame_skip=4, screen_size=84, terminal_on_life_loss=False, clip_reward=False)

    env = DummyVecEnv([factory])
    env.seed(seed)
    return VecFrameStack(env, n_stack=4)


def _record(model, env, artifacts: Path, seed: int, task, output_path: Path | None = None):
    env.seed(seed)
    observation = env.reset()
    frames: list[np.ndarray] = []
    for step in range(4_000):
        frame = env.render(mode="rgb_array")
        if frame is not None and (step % 2 == 0 or step < 20):
            frames.append(np.asarray(frame))
        action, _ = model.predict(observation, deterministic=True)
        observation, _, done, _ = env.step(action)
        if bool(np.asarray(done).any()):
            break
        if len(frames) >= 500:
            break
    env.close()
    return save_gif(frames, output_path or artifacts / f"{task['key']}-learned-policy.gif", fps=20)


def render_preview(key: str, seed: int):
    """Run the latest saved model for ``key`` with an independent rollout seed."""
    task = next(item for item in TASKS if item["key"] == key)
    artifacts = ROOT / "artifacts"
    model_path = artifacts / f"{key}-model.zip"
    if not model_path.is_file():
        raise FileNotFoundError(f"No trained model is saved for {task['title']['en']}. Start training first.")

    from stable_baselines3 import DQN

    rollout_seed = max(0, min(int(seed), 2**32 - 1))
    model = DQN.load(str(model_path), device="cpu")
    environment = _make_vec_env(task["environment"], rollout_seed)
    # A new path forces Gradio/the browser to load the newly rendered GIF rather
    # than reusing a cached file with the same name.
    version = model_path.stat().st_mtime_ns
    output = artifacts / f"{key}-rollout-m{version}-s{rollout_seed}-{time.time_ns()}.gif"
    preview = _record(model, environment, artifacts, rollout_seed, task, output)

    old_replays = sorted(artifacts.glob(f"{key}-rollout-m*-s*.gif"), key=lambda path: path.stat().st_mtime, reverse=True)
    for stale in old_replays[12:]:
        try:
            stale.unlink()
        except OSError:
            pass
    return {
        "preview": preview,
        "seed": rollout_seed,
        "model": model_path.name,
        "model_version": str(version),
        "detail": f"{task['title']['en']} · {model_path.name} · deterministic rollout seed {rollout_seed}",
    }


def run(key: str, budget: int, learning_rate: float, gamma: float, epsilon: float, seed: int):
    task = next(item for item in TASKS if item["key"] == key)
    environment = task["environment"]
    yield from train_sb3(
        root=ROOT,
        task=task,
        make_train_env=lambda: _make_vec_env(environment, seed),
        make_eval_env=lambda: _make_vec_env(environment, seed + 1_000),
        make_record_env=lambda: _make_vec_env(environment, seed),
        budget=budget,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon=epsilon,
        seed=seed,
        record_episode=lambda model, env, artifacts, record_seed: _record(model, env, artifacts, record_seed, task),
    )
