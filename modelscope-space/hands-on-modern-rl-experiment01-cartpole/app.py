"""ModelScope Studio: train PPO on CartPole from the browser."""

from __future__ import annotations

import base64
import contextlib
import html
import io
import os
import re
import time
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import gymnasium as gym
import gradio as gr
import imageio.v2 as imageio
import matplotlib
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)
LOGO_PATH = Path(__file__).parent / "assets" / "readmelogo.png"
LOGO_DATA_URI = f"data:image/png;base64,{base64.b64encode(LOGO_PATH.read_bytes()).decode()}"
SEED = 42
PROJECT_URL = "https://github.com/walkinglabs/hands-on-modern-rl"
COURSE_URL = "https://walkinglabs.github.io/hands-on-modern-rl/"
CHAPTER_URL = f"{COURSE_URL}chapter01_cartpole/training"
NOTEBOOK_GITHUB_PATH = "walkinglabs/hands-on-modern-rl/blob/main/notebooks/cartpole-ppo.ipynb"
MODELSCOPE_NOTEBOOK_URL = f"https://modelscope.cn/notebook/share/github/{NOTEBOOK_GITHUB_PATH}"
SCRIPT_URL = (
    "https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment01-cartpole/"
    "file/view/master/train.py"
)
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


UI_TEXT = {
    "中文": {
        "course": "《动手学现代强化学习》· 第 1 章配套",
        "title": "CartPole 在线训练实验",
        "description": "使用 PPO 从零训练倒立摆策略。启动后可以实时查看奖励曲线、PPO 输出和评估记录，训练结束后下载模型并播放策略动画。全程使用 CPU。",
        "chapter": "阅读配套章节",
        "notebook": "Notebook",
        "script": "训练脚本",
        "project": "GitHub 项目",
        "environment": "环境",
        "algorithm": "算法",
        "device": "设备",
        "threshold": "解决阈值",
        "max_score": "满分",
        "settings_title": "训练设置",
        "settings_copy": "选择总交互步数。系统每训练 2,000 步评估一次策略。",
        "steps_label": "训练步数",
        "steps_info": "建议首次使用 30,000 步",
        "start": "开始训练",
        "run_status": "训练状态",
        "idle": "等待开始",
        "idle_detail": "设置训练步数后启动实验",
        "running": "训练进行中",
        "complete": "训练完成",
        "steps_unit": "步",
        "seconds_unit": "秒",
        "mean_reward": "平均奖励",
        "final_mean_reward": "最终平均奖励",
        "std": "标准差",
        "episodes": "回合",
        "metric_waiting": "训练开始后显示评估结果",
        "chart_title": "奖励曲线",
        "chart_copy": "纵轴为确定性策略的平均奖励，绿色虚线表示 475 分解决阈值。图表标记保留英文。",
        "console_title": "实时训练日志",
        "console_waiting": "等待训练任务...",
        "results_title": "训练结果",
        "results_copy": "任务完成后，这里会显示策略动画并提供模型文件。",
        "animation": "策略动画",
        "download": "下载 PPO 模型",
        "footer": "实验 01",
        "console_name": "CartPole PPO 训练日志",
        "initialization": "PPO 初始化",
        "collecting": "正在收集第一批环境交互",
        "finished": "训练完成",
    },
    "English": {
        "course": "Hands-On Modern RL · Chapter 1 companion",
        "title": "CartPole Online Training Lab",
        "description": "Train a CartPole policy from scratch with PPO. Follow the reward curve, PPO output, and evaluations in real time, then download the model and replay the trained policy. Runs entirely on CPU.",
        "chapter": "Read companion chapter",
        "notebook": "Notebook",
        "script": "Training script",
        "project": "GitHub project",
        "environment": "Environment",
        "algorithm": "Algorithm",
        "device": "Device",
        "threshold": "Solved threshold",
        "max_score": "Maximum score",
        "settings_title": "Training setup",
        "settings_copy": "Choose the total interaction steps. The policy is evaluated every 2,000 steps.",
        "steps_label": "Training steps",
        "steps_info": "30,000 steps is recommended for the first run",
        "start": "Start training",
        "run_status": "Run status",
        "idle": "Ready to start",
        "idle_detail": "Choose the training steps, then start the experiment",
        "running": "Training in progress",
        "complete": "Training complete",
        "steps_unit": "steps",
        "seconds_unit": "s",
        "mean_reward": "Mean reward",
        "final_mean_reward": "Final mean reward",
        "std": "Standard deviation",
        "episodes": "episodes",
        "metric_waiting": "Evaluation results appear after training starts",
        "chart_title": "Reward curve",
        "chart_copy": "The vertical axis is the deterministic policy's mean reward. The green dashed line marks the solved threshold of 475.",
        "console_title": "Live training log",
        "console_waiting": "Waiting for a training run...",
        "results_title": "Training results",
        "results_copy": "When training finishes, the policy animation and model file will appear here.",
        "animation": "Policy animation",
        "download": "Download PPO model",
        "footer": "Experiment 01",
        "console_name": "CartPole PPO training console",
        "initialization": "PPO initialization",
        "collecting": "collecting the first rollout",
        "finished": "training completed",
    },
}
DEFAULT_LANGUAGE = "English"
DEFAULT_COPY = UI_TEXT[DEFAULT_LANGUAGE]


def text_for(language: str) -> dict[str, str]:
    """Return UI copy for a supported language."""
    return UI_TEXT["English" if language == "English" else "中文"]


def evaluate(model: PPO, episodes: int = 5) -> tuple[float, float]:
    """Evaluate the current deterministic policy without rendering."""
    env = gym.make("CartPole-v1")
    rewards, _ = evaluate_policy(
        model,
        env,
        n_eval_episodes=episodes,
        deterministic=True,
        return_episode_rewards=True,
        warn=False,
    )
    env.close()
    return float(np.mean(rewards)), float(np.std(rewards))


def reward_figure(steps: list[int], rewards: list[float]):
    """Build a compact reward chart suitable for Gradio streaming updates."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(steps, rewards, color="#4f46e5", marker="o", linewidth=2)
    ax.axhline(475, color="#16a34a", linestyle="--", linewidth=1.2, label="Solved threshold: 475")
    ax.set(xlabel="Training steps", ylabel="Mean reward", ylim=(0, 510))
    ax.set_title("PPO evaluation reward on CartPole-v1")
    ax.grid(alpha=0.22)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def clean_output(text: str) -> str:
    """Remove terminal control characters before showing library output."""
    return ANSI_ESCAPE.sub("", text).strip()


def log_line(started_at: float, level: str, message: str) -> str:
    """Format a compact console line with elapsed time."""
    elapsed = time.perf_counter() - started_at
    return f"{elapsed:7.1f}s  {level:<7} {message}"


def status_card(state: str, title: str, detail: str, language: str = "中文") -> str:
    """Render a compact run-status summary without nested borders."""
    copy = text_for(language)
    return f"""
    <div class="run-state run-state--{state}">
      <span class="run-state__dot" aria-hidden="true"></span>
      <div class="run-state__body">
        <span class="summary-label">{copy['run_status']}</span>
        <strong>{title}</strong>
        <small>{detail}</small>
      </div>
    </div>
    """


def metric_card(label: str, value: str, detail: str) -> str:
    """Render the latest evaluation result as a compact summary."""
    return f"""
    <div class="live-metric">
      <span class="summary-label">{label}</span>
      <div class="metric-reading"><strong>{value}</strong><small>{detail}</small></div>
    </div>
    """


def record_policy(model: PPO) -> tuple[str, float]:
    """Render one deterministic episode and save it as a browser-friendly GIF."""
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    obs, _ = env.reset(seed=SEED + 1)
    frames: list[np.ndarray] = []
    score = 0.0

    for _ in range(500):
        frames.append(env.render())
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        score += float(reward)
        if terminated or truncated:
            break

    env.close()
    gif_path = ARTIFACT_DIR / "cartpole-trained-policy.gif"
    imageio.mimsave(gif_path, frames, duration=1 / 30, loop=0)
    return str(gif_path), score


def train(total_timesteps: int, language: str):
    """Train in chunks so the browser receives live progress and reward updates."""
    copy = text_for(language)
    total_timesteps = int(total_timesteps)
    chunk_size = 2_000
    started_at = time.perf_counter()
    logs = [
        copy["console_name"],
        "=" * 72,
        log_line(started_at, "CONFIG", "environment=CartPole-v1  algorithm=PPO  device=CPU"),
        log_line(started_at, "CONFIG", f"timesteps={total_timesteps}  seed={SEED}  eval_episodes=5"),
    ]
    env = gym.make("CartPole-v1")
    env.reset(seed=SEED)
    library_output = io.StringIO()
    with contextlib.redirect_stdout(library_output), contextlib.redirect_stderr(library_output):
        model = PPO(
            "MlpPolicy",
            env,
            seed=SEED,
            verbose=1,
            device="cpu",
            n_steps=1_024,
            batch_size=64,
            learning_rate=3e-4,
        )
    initialization_output = clean_output(library_output.getvalue())
    if initialization_output:
        logs.extend(["", copy["initialization"], initialization_output])

    steps: list[int] = [0]
    mean_rewards: list[float] = []
    initial_mean, initial_std = evaluate(model)
    mean_rewards.append(initial_mean)
    logs.extend(
        [
            "",
            log_line(
                started_at,
                "EVAL",
                f"step=0  mean_reward={initial_mean:.1f}  std={initial_std:.1f}",
            ),
            log_line(started_at, "TRAIN", copy["collecting"]),
        ]
    )

    yield (
        status_card(
            "running",
            copy["running"],
            f"0 / {total_timesteps:,} {copy['steps_unit']}",
            language,
        ),
        metric_card(copy["mean_reward"], f"{initial_mean:.1f}", f"{copy['std']} {initial_std:.1f}"),
        reward_figure(steps, mean_rewards),
        None,
        None,
        console_panel("\n".join(logs), language),
    )

    trained = 0
    while trained < total_timesteps:
        current_chunk = min(chunk_size, total_timesteps - trained)
        library_output = io.StringIO()
        with contextlib.redirect_stdout(library_output), contextlib.redirect_stderr(library_output):
            model.learn(
                total_timesteps=current_chunk,
                reset_num_timesteps=False,
                progress_bar=False,
            )
        trained += current_chunk
        ppo_output = clean_output(library_output.getvalue())
        if ppo_output:
            logs.extend(["", f"PPO update · step {trained:,}", ppo_output])
        mean_reward, std_reward = evaluate(model)
        steps.append(trained)
        mean_rewards.append(mean_reward)
        elapsed = time.perf_counter() - started_at
        logs.append(
            log_line(
                started_at,
                "EVAL",
                f"step={trained}  mean_reward={mean_reward:.1f}  std={std_reward:.1f}",
            )
        )
        yield (
            status_card(
                "running",
                copy["running"],
                f"{trained:,} / {total_timesteps:,} {copy['steps_unit']} · {trained / total_timesteps:.0%} · {elapsed:.1f} {copy['seconds_unit']}",
                language,
            ),
            metric_card(copy["mean_reward"], f"{mean_reward:.1f}", f"{copy['std']} {std_reward:.1f}"),
            reward_figure(steps, mean_rewards),
            None,
            None,
            console_panel("\n".join(logs), language),
        )

    model_path = ARTIFACT_DIR / "ppo-cartpole"
    model.save(model_path)
    model_file = str(model_path.with_suffix(".zip"))
    logs.append(log_line(started_at, "SAVE", f"model={model_file}"))
    gif_path, demo_score = record_policy(model)
    logs.append(log_line(started_at, "RENDER", f"animation={gif_path}  episode_reward={demo_score:.0f}"))
    elapsed = time.perf_counter() - started_at
    final_mean, final_std = evaluate(model, episodes=10)
    env.close()
    logs.extend(
        [
            log_line(
                started_at,
                "FINAL",
                f"episodes=10  mean_reward={final_mean:.1f}  std={final_std:.1f}",
            ),
            log_line(started_at, "DONE", f"{copy['finished']} · {elapsed:.1f} {copy['seconds_unit']}"),
        ]
    )

    yield (
        status_card(
            "complete",
            copy["complete"],
            f"{total_timesteps:,} {copy['steps_unit']} · {elapsed:.1f} {copy['seconds_unit']}",
            language,
        ),
        metric_card(
            copy["final_mean_reward"],
            f"{final_mean:.1f}",
            f"10 {copy['episodes']} · {copy['std']} {final_std:.1f}",
        ),
        reward_figure(steps, mean_rewards),
        gif_path,
        model_file,
        console_panel("\n".join(logs), language),
    )


def hero_html(language: str) -> str:
    """Render the bilingual hero and experiment facts."""
    copy = text_for(language)
    return f"""
    <main class="app-shell">
      <section class="hero">
        <div class="brand-lockup"><a href="https://modelscope.cn/organization/walkinglab" target="_blank" rel="noreferrer">WALKINGLAB</a><span>×</span><a href="{PROJECT_URL}" target="_blank" rel="noreferrer">HANDS-ON MODERN RL</a></div>
        <img class="project-mark" src="{LOGO_DATA_URI}" alt="Hands-On Modern RL" />
        <div class="hero-topline">
          <span class="experiment-badge">EXPERIMENT 01</span>
          <span class="hero-course">{copy['course']}</span>
        </div>
        <h1>{copy['title']}</h1>
        <p class="hero-copy">{copy['description']}</p>
        <nav class="hero-links" aria-label="Project links">
          <a class="hero-link primary" href="{PROJECT_URL}" target="_blank" rel="noreferrer">GitHub · walkinglabs/hands-on-modern-rl</a>
          <a class="hero-link" href="https://modelscope.cn/organization/walkinglab" target="_blank" rel="noreferrer">WalkingLab</a>
          <a class="hero-link" href="{CHAPTER_URL}" target="_blank" rel="noreferrer">{copy['chapter']}</a>
          <a class="hero-link" href="{MODELSCOPE_NOTEBOOK_URL}" target="_blank" rel="noreferrer">{copy['notebook']}</a>
          <a class="hero-link" href="{SCRIPT_URL}" target="_blank" rel="noreferrer">{copy['script']}</a>
        </nav>
      </section>
      <section class="lab-strip" aria-label="Experiment configuration">
        <span>{copy['environment']} <strong>CartPole-v1</strong></span>
        <span>{copy['algorithm']} <strong>PPO</strong></span>
        <span>{copy['device']} <strong>CPU</strong></span>
        <span>{copy['threshold']} <strong>475</strong></span>
        <span>{copy['max_score']} <strong>500</strong></span>
      </section>
    </main>
    """


def panel_html(title: str, copy: str, copy_class: str = "panel-copy") -> str:
    """Render a panel heading and its supporting sentence."""
    return f'<h2 class="panel-title">{title}</h2><p class="{copy_class}">{copy}</p>'


def console_header_html(language: str) -> str:
    """Render the localized console title."""
    return f'<div class="console-head"><span class="console-dot"></span>{text_for(language)["console_title"]}</div>'


def console_panel(logs: str, language: str) -> str:
    """Render stable, escaped console output without an input component redraw."""
    return f"""
    <section class="console-panel" aria-live="polite" aria-atomic="true">
      {console_header_html(language)}
      <pre class="console-text">{html.escape(logs)}</pre>
    </section>
    """


def footer_html(language: str) -> str:
    """Render the localized footer label."""
    copy = text_for(language)
    return f'<div class="footer-note">{copy["footer"]} · <a href="{COURSE_URL}" target="_blank" rel="noreferrer">Hands-On Modern RL</a> · WalkingLab</div>'


def switch_language(language: str):
    """Update all visible interface copy when the language changes."""
    copy = text_for(language)
    return (
        hero_html(language),
        panel_html(copy["settings_title"], copy["settings_copy"]),
        gr.Slider(label=copy["steps_label"], info=copy["steps_info"]),
        gr.Button(value=copy["start"]),
        status_card("idle", copy["idle"], copy["idle_detail"], language),
        metric_card(copy["mean_reward"], "—", copy["metric_waiting"]),
        panel_html(copy["chart_title"], copy["chart_copy"]),
        console_panel(copy["console_waiting"], language),
        panel_html(copy["results_title"], copy["results_copy"], "artifact-note"),
        gr.Image(label=copy["animation"]),
        gr.File(label=copy["download"]),
        footer_html(language),
    )


CSS = """
:root {
  --ink: #172033;
  --muted: #68748a;
  --line: #e4e8f0;
  --paper: #ffffff;
  --canvas: #f4f6fa;
  --brand: #5b5ce2;
  --brand-dark: #4446be;
  --green: #13a36f;
}
.gradio-container {
  max-width: 1180px !important;
  margin: 0 auto !important;
  padding: 28px 22px 52px !important;
  background: var(--canvas);
}
.hero-stack {
  position: relative !important;
  margin: 0 !important;
  padding: 0 !important;
  border: 0 !important;
  background: transparent !important;
}
.language-bar {
  position: absolute !important;
  z-index: 5 !important;
  top: 18px !important;
  right: 20px !important;
  align-items: center !important;
  width: auto !important;
  min-width: 0 !important;
  min-height: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}
.language-switch {
  flex: 0 0 216px !important;
  width: 216px !important;
  min-width: 216px !important;
  margin: 0 !important;
  padding: 3px !important;
  border: 1px solid rgba(255,255,255,.18) !important;
  border-radius: 10px !important;
  background: rgba(14, 20, 46, .58) !important;
  box-shadow: 0 7px 20px rgba(5, 8, 24, .22) !important;
  backdrop-filter: blur(12px) !important;
}
.language-switch > div { display: grid !important; grid-template-columns: 1fr 1fr !important; gap: 3px !important; }
.language-switch label { display: flex !important; flex: 1 1 0 !important; cursor: pointer !important; }
.language-switch input { display: none !important; }
.language-switch label span {
  box-sizing: border-box !important;
  width: 100% !important;
  min-height: 34px !important;
  justify-content: center !important;
  padding: 7px 13px !important;
  border-radius: 7px !important;
  border: 1px solid transparent !important;
  color: rgba(255,255,255,.72) !important;
  background: transparent !important;
  font-size: 13px !important;
  font-weight: 700 !important;
  transition: color .16s ease, background .16s ease, box-shadow .16s ease !important;
}
.language-switch label:hover span { color: #ffffff !important; background: rgba(255,255,255,.10) !important; }
.language-switch label:has(input:checked) span,
.language-switch input:checked + span {
  border-color: transparent !important;
  color: #ffffff !important;
  background: linear-gradient(135deg, #6667e8, #7778f2) !important;
  box-shadow: 0 3px 9px rgba(13, 15, 55, .28) !important;
}
.language-switch label:focus-within span {
  outline: 3px solid rgba(91, 92, 226, .20) !important;
  outline-offset: 1px !important;
}
.app-shell { font-family: Inter, "PingFang SC", "Microsoft YaHei", sans-serif; color: var(--ink); }
.hero {
  position: relative;
  overflow: hidden;
  padding: 38px 42px 34px;
  border: 1px solid rgba(129, 140, 248, .2);
  border-radius: 26px;
  color: #f8fafc;
  background:
    radial-gradient(circle at 88% 8%, rgba(125, 127, 255, .42), transparent 31%),
    radial-gradient(circle at 92% 92%, rgba(61, 207, 170, .18), transparent 30%),
    linear-gradient(132deg, #11182c 0%, #25265d 58%, #4546a4 100%);
  box-shadow: 0 22px 54px rgba(25, 32, 56, .16);
}
.hero::after {
  content: "";
  position: absolute;
  width: 290px;
  height: 290px;
  right: -104px;
  top: -136px;
  border: 1px solid rgba(255,255,255,.12);
  border-radius: 50%;
}
.project-mark {
  display: block;
  width: 290px;
  max-width: 72%;
  height: auto;
  margin: 0 0 22px;
  padding: 9px 13px;
  border-radius: 11px;
  background: #ffffff;
  box-shadow: 0 8px 24px rgba(8, 15, 35, .2);
}
.brand-lockup{display:flex;align-items:center;gap:9px;width:max-content;max-width:calc(100% - 230px);margin:0 0 14px;padding:8px 12px;border:1px solid rgba(255,255,255,.28);border-radius:10px;background:rgba(12,17,59,.28);font-size:12px;font-weight:900;letter-spacing:.075em}.brand-lockup a{color:#fff!important;text-decoration:none!important}.brand-lockup a:last-child{color:#cfd5ff!important}.brand-lockup span{color:#aeb7ff}
.hero-topline { display: flex; align-items: center; gap: 11px; margin-bottom: 22px; }
.experiment-badge {
  display: inline-flex;
  align-items: center;
  padding: 6px 11px;
  border: 1px solid rgba(221, 224, 255, .3);
  border-radius: 999px;
  color: #eef0ff;
  background: rgba(255,255,255,.1);
  font-size: 12px;
  font-weight: 700;
  letter-spacing: .06em;
}
.hero-course { color: #b9c0d4; font-size: 13px; font-weight: 650; }
.hero h1 {
  max-width: 760px;
  margin: 0 0 12px;
  color: #ffffff;
  font-size: clamp(32px, 5vw, 48px);
  line-height: 1.1;
  letter-spacing: -.035em;
}
.hero-copy {
  max-width: 700px;
  margin: 0;
  color: #cdd3e2;
  font-size: 15px;
  line-height: 1.7;
}
.hero-links { display: flex; flex-wrap: wrap; gap: 9px; margin-top: 25px; }
.hero-link {
  display: inline-flex;
  align-items: center;
  min-height: 38px;
  padding: 0 14px;
  border: 1px solid rgba(255,255,255,.18);
  border-radius: 9px;
  color: #eef2ff !important;
  background: rgba(255,255,255,.08);
  font-size: 13px;
  font-weight: 650;
  text-decoration: none !important;
  transition: transform .16s ease, background .16s ease;
}
.hero-link:hover { transform: translateY(-1px); background: rgba(255,255,255,.15); }
.hero-link.primary { color: #172554 !important; background: #ffffff; border-color: #ffffff; }
.lab-strip {
  display: flex;
  flex-wrap: wrap;
  gap: 8px 22px;
  margin: 17px 0 22px;
  padding: 13px 18px;
  border: 1px solid var(--line);
  border-radius: 13px;
  background: rgba(255,255,255,.84);
  color: var(--muted);
  font-size: 13px;
  box-shadow: 0 6px 20px rgba(18, 25, 43, .035);
}
.lab-strip strong { margin-left: 5px; color: var(--ink); font-weight: 750; }
.panel-title { margin: 0 0 5px; color: var(--ink); font-size: 19px; font-weight: 780; letter-spacing: -.015em; }
.panel-copy { margin: 0 0 17px; color: var(--muted); font-size: 13px; line-height: 1.6; }
.control-card, .chart-card, .output-card {
  border: 1px solid var(--line) !important;
  border-radius: 17px !important;
  background: #ffffff !important;
  box-shadow: 0 9px 26px rgba(20, 28, 48, .05) !important;
}
.control-card { padding: 22px !important; }
.chart-card { padding: 18px !important; }
.output-card { margin-top: 14px !important; padding: 18px !important; }
.primary-btn {
  min-height: 48px !important;
  border: 0 !important;
  border-radius: 11px !important;
  background: linear-gradient(135deg, var(--brand-dark), #6969ec) !important;
  box-shadow: 0 8px 20px rgba(76, 77, 202, .22) !important;
  font-size: 15px !important;
  font-weight: 750 !important;
}
.status-output, .metric-output {
  min-width: 0 !important;
  margin: 10px 0 0 !important;
  padding: 0 !important;
  border: 0 !important;
  border-radius: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}
.status-output > div, .metric-output > div {
  padding: 0 !important;
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}
.run-state, .live-metric {
  box-sizing: border-box;
  min-height: 88px;
  padding: 15px 16px;
  border: 0;
  border-radius: 13px;
  background: #f5f6fa;
}
.run-state { display: flex; align-items: center; gap: 13px; }
.run-state__dot {
  flex: 0 0 auto;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #98a2b3;
}
.run-state--running .run-state__dot { background: var(--brand); }
.run-state--complete .run-state__dot { background: var(--green); }
.run-state__body { min-width: 0; }
.summary-label {
  display: block;
  margin-bottom: 4px;
  color: #8993a5;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: .055em;
}
.run-state strong { display: block; color: var(--ink); font-size: 15px; line-height: 1.35; }
.run-state small, .live-metric small { display: block; margin-top: 4px; color: var(--muted); font-size: 12px; line-height: 1.45; }
.live-metric { display: flex; flex-direction: column; justify-content: center; }
.metric-reading { display: flex; align-items: baseline; gap: 9px; }
.live-metric strong { color: var(--ink); font-size: 23px; line-height: 1; letter-spacing: -.025em; }
.live-metric small { margin: 0; }
.console-head {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 13px 16px;
  border-bottom: 1px solid #263044;
  color: #d9e0eb;
  background: #151c2b;
  font-size: 13px;
  font-weight: 700;
}
.console-dot { width: 8px; height: 8px; border-radius: 50%; background: #24c689; box-shadow: 0 0 0 4px rgba(36,198,137,.12); }
.console-output, .console-output > div {
  min-width: 0 !important;
  margin: 14px 0 0 !important;
  padding: 0 !important;
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
  animation: none !important;
  transition: none !important;
}
.console-output.generating, .console-output.pending,
.console-output.generating > div, .console-output.pending > div {
  opacity: 1 !important;
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
  animation: none !important;
}
.console-output.generating::before, .console-output.generating::after,
.console-output.pending::before, .console-output.pending::after {
  display: none !important;
  content: none !important;
  animation: none !important;
}
.console-panel {
  overflow: hidden;
  border: 1px solid #263044;
  border-radius: 13px;
  background: #0f1623;
  contain: layout paint;
  transition: none;
}
.console-text {
  box-sizing: border-box;
  height: 300px;
  margin: 0;
  padding: 17px 18px;
  overflow: auto;
  white-space: pre;
  color: #cbd5e1 !important;
  background: #0f1623 !important;
  font: 12px/1.58 "SFMono-Regular", Consolas, "Liberation Mono", monospace !important;
  scrollbar-gutter: stable;
}
.artifact-note { margin: 0 0 12px; color: var(--muted); font-size: 13px; line-height: 1.55; }
.footer-note a { color: var(--brand) !important; font-weight: 650; text-decoration: none !important; }
.footer-note { margin-top: 18px; text-align: center; color: #94a3b8; font-size: 12px; }
@media (max-width: 760px) {
  .gradio-container { padding: 12px 10px 30px !important; }
  .language-bar { top: 14px !important; right: 14px !important; }
  .language-switch { flex: 0 0 196px !important; width: 196px !important; min-width: 196px !important; }
  .hero { padding: 70px 22px 25px; border-radius: 19px; }
  .hero-topline { align-items: flex-start; flex-direction: column; gap: 8px; }
  .lab-strip { gap: 8px 16px; }
}
"""

AUTO_SCROLL_JS = """
() => {
  const consoleSelector = "#live-training-console .console-text";
  let activeConsole = null;
  let followLatest = true;
  let savedScrollTop = 0;
  let internalScroll = false;
  let updateScheduled = false;

  const updateConsolePosition = () => {
    updateScheduled = false;
    const consoleElement = document.querySelector(consoleSelector);
    if (!consoleElement) return;

    if (consoleElement !== activeConsole) {
      activeConsole = consoleElement;
      activeConsole.addEventListener("scroll", () => {
        if (internalScroll) return;
        const bottomGap = activeConsole.scrollHeight - activeConsole.clientHeight - activeConsole.scrollTop;
        followLatest = bottomGap <= 24;
        savedScrollTop = activeConsole.scrollTop;
      }, { passive: true });
    }

    internalScroll = true;
    if (followLatest) {
      activeConsole.scrollTop = activeConsole.scrollHeight;
    } else {
      const maximumScrollTop = Math.max(0, activeConsole.scrollHeight - activeConsole.clientHeight);
      activeConsole.scrollTop = Math.min(savedScrollTop, maximumScrollTop);
    }
    requestAnimationFrame(() => { internalScroll = false; });
  };

  const scheduleConsoleUpdate = () => {
    if (updateScheduled) return;
    updateScheduled = true;
    requestAnimationFrame(updateConsolePosition);
  };

  new MutationObserver(scheduleConsoleUpdate).observe(document.body, {
    childList: true,
    subtree: true,
    characterData: true,
  });
  scheduleConsoleUpdate();
}
"""


with gr.Blocks(title="Experiment 01 · CartPole Online Training") as demo:
    with gr.Column(elem_classes="hero-stack"):
        hero = gr.HTML(hero_html(DEFAULT_LANGUAGE))
        with gr.Row(elem_classes="language-bar"):
            language = gr.Radio(
                choices=[("English", "English"), ("中文", "中文")],
                value=DEFAULT_LANGUAGE,
                label="Language",
                show_label=False,
                elem_classes="language-switch",
            )

    with gr.Row():
        with gr.Column(scale=1, min_width=300, elem_classes="control-card"):
            settings_header = gr.HTML(panel_html(DEFAULT_COPY["settings_title"], DEFAULT_COPY["settings_copy"]))
            timesteps = gr.Slider(
                minimum=10_000,
                maximum=50_000,
                value=30_000,
                step=5_000,
                label=DEFAULT_COPY["steps_label"],
                info=DEFAULT_COPY["steps_info"],
            )
            start = gr.Button(DEFAULT_COPY["start"], variant="primary", elem_classes="primary-btn")
            status = gr.HTML(
                status_card("idle", DEFAULT_COPY["idle"], DEFAULT_COPY["idle_detail"], DEFAULT_LANGUAGE),
                elem_classes="status-output",
            )
            metrics = gr.HTML(
                metric_card(DEFAULT_COPY["mean_reward"], "—", DEFAULT_COPY["metric_waiting"]),
                elem_classes="metric-output",
            )
        with gr.Column(scale=2, elem_classes="chart-card"):
            chart_header = gr.HTML(panel_html(DEFAULT_COPY["chart_title"], DEFAULT_COPY["chart_copy"]))
            curve = gr.Plot(show_label=False)
            console = gr.HTML(
                console_panel(DEFAULT_COPY["console_waiting"], DEFAULT_LANGUAGE),
                elem_id="live-training-console",
                elem_classes="console-output",
            )

    with gr.Row(elem_classes="output-card"):
        with gr.Column(scale=2):
            results_header = gr.HTML(
                panel_html(DEFAULT_COPY["results_title"], DEFAULT_COPY["results_copy"], "artifact-note")
            )
            animation = gr.Image(label=DEFAULT_COPY["animation"], type="filepath")
        with gr.Column(scale=1):
            model_download = gr.File(label=DEFAULT_COPY["download"], interactive=False)

    footer = gr.HTML(footer_html(DEFAULT_LANGUAGE))

    language.change(
        fn=switch_language,
        inputs=language,
        outputs=[
            hero,
            settings_header,
            timesteps,
            start,
            status,
            metrics,
            chart_header,
            console,
            results_header,
            animation,
            model_download,
            footer,
        ],
        queue=False,
    )

    start.click(
        fn=train,
        inputs=[timesteps, language],
        outputs=[status, metrics, curve, animation, model_download, console],
        concurrency_limit=1,
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(css=CSS, js=AUTO_SCROLL_JS)
