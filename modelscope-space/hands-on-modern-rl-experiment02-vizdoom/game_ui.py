from __future__ import annotations

import base64
import html
import io
import json
import math
import os
import time
import traceback
from dataclasses import asdict, is_dataclass
from functools import lru_cache
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", "/tmp/hands-on-modern-rl-matplotlib")

from typing import Any, Iterable

import gradio as gr
import matplotlib
import numpy as np
from PIL import Image, ImageStat

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_URL = "https://github.com/walkinglabs/hands-on-modern-rl"
COURSE_URL = "https://walkinglabs.github.io/hands-on-modern-rl/"
ORGANIZATION_URL = "https://modelscope.cn/organization/walkinglab"
DEFAULT_LANGUAGE = "English"


TEXT = {
    "English": {
        "course": "Open-source reinforcement learning experiments",
        "brand": "WALKINGLAB × HANDS-ON MODERN RL",
        "brand_note": "Learn the idea, change the parameters, and train it online.",
        "organization": "WalkingLab",
        "chapter": "Companion chapter",
        "notebook": "Notebook",
        "source": "Training source",
        "project": "GitHub project",
        "device": "Device",
        "tasks": "Playable tasks",
        "runtime": "Runtime",
        "choose": "Choose a game",
        "choose_copy": "Every card opens a real training recipe. Select a task, review its goal, then start a run.",
        "understand": "UNDERSTAND BEFORE TRAINING",
        "observation": "Observation",
        "action": "Action",
        "algorithm": "Algorithm",
        "status": "Status",
        "ready_runtime": "Ready · preinstalled",
        "hint": "Adjust the parameters below, then start training. The curve and console update throughout the run.",
        "setup": "Experiment setup",
        "setup_copy": "Quick defaults are sized for the selected ModelScope runtime. Increase the budget for a longer run.",
        "selected": "Selected game",
        "budget": "Training budget",
        "budget_info": "Environment steps, episodes, or optimization iterations depending on the selected game",
        "lr": "Learning rate",
        "gamma": "Discount factor γ",
        "epsilon": "Exploration ε",
        "seed": "Random seed",
        "start": "Start training",
        "running_button": "Training…",
        "run_status": "Run status",
        "ready": "Ready to train",
        "ready_detail": "Select a game, review its goal, and start the run",
        "running": "Training in progress",
        "complete": "Training complete",
        "failed": "Run stopped",
        "metric": "Latest evaluation",
        "metric_waiting": "Results appear after training starts",
        "curve": "Learning curve",
        "curve_copy": "The chart updates at each checkpoint. Labels remain in English for readability.",
        "log": "Live training log",
        "log_waiting": "Waiting for a training run…",
        "preview": "Task preview / learned policy",
        "preview_copy": "The task opens with a real environment preview. A completed run replaces it with this run's replay or result visualization.",
        "download": "Download run summary",
        "wait_title": "Training is active",
        "wait_detail": "Preparing the environment, updating the policy, evaluating it, and rendering the learned result.",
    },
    "中文": {
        "course": "开源强化学习在线实验",
        "brand": "WALKINGLAB × 动手学现代强化学习",
        "brand_note": "理解任务、调整参数，并在网页中完成真实训练。",
        "organization": "WalkingLab 主页",
        "chapter": "阅读配套章节",
        "notebook": "Notebook",
        "source": "训练源码",
        "project": "GitHub 项目",
        "device": "设备",
        "tasks": "可训练任务",
        "runtime": "运行环境",
        "choose": "选择一个游戏",
        "choose_copy": "每张卡片都对应一个真实训练配方。选择任务、阅读目标，然后启动训练。",
        "understand": "训练前先理解任务",
        "observation": "观察",
        "action": "动作",
        "algorithm": "算法",
        "status": "状态",
        "ready_runtime": "就绪 · 已预装",
        "hint": "调整下方参数后启动训练。学习曲线和训练日志会持续更新。",
        "setup": "实验设置",
        "setup_copy": "默认参数适配当前 ModelScope 运行环境；增加训练预算可以运行更久。",
        "selected": "已选游戏",
        "budget": "训练预算",
        "budget_info": "根据任务表示环境步数、回合数或优化迭代次数",
        "lr": "学习率",
        "gamma": "折扣因子 γ",
        "epsilon": "探索率 ε",
        "seed": "随机种子",
        "start": "开始训练",
        "running_button": "训练中…",
        "run_status": "训练状态",
        "ready": "等待训练",
        "ready_detail": "选择游戏、阅读目标，然后启动训练",
        "running": "训练进行中",
        "complete": "训练完成",
        "failed": "训练停止",
        "metric": "最新评估",
        "metric_waiting": "训练开始后显示结果",
        "curve": "学习曲线",
        "curve_copy": "每个检查点更新一次曲线；图表标记统一使用英文。",
        "log": "实时训练日志",
        "log_waiting": "等待训练任务…",
        "preview": "任务预览 / 学习后的策略",
        "preview_copy": "打开任务时显示真实环境预览；训练完成后替换为本次运行生成的回放或结果图。",
        "download": "下载运行摘要",
        "wait_title": "训练正在运行",
        "wait_detail": "正在准备环境、更新策略、执行评估并渲染学习结果。",
    },
}


def copy_for(language: str) -> dict[str, str]:
    return TEXT["中文" if language == "中文" else "English"]


def task_value(task: Any, key: str, default: Any = None) -> Any:
    if isinstance(task, dict):
        return task.get(key, default)
    return getattr(task, key, default)


def local_value(value: Any, language: str) -> str:
    if isinstance(value, dict):
        return str(value.get("zh" if language == "中文" else "en") or value.get("en") or next(iter(value.values())))
    return str(value)


def normalize_tasks(tasks: Iterable[Any]) -> list[Any]:
    result = list(tasks)
    if not result:
        raise ValueError("At least one game task is required")
    keys = [task_value(task, "key") for task in result]
    if any(not key for key in keys) or len(keys) != len(set(keys)):
        raise ValueError("Every game task needs a unique non-empty key")
    return result


def get_task(tasks: list[Any], key: str) -> Any:
    return next((task for task in tasks if task_value(task, "key") == key), tasks[0])


def preview_path(root: Path, task: Any) -> str:
    path = Path(str(task_value(task, "preview")))
    if not path.is_absolute():
        path = root / path
    return str(path)


@lru_cache(maxsize=128)
def embedded_image(path_value: str) -> str:
    """Return a compact, proxy-independent task-detail image data URL."""
    path = Path(path_value)
    with Image.open(path) as source:
        frame_count = int(getattr(source, "n_frames", 1))
        if frame_count > 1:
            sample_count = min(frame_count, 18)
            sample_indices = sorted({
                round(index * (frame_count - 1) / (sample_count - 1))
                for index in range(sample_count)
            })
            candidates: list[tuple[tuple[float, float], Image.Image]] = []
            for frame_index in sample_indices:
                source.seek(frame_index)
                candidate = source.convert("RGB")
                grayscale = candidate.convert("L")
                score = (grayscale.entropy(), float(ImageStat.Stat(grayscale).var[0]))
                candidates.append((score, candidate))
            frame = max(candidates, key=lambda item: item[0])[1]
        else:
            source.seek(0)
            has_alpha = "A" in source.getbands() or "transparency" in source.info
            frame = source.convert("RGBA" if has_alpha else "RGB")
        frame.thumbnail((960, 640), Image.Resampling.LANCZOS)
    lossy = io.BytesIO()
    frame.save(lossy, format="WEBP", quality=88, method=4)
    lossless = io.BytesIO()
    frame.save(lossless, format="WEBP", lossless=True, method=4)
    payload = min(lossy.getvalue(), lossless.getvalue(), key=len)
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:image/webp;base64,{encoded}"


def hero_html(space: dict[str, Any], tasks: list[Any], task: Any, language: str, runtime_status: str) -> str:
    copy = copy_for(language)
    title = local_value(space["title"], language)
    description = local_value(space["description"], language)
    course = space.get("course_url", COURSE_URL)
    notebook = space.get("notebook_url")
    source = space.get("source_url", PROJECT_URL)
    project = space.get("project_url", PROJECT_URL)
    organization = space.get("organization_url", ORGANIZATION_URL)
    device = html.escape(str(space.get("device", "CPU")))
    badge = html.escape(str(space.get("badge", "CPU GAME LAB")))
    return f"""
    <main class="app-shell">
      <section class="hero">
        <div class="brand-lockup">
          <a href="{html.escape(organization)}" target="_blank" rel="noreferrer">WALKINGLAB</a>
          <span aria-hidden="true">×</span>
          <a href="{html.escape(project)}" target="_blank" rel="noreferrer">HANDS-ON MODERN RL</a>
        </div>
        <p class="brand-note">{html.escape(copy['brand_note'])}</p>
        <div class="hero-topline"><span class="experiment-badge">{badge}</span><span class="hero-course">{copy['course']}</span></div>
        <h1>{html.escape(title)}</h1>
        <p class="hero-copy">{html.escape(description)}</p>
        <nav class="hero-links">
          <a class="hero-link primary" href="{html.escape(project)}" target="_blank" rel="noreferrer">GitHub · walkinglabs/hands-on-modern-rl</a>
          <a class="hero-link" href="{html.escape(organization)}" target="_blank" rel="noreferrer">{copy['organization']}</a>
          <a class="hero-link" href="{html.escape(course)}" target="_blank" rel="noreferrer">{copy['chapter']}</a>
          {f'<a class="hero-link" href="{html.escape(str(notebook))}" target="_blank" rel="noreferrer">{copy["notebook"]}</a>' if notebook else ''}
          <a class="hero-link" href="{html.escape(source)}" target="_blank" rel="noreferrer">{copy['source']}</a>
        </nav>
      </section>
      <section class="lab-strip">
        <span>{copy['tasks']} <strong>{len(tasks)}</strong></span>
        <span>{copy['device']} <strong>{device}</strong></span>
        <span>{copy['runtime']} <strong>{html.escape(runtime_status)}</strong></span>
        <span>Environment <strong>{html.escape(str(task_value(task, 'environment')))}</strong></span>
      </section>
    </main>
    """


def task_brief(root: Path, task: Any, language: str) -> str:
    copy = copy_for(language)
    preview = embedded_image(preview_path(root, task))
    return f"""
    <section class="task-brief">
      <div class="task-brief__visual"><img src="{html.escape(preview)}" alt="{html.escape(str(task_value(task, 'environment')))}"></div>
      <div class="task-brief__body">
        <span class="task-kicker">{copy['understand']}</span>
        <h3>{html.escape(local_value(task_value(task, 'title'), language))}</h3>
        <p>{html.escape(local_value(task_value(task, 'description'), language))}</p>
        <div class="task-facts">
          <span><b>{copy['observation']}</b>{html.escape(local_value(task_value(task, 'observation'), language))}</span>
          <span><b>{copy['action']}</b>{html.escape(local_value(task_value(task, 'action'), language))}</span>
          <span><b>{copy['algorithm']}</b>{html.escape(str(task_value(task, 'algorithm')))}</span>
          <span><b>{copy['status']}</b>{copy['ready_runtime']}</span>
        </div>
        <p class="task-hint">{copy['hint']}</p>
      </div>
    </section>
    """


def panel_html(title: str, text: str, cls: str = "panel-copy") -> str:
    return f'<h2 class="panel-title">{html.escape(title)}</h2><p class="{cls}">{html.escape(text)}</p>'


def status_card(state: str, title: str, detail: str, language: str) -> str:
    return f"""
    <div class="run-state run-state--{state}">
      <span class="run-state__dot"></span>
      <div><span class="summary-label">{copy_for(language)['run_status']}</span><strong>{html.escape(title)}</strong><small>{html.escape(detail)}</small></div>
    </div>
    """


def metric_card(value: str, detail: str, language: str) -> str:
    return f"""
    <div class="live-metric">
      <span class="summary-label">{copy_for(language)['metric']}</span>
      <div class="metric-reading"><strong>{html.escape(value)}</strong><small>{html.escape(detail)}</small></div>
    </div>
    """


def console_panel(logs: str, language: str) -> str:
    return f"""
    <section class="console-panel" aria-live="polite">
      <div class="console-head"><span class="console-dot"></span>{copy_for(language)['log']}</div>
      <pre class="console-text">{html.escape(logs)}</pre>
    </section>
    """


def waiting_panel(language: str) -> str:
    copy = copy_for(language)
    elapsed_label = "elapsed" if language == "English" else "已等待"
    return f"""
    <section class="run-wait" role="status" aria-live="polite">
      <span class="run-wait__spinner" aria-hidden="true"></span>
      <div><strong>{copy['wait_title']}</strong><small>{copy['wait_detail']}</small>
      <em class="run-wait__elapsed" data-start-ms="{int(time.time() * 1000)}" data-label="{elapsed_label}">0s {elapsed_label}</em></div>
    </section>
    """


def learning_figure(x: list[float], y: list[float], title: str, ylabel: str = "Evaluation score"):
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    ax.plot(x, y, color="#5b5ce2", linewidth=2.2)
    if x:
        ax.scatter([x[-1]], [y[-1]], color="#15a873", s=36, zorder=3)
    ax.set_title(title)
    ax.set_xlabel("Training progress")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    # Gradio serializes the Figure object after this function returns.  Closing
    # it only unregisters the pyplot manager, so repeated checkpoints do not
    # retain every historical chart in the long-running Studio process.
    plt.close(fig)
    return fig


def result_image(root: Path, task: Any, status: str, score: float | None, x: list[float], y: list[float], note: str) -> str:
    artifact_dir = root / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    key = str(task_value(task, "key"))
    path = artifact_dir / f"{key}-result.png"
    fig = plt.figure(figsize=(9.6, 5.0), facecolor="#f7f8fc")
    grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.8], wspace=.3)
    info = fig.add_subplot(grid[0, 0])
    plot = fig.add_subplot(grid[0, 1])
    info.set_facecolor("#20245b")
    info.set_xticks([]); info.set_yticks([])
    for spine in info.spines.values():
        spine.set_visible(False)
    info.text(.09, .86, status.upper(), color="#a5b4fc", fontsize=10, fontweight="bold", transform=info.transAxes)
    info.text(.09, .67, "—" if score is None else f"{score:.2f}", color="white", fontsize=28, fontweight="bold", transform=info.transAxes)
    info.text(.09, .57, "Final evaluation score", color="#cbd5e1", fontsize=9, transform=info.transAxes)
    info.text(.09, .39, str(task_value(task, "environment")), color="white", fontsize=12, fontweight="bold", wrap=True, transform=info.transAxes)
    info.text(.09, .24, str(task_value(task, "algorithm")), color="#cbd5e1", fontsize=10, transform=info.transAxes)
    info.text(.09, .07, note[:180], color="#aeb7ca", fontsize=8.5, wrap=True, transform=info.transAxes)
    if x and y:
        plot.plot(x, y, color="#5b5ce2", linewidth=2.3)
        plot.scatter([x[-1]], [y[-1]], color="#13a36f", s=44, zorder=3)
        plot.set_xlabel("Training progress")
        plot.set_ylabel("Evaluation score")
        plot.grid(alpha=.2)
        plot.set_title("Learned policy result", loc="left", fontweight="bold")
    else:
        plot.axis("off")
        plot.text(.5, .55, "RESULT", ha="center", color="#5b5ce2", fontsize=22, fontweight="bold")
        plot.text(.5, .4, note[:220], ha="center", color="#68748a", fontsize=10, wrap=True)
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return str(path)


def save_summary(root: Path, task: Any, payload: dict[str, Any]) -> str:
    artifact_dir = root / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / f"{task_value(task, 'key')}-run-summary.json"
    if is_dataclass(task):
        task_payload = asdict(task)
    elif isinstance(task, dict):
        task_payload = task
    else:
        task_payload = {name: getattr(task, name) for name in dir(task) if not name.startswith("_") and not callable(getattr(task, name))}
    data = {"task": task_payload, **payload}
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return str(path)


def event_value(event: Any, key: str, default: Any = None) -> Any:
    if isinstance(event, dict):
        return event.get(key, default)
    return getattr(event, key, default)


def slider_spec(task: Any, name: str, fallback: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    raw = task_value(task, name, fallback)
    return tuple(float(value) for value in raw)


def slider_update(label: str, spec: tuple[float, float, float, float], info: str | None = None):
    return gr.Slider(minimum=spec[0], maximum=spec[1], value=spec[2], step=spec[3], label=label, info=info)


def build_demo(space_module: Any):
    root = Path(space_module.__file__).resolve().parent
    tasks = normalize_tasks(space_module.TASKS)
    space = dict(space_module.SPACE)
    runtime_status = str(space_module.runtime_status())
    default_language = DEFAULT_LANGUAGE
    default_task = tasks[0]

    def gallery_items(language: str):
        return [
            (preview_path(root, task), f"{local_value(task_value(task, 'title'), language)}\n{task_value(task, 'algorithm')}")
            for task in tasks
        ]

    def choose_task(language: str, event: gr.SelectData):
        index = max(0, min(int(event.index), len(tasks) - 1))
        task = tasks[index]
        copy = copy_for(language)
        budget = slider_spec(task, "budget", (100, 10000, 1000, 100))
        lr = slider_spec(task, "learning_rate", (1e-5, .1, 3e-4, 1e-5))
        gamma = slider_spec(task, "gamma", (0, 1, .99, .01))
        epsilon = slider_spec(task, "epsilon", (0, 1, .1, .01))
        return (
            task_value(task, "key"),
            hero_html(space, tasks, task, language, runtime_status),
            task_brief(root, task, language),
            gr.Slider(minimum=budget[0], maximum=budget[1], value=budget[2], step=budget[3], label=copy["budget"], info=copy["budget_info"]),
            gr.Slider(minimum=lr[0], maximum=lr[1], value=lr[2], step=lr[3], label=copy["lr"]),
            gr.Slider(minimum=gamma[0], maximum=gamma[1], value=gamma[2], step=gamma[3], label=copy["gamma"]),
            gr.Slider(minimum=epsilon[0], maximum=epsilon[1], value=epsilon[2], step=epsilon[3], label=copy["epsilon"]),
            status_card("idle", copy["ready"], copy["ready_detail"], language),
            metric_card("—", copy["metric_waiting"], language),
            console_panel(copy["log_waiting"], language),
            preview_path(root, task),
            None,
        )

    def switch_language(language: str, key: str, seed: float):
        task = get_task(tasks, key)
        copy = copy_for(language)
        return (
            hero_html(space, tasks, task, language, runtime_status),
            panel_html(copy["choose"], copy["choose_copy"]),
            gr.Gallery(value=gallery_items(language)),
            task_brief(root, task, language),
            panel_html(copy["setup"], copy["setup_copy"]),
            gr.Textbox(value=key, label=copy["selected"]),
            gr.Number(value=seed, label=copy["seed"], precision=0),
            gr.Button(value=copy["start"]),
            status_card("idle", copy["ready"], copy["ready_detail"], language),
            metric_card("—", copy["metric_waiting"], language),
            panel_html(copy["curve"], copy["curve_copy"]),
            console_panel(copy["log_waiting"], language),
            panel_html(copy["preview"], copy["preview_copy"], "artifact-note"),
            gr.File(label=copy["download"]),
        )

    def train_with_ui(key: str, budget: float, learning_rate: float, gamma: float, epsilon: float, seed: float, language: str):
        task = get_task(tasks, key)
        copy = copy_for(language)
        params = {
            "budget": int(budget),
            "learning_rate": float(learning_rate),
            "gamma": float(gamma),
            "epsilon": float(epsilon),
            "seed": int(seed),
        }
        device = str(space.get("device", "CPU"))
        logs = [f"0.0s  CONFIG  environment={task_value(task, 'environment')} algorithm={task_value(task, 'algorithm')} device={device}"]
        logs.append(f"0.0s  CONFIG  budget={params['budget']} seed={params['seed']}")
        started = time.perf_counter()
        last_x: list[float] = []
        last_y: list[float] = []
        last_score: float | None = None
        preview = preview_path(root, task)
        wait = waiting_panel(language)
        yield (
            status_card("running", copy["running"], "Environment initialization", language),
            metric_card("—", "Preparing runtime", language),
            None,
            preview,
            None,
            console_panel("\n".join(logs), language),
            gr.HTML(value=wait, visible=True),
            gr.Button(value=copy["running_button"], interactive=False),
        )
        try:
            for event in space_module.run(key, **params):
                message = str(event_value(event, "log", "")).strip()
                if message:
                    if not message.lstrip().startswith(tuple("0123456789")):
                        message = f"{time.perf_counter() - started:7.1f}s  TRAIN   {message}"
                    logs.extend(message.splitlines())
                    if len(logs) > 1_200:
                        logs = logs[:2] + ["... older log lines omitted from the live view ..."] + logs[-1_197:]
                x = event_value(event, "x", last_x)
                y = event_value(event, "y", last_y)
                if x is not None and y is not None:
                    last_x = [float(value) for value in x]
                    last_y = [float(value) for value in y]
                score = event_value(event, "score", last_score)
                if score is not None and math.isfinite(float(score)):
                    last_score = float(score)
                event_preview = event_value(event, "preview")
                if event_preview:
                    preview = str(event_preview)
                phase = str(event_value(event, "phase", "training"))
                detail = str(event_value(event, "detail", f"{int(event_value(event, 'step', 0)):,}/{params['budget']:,}"))
                metric_detail = str(event_value(event, "metric_detail", "Mean evaluation score"))
                curve = learning_figure(last_x, last_y, f"{task_value(task, 'environment')} · {task_value(task, 'algorithm')}") if last_x else None
                yield (
                    status_card("running", copy["running"], detail, language),
                    metric_card("—" if last_score is None else f"{last_score:.2f}", metric_detail, language),
                    curve,
                    preview,
                    None,
                    console_panel("\n".join(logs), language),
                    gr.HTML(value=wait, visible=True),
                    gr.Button(value=copy["running_button"], interactive=False),
                )
            if preview == preview_path(root, task):
                preview = result_image(root, task, "training complete", last_score, last_x, last_y, "The environment did not expose replay frames; this plot records the learned result.")
            summary = save_summary(root, task, {
                "status": "complete",
                "parameters": params,
                "score": last_score,
                "curve": {"x": last_x, "y": last_y},
                "preview": preview,
                "logs": logs,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            })
            logs.append(f"{time.perf_counter() - started:7.1f}s  DONE    training, evaluation, and visualization complete")
            curve = learning_figure(last_x, last_y, f"{task_value(task, 'environment')} · {task_value(task, 'algorithm')}") if last_x else None
            yield (
                status_card("complete", copy["complete"], f"{time.perf_counter() - started:.1f}s elapsed", language),
                metric_card("—" if last_score is None else f"{last_score:.2f}", "Final evaluation score", language),
                curve,
                preview,
                summary,
                console_panel("\n".join(logs), language),
                gr.HTML(value="", visible=False),
                gr.Button(value=copy["start"], interactive=True),
            )
        except Exception as exc:
            logs.append(f"{time.perf_counter() - started:7.1f}s  ERROR   {type(exc).__name__}: {exc}")
            logs.extend(traceback.format_exc(limit=8).splitlines())
            preview = result_image(root, task, "run stopped", last_score, last_x, last_y, f"{type(exc).__name__}: {exc}")
            summary = save_summary(root, task, {
                "status": "failed",
                "parameters": params,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "logs": logs,
            })
            yield (
                status_card("error", copy["failed"], str(exc), language),
                metric_card("—" if last_score is None else f"{last_score:.2f}", "Last valid evaluation", language),
                learning_figure(last_x, last_y, f"{task_value(task, 'environment')} · stopped") if last_x else None,
                preview,
                summary,
                console_panel("\n".join(logs), language),
                gr.HTML(value="", visible=False),
                gr.Button(value=copy["start"], interactive=True),
            )

    copy = copy_for(default_language)
    initial_budget = slider_spec(default_task, "budget", (100, 10000, 1000, 100))
    initial_lr = slider_spec(default_task, "learning_rate", (1e-5, .1, 3e-4, 1e-5))
    initial_gamma = slider_spec(default_task, "gamma", (0, 1, .99, .01))
    initial_epsilon = slider_spec(default_task, "epsilon", (0, 1, .1, .01))

    with gr.Blocks(title=f"Hands-On Modern RL · {local_value(space['title'], 'English')}") as demo:
        with gr.Column(elem_classes="hero-stack"):
            hero = gr.HTML(hero_html(space, tasks, default_task, default_language, runtime_status))
            with gr.Row(elem_classes="language-bar"):
                language = gr.Radio(choices=["English", "中文"], value=default_language, show_label=False, elem_classes="language-switch")

        with gr.Column(elem_classes="catalog-card"):
            catalog_header = gr.HTML(panel_html(copy["choose"], copy["choose_copy"]))
            gallery = gr.Gallery(value=gallery_items(default_language), show_label=False, columns=3, rows=2, object_fit="cover", height="auto", allow_preview=False, buttons=[], elem_classes="experiment-gallery")

        task_info = gr.HTML(task_brief(root, default_task, default_language), elem_id="selected-task-detail")

        with gr.Row():
            with gr.Column(scale=1, min_width=310, elem_classes="control-card"):
                settings_header = gr.HTML(panel_html(copy["setup"], copy["setup_copy"]))
                selected = gr.Textbox(value=task_value(default_task, "key"), label=copy["selected"], interactive=False, elem_classes="selected-experiment")
                budget = gr.Slider(minimum=initial_budget[0], maximum=initial_budget[1], value=initial_budget[2], step=initial_budget[3], label=copy["budget"], info=copy["budget_info"])
                learning_rate = slider_update(copy["lr"], initial_lr)
                gamma = slider_update(copy["gamma"], initial_gamma)
                epsilon = slider_update(copy["epsilon"], initial_epsilon)
                seed = gr.Number(value=42, precision=0, label=copy["seed"])
                start = gr.Button(copy["start"], variant="primary", elem_classes="primary-btn")
                status = gr.HTML(status_card("idle", copy["ready"], copy["ready_detail"], default_language))
                metric = gr.HTML(metric_card("—", copy["metric_waiting"], default_language))
            with gr.Column(scale=2, elem_classes="chart-card"):
                chart_header = gr.HTML(panel_html(copy["curve"], copy["curve_copy"]))
                wait_state = gr.HTML(value="", visible=False)
                curve = gr.Plot(show_label=False)
                console = gr.HTML(console_panel(copy["log_waiting"], default_language), elem_id="live-training-console")

        with gr.Row(elem_classes="output-card"):
            with gr.Column(scale=2):
                preview_header = gr.HTML(panel_html(copy["preview"], copy["preview_copy"], "artifact-note"))
                preview = gr.Image(value=preview_path(root, default_task), show_label=False, interactive=False, elem_classes="policy-preview")
            with gr.Column(scale=1):
                artifact = gr.File(label=copy["download"], interactive=False)

        gr.HTML(f'<div class="footer-note">{html.escape(local_value(space["title"], "English"))} · <a href="{COURSE_URL}" target="_blank">Hands-On Modern RL</a> · WalkingLab</div>')

        gallery.select(choose_task, inputs=[language], outputs=[selected, hero, task_info, budget, learning_rate, gamma, epsilon, status, metric, console, preview, artifact], queue=False, show_progress="hidden")
        language.change(switch_language, inputs=[language, selected, seed], outputs=[hero, catalog_header, gallery, task_info, settings_header, selected, seed, start, status, metric, chart_header, console, preview_header, artifact], queue=False, show_progress="hidden")
        start.click(train_with_ui, inputs=[selected, budget, learning_rate, gamma, epsilon, seed, language], outputs=[status, metric, curve, preview, artifact, console, wait_state, start], concurrency_limit=1)

    return demo


CSS = r"""
:root{--ink:#172033;--muted:#68748a;--line:#e4e8f0;--canvas:#f4f6fa;--brand:#5b5ce2;--green:#13a36f}
.gradio-container{max-width:1180px!important;margin:0 auto!important;padding:28px 22px 52px!important;background:var(--canvas)}
.hero-stack{position:relative!important;margin:0!important;padding:0!important;border:0!important;background:transparent!important}
.language-bar{position:absolute!important;z-index:5!important;top:18px!important;right:20px!important;width:auto!important;min-width:0!important;margin:0!important;padding:0!important;border:0!important;background:transparent!important}
.language-switch{width:216px!important;min-width:216px!important;margin:0!important;padding:5px!important;border:1px solid rgba(255,255,255,.82)!important;border-radius:12px!important;background:rgba(255,255,255,.96)!important;box-shadow:0 10px 28px rgba(20,24,74,.2)!important;backdrop-filter:blur(12px)}
.language-switch>.wrap:not([data-testid]){display:grid!important;grid-template-columns:1fr 1fr!important;gap:5px!important}.language-switch label{display:flex!important;align-items:center!important;justify-content:center!important;min-height:38px!important;margin:0!important;padding:0 12px!important;border:0!important;border-radius:8px!important;background:transparent!important}.language-switch label input{position:absolute!important;opacity:0!important;pointer-events:none!important}.language-switch label span{color:#343b68!important;font-weight:800!important}.language-switch label.selected,.language-switch label:has(input:checked){background:#554ee6!important;box-shadow:0 4px 12px rgba(72,65,201,.28)!important}.language-switch label.selected span,.language-switch label:has(input:checked) span{color:#fff!important}
.app-shell{margin-bottom:18px}.hero{position:relative;overflow:hidden;padding:28px 34px 30px;border-radius:24px 24px 0 0;color:white;background:radial-gradient(circle at 84% 12%,rgba(139,92,246,.45),transparent 28%),linear-gradient(135deg,#1f255f,#3731a8 56%,#5b3fc5);box-shadow:0 24px 55px rgba(30,37,95,.18)}
.hero:after{content:"";position:absolute;inset:auto -80px -120px auto;width:280px;height:280px;border:44px solid rgba(255,255,255,.07);border-radius:50%}.brand-lockup{position:relative;z-index:2;display:flex;align-items:center;gap:9px;width:max-content;max-width:calc(100% - 230px);padding:8px 12px;border:1px solid rgba(255,255,255,.28);border-radius:10px;background:rgba(12,17,59,.28);box-shadow:0 8px 24px rgba(10,13,50,.14);font-size:12px;font-weight:900;letter-spacing:.075em}.brand-lockup a{color:#fff!important;text-decoration:none!important}.brand-lockup a:last-child{color:#cfd5ff!important}.brand-lockup span{color:#aeb7ff}.brand-note{position:relative;z-index:1;margin:9px 0 18px!important;color:#dfe3ff!important;font-size:12px!important}.hero-topline{display:flex;gap:12px;align-items:center;margin-bottom:15px}.experiment-badge{display:inline-flex;padding:7px 11px;border:1px solid rgba(255,255,255,.3);border-radius:999px;color:#fff!important;background:rgba(255,255,255,.12);font-size:11px;font-weight:850;letter-spacing:.13em}.hero-course{font-size:13px;color:#d9ddff!important}.hero h1{position:relative;z-index:1;max-width:780px;margin:0 0 12px;color:#fff!important;font-size:clamp(30px,5vw,56px);line-height:1.04;letter-spacing:-.035em}.hero-copy{position:relative;z-index:1;max-width:760px;margin:0;color:#e1e4ff!important;font-size:16px;line-height:1.65}.hero-links{position:relative;z-index:1;display:flex;flex-wrap:wrap;gap:9px;margin-top:22px}.hero-link{display:inline-flex;padding:10px 14px;border:1px solid rgba(255,255,255,.22);border-radius:10px;color:white!important;text-decoration:none!important;background:rgba(255,255,255,.08);font-size:13px;font-weight:750}.hero-link.primary{color:#272b72!important;background:white}.lab-strip{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;overflow:hidden;border:1px solid #dde2ec;border-top:0;border-radius:0 0 18px 18px;background:#dde2ec}.lab-strip span{padding:13px 16px;color:var(--muted);background:#fff;font-size:11px;text-transform:uppercase;letter-spacing:.07em}.lab-strip strong{display:block;margin-top:3px;color:var(--ink);font-size:13px;text-transform:none;letter-spacing:0}
.catalog-card,.control-card,.chart-card,.output-card>div,.task-brief{border:1px solid #e1e6ef!important;border-radius:17px!important;background:#fff!important;box-shadow:0 14px 34px rgba(31,42,77,.055)!important}.catalog-card{padding:24px!important;margin-bottom:18px!important}.panel-title{margin:0 0 5px!important;color:var(--ink);font-size:18px!important}.panel-copy,.artifact-note{margin:0 0 18px!important;color:var(--muted)!important;font-size:13px!important;line-height:1.55!important}
.experiment-gallery{margin-top:6px!important;border:0!important;background:transparent!important}.experiment-gallery .grid-wrap{height:auto!important;min-height:0!important}.experiment-gallery .grid-container{display:grid!important;grid-template-columns:repeat(3,minmax(0,1fr))!important;gap:13px!important}.experiment-gallery button,.experiment-gallery .thumbnail-item{position:relative!important;height:auto!important;aspect-ratio:16/9!important;overflow:hidden!important;border:2px solid transparent!important;border-radius:14px!important;background:#101532!important;transition:transform .16s ease,border-color .16s ease!important}.experiment-gallery button:hover{transform:translateY(-2px);border-color:#7778eb!important}.experiment-gallery img{width:100%!important;height:100%!important;object-fit:cover!important}.experiment-gallery .caption-label{position:absolute!important;z-index:2!important;inset:auto 0 0!important;padding:34px 13px 12px!important;color:#fff!important;background:linear-gradient(transparent,rgba(4,8,28,.92))!important;font-size:12px!important;font-weight:800!important;line-height:1.35!important;white-space:pre-line!important;text-align:left!important}
.task-brief{display:grid;grid-template-columns:minmax(270px,.9fr) minmax(0,1.7fr);gap:26px;margin:18px 0!important;padding:13px!important;background:linear-gradient(135deg,#fff,#f7f9ff)!important}.task-brief__visual{overflow:hidden;border-radius:12px;background:#101532}.task-brief__visual img{display:block;width:100%;height:100%;min-height:205px;object-fit:cover}.task-brief__body{padding:12px 12px 8px 0}.task-kicker{display:block;margin-bottom:7px;color:#5b5ce2;font-size:10px;font-weight:900;letter-spacing:.13em}.task-brief h3{margin:0 0 5px;color:var(--ink);font-size:23px}.task-brief p{margin:0;color:var(--muted);font-size:13px;line-height:1.55}.task-facts{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:14px 0}.task-facts span{padding:9px 11px;border:1px solid #e1e6ef;border-radius:9px;background:rgba(255,255,255,.88);color:var(--ink);font-size:11px}.task-facts b{display:block;margin-bottom:3px;color:#7a879d;font-size:8px;letter-spacing:.12em;text-transform:uppercase}.task-hint{font-weight:650;color:#465166!important}
.control-card,.chart-card{padding:25px!important}.primary-btn{min-height:46px!important;border:0!important;border-radius:10px!important;background:linear-gradient(135deg,#5b5ce2,#7c4dff)!important;font-weight:850!important}.run-state,.live-metric{display:flex;gap:11px;align-items:center;margin-top:10px;padding:14px 15px;border:1px solid #e3e7ef;border-radius:12px;background:#fafbfe}.run-state__dot{width:9px;height:9px;border-radius:50%;background:#8b95a8;box-shadow:0 0 0 5px rgba(139,149,168,.12)}.run-state--running .run-state__dot{background:#8b5cf6;animation:pulse 1.2s infinite}.run-state--complete .run-state__dot{background:#13a36f}.run-state--error .run-state__dot{background:#e05252}.summary-label{display:block;margin-bottom:2px;color:#7b879c;font-size:8px;font-weight:850;letter-spacing:.12em;text-transform:uppercase}.run-state strong,.live-metric strong{display:block;color:var(--ink);font-size:14px}.run-state small,.live-metric small{display:block;margin-top:2px;color:var(--muted);font-size:11px}.metric-reading{display:flex;gap:9px;align-items:baseline}.metric-reading strong{font-size:20px}
.console-panel{overflow:hidden;margin-top:14px;border:1px solid #29315e;border-radius:12px;background:#11162d}.console-head{padding:9px 13px;border-bottom:1px solid #28305b;color:#dbe1ff;font-size:11px;font-weight:800}.console-dot{display:inline-block;width:7px;height:7px;margin-right:8px;border-radius:50%;background:#31d39b}.console-text{height:300px!important;margin:0!important;padding:13px!important;overflow:auto!important;color:#d5dcf4!important;background:#11162d!important;font:11px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace!important;white-space:pre-wrap!important}.run-wait{display:flex;gap:12px;align-items:center;margin:0 0 13px;padding:13px 15px;border:1px solid #d6d8ff;border-radius:11px;background:#f8f7ff;color:#313774}.run-wait strong,.run-wait small,.run-wait em{display:block}.run-wait small{margin-top:2px;color:#68748a;font-size:11px}.run-wait em{margin-top:4px;color:#5b5ce2;font-size:10px;font-style:normal;font-weight:800}.run-wait__spinner{width:20px;height:20px;border:2px solid #d7d9ff;border-top-color:#5b5ce2;border-radius:50%;animation:spin .75s linear infinite}
.output-card{margin-top:18px!important}.output-card>div{padding:24px!important}.policy-preview,.policy-preview .image-container,.policy-preview img{min-height:320px!important}.policy-preview img{max-height:520px!important;object-fit:contain!important;background:#0f1430}.footer-note{padding:27px 0 0;text-align:center;color:#8390a6;font-size:11px}.footer-note a{color:#5b5ce2!important;font-weight:750}
@keyframes spin{to{transform:rotate(360deg)}}@keyframes pulse{50%{box-shadow:0 0 0 8px rgba(139,92,246,.08)}}
@media(max-width:900px){.lab-strip{grid-template-columns:1fr 1fr}.experiment-gallery .grid-container{grid-template-columns:repeat(2,minmax(0,1fr))!important}.task-brief{grid-template-columns:1fr}.task-brief__body{padding:7px}.language-bar{position:static!important;margin:-58px 12px 16px auto!important}.brand-lockup{max-width:calc(100% - 220px)}.hero{padding-top:24px}}
@media(max-width:620px){.gradio-container{padding:10px!important}.hero{padding:22px 20px 24px;border-radius:18px 18px 0 0}.brand-lockup{max-width:100%;font-size:10px;letter-spacing:.045em}.experiment-gallery .grid-container{grid-template-columns:1fr!important}.task-facts{grid-template-columns:1fr}.lab-strip{grid-template-columns:1fr}.language-switch{width:190px!important;min-width:190px!important}}
"""


AUTO_SCROLL_JS = r"""
function initializeGameLabUi(){
  if(window.__gameLabUiReady)return;window.__gameLabUiReady=true;
  let active=null,follow=true,saved=0,internal=false;
  const update=()=>{
    const element=document.querySelector("#live-training-console .console-text");
    if(element&&element!==active){active=element;follow=true;saved=0;active.addEventListener("scroll",()=>{if(internal)return;follow=active.scrollHeight-active.clientHeight-active.scrollTop<=24;saved=active.scrollTop},{passive:true})}
    if(active){internal=true;if(follow)active.scrollTop=active.scrollHeight;else active.scrollTop=Math.min(saved,Math.max(0,active.scrollHeight-active.clientHeight));requestAnimationFrame(()=>{internal=false})}
    const timer=document.querySelector(".run-wait__elapsed");if(timer){const elapsed=Math.max(0,Math.floor((Date.now()-Number(timer.dataset.startMs))/1000));timer.textContent=`${elapsed}s ${timer.dataset.label}`}
  };
  new MutationObserver(()=>requestAnimationFrame(update)).observe(document.body,{childList:true,subtree:true,characterData:true});setInterval(update,1000);update();
}
initializeGameLabUi();
"""


def launch(space_module: Any) -> None:
    demo = build_demo(space_module)
    demo.queue(default_concurrency_limit=1, max_size=12).launch(css=CSS, js=AUTO_SCROLL_JS, footer_links=[])
