#!/usr/bin/env python3
"""Generate the ModelScope companion notebooks for every online experiment.

The Studio runtime remains the single source of truth.  Each notebook clones
this repository, installs the matching Studio dependencies once, imports the
same runtime module, and streams the same training events used by Gradio.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "code" / "online-experiments"
REPOSITORY = "walkinglabs/hands-on-modern-rl"
MODELSCOPE_PREFIX = "https://modelscope.cn/notebook/share/github"


EXPERIMENTS = [
    {
        "slug": "hands-on-modern-rl-experiment01-cartpole",
        "title": "Experiment 01 · CartPole PPO",
        "summary": "Train PPO on CartPole-v1 with CPU, inspect evaluation rewards, and render the learned policy.",
        "resource": "CPU",
        "kind": "cartpole",
        "task": "CartPole-v1",
        "budget": 30_000,
        "full_budget": "30,000 environment steps (usually under one minute on a notebook CPU)",
    },
    {
        "slug": "hands-on-modern-rl-experiment-gymnasium",
        "title": "Gymnasium CPU Playground",
        "summary": "Run the same curated Gymnasium recipes as the Studio, from bandits and grids to control tasks.",
        "resource": "CPU",
        "kind": "gymnasium",
        "task": "Bandit · ε-greedy",
        "budget": 2_000,
        "full_budget": "Use each task's default budget from EXPERIMENTS for a full comparison",
    },
    {
        "slug": "hands-on-modern-rl-experiment02-vizdoom",
        "title": "Experiment 02 · ViZDoom CPU Arena",
        "summary": "Train a DQN agent in a real Doom-engine scenario and record the learned first-person policy.",
        "resource": "CPU",
        "kind": "runtime",
        "task": "basic",
        "budget": 2_000,
        "full_budget": "12,000+ environment steps for the Basic scenario",
    },
    {
        "slug": "hands-on-modern-rl-experiment03-atari",
        "title": "Experiment 03 · Atari CPU Arcade",
        "summary": "Train DQN from ALE pixels, evaluate checkpoints, and replay the learned Atari policy.",
        "resource": "CPU",
        "kind": "runtime",
        "task": "freeway",
        "budget": 2_000,
        "full_budget": "20,000+ environment steps for a visible Freeway learning curve",
    },
    {
        "slug": "hands-on-modern-rl-experiment04-board-selfplay",
        "title": "Experiment 04 · Board Games & Self-Play",
        "summary": "Run CFR+ or tabular self-play in OpenSpiel and inspect the learned game policy.",
        "resource": "CPU",
        "kind": "runtime",
        "task": "kuhn-poker",
        "budget": 500,
        "full_budget": "2,000+ CFR iterations for Kuhn Poker",
    },
    {
        "slug": "hands-on-modern-rl-experiment05-multiagent-games",
        "title": "Experiment 05 · Multi-Agent Games",
        "summary": "Train a shared PPO policy in PettingZoo and replay all agents in one synchronized result.",
        "resource": "CPU",
        "kind": "runtime",
        "task": "simple-spread",
        "budget": 2_000,
        "full_budget": "20,000+ environment steps for cooperative navigation",
    },
    {
        "slug": "hands-on-modern-rl-experiment06-minigrid-adventure",
        "title": "Experiment 06 · MiniGrid Adventures",
        "summary": "Train an exploration policy in a partially observed maze and render its learned route.",
        "resource": "CPU",
        "kind": "runtime",
        "task": "empty",
        "budget": 2_000,
        "full_budget": "12,000+ environment steps for Empty-6x6",
    },
    {
        "slug": "hands-on-modern-rl-experiment07-jax-games",
        "title": "Experiment 07 · JAX MinAtar Games",
        "summary": "JIT-compile a policy-gradient update on CPU and train inside a compact MinAtar game.",
        "resource": "CPU",
        "kind": "runtime",
        "task": "breakout",
        "budget": 100,
        "full_budget": "1,000+ episodes after the first JAX compilation",
    },
    {
        "slug": "hands-on-modern-rl-experiment08-maniskill",
        "title": "Experiment 08 · ManiSkill Robot Lab",
        "summary": "Train a robot policy with CUDA PPO and PhysX, then render the learned manipulation trajectory.",
        "resource": "xGPU",
        "kind": "runtime",
        "task": "push-cube",
        "budget": 10_000,
        "full_budget": "100,000+ transitions with a scheduled CUDA notebook",
    },
    {
        "slug": "hands-on-modern-rl-experiment10-minestudio",
        "title": "Experiment 10 · MineStudio Minecraft Agent",
        "summary": "Start the real Minecraft simulator, train visual PPO on xGPU, and record a first-person replay.",
        "resource": "xGPU",
        "kind": "runtime",
        "task": "mine-dirt",
        "budget": 1_024,
        "full_budget": "10,000+ simulator steps after the one-time engine download",
    },
    {
        "slug": "hands-on-modern-rl-experiment11-unity-mlagents",
        "title": "Experiment 11 · Unity ML-Agents Arena",
        "summary": "Train Huggy the Dog to fetch a stick in a real Unity Linux scene, or switch to another ML-Agents task.",
        "resource": "xGPU",
        "kind": "runtime",
        "task": "unity-huggy",
        "budget": 100_000,
        "gamma": "0.995",
        "epsilon": "0.20",
        "full_budget": "up to 2,000,000 Unity steps after the one-time 39 MB Huggy scene download",
    },
    {
        "slug": "hands-on-modern-rl-experiment12-ai2thor-embodied",
        "title": "Experiment 12 · AI2-THOR Embodied Home",
        "summary": "Train a visual ObjectNav policy in an interactive AI2-THOR room and replay the learned route.",
        "resource": "xGPU",
        "kind": "runtime",
        "task": "find-mug",
        "budget": 2_000,
        "full_budget": "10,000+ simulator steps with Xvfb and CUDA",
    },
]


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(source).strip() + "\n"}


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip() + "\n",
    }


def notebook_url(slug: str) -> str:
    path = f"code/online-experiments/{slug}.ipynb"
    return f"{MODELSCOPE_PREFIX}/{REPOSITORY}/blob/main/{path}"


def intro_cell(spec: dict) -> dict:
    studio = f"https://modelscope.cn/studios/walkinglab/{spec['slug']}"
    source = f"https://github.com/{REPOSITORY}/tree/main/modelscope-space/{spec['slug']}"
    return markdown_cell(
        f"""
        # {spec['title']}

        **WalkingLab × Hands-On Modern RL companion experiment notebook**

        {spec['summary']}

        - Resource profile: **{spec['resource']}**
        - Quick run in this notebook: **{spec['budget']:,}** training units
        - Full experiment: {spec['full_budget']}
        - [Live ModelScope Studio]({studio})
        - [Experiment source]({source})
        - [Hands-On Modern RL](https://github.com/{REPOSITORY}) · [WalkingLab](https://modelscope.cn/organization/walkinglab)

        The notebook imports the exact runtime used by the Studio. Change the parameters below, run the cells in order,
        and compare the checkpoint curve with the final policy GIF or result image. The first setup can take longer because
        native environments and simulator assets are cached; later runs reuse `/mnt/workspace/hands-on-modern-rl-notebooks`.
        """
    )


def learning_cell(spec: dict) -> dict:
    gpu_note = (
        "A scheduled ModelScope **xGPU Notebook** is required. The run cell stops early if CUDA is unavailable."
        if spec["resource"] == "xGPU"
        else "A normal ModelScope **CPU Notebook** is sufficient; no GPU is required."
    )
    return markdown_cell(
        f"""
        ## 1. Question and run boundary

        This experiment asks whether the selected policy improves on the task's evaluation metric as its training budget
        increases. Start with the quick budget to verify the environment and logs. Then increase the budget only after the
        complete result cell produces a curve and an artifact.

        {gpu_note}

        A short smoke run proves that the pipeline executes; it does not prove convergence. Use the full budget above when
        comparing algorithms or reporting a learned behavior.
        """
    )


def setup_cell(spec: dict) -> dict:
    return code_cell(
        f'''
        from __future__ import annotations

        import hashlib
        import os
        import subprocess
        import sys
        from pathlib import Path

        REPOSITORY_URL = "https://github.com/{REPOSITORY}.git"
        SPACE_SLUG = "{spec['slug']}"
        INSTALL_DEPENDENCIES = True

        def locate_or_clone_repo() -> Path:
            here = Path.cwd().resolve()
            for candidate in (here, *here.parents):
                if (candidate / "modelscope-space" / SPACE_SLUG).is_dir():
                    return candidate
            workspace = Path("/mnt/workspace") if Path("/mnt/workspace").is_dir() else Path.cwd()
            target = workspace / "hands-on-modern-rl-notebooks" / "source"
            target.parent.mkdir(parents=True, exist_ok=True)
            if (target / ".git").is_dir():
                subprocess.run(["git", "-C", str(target), "pull", "--ff-only"], check=True)
            else:
                subprocess.run(["git", "clone", "--depth", "1", REPOSITORY_URL, str(target)], check=True)
            return target

        REPO_ROOT = locate_or_clone_repo()
        SPACE_DIR = REPO_ROOT / "modelscope-space" / SPACE_SLUG
        requirements = SPACE_DIR / "requirements.txt"
        packages = SPACE_DIR / "packages.txt"
        cache_root = Path("/mnt/workspace/hands-on-modern-rl-notebooks") if Path("/mnt/workspace").is_dir() else REPO_ROOT / ".cache" / "online-experiments"
        cache_root.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(requirements.read_bytes() + (packages.read_bytes() if packages.exists() else b"")).hexdigest()[:12]
        marker = cache_root / f"{{SPACE_SLUG}}-{{digest}}.ready"

        if INSTALL_DEPENDENCIES and not marker.exists():
            if packages.exists() and sys.platform.startswith("linux") and hasattr(os, "geteuid") and os.geteuid() == 0:
                system_packages = [line.strip() for line in packages.read_text().splitlines() if line.strip() and not line.startswith("#")]
                subprocess.run(["apt-get", "update"], check=True)
                subprocess.run(["apt-get", "install", "-y", "--no-install-recommends", *system_packages], check=True)
            subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "-r", str(requirements)], check=True)
            marker.touch()
        else:
            print(f"Dependency cache ready: {{marker}}")

        os.chdir(SPACE_DIR)
        if str(SPACE_DIR) not in sys.path:
            sys.path.insert(0, str(SPACE_DIR))
        print(f"Repository: {{REPO_ROOT}}")
        print(f"Experiment runtime: {{SPACE_DIR}}")
        '''
    )


def runtime_parameter_cell(spec: dict) -> dict:
    return code_cell(
        f'''import importlib

if SPACE_SLUG.endswith("experiment10-minestudio"):
    subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-deps", "minestudio==1.1.6"], check=True)
if SPACE_SLUG.endswith("experiment11-unity-mlagents"):
    from bootstrap_mlagents import ensure_mlagents
    ensure_mlagents()

runtime = importlib.import_module("space_runtime")
tasks = {{item["key"]: item for item in runtime.TASKS}}
print("Available tasks:")
for key, item in tasks.items():
    title = item.get("title", {{}})
    print(f"  {{key:20s}} {{title.get('en', title)}} · {{item.get('environment', 'environment provided by runtime')}}")

TASK_KEY = "{spec['task']}"
TRAINING_BUDGET = {spec['budget']}
LEARNING_RATE = 3e-4
GAMMA = {spec.get('gamma', '0.99')}
EPSILON = {spec.get('epsilon', '0.10')}
SEED = 42

if TASK_KEY not in tasks:
    raise ValueError(f"Unknown TASK_KEY={{TASK_KEY!r}}. Choose one of {{list(tasks)}}")
selected_task = tasks[TASK_KEY]
print("\\nSelected:", selected_task.get("title", {{}}).get("en", TASK_KEY))
print("Algorithm:", selected_task.get("algorithm"))
print("Budget:", TRAINING_BUDGET)
'''
    )


def runtime_run_cell(spec: dict) -> dict:
    gpu_check = """import torch
if not torch.cuda.is_available():
    raise RuntimeError("This experiment requires a scheduled ModelScope xGPU Notebook; CUDA is not visible.")
print("CUDA:", torch.cuda.get_device_name(0))""" if spec["resource"] == "xGPU" else "print('Device: CPU')"
    return code_cell(
        f'''from IPython.display import display
import matplotlib.pyplot as plt

{gpu_check}

print("Starting the same training generator used by the live Studio...\\n")
events = []
for event in runtime.run(TASK_KEY, TRAINING_BUDGET, LEARNING_RATE, GAMMA, EPSILON, SEED):
    events.append(dict(event))
    message = event.get("log") or event.get("detail")
    if message:
        print(message, flush=True)

if not events:
    raise RuntimeError("The runtime returned no training events")
final_event = events[-1]
print("\\nFinal phase:", final_event.get("phase", "complete"))
print("Final score:", final_event.get("score", "reported in the log"))
'''
    )


def runtime_result_cell() -> dict:
    return code_cell(
        '''
        from IPython.display import Image as NotebookImage

        x = final_event.get("x", [])
        y = final_event.get("y", [])
        if x and y:
            fig, ax = plt.subplots(figsize=(8, 4.2))
            ax.plot(x, y, marker="o", color="#5b5ce2", linewidth=2)
            ax.set_title(f"{TASK_KEY} · checkpoint evaluation")
            ax.set_xlabel("Training progress")
            ax.set_ylabel("Evaluation score")
            ax.grid(alpha=0.25)
            plt.show()
        else:
            print("This task reports its result through the artifact rather than a scalar learning curve.")

        preview = final_event.get("preview")
        if preview and Path(preview).exists():
            print("Learned-policy artifact:", preview)
            display(NotebookImage(filename=str(preview)))
        else:
            print("No replay path was returned. Inspect the final log and the artifacts directory:", SPACE_DIR / "artifacts")

        artifact = final_event.get("artifact") or final_event.get("model")
        if artifact:
            print("Downloadable artifact:", artifact)
        '''
    )


def cartpole_cells(spec: dict) -> list[dict]:
    return [
        code_cell(
            f'''
            from train import train

            TIMESTEPS = {spec['budget']}
            OUTPUT_DIR = cache_root / "results" / SPACE_SLUG
            train(total_timesteps=TIMESTEPS, output_dir=OUTPUT_DIR)
            '''
        ),
        code_cell(
            '''
            from IPython.display import Image as NotebookImage, display

            curve = OUTPUT_DIR / "reward-curve.png"
            replay = OUTPUT_DIR / "cartpole-trained-policy.gif"
            model = OUTPUT_DIR / "ppo-cartpole.zip"
            for path in (curve, replay, model):
                if not path.exists():
                    raise FileNotFoundError(path)
            display(NotebookImage(filename=str(curve)))
            display(NotebookImage(filename=str(replay)))
            print("Saved model:", model)
            '''
        ),
    ]


def gymnasium_cells(spec: dict) -> list[dict]:
    return [
        code_cell(
            f'''
            import app as playground

            EXPERIMENT = "{spec['task']}"
            TRAINING_BUDGET = {spec['budget']}
            LEARNING_RATE = 0.1
            GAMMA = 0.99
            EPSILON = 0.10
            SEED = 42

            print("Curated experiments:")
            for name in playground.EXPERIMENTS:
                print(" ", name)
            if EXPERIMENT not in playground.EXPERIMENTS:
                raise ValueError(f"Choose EXPERIMENT from {{list(playground.EXPERIMENTS)}}")
            '''
        ),
        code_cell(
            '''
            import html as html_module
            import re
            from IPython.display import Image as NotebookImage, display

            results = []
            for result in playground.train(EXPERIMENT, TRAINING_BUDGET, LEARNING_RATE, GAMMA, EPSILON, SEED, "English"):
                results.append(result)
                console = result[-1]
                value = getattr(console, "value", str(console))
                text = html_module.unescape(re.sub(r"<[^>]+>", "", str(value)))
                print(text[-1800:], flush=True)

            if not results:
                raise RuntimeError("The playground returned no training result")
            status, metric, curve, preview, artifact, console = results[-1]
            display(curve)
            if isinstance(preview, (str, Path)) and Path(preview).exists():
                display(NotebookImage(filename=str(preview)))
            else:
                display(preview)
            print("Result artifact:", getattr(artifact, "value", artifact))
            '''
        ),
    ]


def reflection_cell(spec: dict) -> dict:
    return markdown_cell(
        f"""
        ## 4. Read the result before increasing the budget

        Compare the first and last checkpoint values, then inspect the replay. A rising curve with an implausible replay can
        indicate reward shaping, evaluation, or rendering problems. A flat quick run is also inconclusive: this notebook's
        default budget is a pipeline check. For a training claim, rerun with **{spec['full_budget']}**, keep the seed fixed,
        and compare at least three seeds before drawing a conclusion.
        """
    )


def build_notebook(spec: dict) -> dict:
    cells = [intro_cell(spec), learning_cell(spec), markdown_cell("## 2. Prepare the matching Studio runtime"), setup_cell(spec)]
    if spec["kind"] == "runtime":
        cells.extend([
            markdown_cell("## 3. Choose a task and train"),
            runtime_parameter_cell(spec),
            runtime_run_cell(spec),
            runtime_result_cell(),
        ])
    elif spec["kind"] == "cartpole":
        cells.extend([markdown_cell("## 3. Train PPO and inspect the artifacts"), *cartpole_cells(spec)])
    else:
        cells.extend([markdown_cell("## 3. Choose a recipe and train"), *gymnasium_cells(spec)])
    cells.append(reflection_cell(spec))
    notebook_id = hashlib.sha1(spec["slug"].encode("utf-8")).hexdigest()[:10]
    for index, cell in enumerate(cells):
        cell["id"] = f"{notebook_id}-{index:02d}"
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
            "homrl": {
                "studio": spec["slug"],
                "resource": spec["resource"],
                "modelscope_notebook_url": notebook_url(spec["slug"]),
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for spec in EXPERIMENTS:
        path = OUTPUT / f"{spec['slug']}.ipynb"
        path.write_text(json.dumps(build_notebook(spec), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(path.relative_to(ROOT), "->", notebook_url(spec["slug"]))


if __name__ == "__main__":
    main()
