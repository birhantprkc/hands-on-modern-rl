from __future__ import annotations

import math
import random
from collections import defaultdict
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent

SPACE = {
    "title": {"en": "Board Games & Self-Play Lab", "zh": "棋盘游戏与自博弈训练场"},
    "description": {
        "en": "Train tabular self-play and counterfactual-regret agents in real OpenSpiel games, then replay the policy's decisions move by move.",
        "zh": "在真实 OpenSpiel 游戏中训练表格自博弈和反事实遗憾最小化策略，并逐步回放策略决策。",
    },
    "badge": "EXPERIMENT 04 · SELF-PLAY",
    "course_url": "https://walkinglabs.github.io/hands-on-modern-rl/chapter32_selfplay/",
    "source_url": "https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment04-board-selfplay/file/view/master/space_runtime.py",
}


def task(key, title, environment, description, observation, action, algorithm, preview, budget):
    return {
        "key": key,
        "title": {"en": title, "zh": title},
        "environment": environment,
        "description": description,
        "observation": observation,
        "action": action,
        "algorithm": algorithm,
        "preview": preview,
        "budget": budget,
        "learning_rate": (0.01, 1.0, 0.25, 0.01),
        "gamma": (0.8, 1.0, 1.0, 0.01),
        "epsilon": (0.0, 1.0, 0.15, 0.01),
        "checkpoints": 8,
    }


TASKS = [
    task("kuhn-poker", "Kuhn Poker", "kuhn_poker", {"en": "Learn an equilibrium strategy in a tiny imperfect-information poker game.", "zh": "在小型不完全信息扑克游戏中学习均衡策略。"}, {"en": "Private card and betting history", "zh": "私有牌和下注历史"}, {"en": "Pass or bet", "zh": "过牌或下注"}, "CFR+", "assets/kuhn-poker.png", (50, 20_000, 2_000, 50)),
    task("leduc-poker", "Leduc Poker", "leduc_poker", {"en": "Balance betting, folding, and hidden information across two betting rounds.", "zh": "在两轮下注中权衡下注、弃牌和隐藏信息。"}, {"en": "Private/public cards and betting history", "zh": "私有牌、公共牌和下注历史"}, {"en": "Fold / call / raise", "zh": "弃牌、跟注、加注"}, "CFR+", "assets/leduc-poker.png", (50, 10_000, 1_000, 50)),
    task("tic-tac-toe", "Tic-Tac-Toe", "tic_tac_toe", {"en": "Discover blocking, forks, and winning lines through tabular self-play.", "zh": "通过表格自博弈发现阻挡、双威胁和获胜连线。"}, {"en": "3×3 board", "zh": "3×3 棋盘"}, {"en": "Place a mark in a legal cell", "zh": "在合法空格落子"}, "Self-play Q-Learning", "assets/tic-tac-toe.png", (100, 100_000, 10_000, 100)),
    task("connect-four", "Connect Four", "connect_four", {"en": "Learn vertical, horizontal, and diagonal threats through self-play.", "zh": "通过自博弈学习纵向、横向和斜向威胁。"}, {"en": "7×6 board", "zh": "7×6 棋盘"}, {"en": "Drop a piece into a legal column", "zh": "在合法列中投入棋子"}, "Self-play Q-Learning", "assets/connect-four.png", (500, 200_000, 20_000, 500)),
]


def runtime_status():
    try:
        import pyspiel

        game = pyspiel.load_game("tic_tac_toe")
        game.new_initial_state()
        return "OpenSpiel · READY"
    except Exception as exc:
        return f"startup check pending · {type(exc).__name__}"


def _font(size: int):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _frame(title: str, state_text: str, footer: str, size=(760, 460)) -> np.ndarray:
    image = Image.new("RGB", size, "#f4f6fa")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((18, 18, size[0] - 18, size[1] - 18), radius=22, fill="#ffffff", outline="#dfe4ef", width=2)
    draw.rounded_rectangle((34, 34, 235, size[1] - 34), radius=16, fill="#20245b")
    draw.text((55, 62), "LEARNED POLICY", font=_font(15), fill="#a5b4fc")
    draw.multiline_text((55, 105), title, font=_font(25), fill="white", spacing=7)
    draw.multiline_text((55, 310), footer, font=_font(14), fill="#cbd5e1", spacing=6)
    draw.text((275, 50), "Game state", font=_font(20), fill="#172033")
    draw.multiline_text((275, 94), state_text, font=_font(22), fill="#27324a", spacing=9)
    return np.asarray(image)


def _save_replay(key: str, title: str, states: list[tuple[str, str]]) -> str:
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    frames = [_frame(title, state, footer) for state, footer in states]
    path = artifacts / f"{key}-learned-policy.gif"
    imageio.mimsave(path, frames, duration=.75, loop=0)
    return str(path)


def _sample(probabilities: dict[int, float], rng: random.Random) -> int:
    actions = list(probabilities)
    weights = np.asarray([max(0.0, probabilities[action]) for action in actions], dtype=float)
    if weights.sum() <= 0:
        return rng.choice(actions)
    weights /= weights.sum()
    return actions[int(np.searchsorted(np.cumsum(weights), rng.random(), side="right").clip(0, len(actions) - 1))]


def _cfr_run(task, budget: int, seed: int):
    import pyspiel
    from open_spiel.python.algorithms import cfr, exploitability

    game = pyspiel.load_game(task["environment"])
    solver = cfr.CFRPlusSolver(game)
    checkpoint = max(1, budget // int(task["checkpoints"]))
    x: list[float] = []
    y: list[float] = []
    for iteration in range(1, budget + 1):
        solver.evaluate_and_update_policy()
        if iteration == 1 or iteration % checkpoint == 0 or iteration == budget:
            gap = float(exploitability.exploitability(game, solver.average_policy()))
            score = -gap
            x.append(float(iteration)); y.append(score)
            yield {"step": iteration, "score": score, "x": x, "y": y, "metric_detail": f"negative exploitability · gap={gap:.6f}", "log": f"CFR+ iteration={iteration:,} exploitability={gap:.8f}"}

    rng = random.Random(seed + 1000)
    state = game.new_initial_state()
    policy = solver.average_policy()
    replay: list[tuple[str, str]] = [(str(state), "Initial state")]
    while not state.is_terminal():
        if state.is_chance_node():
            outcomes = dict(state.chance_outcomes())
            action = _sample(outcomes, rng)
            label = f"Chance → {action}"
        else:
            player = state.current_player()
            probabilities = policy.action_probabilities(state, player)
            action = _sample(probabilities, rng)
            label = f"Player {player} → {state.action_to_string(player, action)}"
        state.apply_action(action)
        replay.append((str(state), label))
    replay.append((str(state), f"Returns: {state.returns()}"))
    preview = _save_replay(task["key"], task["title"]["en"], replay)
    yield {"phase": "complete", "step": budget, "score": y[-1], "x": x, "y": y, "preview": preview, "log": "Sampled a terminal game from the learned average policy"}


def _state_key(state, player: int) -> str:
    try:
        return state.information_state_string(player)
    except Exception:
        try:
            return state.observation_string(player)
        except Exception:
            return str(state)


def _choose_q(q, state, player: int, epsilon: float, rng: random.Random) -> int:
    legal = state.legal_actions(player)
    if rng.random() < epsilon:
        return rng.choice(legal)
    key = _state_key(state, player)
    values = [q[(player, key, action)] for action in legal]
    best = max(values)
    return rng.choice([action for action, value in zip(legal, values) if value == best])


def _evaluate_q(game, q, seed: int, episodes: int = 100) -> float:
    rng = random.Random(seed)
    wins = 0.0
    for episode in range(episodes):
        learner = episode % 2
        state = game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                action = _sample(dict(state.chance_outcomes()), rng)
            else:
                player = state.current_player()
                action = _choose_q(q, state, player, 0.0, rng) if player == learner else rng.choice(state.legal_actions(player))
            state.apply_action(action)
        result = state.returns()[learner]
        wins += 1.0 if result > 0 else .5 if result == 0 else 0.0
    return wins / episodes


def _q_run(task, budget: int, alpha: float, gamma: float, epsilon: float, seed: int):
    import pyspiel

    game = pyspiel.load_game(task["environment"])
    q = defaultdict(float)
    rng = random.Random(seed)
    checkpoint = max(1, budget // int(task["checkpoints"]))
    x: list[float] = []
    y: list[float] = []
    for episode in range(1, budget + 1):
        state = game.new_initial_state()
        last: dict[int, tuple[str, int]] = {}
        while not state.is_terminal():
            if state.is_chance_node():
                state.apply_action(_sample(dict(state.chance_outcomes()), rng))
                continue
            player = state.current_player()
            key = _state_key(state, player)
            action = _choose_q(q, state, player, epsilon, rng)
            previous = last.get(player)
            if previous is not None:
                legal = state.legal_actions(player)
                bootstrap = max((q[(player, key, candidate)] for candidate in legal), default=0.0)
                q[(player, previous[0], previous[1])] += alpha * (gamma * bootstrap - q[(player, previous[0], previous[1])])
            last[player] = (key, action)
            state.apply_action(action)
        returns = state.returns()
        for player, (key, action) in last.items():
            q[(player, key, action)] += alpha * (returns[player] - q[(player, key, action)])
        if episode == 1 or episode % checkpoint == 0 or episode == budget:
            win_rate = _evaluate_q(game, q, seed + episode, episodes=80)
            x.append(float(episode)); y.append(win_rate)
            yield {"step": episode, "score": win_rate, "x": x, "y": y, "metric_detail": "win/draw rate versus random", "log": f"SELFPLAY episode={episode:,} states={len(q):,} win_or_draw_rate={win_rate:.3f}"}

    state = game.new_initial_state()
    replay: list[tuple[str, str]] = [(str(state), "Initial board")]
    while not state.is_terminal():
        player = state.current_player()
        action = _choose_q(q, state, player, 0.0, rng)
        label = f"Player {player} → {state.action_to_string(player, action)}"
        state.apply_action(action)
        replay.append((str(state), label))
    replay.append((str(state), f"Returns: {state.returns()}"))
    preview = _save_replay(task["key"], task["title"]["en"], replay)
    yield {"phase": "complete", "step": budget, "score": y[-1], "x": x, "y": y, "preview": preview, "log": "Rendered a complete greedy self-play game from the learned Q table"}


def run(key: str, budget: int, learning_rate: float, gamma: float, epsilon: float, seed: int):
    task = next(item for item in TASKS if item["key"] == key)
    if task["algorithm"] == "CFR+":
        yield from _cfr_run(task, budget, seed)
    else:
        yield from _q_run(task, budget, learning_rate, gamma, epsilon, seed)
