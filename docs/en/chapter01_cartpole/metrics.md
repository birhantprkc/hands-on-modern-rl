---
title: 1.2 Rewards and Training Metrics
outline: [2, 3]
---

# 1.2 Rewards and Training Metrics

> **Evidence used in this section**: [pure PyTorch PPO](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code/chapter01_cartpole/2-pytorch_ppo.py) · [raw metrics CSV](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code/chapter01_cartpole/output/training_metrics_seed42.csv) · [plotting script](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code/chapter01_cartpole/plot_curves.py)

Section 1.1 described the state, actions, and termination conditions. Once training starts, we need evidence that the policy actually improves. A program that merely finishes without an exception is not enough. We should check episode return, an independent evaluation, and the size of policy updates.

## Where the data comes from

This page analyzes one explicitly recorded run. It does not combine different scripts, random seeds, or old dashboard exports.

| Item              | Measured-run setting                                         |
| ----------------- | ------------------------------------------------------------ |
| Date              | 2026-08-13                                                   |
| Environment       | `CartPole-v1`                                                |
| Algorithm         | the repository's pure PyTorch PPO                            |
| Random seed       | `42`                                                         |
| Training budget   | 40 iterations × 2,048 steps = 81,920 environment steps       |
| Evaluation        | 20 independent episodes with a deterministic policy          |
| Measured software | Python 3.12.13, PyTorch 2.13.0, Gymnasium 1.3.0, NumPy 2.5.2 |

Use this command to reproduce the run. `--swanlab-mode disabled` disables dashboard logging only; it does not change training.

```bash
cd code/chapter01_cartpole
python 2-pytorch_ppo.py \
  --seed 42 \
  --iterations 40 \
  --steps-per-rollout 2048 \
  --swanlab-mode disabled \
  --log-csv output/training_metrics_seed42.csv

python plot_curves.py \
  --input output/training_metrics_seed42.csv \
  --output-dir output
```

The following lines are taken from that run. Every value can be checked against the CSV.

```text
Iteration  1/40 | episodes: 91 | mean reward:  21.4 | KL: 0.0087 | clip: 14.5%
Iteration  5/40 | episodes: 15 | mean reward: 133.4 | KL: 0.0043 | clip:  5.2%
Iteration 10/40 | episodes:  4 | mean reward: 500.0 | KL: 0.0059 | clip:  7.4%
Iteration 11/40 | episodes:  5 | mean reward: 460.4 | KL: 0.0077 | clip: 12.8%
Iteration 20/40 | episodes:  4 | mean reward: 500.0 | KL: 0.0019 | clip:  1.6%
Iteration 40/40 | episodes:  4 | mean reward: 500.0 | KL: 0.0004 | clip:  0.0%
20-episode evaluation: 500.0 +/- 0.0
```

## What episode return tells us

CartPole gives `+1` for every step survived. An episode starts at `reset` and ends when the cart leaves the allowed region, the pole angle crosses its threshold, or the 500-step time limit is reached. Episode return and episode length are therefore numerically equal in this environment.

The script collects 2,048 steps per training iteration, then averages the **complete episodes** that ended in that segment. An episode that crosses a rollout boundary must continue accumulating until it actually ends. Resetting its counter at every rollout boundary would make the logged return too small.

![Measured reward curve for seed 42](../../chapter01_cartpole/images/cartpole_reward_seed42.png)

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em>Figure 1-2: raw measured data for seed 42. Each point is the mean of complete episodes ending within that 2,048-step rollout. No smoothing or manual adjustment is applied. The dashed line marks the 500-step episode limit.</em>
</div>

This curve supports three specific observations:

1. The 91 complete episodes in the first rollout averaged `21.35` points.
2. At iteration 10, or `20,480` environment steps, the four complete episodes averaged `500.0`.
3. Iteration 11 fell back to `460.4`; a training curve need not improve monotonically. Most later rollouts returned to 500.

This remains a single-seed result. It shows that this implementation solved the task in this run. It does not prove that every machine, dependency version, or seed reaches 500 at 20,480 steps. Algorithm comparisons require multiple seeds and a shared evaluation protocol.

## Training return versus independent evaluation

Training uses stochastic action sampling to preserve exploration. Each plotted point also averages a different number of completed episodes: dozens early in training and usually four near the 500-step limit.

Independent evaluation selects the highest-probability action at every step. After training, the saved policy was evaluated for 20 episodes:

```text
mean = 500.0
std  = 0.0
```

The training curve describes how learning progressed. Independent evaluation describes how the final policy performed under a fixed protocol. We report both.

## Four diagnostic metrics

Return answers whether the policy performs the task. Diagnostic metrics help explain failed training or updates that move too far. The four curves below come from the same CSV as the reward curve.

![Four diagnostics from the same measured run](../../chapter01_cartpole/images/cartpole_diagnostics_seed42.png)

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em>Figure 1-3: raw Value Loss, Policy Entropy, Approximate KL, and Clip Fraction from the same run as Figure 1-2.</em>
</div>

### Policy entropy

Policy entropy measures uncertainty in the action distribution. CartPole has two actions, so a policy assigning about `50%` to each starts near `ln 2 ≈ 0.693`. In this run entropy fell from `0.685` to `0.421`. The policy became more selective without choosing one fixed direction in every state.

Falling entropy does not by itself prove improvement. If return stays low, a low-entropy policy may simply have committed to a poor action pattern.

### Value loss

The Critic predicts discounted return from the current state. Value loss is the mean squared error between predictions and return targets:

$$
L_V = \frac{1}{|B|}\sum_{i \in B}\left(V(s_i)-\hat G_i\right)^2.
$$

In this run Value Loss fluctuated between `41.8` and `64.1` early on, then fell overall and ended at `0.00014`. The intermediate spikes show the Critic adapting to states generated by a changing policy. A lower value loss means better fitting of these targets; policy quality is still judged by return and evaluation.

### Approximate KL

Approximate KL measures how action probabilities change between the old and updated policies on the sampled batch. The maximum across these 40 iterations was `0.00871`; the last value was `0.00038`.

These numbers are useful for comparing settings within the same implementation. This code does not implement KL-based early stopping, so this page does not invent a universal warning threshold.

### Clip fraction

Clip fraction reports the fraction of samples whose probability ratio lies outside the PPO clipping interval:

$$
\text{clipfrac}=\frac{1}{|B|}\sum_{i \in B}
\mathbf 1\left[|r_i(\theta)-1|>\epsilon\right].
$$

It was `14.48%` in the first iteration, also the maximum, and `0%` in the final three iterations. The learning rate approaches zero late in training, so policy changes and clipping both become smaller. Clip fraction must be interpreted together with KL, learning rate, and return.

## Minimal diagnostic order

When a result looks wrong, follow the data flow:

1. Check `terminated` and `truncated`. True termination uses `V(s')=0`; time truncation still bootstraps from `V(s')`.
2. Check that GAE stops at every environment reset. Advantages cannot propagate into a new episode.
3. Check that logs contain complete episodes and carry an unfinished episode across rollout boundaries.
4. Check whether both training return and independent evaluation improve.
5. If return is abnormal, use KL, clip fraction, entropy, and value loss to locate update-size or Critic-fitting problems.

## Summary

In CartPole, episode return equals the number of steps survived. The training curve shows the stochastic policy during learning, while independent evaluation tests the final deterministic policy. Diagnostic metrics explain training behavior; they do not replace task return.

The next section runs this complete pipeline and captures rendered frames from the trained model.
