# 1.3 动手：PPO 训练可视化

> **本节目标**：运行纯 PyTorch PPO，保存原始指标，生成训练曲线，并用训练后的模型完成一次可见的 CartPole 评估。

> **学习路径**：[1.1 CartPole 控制原理](./principles) → [1.2 奖励与训练指标](./metrics) → **1.3 PPO 训练可视化**

> **本节代码**：[训练脚本](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code/chapter01_cartpole/2-pytorch_ppo.py) · [绘图脚本](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code/chapter01_cartpole/plot_curves.py) · [环境帧脚本](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code/chapter01_cartpole/capture_frames.py) · [原始 CSV](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code/chapter01_cartpole/output/training_metrics_seed42.csv)

前两节已经给出了判断标准：奖励必须来自完整回合，训练结束后还要做独立评估。现在把整条证据链跑一遍。

## 安装与运行

```bash
cd code/chapter01_cartpole
pip install -r requirements.txt

python 2-pytorch_ppo.py \
  --seed 42 \
  --iterations 40 \
  --steps-per-rollout 2048 \
  --swanlab-mode disabled \
  --log-csv output/training_metrics_seed42.csv
```

训练结束后会得到两个本地文件：

- `output/pytorch_ppo_cartpole.pth`：训练后的模型参数；
- `output/training_metrics_seed42.csv`：每一轮未经平滑的训练指标。

若要使用本地 SwanLab 看板，把 `--swanlab-mode disabled` 改为 `--swanlab-mode local`，训练结束后执行：

```bash
swanlab watch swanlog
```

## PPO 的数据流

每一轮训练依次执行三个步骤：

```mermaid
flowchart LR
    A["用当前策略收集 2048 步"] --> B["计算 TD 误差与 GAE"]
    B --> C["同一批数据做 10 轮 PPO 更新"]
    C --> A
```

### 1. 收集轨迹

环境每一步返回 `terminated` 和 `truncated`。二者都会触发 `reset`，但价值目标不同：

```python
next_obs, reward, terminated, truncated, _ = env.step(action.item())

with torch.no_grad():
    if terminated:
        next_value = 0.0
    else:
        _, next_value_tensor = model(torch.FloatTensor(next_obs))
        next_value = next_value_tensor.item()
```

`terminated=True` 表示杆子倒下或小车越界，回合的未来回报确实为 0。`truncated=True` 表示达到 500 步时间上限，此时状态本身仍有价值，需要用 `V(s')` bootstrap。

采样可能在一个回合中间结束。配套代码会保存最后一个 `next_value`，也会保留尚未结束回合已经累积的奖励和长度，下一轮继续累计。训练数据与日志统计因此分别满足：

- GAE 不跨越环境 reset；
- 回合奖励不在 rollout 边界被截断。

### 2. 计算 GAE

每一步先计算 TD 误差：

$$
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t).
$$

再从后向前累积优势：

```python
episode_end = t["terminated"] or t["truncated"]
delta = t["reward"] + gamma * t["next_value"] - t["value"]
gae = delta + gamma * lam * (1.0 - float(episode_end)) * gae
```

乘数 `1 - episode_end` 在每次 reset 处把递推切断。时间截断处仍然通过 `next_value` 计算当前步的 TD 误差，但不会把下一个新回合的优势传回来。

Critic 使用未归一化的目标：

```python
returns = raw_advantages + values
advantages = (raw_advantages - raw_advantages.mean()) / (
    raw_advantages.std(unbiased=False) + 1e-8
)
```

归一化只改变用于策略梯度的优势尺度，不改变 Critic 要拟合的回报目标。

### 3. PPO 裁剪更新

新旧策略对已采样动作的概率比为：

$$
r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}.
$$

配套代码用 `clip_eps=0.2` 限制一次更新从旧策略偏离太远：

```python
ratio = torch.exp(new_log_probs - batch_old_log_probs)
surr1 = ratio * batch_advantages
surr2 = torch.clamp(ratio, 0.8, 1.2) * batch_advantages
policy_loss = -torch.min(surr1, surr2).mean()
```

## 从原始 CSV 画图

训练脚本直接导出 CSV。绘图脚本只读取这份文件，因此图上的每一个点都能追溯到一行原始记录。

```bash
python plot_curves.py \
  --input output/training_metrics_seed42.csv \
  --output-dir output
```

该命令生成奖励曲线和四指标诊断图。仓库页面展示的是同一命令在 2026-08-13、seed=42 下生成的结果：

![seed=42 的 CartPole PPO 奖励曲线](./images/cartpole_reward_seed42.png)

本次运行在第 1 轮得到 `21.35` 分，在第 10 轮的 4 个完整回合中得到 `500.0` 分。第 11 轮回落到 `460.4`，说明单次训练曲线不会严格单调。最终确定性策略的 20 回合评估为 `500.0 ± 0.0`。

这是一条单种子实测曲线。它可以验证代码能够收敛，不能代替多种子算法比较。

## 捕获真实环境画面

下面的图由 `capture_frames.py` 加载本次训练保存的模型，在 Gymnasium 的 `CartPole-v1` 中运行确定性评估并调用 `env.render()` 得到。它不是手绘示意图。

```bash
python capture_frames.py \
  --model output/pytorch_ppo_cartpole.pth \
  --output output/cartpole_frames_seed42.png \
  --seed 10042
```

![训练后策略在 Gymnasium CartPole-v1 中的实测帧](./images/cartpole_frames_seed42.png)

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em>图 1-4：同一个确定性评估回合的第 0、125、250、375 和 500 步。该回合得到 500 分；标题中的角度直接来自对应时刻的环境观测。</em>
</div>

一张静态帧只能证明某个时刻的姿态，不能证明策略持续控制了 500 步。这里同时给出完整回合得分、多个时间点的帧和独立评估统计，三者共同构成验证证据。

## 改参数时怎样报告

学习率、裁剪范围或 GAE 参数会改变曲线，但一次单种子运行不能给出普遍结论。比较设置时至少保持环境版本、训练步数、网络结构和评估回合数一致，并使用多个随机种子。

建议把问题写成可检验的形式：

- `lr=1e-4` 是否比 `3e-4` 需要更多环境步才能达到相同评估分数？
- `clip_eps=0.1` 是否降低 KL，同时减慢奖励上升？
- 错误地让 GAE 跨回合传播，会让多少个种子的最终评估失败？

结果应报告每个种子的原始曲线、首次达到目标的环境步数和最终评估分数。不要把某一次漂亮曲线写成算法保证。

## 本节小结

完整实验包含五个可检查环节：固定随机种子、保存原始指标、从原始指标绘图、独立评估、从真实环境捕获回放画面。只要其中一环无法追溯，页面上的数字和图片就不能称为实测结果。

下一章从多臂老虎机开始，把这里已经运行过的“状态—动作—奖励—更新”过程写成正式的强化学习问题。
