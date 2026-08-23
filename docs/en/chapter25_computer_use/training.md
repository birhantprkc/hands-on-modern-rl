# 22.1 GUI Agent Training: From Screenshots to Executable Actions

Start with a one-step task. A web form already contains an email address and verification code; only the **Submit** button remains. A person naturally moves the pointer to the button. The model receives a pixel matrix. It must first identify the button region, then translate “submit” into a coordinate action such as `click(214, 209)`.

The task does not end after the click. The page may show a success screen, or an expired-code error. The model must read the next screenshot, decide whether the action worked, and then stop, wait, or recover. **GUI-agent training learns this closed loop: find task-relevant interface information, produce executable actions, and use environment feedback to correct later behavior.**

<img src="../../chapter25_computer_use/images/gui-agent-one-step.svg" alt="A GUI operation proceeds from goal recognition through coordinate grounding to outcome verification">

The figure separates one click into four judgments. Training only button location stops at grounding; training only final success cannot identify whether recognition, localization, or verification caused a failure. Demonstrations, online RL, curriculum sampling, and progress rewards supply learning signals to different parts of this chain.

[Chapter 19: Agentic RL](../chapter22_agentic/overview) usually exposes tools with explicit function names and arguments. A GUI exposes screenshots plus mouse and keyboard input. The same “submit this form” objective can therefore generate different trajectories under different window sizes, themes, pop-ups, and loading states.

<img src="../../chapter25_computer_use/images/gui-agent-training-loop.svg" alt="The GUI-agent training loop connects screenshots, actions, environment transitions, verifiers, and rewards">

Each action changes the next screenshot, and a verifier judges whether the task advanced. This observation–action–new-observation loop is the starting point for formulating GUI operation as reinforcement learning.

## The Computer Use Paradigm

The tools in [Chapter 19: Tool Use and Trajectory](../chapter22_agentic/tool-use-and-trajectory) are **structured APIs** — `def search(query): return results`, with input and output as strings. However, in the real world, many software applications have only one interface: the **GUI**. Browsers, Excel, enterprise internal OA systems, Photoshop, and games — none of them expose public APIs; they only provide screens and mouse and keyboard events.

The **Computer Use** paradigm treats the entire operating system as the agent's environment:

- **Observation**: A screen capture $o_t \in \mathbb{R}^{H \times W \times 3}$ (1–4 frames per second)
- **Action**: Atomic GUI events (mouse movement, click, scroll, keyboard input, wait)
- **Reward**: A binary signal indicating task completion ("whether a flight was successfully booked")

This MDP is entirely different from traditional RL benchmarks. CartPole has a 4-dimensional state, 2-dimensional actions, and dense per-step rewards; in the Computer Use paradigm, the state is millions of dimensions of pixels, the action space is mixed-type, and the reward is sparse, only given at the final step.

### Mainstream Products

| Product             | Organization   | Release | Features                                                          |
| ------------------- | -------------- | ------- | ----------------------------------------------------------------- |
| **Computer Use**    | Anthropic      | 2024.10 | Native support for screenshot-action pairs with Claude 3.5 Sonnet |
| **Operator**        | OpenAI         | 2025.01 | CU Agent + GPT-4o Vision, browser-specific                        |
| **Project Mariner** | Google         | 2024.12 | Gemini-driven, deep integration with Chrome                       |
| **UI-TARS-2**       | ByteDance Seed | 2025.09 | End-to-end VLM + RL training                                      |
| **Open-AutoGLM**    | Zhipu          | 2025.12 | Open-source upgraded version of AutoGLM                           |

### Core Action Space

The primitive actions for Anthropic Computer Use are defined as follows (similar to OpenAI Operator and Google Mariner):

```python
ACTIONS = {
    "click":      {"x": int, "y": int, "button": "left|right|middle"},
    "double":     {"x": int, "y": int},
    "drag":       {"start": [x,y], "end": [x,y]},
    "type":       {"text": str},
    "key":        {"keys": "ctrl+c|enter|tab"},   # Combination keys
    "scroll":     {"x": int, "y": int, "dy": int},
    "wait":       {"ms": int},
    "screenshot": {},
    "done":       {"summary": str},
}
```

Note three key design choices:

1. **Actions are a mixture of discrete tokens and continuous coordinates** — `click` requires both selecting a token and predicting $(x, y)$. This is a challenge for LLMs to handle natively: standard transformer outputs discrete tokens, while $(x, y) \in [0, W] \times [0, H]$ are continuous values.
2. **Screenshot frequency is much lower than human visual perception** — humans perceive 30–60 frames per second, while Computer Use captures 1–4 frames per second. This means that the state transition $P(s_{t+1} \mid s_t, a_t)$ involves a large number of hidden state changes between observations.
3. **The `wait` action** — GUI animations, network loading, and popup transitions require waiting. This is a unique "active time consumption" action not present in traditional RL.

### MDP Formalization

Define the Computer Use MDP as $\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma, T)$:

$$\mathcal{S} = \{\text{screenshots}\}, \quad \mathcal{\mathcal{A}} = \{\text{click, type, scroll, key, wait, done}\}$$

The task description (e.g., "Help me convert this PDF into Markdown") is appended as an initial prompt $q$ before each observation. The policy is a conditional distribution:

$$\pi_\theta(a_t \mid q, o_{1:t}, a_{1:t-1})$$

The reward $R$ is typically sparse and binary: $r_T = \mathbb{1}[\text{task completed}]$, and intermediate steps $r_{t<T} = 0$. This makes credit assignment extremely difficult — a single browser automation task may involve 50 actions, with only the last step receiving a reward, making it impossible to determine which steps were correct or incorrect.

::: warning RL's True Challenges
Sparse rewards + long sequences (50–500 steps) + high-dimensional observations (screenshots of 1344×756 pixels) + mixed action spaces — Computer Use simultaneously hits all the pain points of RL. This is why, before 2024, almost all Computer Use systems were based on **pure prompt engineering** (prompt engineering). It wasn't until 2025 that RL training truly entered industrial deployment.
:::

## GUI Grounding RL

The first challenge in computer use is not decision-making, but **grounding**: how does the model know where the "submit" button is on the screen at coordinate $(x, y)$?

### Set-of-Mark Prompting

Yang et al. 2023 propose the **Set-of-Mark (SoM)** prompting approach: first, use OCR or object detection to box all interactive elements on the screen, labeling them as $1, 2, \ldots, K$. When the agent outputs actions, it only needs to refer to the labels:

```
[Screen shot + Box 1: Input field "Username", Box 2: Input field "Password", Box 3: Button "Login"]

Agent: type("alice") → click(Box 1) → type("***") → click(Box 2) → click(Box 3)
```

This transforms the problem of continuous coordinate prediction into a **discrete selection** problem. However, the cost is the reliance on external detectors, and the agent is helpless when the detector misses elements.

### Visual Grounding

Models like UI-TARS and CogAgent take a different approach: **let the VLM directly output coordinates**. The model architecture has two heads:

$$\text{VLM}(o_t, q) \to \underbrace{(\text{thought}, \text{action token})}_{\text{language head}} + \underbrace{(x, y) \in [0,1]^2}_{\text{grounding head}}$$

The grounding head is typically an MLP that outputs normalized coordinates $(x, y) \in [0, 1]^2$, which are then scaled to pixel coordinates by multiplying with the screen size.

Training the grounding head uses **supervised imitation**: human-labeled "center points" $(x_i, y_i)$ of buttons, with loss defined as:

$$\mathcal{L}_{\text{ground}} = \frac{1}{N}\sum_i \|\hat{p}_\theta(o_i) - p_i\|_2^2$$

However, pure supervision has a problem: the model might output **empty space**. Supervision only learns "where the button is," not "the button should be pressed." Here, reinforcement learning comes into play.

### Joint RL for Grounding and Decision Making

We combine grounding and action selection into a single PPO objective:

$$\mathcal{J}(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^T \gamma^t r_t\right] - \beta \cdot \mathcal{L}_{\text{ground}}(\theta)$$

The second term is a supervised loss for grounding, acting as a regularizer. This **SFT + RL joint training** is the standard recipe for GUI Agents — first imitate to learn basic operations, then use RL to optimize task success rate.

UI-TARS-2 takes this idea to its extreme: it outputs three parts — thought, action, and coordinate — as a **single sequence**, and optimizes them all simultaneously with RL:

```python
def ui_tars_forward(self, screenshot, task):
    # Encode image
    visual_tokens = self.vision_encoder(screenshot)  # [B, N_vis, d]

    # Concatenate prompt
    prompt = f"<task>{task}</task>\n<image>{visual_tokens}</image>\n"

    # Autoregressively generate thought + action + coord
    # Key: coord is wrapped with special tokens <coord_x> <coord_y>
    output = self.llm.generate(prompt, max_new_tokens=256)

    # Parse output: "<thought>...</thought>\n<action>click</action>\n<coord>(0.45, 0.62)</coord>"
    thought, action, coord = parse_action(output)
    return thought, action, coord
```

### Generating RL Training Data

Real GUI tasks cannot be extensively manually labeled — a 50-step browser task requires about 30 minutes of human demonstration. The solution is **programmatic task generation**:

1. **Real Website Crawling**: UI-TARS collects over 200 real applications, each generating more than 1,000 task templates automatically.
2. **Environment Snapshots**: Record the human operation process, saving screenshots and actions at each step as SFT data.
3. analogical Task Validator\*\*: Use programmatic rules to check whether the task is completed ("Has the success prompt appeared on the page?").
4. **RL Rollout**: The agent executes the task in a virtual machine, and the validator provides the final reward.

```python
class GUIEnv:
    def reset(self, task_id):
        self.vm.restore_snapshot(task_id)  # Restore the virtual machine to the initial state of the task
        self.task = self.tasks[task_id]
        return self.screenshot()

    def step(self, action):
        self.vm.execute(action)            # Inject mouse and keyboard events
        obs = self.screenshot()
        done = self.task.verifier(obs, self.vm.state)
        reward = 1.0 if done else 0.0
        return obs, reward, done, {}
```

::: details Why Not Use Real Mouse
Directly controlling the operating system's mouse would cause conflicts between the agent's input and human users' input. In industrial practice, the agent runs in a **virtual machine + VNC remote desktop**, with mouse and keyboard events injected through the RDP/VNC protocol, isolating the agent from human users. This is why Computer Use systems typically execute only 1–2 actions per second — due to the delay from screenshot capture and VNC injection.
:::

## Summary of This Section

Computer Use treats GUI pixel streams as the state space for RL and mouse and keyboard events as the action space, which amplifies all traditional RL challenges (sparse rewards, long sequences, high-dimensional observations) simultaneously. **Set-of-Mark** and **Visual Grounding** are the two mainstream approaches to solving the "localization" problem: the former relies on external detectors to simplify the action space, while the latter uses VLMs to end-to-end output coordinates.

The next section [22.1 GUI Agent Training Practice](./training) will take us into industrial practice — you will see how systems such as UI-TARS-2, AutoGLM, MobileRL, and ComputerRL transform this theoretical framework into a reproducible training pipeline.

The previous sections have clarified the MDP modeling of Computer Use and the visual alignment of GUI Grounding. This section addresses the next engineering question: **How to train a VLM into a GUI Agent?** This involves data synthesis, curriculum design, reward engineering, and virtual environments — a complete industrial training pipeline. Below, we will compare the technical approaches of UI-TARS-2, AutoGLM, MobileRL, ComputerRL, and CogAgent, based on representative work from Chinese laboratories in 2025–2026.

## A Concentrated Outburst of GUI Agents in Chinese Laboratories

In the second half of 2025, Chinese laboratories experienced a concentrated outburst in GUI Agent RL training. This was not a coincidence — three conditions simultaneously matured:

1. **Mature VLM Foundation**: Open-source VLMs such as Qwen2.5-VL, InternVL3, and GLM-4.5V provided a high-quality starting point.
2. **Virtual Environment Toolchains**: Benchmarks such as Android Worldwide, AndroidWorld, OSWorld, and WebArena provided reproducible training and evaluation environments.
3. **Reduced Compute Costs**: The stable prices of 4090 and H100 GPUs made RL training of 7B models affordable.

Comparison of Representative Works:

| Model            | Institution     | arXiv      | Parameter Scale | Core Innovation                                             |
| ---------------- | --------------- | ---------- | --------------- | ----------------------------------------------------------- |
| **UI-TARS-2**    | Byte Seed       | 2509.02544 | 7B / 72B        | End-to-end VLM + Long-term Task RL + Reflection Enhancement |
| **Open-AutoGLM** | Zhipu AI        | 2411.00820 | 9B              | Multilingual GUI + Mobile + Full Open Source                |
| **MobileRL**     | Tencent         | 2509.18119 | 7B              | Mobile App Difficulty Curriculum Learning                   |
| **ComputerRL**   | Shanghai AI Lab | 2508.14040 | 7B              | Reverse Curriculum + Intermediate Exploration Rewards       |
| **CogAgent-9B**  | Zhipu AI        | 2408.16500 | 9B              | High-Resolution Visual Encoding + Dual-Branch Fusion        |

## UI-TARS-2: End-to-End RL

UI-TARS-2 treats Computer Use as a **pure LLM RL problem** — a single VLM simultaneously handles perception, reasoning, and action output. The model architecture has no explicit planner / actor division; all logic resides within a single transformer.

### Four-Stage Training Process

```
Stage 1: Vision-Language Pretraining
  └─ GUI screenshot + text pairs → foundational visual capabilities

Stage 2: Supervised Fine-Tuning (SFT)
  └─ Human demonstrations + model self-generated trajectories → foundational action capabilities

Stage 3: Reflective RL (Reflection-enhanced RL)
  └─ Multiple candidate trajectories + verifier selection → rejection sampling + SFT

Stage 4: Online RL (Online Reinforcement Learning)
  └─ Real GUI environment rollouts → PPO optimizes task completion rate
```

Stage 3's rejection sampling is the key transition: the model generates $K=8$ trajectories for the same task, and a programmatically defined verifier determines which ones are successful, then feeds the successful trajectories back as high-quality SFT data. This is more stable than direct online RL — online RL learns almost nothing when success rate is below 10%.

### Reflection Augmentation

The core innovation of Phase 4 is the reflection mechanism. It allows the agent to explicitly output a `<reflection>` tag when it fails:

```
<thought>I need to click the "submit" button</thought>
<action>click(450, 320)</action>
<observation>The button turns gray, but the page does not jump</observation>
<reflection>It is possible that I clicked outside the clickable area. The clickable area of the "submit" button is (440-470, 310-330), and I clicked outside of it. I should move closer to the center on retry.</reflection>
<action>click(455, 320)</action>
<observation>The page jumps to the success page</observation>
<action>done</action>
```

This self-correcting capability cannot be learned solely through SFT — it must be learned through the trial-and-error signals of RL. During RL training, we give an additional +0.3 reward to trajectories that successfully correct errors through reflection, encouraging the model to learn how to reflect.

### Multi-Task RL Reward

The total reward function of UI-TARS-2 is:

$$r = r_{\text{task}} + \alpha \cdot r_{\text{format}} + \beta \cdot r_{\text{reflection}} - \gamma \cdot r_{\text{invalid}}$$

- $r_{\text{task}} \in \{0, 1\}$: Whether the task is completed
- $r_{\text{format}} \in \{0, 1\}$: Whether the output format is valid (XML tags are properly closed, coordinates are within range)
- $r_{\text{reflection}} \in [0, 0.3]$: Quality of successful error correction through reflection
- $r_{\text{invalid}}$: Penalty for executing unauthorized actions (e.g., attempting to close the browser)

In practice, the weights are set as $\alpha = 0.1, \beta = 0.3, \gamma = 2.0$. $\gamma$ is intentionally large — the cost of an unauthorized action is much higher than the reward for completing a task.

## Open-AutoGLM: Open-Source Training Pipeline

The AutoGLM series by Zhipu (Open-AutoGLM was open-sourced in December 2025) is optimized for the **Chinese internet environment** — platforms such as Weibo, Taobao, and WeChat Mini Programs perform poorly on English models (e.g., Operator, Mariner). Its training innovations include:

### Chinese GUI Data Synthesis

The data sources for English models are Common Crawl + RPA recordings, but Chinese GUI data is scarce. Open-AutoGLM's approach includes:

1. **WeChat Mini Program Crawling**: Using the Android automation framework Appium to control over 100 real devices, automatically exploring mini programs, and recording screenshots and actions for each step.
2. - **Chinese E-commerce Task Synthesis**: Automatically generating "search product → price comparison → add to cart → place order (or not)" task templates on platforms like Taobao, JD.com, and Pinduoduo.
3. **Chinese Social Tasks**: Such as Weibo posts, Douyin comments, Xiaohongshu collections, etc.

In total, **2.3 million Chinese GUI trajectories** were collected, which is **2.9 times** the number of English trajectories (800K).

### Unified Action Space Across Platforms

A key design of Open-AutoGLM is **cross-platform unification** — the same model can work on desktop browsers, Android apps, and iOS apps (via WebDriverAgent). The unified action space is defined as:

```python
UNIFIED_ACTIONS = {
    "tap":       {"x": float, "y": float},           # Tap/Touch
    "long_press":{"x": float, "y": float, "ms": int},
    "swipe":     {"start": [x,y], "end": [x,y]},     # Swipe/Drag
    "type":      {"text": str},
    "key":       {"name": str},                      # back, home, enter
    "scroll":    {"dy": int},
    "wait":      {"ms": int},
    "done":      {"summary": str},
}
```

The desktop "click" and mobile "tap" are unified as the same action `tap` — semantic differences across platforms are handled by environment adapters.

### Complete Open-Source

Open-AutoGLM opensources **model weights, training data, environment simulators, and training scripts**, making it the most complete open-source GUI Agent training framework to date:

```bash
git clone https://github.com/zai-org/Open-AutoGLM
cd Open-AutoGLM

# 1. Download pre-trained weights
huggingface-cli download zhipuai/Open-AutoGLM-9B

# 2. Start Android emulator
bash scripts/start_emulator.sh

# 3. RL Training (Single machine with 8×H100)
bash train.sh \
    --model Open-AutoGLM-9B \
    --algo grpo \
    --platform android \
    --tasks curated-1k.jsonl
```

In practice, on 8×H100, a single GRPO step processes about 256 prompts in approximately 4 minutes. Training for 5000 steps to convergence takes about 14 days.

## MobileRL: Mobile RL

Tencent's MobileRL (arXiv:2509.18119) is specifically designed to address automation in mobile apps. Mobile environments are more challenging than desktop environments, for three reasons:

- **Small screen, dense elements**: A mobile app's home page may have 30 clickable elements densely arranged.
- **Complex gestures**: Long press, swipe, pinch, 3D Touch, and other gestures are far more diverse than mouse clicks.
- **Frequent app switching**: Push notifications, incoming calls, and low battery alerts can interrupt tasks at any time.

### Gradual Difficulty Curriculum

The core innovation of MobileRL is the **Gradual Difficulty Curriculum** (Curriculum Learning):

$$\text{Curriculum}(\pi_\theta) = \arg\max_{\text{task } \tau} \; \text{Difficulty}(\tau) \quad \text{s.t.} \quad 0.3 \leq P_\theta(\text{success} \mid \tau) \leq 0.7$$

Tasks are sampled only from the "Zone of Proximal Development" where the model's current success rate is between 30% and 70%, avoiding overly difficult tasks (too sparse signals) and overly easy tasks (no learning signal).

### Quantification of Task Difficulty

MobileRL defines task difficulty as a weighted sum of four dimensions:

$$\text{Difficulty}(\tau) = w_1 \cdot \text{Steps}(\tau) + w_2 \cdot \text{Apps}(\tau) + w_3 \cdot \text{GestureComplexity}(\tau) + w_4 \cdot \text{Distraction}(\tau)$$

- $\text{Steps}$: Minimum number of steps to complete the task (5–50)
- $\text{Apps}$: Number of apps to switch between (1–4)
- $\text{GestureComplexity}$: Number of gesture types required (tap=1, swipe=2, long_press=3, multi-touch=5)
- $\text{Distraction}$: Number of simulated distraction events (push notifications, incoming calls)

Empirical weights are $w_1=0.4, w_2=0.2, w_3=0.2, w_4=0.2$.

### Curriculum Scheduler

```python
class CurriculumSampler:
    def __init__(self, tasks, model):
        self.tasks = tasks
        self.model = model
        self.success_rate = {}  # task_id -> moving average success rate

    def sample(self, batch_size):
        # 1. Evaluate the success rate of each task under the current model
        for tau in self.tasks:
            if tau.id not in self.success_rate:
                self.success_rate[tau.id] = self._estimate(tau)

        # 2. Filter out tasks with success rate between 30% and 70%
        candidates = [t for t in self.tasks
                      if 0.3 <= self.success_rate[t.id] <= 0.7]

        # 3. Sample with difficulty-weighted probability
        weights = [t.difficulty for t in candidates]
        return weighted_sample(candidates, weights, batch_size)

    def _estimate(self, task):
        # Run 10 rollouts to estimate the success rate
        successes = sum(self._rollout(task) for _ in range(10))
        return successes / 10
```

The success rate of each task is re-evaluated at each epoch, allowing the curriculum to dynamically adjust according to the model's current capabilities.

## ComputerRL: Backward Curriculum and Exploration Rewards

ComputerRL (arXiv:2508.14040) from Shanghai AI Lab discovers that pure task completion rewards are too sparse for long-horizon tasks (over 50 steps). Their solution is **backward curriculum + intermediate exploration rewards**.

### Backward Curriculum

Traditional curriculum learning progresses from easy to hard — first learning a 5-step task, then a 10-step, then a 20-step. Backward curriculum reverses this: **starting from the task endpoint**.

Consider a 50-step task $T = (s_0, a_1, s_1, \ldots, a_{50}, s_{50})$. The training order for backward curriculum is:

```
Round 1: Start from $s_{49}$, execute $a_{50}$ → done (1-step task)
Round 2: Start from $s_{48}$, execute $a_{49}, a_{50}$ → done (2-step task)
Round 3: Start from $s_{47}$, execute $a_{48}, a_{49}, a_{50}$ → done (3-step task)
...
Round 50: Start from $s_0$, complete the full task (50-step task)
```

**Why is this effective?** Backward curriculum ensures that the RL agent is always trained in states that are close to the reward. In forward training, the agent at $s_0$ sees no reward signal; in backward training, the agent at $s_{49}$ receives the reward in one step. This makes credit assignment much simpler — the most recent action receives immediate feedback.

### Intermediate Exploration Rewards

The inverse curriculum addresses the issue of "sparse terminal rewards being too far," but intermediate steps still lack signals. ComputerRL introduces **intermediate state rewards**:

$$r_t = \underbrace{r_{\text{task}}(t=T)}_{\text{Sparse Terminal Reward}} + \lambda \cdot \underbrace{r_{\text{progress}}(s_t, s_{t+1})}_{\text{Dense Progress Reward}}$$

Here, $r_{\text{progress}}$ is generated by an independent "progress evaluator" LLM:

```python
def compute_progress_reward(s_t, s_{t+1}, task):
    prompt = f"""
    Task: {task}
    State before: {describe(s_t)}
    State after: {describe(s_{t+1})}
    Question: did the agent make progress toward the task?
    Answer with a score in [0, 1]:
    - 1.0: significant progress (e.g., filled a required field)
    - 0.5: minor progress (e.g., navigated closer)
    - 0.0: no progress (e.g., clicked irrelevant element)
    - -0.5: regression (e.g., closed important dialog)
    """
    return float(llm_judge(prompt))
```

This LLM-as-judge approach for intermediate rewards is similar to the idea in [Chapter 17: Process Reward Model](../chapter20_prm_search/inference-time-search), where LLMs are used to evaluate the quality of intermediate steps.

### Comparison with Forward Curriculum

The paper ComputerRL reports comparative experiments:

| Method                                   | OSLevel-3 Success Rate | Average Steps | Training Cost |
| ---------------------------------------- | ---------------------- | ------------- | ------------- |
| Forward Curriculum + Terminal Reward     | 12.3%                  | 47            | 1×            |
| Forward Curriculum + Progress Reward     | 27.7%                  | 35            | 2.3×          |
| **Reverse Curriculum + Progress Reward** | **51.2%**              | **28**        | 2.8×          |

Reverse curriculum increases the success rate from 12% to 51%, but the training cost increases by 2.8 times — mainly due to the computational overhead of the progress evaluator LLM.

## CogAgent: The Cost of High-Resolution Vision

The CogAgent-9B from Zhipu (arXiv:2408.16500) takes a different approach: **using higher-resolution visual encoding to improve accuracy**.

### High-Resolution Visual Branch

Standard VLMs input images at a resolution of 448×448, while CogAgent uses 1120×1120 — four times the number of pixels, which means four times the number of visual tokens, but allows for better recognition of small text on UIs (e.g., 9-point font in tables, PowerPoint toolbar icons).

CogAgent's architectural insight is the **dual-branch fusion**:

```
┌──────────────────────────────────────────┐
│ Input screenshot (1120×1120)              │
└────────────┬─────────────────────────────┘
             ↓
   ┌─────────┴─────────┐
   │                   │
   ↓                   ↓
High-Resolution Branch   Low-Resolution Branch
(EVA-CLIP)              (SigLIP)
1120×1120               448×448
→ 3136 tokens          → 256 tokens
   │                   │
   └─────────┬─────────┘
             ↓
        Cross-Attention
             ↓
         LLM Decoder
```

The low-resolution branch provides global context ("This is a shopping page"), while the high-resolution branch provides details ("The shopping cart button is in the top right corner"). The two branches are fused through cross-attention, avoiding the computational overhead of having the LLM process all 3136 tokens.

### Trade-off Between Accuracy and Latency

The cost is computational: high-resolution visual tokens make inference 3–5 times slower.

| Configuration           | Visual Tokens | Inference Latency | OSWorld Accuracy |
| ----------------------- | ------------- | ----------------- | ---------------- |
| 448×448 Single Branch   | 256           | 0.8s              | 38.2%            |
| 1120×1120 Single Branch | 3136          | 4.2s              | 47.5%            |
| **Dual-Branch Fusion**  | 3392          | 1.6s              | **46.8%**        |

The dual-branch approach maintains an accuracy close to that of the high-resolution branch while increasing latency only by a factor of 1. This trade-off is at the core of engineering decisions in GUI Agents—accuracy versus latency.

## Three Challenges in Industrial Deployment

Moving the above system from paper to production environment will encounter three challenges that are not thoroughly discussed in the literature.

### Distributional Shift in Environments

The training environments in the papers are controlled benchmarks such as OSWorld and AndroidWorld. In production, the environment is the real user's computer — each user has a different system version, browser plugins, and font size.

**Solutions**:

- **Data Diversity**: UI-TARS-2 collects training environments with over 50 different Windows/macOS/Linux configurations.
- **Domain Randomization**: During training, the UI theme, font, and resolution are randomly changed.
- **Continual Learning**: After deployment, failure cases are collected, and the model is retrained periodically.

### Long-Tail Tasks

The benchmark tasks in the papers are all "mainstream tasks" (book a flight, check a calendar, write an email). In production, users may ask "Help me change this computer's BIOS to UEFI mode" — a task with very little training data.

**Solutions**:

- **Task Hierarchization**: Use pre-trained strategies for common tasks; for rare tasks, fall back to "tree search + LLM planning".
- **Human-in-the Loop**: Actively ask the user for confirmation when the model's confidence is low.

### Safety Boundaries

GUI Agents can perform destructive actions — delete files, transfer money, send emails. The production environment must have clear safety boundaries.

**Solutions**:

- **Whitelisted Actions**: By default, prohibit `rm -rf`, money transfers over $100, and mass email sending.
- **Double-Confirmation**: Pop-up confirmation is required for high-risk operations.
- **Audit Logs**: All operations are logged and traceable.

See [22.2 Prompt Injection and Instruction-Level](./safety-swarm).

## Summary of This Section

Chinese laboratories have formed four clear paths in the training of GUI Agents using Reinforcement Learning (RL):

- **UI-TARS-2**: End-to-end Vision-Language Model (VLM) + Reflection Enhancement, treating "Computer Use" as a pure Large Language Model (LLM) RL problem.
- **Open-AutoGLM**: Chinese GUI data synthesis + cross-platform unification, with the highest level of engineering completeness.
- **MobileRL**: Progressive difficulty curriculum, focusing on mobile apps.
- **ComputerRL**: Reverse curriculum + intermediate exploration rewards, targeting long-term tasks.
- **CogAgent**: High-resolution visual encoding, focusing on small text recognition.

These four paths are not mutually exclusive. For instance, UI-TARS-2 later incorporated a reflection curriculum (similar to MobileRL's idea), and Open-AutoGLM also used a reverse curriculum (similar to ComputerRL's idea). **Industrial systems are often combinations of multiple approaches**.

The next section [22.2 Prompt Injection and Instruction Level](./safety-swarm) shifts toward safety — once agents are truly deployed on users' computers, how to prevent malicious websites, forged UIs, and cross-application attacks from hijacking the system.
