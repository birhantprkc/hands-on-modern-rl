# 18.5 Large-Scale RL Data Engineering

[18.2](./industrial-post-training) has already explained that industrial post-training continuously sends failure samples back to the next training round. Now, tracing back along a failure trajectory: how does the team find the original task, how do they recover the environment as it was running, how do they determine whether the model completed the task, and how do they decide whether this trajectory is worth training?

Let's look at a code task. A user reports, "The page does not update after clicking Save." With only this sentence, the model can provide a segment of code modification suggestions, but it cannot obtain reliable rewards through real execution. To turn this into RL data, one also needs to prepare a code repository, the commit before the problem occurred, dependencies, startup commands, tests, and a sandbox. The model's process of reading files, modifying code, and running tests must also be fully recorded. Only after the tests pass can this trajectory become a verifiable training sample.

Therefore, a typical data sample in large-scale RL usually contains the following parts:

```text
Task Objective
  + Initial Environment
  + Complete Interaction Trajectory of the Model and Tools
  + Sub-item Results Provided by the Verifier
  + Generated, Environment, and Model Versions
```

This is significantly different from ordinary SFT data. SFT typically uses "Question—Demonstration Answer" format; RL data must also be **runnable, scorable, replayable, and version-traceable**.

## 1. Production Tasks: From Requirements to Training Data

The code problem we discussed earlier is not yet ready for training. It must first be transformed into a clear task, a recoverable environment, a reliable verifier, and a complete trajectory, before it can be added to the training set or evaluation set. The entire process can be broken down into six steps:

```text
Real-world requirements, public data, or synthetic seeds
              ↓
        1. Construct the Task
              ↓
        2. Package the Environment
              ↓
        3. Build the Verifier
              ↓
        4. Sample Trajectories Multiple Times
              ↓
        5. Filter and Annotate
              ↓
        6. Enter Training or Evaluation
              ↓
       Reconstruct After Failure Analysis
```

### 1.1 Defining Tasks from Real Requirements

The data source determines the problems the model will eventually encounter. Mathematical RL can start from competition problems, textbooks, and synthetic problems; code agents can begin from GitHub issues, PRs, and commits; search agents can start from questions requiring multi-page evidence; and desktop agents come from real workflows or controlled simulation tasks.

Raw records are not yet training tasks. The data team must clearly define the objective, remove fields that leak answers, and determine the initial state of the task. A GitHub PR can be rewritten into multiple tasks: fixing a bug, adding tests, reviewing a patch, or optimizing performance. The same source can generate data with different capabilities, but they must retain a common source identifier to prevent being mistakenly treated as independent samples.

### 1.2 Making the Environment Restartable

RL samples the same task multiple times. Each sample must start from the same state, otherwise, the rewards cannot be compared.

A code environment must at least lock the repository commit, dependency versions, system image, startup command, and network permissions. Browser tasks need to lock the website snapshot, account status, and available tools. Long-running agents also need to save intermediate files, external service states, or recoverable checkpoints.

An environment build failure is also a data result. It indicates that the sample has not yet met the training standard and cannot be simply attributed to model failure.

### 1.3 Validating Tasks and Verifiers

The verifier is responsible for converting execution results into rewards. When building a verifier, it is essential to first run a known correct solution to confirm that the task is solvable; then test with clearly incorrect solutions to ensure the verifier does not overlook errors.

Code tasks often use two sets of tests: one that fails before the fix and passes after the fix, proving the problem has been resolved; and another that was already passing to check whether the changes have broken existing functionality. Application development tasks are difficult to pre-write all tests, so one can let the verification agent launch the application, interact with the interface, and then score based on executability, interaction results, and visual requirements.

A reliable reward should retain sub-item results:

| Reward Item          | What It Checks                                            | What Problem It Can Identify                    |
| -------------------- | --------------------------------------------------------- | ----------------------------------------------- |
| Environment Validity | Whether the environment successfully starts               | Data or infrastructure errors                   |
| Task Completion      | Whether core tests or objectives are met                  | The model has not solved the main problem       |
| Regression Check     | Whether existing capabilities remain normal               | Side effects from changes                       |
| Behavior Constraints | Whether there is overstepping, timeouts, or invalid loops | Tool strategies and safety issues               |
| Cost                 | Token, time, and tool usage                               | Whether the task is completed too inefficiently |

Saving only a total score will prevent subsequent teams from understanding why the reward changes.

### 1.4 Recording the Full Trajectory of the Current Policy

Multiple trajectories need to be generated for the same task, as the model may attempt different approaches. Each trajectory should record the input, model action, tool parameters, environment return, time, and termination reason for each step. The model version, sampling parameters, and agent scaffold that generated the trajectory must also be saved.

These version fields directly influence the training process. If the trajectories in an asynchronous system come from an earlier model, the training side needs to decide whether to continue using them, apply importance sampling corrections, or discard them. If a task only appears in one scaffold, the model may learn the outer template without truly learning to solve the task.

### 1.5 Distinguishing Between Invalid Failures and Learnable Failures

Successful trajectories are suitable for use as SFT demonstrations and can also participate in RL. Failed trajectories cannot be entirely deleted: the training value of three types of failure—model making a mistake in the last step, calling the wrong tool, or the environment crashing midway—is completely different.

When filtering, first distinguish among four categories of results:

| Result              | Eligible for Training | Handling Method                                               |
| ------------------- | --------------------- | ------------------------------------------------------------- |
| Invalid Environment | No                    | Resample after fixing the environment                         |
| Unreliable Verifier | No                    | Modify rules and recalculate historical samples               |
| Learnable Failure   | Yes                   | Retain the trajectory, failure location, and per-step rewards |
| Model Success       | Yes                   | Check for exploiting loopholes, then enter SFT or RL          |

RL also requires controlling difficulty. If the model successfully samples all instances of a particular problem, there is typically little learning signal left; if all samples fail, it may indicate that the task is too difficult or the verifier is problematic. DAPO's dynamic sampling prioritizes retaining questions that have both successful and failed samples within the group.

### 1.6 Split Training, Replay, and Evaluation

Cleaned data should not be directly placed into the same directory. At least three data pools should be established with clear purposes:

- **Task Pool** stores tasks and environments that have not yet been sampled or are ready for further sampling.
- **Trajectory Pool** stores successful, failed, and intermediate states, used for SFT, RL, distillation, and replay.
- **Evaluation Pool** is only used for independent checks, not involved in training, and cannot be indirectly seen by the task synthesizer.

Samples generated from the same GitHub repository, the same web source, or the same synthetic template should be grouped and split by their source. Randomly splitting each sample individually can easily lead to highly similar tasks entering both the training set and the evaluation set simultaneously, resulting in artificially inflated performance metrics.

## 2. Saving Data: Making Each Trajectory Trackable and Replayable

After the production line obtains tasks and trajectories, the next step is to save them in a format that allows querying, replaying, and re-evaluation. The field table specifies what to record, hierarchical storage preserves the origin of the data, and replay records ensure that historical trajectories can be rerun.

### 2.1 What Fields Should Be Recorded for a Trajectory

After completing the six steps, the task, environment, actions, and rewards must be able to be re-associated. Below is a minimal field table. Real systems can split these into databases and object storage, but they must not lose the relationships between the fields.

| Category    | Key Fields                                                                                             |
| ----------- | ------------------------------------------------------------------------------------------------------ |
| Task        | `task_id`, capability domain, source, goal, difficulty, data license                                   |
| Environment | Image or snapshot version, initial state, tool list, network and permission policies                   |
| Generation  | Model version, policy version, sampling parameters, scaffold, start and end time                       |
| Trajectory  | Observation per step, action, tool response, token count, termination reason                           |
| Validation  | Validator version, sub-reward, total reward, test log, whether human re-evaluation is needed           |
| Governance  | Deduplication cluster, training/evaluation split, quality status, creation time, upstream data version |

Among these, the **validator version** is often overlooked. After rule changes, the reward of the same trajectory may change. Without a version number, the team cannot explain why the training curve suddenly changes, nor can they recalculate historical data.

### 2.2 Retaining Data at Each Layer with UltraData

Wallace and Tsinghua University's UltraData divides general data into layers L0 to L4: from traceable raw data, through cleaning, deduplication, quality selection, and deep processing, toward organized, verifiable knowledge resources. The paper discusses data management that spans pre-training, mid-training, and alignment, and does not directly define RL trajectories as another set of L0 to L4.

In RL engineering, we can borrow its idea of "layered without overriding upstream data":

| Borrowed Layer   | RL Data Example                                                                                   |
| ---------------- | ------------------------------------------------------------------------------------------------- |
| Raw Layer        | Issues, PRs, web pages, user tasks, or public problems                                            |
| Cleaned Layer    | Deduplicated, anonymized, license-checked, and traceable task candidates                          |
| Executable Layer | Locked environments, tools, and verifiers, capable of repeatable tasks                            |
| Trajectory Layer | Successful and failed trajectories with model versions, environment returns, and per-step rewards |
| Training Layer   | Bucketed by difficulty, domain, and purpose, suitable for SFT, RL, OPD, or evaluation             |

This table represents an engineering adaptation of the UltraData idea, not the official RL data classification proposed in the paper. The key principle is to retain upstream data: if only the final training JSONL is saved, it will be impossible to reconstruct the data from the original tasks when validation rules fail.

### 2.3 How to Save Replayable Trajectories

The following JSON illustrates the field relationships. Large logs, container images, and per-token probabilities are typically stored in object storage, with the records storing the location and checksum.

```json
{
  "task_id": "swe-01942",
  "source": {
    "type": "github_pr",
    "group_id": "repo-318",
    "license_status": "approved"
  },
  "environment": {
    "image_digest": "sha256:...",
    "base_commit": "8c27...",
    "tools_version": "code-agent-v4"
  },
  "generation": {
    "policy_version": "step-1840",
    "scaffold": "test-driven-v2",
    "temperature": 0.8
  },
  "trajectory_uri": "object://rl-runs/run-77/trajectory.jsonl",
  "verification": {
    "verifier_version": "swe-verifier-v6",
    "environment_valid": true,
    "task_reward": 1.0,
    "regression_reward": 1.0,
    "cost_penalty": -0.08,
    "termination": "success"
  },
  "governance": {
    "split": "train",
    "dedup_cluster": "cluster-9021",
    "quality_status": "accepted"
  }
}
```

When replaying, the system restores the container based on the environment summary, starts from the same commit, re-executes the trajectory according to the recorded tool protocol, and calculates the rewards using the same version of the verifier. Only when the replay results are consistent does the training data possess basic auditability.

## 3. Quality Control: Stopping Errors Before They Enter Training

Data has been saved, but that does not mean it is suitable for training. Next, we need to set up quality gates along the production line, use training dashboards to observe the data that has entered the system, and finally combine this with open projects to understand how these checks are implemented in real-world systems.

### 3.1 How Data Errors Propagate Along the Production Line

Data errors can amplify along the production line, so we cannot only check once when exporting JSONL. Each stage should answer a question before the data proceeds to the next step.

1. **Source Gate**: Is the data allowed to be used? Does it contain privacy information, keys, or evaluation answers?
2. **Task Gate**: Is the goal clear? Can the task be completed from the specified initial state?
3. **Environment Gate**: Can the environment be restarted? Are external dependencies stable?
4. a **Validation Gate**: Can correct solutions pass, and do obvious errors fail? Is the reward easy to exploit?
5. **Trajectory Gate**: Are the actions and tool returns complete? Are there truncations, timeouts, or service errors?
6. **Training Gate**: Are the difficulty, domain, length, and success rate balanced? Is there overlap with the evaluation set?

Early errors can propagate to later stages. A missing dependency in a code environment can cause all rollouts to fail. If the system attributes this to model capability, the difficulty scheduling will continue to increase such tasks, and training compute will be continuously wasted on invalid samples. The purpose of phased gates is to explain the origin of errors before they enter downstream processes.

### 3.2 How Training Dashboards Identify Data Failures

In addition to the total reward, training dashboards should at least preserve the following metrics:

- **Environment Success Rate**: Whether the environment can be started, dependencies are installed, and services are called correctly.
- **Effective Trajectory Rate**: After excluding service errors, truncations, and format damage, how many trajectories remain?
- **Task Success Rate Distribution**: Observe by domain, difficulty, source, and environment version to avoid local failures being masked by averages.
- **Intra-Group Reward Variance**: Determine whether there is still learnable signal in a batch of tasks.
- **Validator Disagreement Rate**: Whether there is a systematic disagreement between rules, model judges, and human verification.
- **Trajectory Length and Cost**: When success rate increases, are token, time, and tool call costs out of control?
- **Strategy Staleness**: How many steps behind the current training version is the trajectory generation?
- **Redundancy and Evaluation Contamination**: Does the new data overlap with existing tasks, evaluation questions, or source-related variations?

When a reward spike occurs, first check the environment success rate and validator version, then check the model capability. Validator widening, test service timeouts defaulting to pass, and webpage snapshots changing can all create false reward improvements.

### 3.3 How Public Projects Implement This Data Line

The previous six steps form a general production line, and different projects reinforce different aspects within this framework. Below, we only compare the confirmed practices from publicly available materials, without inferring undisclosed data volume, human processes, or mixing ratios.

| Team and Public Project | Data Engineering Focus                                                                                                       | What Can Be Learned                                                                         |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| FeiWall MiniCPM5        | Domain-specific RL Teachers, Reusing Domain Prompts for OPD, UltraData Hierarchical Management                               | How expert capabilities are integrated into a small model                                   |
| Moonshot Kimi K3        | Knowledge Graph-Guided Task Synthesis, Composable White-Box Environments, Multi-Stage Validation and Human Review            | How tasks, tools, contexts, and scaffolds together form a training configuration            |
| MiniMax M2.1            | GitHub PR to Docker Environment, Diverse Tasks, Multiple Scaffolds, Validation Agent                                         | How real software events become large-scale verifiable data                                 |
| Qwen-AgentWorld         | Containers, MCP, and Multi-Type Simulators, Next-State Prediction, Turn-Level Filtering, Hybrid Rewards                      | How to filter out the parts of a trajectory that truly contain environmental information    |
| Byte Seed / DAPO        | DAPO-Math-17k, Dynamic Sampling, Long Trajectory Handling                                                                    | How to allocate computing power to problems that still have learning signals                |
| GLM-5.2 / slime         | Environment and Validator Interface, Continuous Data Buffering, Single-Trajectory Asynchronous RL, Policy Version Management | How agent trajectories of varying lengths can be continuously fed into training             |
| NVIDIA Cascade 2        | Multi-Domain Cascade RL, Staged Teachers, On-Policy Distillation, Open RL Data                                               | How to use intermediate teachers to fix capability regression caused by subsequent training |

These projects employ different algorithms, yet their data pipeline is highly consistent: **transform the task into an executable environment, record the full trajectory of the current policy, use a reliable verifier to score, and then select the next batch of data based on difficulty, domain, and version.**

## 4. Data Recycling: Feeding Failures and Expert Signals into the Next Round

Trajectories that have passed quality control have two destinations: successful and failed samples enter the next round of training, while domain teachers provide more detailed per-token signals. The OPD of MiniCPM5 demonstrates the second approach, which also explains why data recycling still relies on the previous steps of task, environment, trajectory, and version recording.

### 4.1 How MiniCPM5 Merges Expert Capabilities with OPD

The MiniCPM5-1B, released by FaceMe and OpenBMB, adopts a three-step training process: SFT, RL, and OPD. During the RL phase, domain-specific teachers are trained for tasks such as mathematics, coding, closed-book QA, and writing. The OPD then merges the capabilities of these teachers into a released model.

Let us first understand OPD intuitively. The student model faces a coding problem and generates an answer based on its current policy. The teacher observes the prefix already generated by the student and provides a more suitable probability distribution for the next token. The student gradually adjusts its distribution. Since the trajectory comes from the student's current policy, the teacher corrects the states the student would actually encounter.

This process reuses the prompt pool used during the training of domain teachers, reducing the need for additional question curation. The data system still needs to record several additional pieces of information: which domain the prompt belongs to, which student policy sampled it, which teacher was used, whether the distillation signal was normal, and how the length and accuracy changed before and after training. The full logits of the teacher can be transmitted online and do not necessarily need to be stored long-term; versions and aggregated statistics must be retained to reproduce the experiment.

OPD addresses the question of **how to merge multiple experts into a single model.** The domain tasks, environments, and verifiers still need to be prepared by data engineering. It cannot replace the previous six-step production line.

### 4.2 Token-wise Training Signal for OPD

Let the student's distribution be $p_\theta(\cdot\mid x_{<t})$ given the prefix $x_{<t}$, and the teacher's distribution be $q(\cdot\mid x_{<t})$. The implementation of MiniCPM5 published uses the reverse KL divergence:

$$
D_{\mathrm{KL}}\!\left(p_\theta\|q\right)
=
\sum_a p_\theta(a\mid x_{<t})
\log\frac{p_\theta(a\mid x_{<t})}{q(a\mid x_{<t})}.
$$

The student first samples a trajectory according to $p_\theta$, then compares the student and teacher distributions at each position. To reduce computation and communication, MiniCPM5 takes the top-$k$ tokens from both distributions and approximates the calculation over their union. This signal is denser than a single 0/1 reward at the end of the response.

Each term in the summation can be understood as a token-wise comparison. If the student assigns a probability of $0.6$ to token A, and the teacher assigns $0.3$, then $\log(0.6/0.3) = \log 2 > 0$, and this term will push the student to reduce overconfidence in A. If the probabilities are the same on both sides, the term is zero. The summation is weighted by the student's probability $p_\theta$, so the training focuses on the tokens that the student is most likely to choose. The top-k approximation retains only the most probable candidates from both sides, thus preserving the main differences with minimal communication overhead.

On the data side, three things need to be carefully checked: the student's trajectory and the teacher's score use the same tokenizer; the chat template and tool protocol of the teacher and the student are consistent; and each trajectory can be traced back to a specific version of the teacher and the student. If any of these aspects drift, the token-wise signal will lose its comparability.

## Summary of This Section

- The basic unit of large-scale RL data is a combination of tasks, environments, trajectories, verification results, and version information.
- Data production proceeds sequentially through task construction, environment packaging, verifier setup, trajectory sampling, filtering and annotation, and training partitioning.
- Task pools, trajectory pools, and evaluation pools serve different purposes; samples from the same source should be grouped and split to prevent evaluation leakage.
- OPD can consolidate the capabilities of multiple domain teachers, provided that reliable domain tasks, environments, and prompt pools are prepared first.
- The goal of data engineering is to ensure that each reward is explainable, each trajectory is replayable, and each failure leads to improvements in the next round.

## Open Resources

- [OpenBMB: MiniCPM5-1B Training Process and OPD](https://github.com/OpenBMB/MiniCPM#-minicpm5-1b)
- [UltraData: Tiered Data Management](https://arxiv.org/abs/2602.09003)
- [Thinking Machines Lab: On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/)
- [Moonshot AI: Kimi K3](https://github.com/MoonshotAI/Kimi-K3)
- [MiniMax M2.1: Post-Training Experience for Agent Models](https://www.minimax.io/news/post-training-experience-and-insights-for-agent-models)
- [Qwen-AgentWorld](https://qwen.ai/blog?id=qwen-agentworld)
- [Qwen-AgentWorld Official Repository](https://github.com/QwenLM/Qwen-AgentWorld)
- [Byte Seed and Tsinghua AIR: DAPO](https://github.com/BytedTsinghua-SIA/DAPO)
- [THUDM: slime](https://github.com/THUDM/slime)
- [GLM-5.2: Single-rollout Asynchronous Optimization](https://arxiv.org/abs/2607.07508)
- [NVIDIA: Nemotron-Cascade 2](https://research.nvidia.com/labs/nemotron/nemotron-cascade-2/)
- [AI2: Open Instruct / Tülu 3 RLVR](https://github.com/allenai/open-instruct)
