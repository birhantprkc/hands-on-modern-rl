# 18.2 工业后训练流水线

上一节我们把一次 RL 训练拆成了**生成、奖励、更新和参数同步**——这解决了"算法怎么在多卡上跑起来"的问题。但真正开始做工业训练时，你会遇到更大的问题：**第一批训练数据从哪里来？** 模型训完以后，怎么知道它是**真的学会了**，还是只是记住了训练集？那些做错的题目，怎么让它在下一轮训练中**不再犯**？

先看一个代码模型的真实例子。团队从真实 GitHub 仓库中收集 bug 报告，为每个问题准备可以运行的环境、**依赖锁定**和测试用例。模型先通过 **SFT** 学会读取仓库、调用工具和提交补丁，再通过 **RL** 反复尝试修复；测试程序给每次尝试打分。训练完成后，团队在**完全没见过的新仓库**上评测模型，把失败任务归类，补充新的数据和验证规则，然后开始下一轮训练。

这就形成了工业后训练的**基本闭环**——不是训完一轮就结束，而是**一轮接一轮地迭代**：

```text
定义目标能力
    ↓
构造数据与可验证环境
    ↓
SFT 教会基本行为
    ↓
采样、筛选并进行 RL 或偏好优化
    ↓
在独立评测集和真实场景中检查结果
    ↓
分析失败样本，回流新数据与新奖励
    └──────────────────────► 下一轮训练
```

## 后训练为什么是一个闭环

第一轮数据只能覆盖团队已经想到的问题。模型进入真实场景后，会暴露新的失败方式：代码能够编译但没有解决需求，工具调用格式正确但一直重复搜索，数学答案正确但推理过程碰巧猜中。评测发现这些问题后，团队需要补充任务、环境或奖励，再重新训练。

因此，工业后训练的产物不只有模型权重，还包括能够持续复用的数据集、环境、验证器和评测体系。模型能力正是在多轮闭环中逐步建立的。

## 一轮完整后训练的六个阶段

| 阶段        | 要回答的问题                           | 典型产物                         |
| ----------- | -------------------------------------- | -------------------------------- |
| 目标定义    | 模型最终要完成什么任务                 | 能力清单、成功标准与安全边界     |
| 数据与环境  | 怎样让任务可以训练、可以重复、可以判分 | 提示、轨迹、沙箱、测试与验证器   |
| SFT         | 模型是否会按正确格式开始行动           | 具备基本行为的冷启动模型         |
| RL/偏好优化 | 模型能否通过尝试提高成功率             | 针对目标能力优化后的模型         |
| 评测        | 提升是否来自真实能力，其他能力是否退化 | 独立指标、失败样本与回归测试     |
| 数据回流    | 下一轮应该补什么                       | 新任务、新轨迹和修改后的奖励规则 |

公开方案使用的算法名称很多，初学时先问它改变了表中的哪一个阶段。下面继续用代码修复任务走完六步，再把不同公司的方法放回对应位置。

### 定义目标能力：先写清楚什么叫"修好"

假设团队要训练一个代码模型修复 Python 项目中的真实 bug。第一步先写清楚什么叫"修好"：新增测试必须通过，原有测试不能退化，模型不能修改无关文件，也不能绕过测试。由此得到能力清单、成功标准和安全边界。部署时如果还限制运行时间、工具次数和回答长度，这些约束也要在此时确定。

### 构造数据与环境：让每个任务都能重复启动

GitHub issue 只有问题描述，还不是可训练任务。团队要保存出错前的仓库版本，锁定依赖和启动命令，准备新增测试与回归测试，再把整个项目封装进可以重复启动的沙箱。同一个任务每次都从相同状态开始，模型得到的奖励才可以比较。

环境型任务还要覆盖执行方式的变化。MiniMax 会使用不同的 Agent Scaffold 采样同一类任务；Kimi K3 会改变工具、提示和上下文管理；Qwen-AgentWorld 同时使用容器、MCP、移动端和网页环境。这样可以减少模型只记住某一种外层模板的风险。

### SFT 冷启动：先教模型基本行为格式

模型一开始可能不会读取仓库、调用测试命令或提交补丁。团队先收集成功轨迹做 SFT，让模型学会工具格式、操作顺序和停止条件。SFT 在这里负责建立基本行为，还不要求模型通过反复试错找到更好的修复策略。

### RL 提升：通过反复试错提高成功率

具备基本行为以后，当前模型对同一个任务尝试多次。测试程序给每条轨迹计算奖励，PPO、GRPO 或偏好优化提高成功轨迹的概率。数学、代码和写作等能力也可以先训练成不同领域专家，再通过 On-Policy Distillation 合回一个模型；MiniCPM5、Kimi K3 和 NVIDIA Nemotron-Cascade 2 都公开了这条路线。

### 独立评测：在没见过的任务上检验真实能力

训练结束后，团队在未参加训练的新仓库上检查补丁通过率、无效工具调用、平均成本和通用能力回归。评测配置要接近真实部署：长度预算、思考强度、量化方式、工具协议和上下文管理不同，都可能改变最终结果。训练奖励升高而独立评测不变时，模型可能只适应了训练环境或利用了验证器漏洞。

### 失败回流：把问题送回下一轮训练

评测失败要按原因回流。任务覆盖不足就补充新问题；环境无法恢复就修复沙箱；测试过弱就升级验证器；模型不会某种工具操作就补成功轨迹；模型已经会操作但成功率仍低，再进入下一轮 RL。团队必须记录任务、环境、轨迹、奖励和模型版本，才能重放失败并判断应该修改哪一层。

这样看，六个阶段并非六套独立工作。前一阶段的产物是后一阶段的输入，最后的评测又会回到最前面的目标、数据和环境。[18.5 大规模 RL 数据工程](./data-engineering) 将继续展开其中的数据生产线。

## 环境型任务对闭环的扩展

数学 RL 的环境通常只有题目和答案验证器。代码仓库、网页和桌面应用把每个阶段都扩展了：目标中出现工具和成本约束，数据要携带环境，SFT 要教授操作协议，RL 要处理长轨迹，评测要复现部署条件，失败回流还要保存完整状态。

| 阶段        | 环境型任务增加的内容                     | 代表案例                        |
| ----------- | ---------------------------------------- | ------------------------------- |
| 目标定义    | 工具权限、时间预算、长度与安全边界       | Apple 端侧约束、OpenAI 工具任务 |
| 数据与环境  | 仓库、容器、网页状态、工具和可恢复检查点 | MiniMax M2.1、Kimi K3           |
| SFT         | 工具格式、环境观察、动作顺序和停止条件   | Qwen-AgentWorld、UI-TARS        |
| RL/偏好优化 | 多轮轨迹奖励、异步经验和领域专家训练     | GLM-5.2、DAPO、MiniCPM5 OPD     |
| 评测        | 真实环境、独立任务、成本与安全回归       | Claude 系统卡、Computer Use     |
| 数据回流    | 失败分类、拒绝采样、蒸馏与验证器更新     | DeepSeek-R1、Nemotron-Cascade 2 |

## 从四个公开案例看闭环的不同实现

下面四个案例分别改变了闭环中的不同位置。MiniCPM5 处理多个领域专家怎样合回一个模型；Kimi K3 把环境配置本身变成训练数据；Qwen-AgentWorld 在训练动作前先学习环境变化；GLM-5.2 则让长短不一的轨迹异步进入训练。阅读时先找它改变了六个阶段中的哪一步，再看使用了什么算法名称。

### MiniCPM5：领域专家 RL 与 OPD 能力合并

MiniCPM5-1B 的公开流程是 SFT、RL、OPD。团队先分别训练数学、代码、闭卷问答、写作等领域的 RL 教师模型，再让最终模型对同一批领域提示进行采样。教师模型对学生生成的每个位置给出概率分布，学生据此调整自己的输出分布。这就是 On-Policy Distillation（OPD）。

它和普通离线蒸馏的差别在数据来源。离线蒸馏先保存教师写好的答案，学生反复模仿这些固定答案；OPD 的回答由当前学生自己生成，因此教师纠正的是学生此刻真正会走到的状态。MiniCPM5 还复用了训练各领域教师时的提示，不需要再构造一套蒸馏题目。

从数据工程角度看，这套方法要求每条样本同时记录领域、提示版本、学生策略版本、教师版本和逐 token 的教师信号。OPD 的公式与日志结构放在 [18.5](./data-engineering) 的进阶部分。

### Kimi K3：可组合训练环境

Kimi K3 把工具、提示、上下文管理、技能、记忆和子智能体组合成统一训练环境。同一个任务可以换用不同工具配置或执行框架，避免模型只熟悉一种 harness。任务侧还可以从知识图谱选择相关节点，再检索公开文章、博客和代码仓库，合成长程研究任务。

Kimi K3 同样先训练不同领域和思考强度的专家，再用多教师 OPD 合并能力。对百万 token 的长任务，系统保存未完成轨迹和外部 KV 状态，下一轮从中间继续，而不必每次从头生成。

### Qwen-AgentWorld：环境建模与动作学习

Qwen-AgentWorld 使用容器沙箱、MCP 工具、Android、网页和操作系统模拟器构造任务。训练分为三个阶段：继续预训练让模型学习环境知识，SFT 让模型预测下一状态并学会基本操作，最后用 GSPO 和混合奖励训练完成任务。

它还会挑出轨迹中真正包含环境变化的轮次重点训练。纯格式重复、没有新观察的轮次提供的信息较少，不必和关键操作获得相同权重。这个例子说明，轨迹筛选已经从“整条保留或删除”深入到回合级别。

### GLM-5.2：异步单轨迹优化

工具任务的时长差异很大，按组等待多个 rollout 会浪费算力。GLM-5.2 公开的 SAO 让每个提示只生成一条轨迹，并持续异步送入训练；同时处理旧策略数据、价值估计和 token 裁剪。slime 则在工程上连接训练后端、SGLang rollout、自定义环境、验证器与数据缓冲区。

这说明数据队列并非简单缓存。异步训练中的每条轨迹必须携带生成它的策略版本，训练侧才能判断数据是否过旧、是否需要重要性修正或丢弃。

### 案例资料来源

- [MiniCPM5-1B 官方训练说明](https://github.com/OpenBMB/MiniCPM#-minicpm5-1b)
- [Kimi K3 官方报告与仓库](https://github.com/MoonshotAI/Kimi-K3)
- [Qwen-AgentWorld 官方说明](https://qwen.ai/blog?id=qwen-agentworld)
- [slime 官方仓库](https://github.com/THUDM/slime)
- [SAO：GLM-5.2 的异步 Agentic RL](https://arxiv.org/abs/2607.07508)
- [Nemotron-Cascade 2](https://research.nvidia.com/labs/nemotron/nemotron-cascade-2/)

## 按六个阶段查阅公开实践

下面保留各团队的完整公开实践，作为前面六个阶段的案例库。同一团队通常覆盖多个阶段，不适合强行拆散；查阅时可以从当前问题进入：

| 当前要解决的阶段 | 优先查看的案例                                                      |
| ---------------- | ------------------------------------------------------------------- |
| 目标定义         | OpenAI Operator/Codex、Apple、IBM Granite、xAI Grok                 |
| 数据与环境       | MiniMax SWE Scaling、WebExplorer、UI-TARS、LongCat、Qwen-AgentWorld |
| SFT              | Qwen2.5、InstructGPT、Meta Llama、Microsoft Phi                     |
| RL/偏好优化      | DeepSeek、Qwen3、DAPO/VAPO、Kimi、GLM、Hunyuan                      |
| 评测             | Claude 系统卡、Gemini、Llama 4、StepFun、多模态与安全案例           |
| 数据回流         | Rejection Sampling、Long-to-Short、OPD、Expert Soup、Tulu 3         |

接下来的公司标题只表示资料来源，并不代表新的训练步骤。阅读一个案例时，先找目标、数据、SFT、RL、评测和回流分别写在哪里，再看它新增了哪一种方法。

---

## 有些事不是没有药：二十多个团队的公开实践

六个阶段讲完了，四篇代表性案例也拆解完了。但真正动手做的时候，你还是会遇到一堆具体问题：SFT 数据到底从哪来？长度奖励什么时候加？异步 RL 怎么保证数据不过期？多模态 reward 怎么平衡？

这些问题不是没有答案——过去两年里，国内外二十多个团队已经把自己踩过的坑、试过的配方、跑通的系统，写进了技术报告、博客和论文里。下面把这些实践按团队整理出来，你不需要一口气读完。遇到具体问题时，回到前面的阶段表，找到对应的团队，看他们是怎么解决的就好。

### 国内团队公开实践汇编

后面的材料按团队整理，适合在理解六阶段闭环以后查阅。每个案例都可以从三个问题进入：训练任务从哪里来，奖励怎样判断成功，失败怎样回到下一轮。公司名称只表示资料来源，不代表另一套独立的学习路线。

#### MiniMax

> **资料入口**：[MiniMax M2.1: Post-Training Experience and Insights for Agent Models][^minimax_m2_1]、[MiniMax-M1][^minimax_m1]、[WebExplorer][^minimax_webexplorer]

MiniMax 的公开资料应该分三条线读：M2.1 的 agent 后训练经验、MiniMax-M1 的长思考 RL scaling、WebExplorer 的长程网页 agent 数据合成与 RL。三份资料覆盖了代码 agent、应用开发 agent、网页搜索 agent 和长上下文 reasoning model，不应只概括成“可验证环境 + reward”。

#### M2.1 SWE Scaling：从 GitHub 数据构建可运行环境

**动机**：软件工程任务天然可验证，但原始 GitHub 数据不能直接训练 agent。一个 issue、PR 或 commit 只有文本和代码差异，不等于 RL 样本；必须恢复仓库状态、构造任务描述、准备依赖、确定验证命令，才能变成“模型行动 -> 环境反馈 -> reward”的闭环。

**数据构造**可以拆成五步：

- **挖掘真实事件**：从 GitHub merged PR、commit、issue、测试变更和代码 diff 中筛出有明确修复目标的样本。
- **改写任务形态**：SWE-Resolve 要求模型修 bug 或实现需求；SWE-Test 反过来要求模型写测试，使其在补丁前失败、补丁后通过；SWE-Review 要求模型审查代码变更并指出问题。
- **封装可执行环境**：checkout 到补丁前状态，安装依赖，准备测试命令，构建 Docker / sandbox，保证模型提交 patch 后可以自动验证。
- **补齐训练字段**：生成 original problem description、test-case reward、runnable environment 等字段，形成既可 SFT 又可 RL 的样本。
- **扩语言和场景**：M2.1 报告里提到覆盖 10+ 主要编程语言、10,000+ runnable PRs 和 140,000+ variable tasks。

**奖励设计**也要按任务类型拆：

- **SWE-Resolve**：主 reward 是测试是否通过，同时检查是否破坏已有测试、是否修改无关文件、是否只硬编码测试。
- **SWE-Test**：测试必须在 patch 前失败、patch 后通过，才能说明测试确实覆盖目标 bug。
- **SWE-Review**：它不是完全可执行任务，只能近似验证，例如用另一个 LLM 检查 review 是否命中真实问题，同时控制 hallucination rate。

**关键方法**是“把一个真实开发事件改写成多个等价任务”。这样同一份 GitHub 数据可以产出修复、测试、审查等多种训练信号。

**Multi-scaffold** 是 M2.1 的另一条主线。scaffold 指 agent 外层执行框架，例如上下文管理方式、工具调用协议、反思/计划模板、文件编辑接口。如果 SFT 和 RL 都只在一个 ReAct loop 上做，模型会过拟合这个 loop 的格式。MiniMax 的做法是用 multi-scaffold rejection sampling 生成 SFT 数据，并在 RL 中让不同 scaffold 参与 rollout。

**最小复现**可以先准备两到三种简单 scaffold：直接 ReAct、plan-then-edit、test-driven loop；同一批 SWE 任务分别跑这些 scaffold，保留成功轨迹做 SFT，再在 RL 中随机 scaffold，让模型学任务本身而不是学模板。

#### M2.1 AppDev：从固定测试到 Agent-as-a-Verifier

**动机**：“从零开发应用”很难预写完整测试。一个前端/后端/移动端应用是否完成，不只看函数输出，还要看能否启动、交互是否正确、视觉是否合理、业务逻辑是否闭环。

**专家数据**：MiniMax 引入 experts-in-the-loop。前端、后端、Android、iOS 专家负责设计 prompts、meta-queries、rubric-based rewards 和系统提示。专家提示里包含最佳实践；训练时可以把系统提示从轨迹中去掉，让模型把专家启发蒸馏进默认行为。

**Reward 三层结构**：

- **Execution level**：检查代码是否能编译、启动、运行。
- **Interaction level**：通过 Playwright 等工具真实点击页面、输入表单、检查状态变化，判断业务逻辑是否正确。
- **Visual level**：按相对一致的审美标准评分，例如布局是否明显错位、关键信息是否可见、交互控件是否可用。

**和普通 LLM-as-a-judge 的区别**：judge 不是只看静态截图或最终文本，而是作为 agent 进入 sandbox 交互。

**最小复现**可以做一个 Todo App / 登录页 / 数据表格任务集：给模型需求，让它生成项目；自动 `npm install && npm run dev`；用 Playwright 跑 5-10 个交互检查；最后用 rubric judge 补视觉和需求覆盖评分。

#### WebExplorer：构造长程网页 Agent 难题

**动机**：现有开源 web agents 在 BrowseComp、GAIA、WebWalkerQA、FRAMES 等复杂信息检索任务上弱，一个核心原因是缺少高质量、长程、可训练的数据。人工写这类题很贵；简单 query evolution 又容易变成不自然的问题；图构造方法需要复杂的节点扩展和启发式选择。

**核心思路**：WebExplorer 用 model-based exploration 让强模型自主探索信息空间，再用 iterative long-to-short query evolution 生成更难的问题。

**WebExplorer-QA 的合成流程**可以按“探索”和“演化”两阶段理解：

- **探索阶段**：从 Wikipedia seed entities 出发，在 prompt 中给三个 BrowseComp-en 的 QA exemplars，让模型围绕 seed entity 执行 search / browse，构建一个局部信息空间。这个阶段不是直接产题，而是让模型找到相关实体、网页、事实链和潜在可问的答案。
- **演化阶段**：做 long-to-short query evolution。初始问题通常包含较多显式线索，后续迭代逐步删除或压缩显著信息，让问题更短、更隐含、更需要多步搜索。论文使用 5 次 evolution，最终合成约 40K WebExplorer-QA。
- **设计重点**：先探索出答案和证据，再把问题变难，而不是凭空写难题。

**训练 recipe** 分两段：

- **冷启动 SFT**：先教模型正确使用 search / browse 工具，并形成基本长程搜索格式。论文用 Qwen3-8B 作为底座，约 13K SFT 样本，batch size 32，learning rate 1e-5，训练 4 epochs。
- **GRPO 阶段**：不需要人工轨迹，只需要合成 QA 对，让模型自己探索不同搜索路径。论文用约 12K 样本做 GRPO，每组 8 rollouts，batch size 64，learning rate 1e-6，并逐步把最大响应长度扩到 128K、最大工具调用轮数扩到 100。

**Reward 公式**是复合形式：`R = 0.2 * R_format + R_correct`。

- **`R_format`**：检查工具调用、思考结构、回答格式是否符合协议。
- **`R_correct`**：用 DeepSeek-V3 作为自动 judge，根据 ground-truth answer 判断最终回答是否正确。
- **权重动机**：web agent RL 中，如果不奖励格式，模型会在 search / browse 协议上漂；如果只奖励格式，模型会学会空转工具。所以格式只占小权重，正确性是主信号。

**最小复现**可以这样做：选 500-1000 个 Wikipedia seed entities；为每个 seed 让强模型用搜索工具探索 5-10 个相关网页并产出候选 QA；用另一个模型过滤答案不唯一、证据不足、过于简单的问题；做 2-3 轮 query compression，把显式实体名、日期、地点等线索逐步隐藏；保留答案和证据 URL；用 1K-5K 轨迹做 SFT，让 7B/8B 模型学会 search/browse 格式；再用 1K QA 做 GRPO，reward 用格式检查 + LLM judge / exact match。

**评估指标**至少看三项：平均工具调用轮数是否增加、正确率是否提升、是否出现无效搜索循环。

#### MiniMax-M1：长思考 RL 的训练效率

**动机**：M1 不是在讲 agent 环境，而是在解决 test-time compute scaling 太贵的问题。长 CoT 输出到 64K、80K、100K token 时，普通 attention 和普通 RL 都会非常昂贵。

**模型与 RL**：M1 用 hybrid MoE + Lightning Attention 支持 1M 输入上下文和 40K/80K thinking budget，再用 CISPO 做高效 RL。CISPO 的要点是 clip importance sampling weights，而不是直接 clip token updates；这样在长响应里仍然保留所有 token 的梯度贡献，并降低方差。

**数据和 curriculum**：

- **继续预训练**：从 MiniMax-Text-01 继续训练，强化 STEM、代码、书籍、reasoning、长上下文。
- **SFT 冷启动**：注入 CoT 模式，为 RL 做起步策略。
- **RL 数据混合**：可验证任务包括数学、竞赛编程、SynLogic 生成的 41 类逻辑任务、SWE-bench 派生的软件工程 sandbox；不可验证任务如 QA、创意写作，则用生成式 reward model 反馈。
- **逐步扩长度**：扩到 80K 输出时，不是直接放开长度，而是从 40K 逐步扩到 48K、56K、64K、72K、80K。
- **晋级指标**：用生成序列 perplexity 是否收敛、99 分位输出长度是否逼近窗口上限等指标判断是否进入下一阶段。
- **数据筛选**：报告还提到使用 40K 模型筛数据、移除过易样本、增加困难数学和代码比例、下采样会导致重复同质化的 synthetic reasoning 数据。

**可复现重点**不是复刻 512 H800，而是在小模型上验证三件事：长输出训练要逐步扩窗口；过易样本要过滤，否则没有 RL 信号；合成 reasoning 数据如果模式太单一，会让长上下文 RL 变得重复和不稳定。

#### 阿里 Qwen / 通义

> **资料入口**：[Qwen2.5 Technical Report][^qwen2_5]、[Qwen2.5-Math][^qwen2_5_math]、[QwQ-32B][^qwq_32b]、[Qwen3][^qwen3]、[Qwen3-Coder][^qwen3_coder]、[Qwen3-Coder-Next][^qwen3_coder_next]、[Tongyi DeepResearch][^tongyi_dr]

Qwen 的资料应该分成四层读：通用 instruct 后训练、数学自改进、reasoning RL、agentic coding / deep research。它的价值在于公开材料把“可验证任务”和“通用体验回填”都讲到了，而不是只停在某个 GRPO 名字上。

#### Qwen2.5：通用指令模型的多阶段后训练

Qwen2.5 报告的 post-training 不只是 SFT。它先用百万级监督样本覆盖通用问答、代码、数学、多语言、结构化数据分析、长文本生成和安全，再做多阶段 RL。这里的 motivation 是：base model 有知识，但不会稳定按照用户意图输出；单轮 SFT 又容易把模型训练成“会答但不够偏好对齐”。因此 Qwen2.5 把 instruction following、长文本、结构化输出、专业能力和人类偏好分阶段处理。最小复现时，应把通用 SFT 数据按能力域分桶，而不是混成一个大 JSONL；每个桶都保留独立 eval，例如长文摘要、JSON 输出、代码、数学、拒答和多语言。

#### Qwen2.5-Math：CoT 与 TIR 自改进闭环

这份报告的 motivation 是普通 CoT 在精确计算、符号操作和算法推理上会出错，所以 Qwen2.5-Math 同时训练 Chain-of-Thought 和 Tool-Integrated Reasoning。数据构造不是只收集人工解题，而是让模型生成候选 CoT / TIR 解法，再用答案解析器、Python executor、majority voting 和 reward model 筛选。Qwen2.5-Math-RM-72B 被用作数学 reward model，既服务 rejection sampling，也服务后续 RL。

TIR 的训练细节尤其重要。模型生成的 token 包括自然语言思考、Python 代码、工具返回和最终答案；其中 executor 返回不是模型自己写的内容，训练时要 mask 掉 executor output token 的 loss。否则模型会学到错误目标：把环境返回当作自己应生成的文本。RL 阶段使用 GRPO，在数学题上用规则 verifier / reward model 打分，并用 KL 系数约束策略漂移。报告里还强调 Qwen2.5-Math 主要面向中英文数学，不建议当通用聊天模型用，这说明专业后训练会带来能力偏置，之后若要产品化必须做通用能力回填。

最小复现路径是：选 GSM8K / MATH / OlympiadBench 子集；让强模型为每题生成 4-16 条 CoT 和 Python TIR 候选；用答案 parser、`sympy`、Python 执行和 majority vote 过滤；把正确轨迹做 SFT；再对每题采样 8 个响应，用规则 verifier 给 0/1 或部分分，做 GRPO。对 TIR 数据单独处理 loss mask，并记录工具错误率、格式错误率和最终答案正确率。

#### QwQ-32B：基于结果奖励的两阶段 RL

QwQ-32B 的公开博客把 RL 拆得很清楚：从 cold-start checkpoint 出发，第一阶段只扩 math 和 coding RL，不用传统 reward model，而是数学 accuracy verifier 和代码执行服务器。数学 reward 看最终答案是否正确；代码 reward 看生成代码是否通过预定义测试。这个阶段的目标是把“能不能解题”推上去。

第二阶段再做 general capabilities RL，奖励来自 general reward model 和一部分 rule-based verifier，用少量步数提升 instruction following、人类偏好对齐和 agent performance，同时避免数学和代码明显回落。这是一个非常实用的 recipe：先在高置信可验证任务上放大能力，再用通用偏好和规则数据修复体验。最小复现时，第一阶段可以只用数学和代码，第二阶段混入 5%-20% 通用指令/偏好/安全数据，观察 math、code、chat、agent 四类 eval 是否互相挤压。

#### Qwen3：Thinking 与 Non-Thinking 的统一后训练

Qwen3 的关键不是单纯“用了 GRPO”，而是把思考模式做成产品可控能力。后训练前两阶段是 long-CoT cold-start 和 reasoning RL：cold-start 用少量高质量长思考数据教格式、推理组织和答案边界；reasoning RL 用 query-verifier pairs 做 GRPO。报告披露了筛 query-verifier pair 的四条标准：不能出现在 cold-start 数据里；对 cold-start 模型可学习；尽量有挑战；覆盖广泛子领域。最终收集 3,995 个 query-verifier pairs，并强调大 batch、每题多 rollout、off-policy 提升样本效率，以及通过控制 entropy 稳住探索和利用。

后两阶段解决“模型只会长想”的问题。Qwen3 把有 reasoning path 和没有 reasoning path 的数据合成统一训练集，让模型同时支持 thinking 和 non-thinking；最后用 general-domain RL 回填通用能力、安全、多语言和工具体验。这个顺序对应一个可复现模板：先训练 `<think>` 格式和长 CoT；再只在 verifier 高置信任务上做 RL；然后混入短回答、普通聊天和工具指令，让模型学会什么时候不展开长思考；最后评估平均输出长度、正确率、用户偏好和非思考模式质量。

#### Qwen3-Coder 与 Tongyi DeepResearch：从答案奖励到过程奖励

Qwen3-Coder 的训练对象是 repository-level action：读文件、定位 bug、写 patch、运行测试、处理失败、提交修改。reward 的主信号来自单测、静态检查、编译、issue 需求覆盖和 patch 合理性。Tongyi DeepResearch 的训练对象是 search / read / synthesize 过程：任务不是回答一个事实，而是搜索证据、去重来源、比较冲突信息、组织带引用的报告。它们共同说明 Qwen 的 agent 后训练已经把“prompt -> answer”改成“environment episode -> verified outcome”。可复现时先做小规模 SWE-bench Lite 或网页 QA：固定工具协议，保留成功轨迹做 SFT，再用测试通过率或 answer judge 做 RL。

#### Moonshot Kimi

> **资料入口**：[Kimi k1.5][^kimi_k1_5]、[Kimi K2][^kimi_k2]、[Kimi-Researcher][^kimi_researcher]

Kimi 的三条公开线索分别对应 reasoning scaling、开放 agentic model、research agent。这里最值得拆的是 k1.5 和 Kimi-Researcher：前者回答“长思考 RL 怎么稳定”，后者回答“研究型 agent 怎么从端到端 RL 里长出来”。

#### Kimi k1.5：简化长思考 RL 训练

k1.5 的 motivation 是训练 test-time compute scaling。它没有把系统做得特别复杂，而是强调一个简洁框架：policy 采样多个响应，reward 只看 outcome，policy optimization 在 KL 约束下把高 reward 轨迹概率推高。报告明确把它和 MCTS、value function、process reward model 区分开：重点不是训练一个逐步打分器，而是让模型在足够多 rollout 中自己探索更有效的推理路径。

数据上，k1.5 把任务分成可验证和偏好型两类。数学、代码、选择题更适合 rule / execution verifier；开放问答、写作、复杂偏好需要 reward model 或 judge。训练时同一 prompt 会采样多个候选，reward 后形成相对优劣，再用 policy mirror descent 更新。可复现的关键是让采样数足够大，因为单个响应的 reward 噪声很高；同题多采样才能看出“哪些推理路径更稳”。

#### 长度奖励：控制过度思考

k1.5 报告专门讨论 overthinking：模型学会长 CoT 后，可能把 token 写得越来越多，甚至在已经得到答案后继续绕。它的 length reward 不是无条件奖励短，而是在同一题的多个候选里比较：正确答案中更短的响应得到额外奖励；错误答案不会因为短而被奖励，甚至长且错会被惩罚。这个设计把“正确且高效”写进 reward，而不是只靠 generation max length。

length reward 还要 warm up。训练早期模型还不会稳定解题，如果过早惩罚长度，会压制探索，模型可能学不到完整推理；等准确率上来后再加入长度项，才能把长思考压缩成有效思考。最小复现实验可以这样做：前 30%-50% RL steps 只用 correctness reward；之后在每题的正确样本里按长度排序，把较短正确解加 0.1-0.3 奖励，把过长错误解扣分；同时监控准确率和平均 response length，确认不是靠变短牺牲正确率。

#### Long-to-Short：从完整推理到精简回答

k1.5 的 long-to-short 思路和 length reward 配套。第一阶段允许模型用很长推理拿到正确答案，得到复杂问题的策略；第二阶段通过蒸馏、偏好或长度奖励，把冗余步骤压掉。这个流程和“直接训练短答案模型”不同：短模型要保留长模型学到的搜索和自检能力，只减少无用表达。复现时可以保留长 CoT 成功轨迹，然后让强模型或同模型生成 concise solution，做一轮 SFT / DPO，再用 verifier 确认短解仍然正确。

#### Kimi K2：Agentic Intelligence 的数据与工具闭环

K2 的公开报告强调 open agentic intelligence，重点不是单 benchmark，而是让模型在工具、代码和复杂任务里具备行动能力。对应的后训练样本应包含任务目标、工具协议、观察、动作、错误恢复和最终结果。K2 的学习点是 agent 数据不能只靠人工写演示，必须结合真实任务、合成任务、工具执行结果、verifier 和 judge，不断筛出成功轨迹，再回流到 SFT / RL。

#### Kimi-Researcher：覆盖证据链的研究 Agent 奖励

Kimi-Researcher 面向长程研究任务。它的训练单位是一个 research episode：模型提出搜索计划，调用搜索/浏览工具，阅读多个来源，提取证据，合并冲突信息，写出带引用的回答。最终 reward 不能只看“答案像不像”，还要看引用是否存在、证据是否支持结论、来源是否覆盖关键角度、是否遗漏反例、是否重复搜索低价值网页。最小复现路径是：构造 200-500 个需要多网页证据的问题；用浏览器工具记录轨迹；让 judge 分别打 evidence coverage、citation correctness、answer faithfulness、redundant-search penalty；先 SFT 成功轨迹，再用 episode-level reward 做 GRPO / DPO。

#### 字节 Seed / Doubao

> **资料入口**：[Seed1.5-Thinking][^seed1_5_thinking]、[VAPO][^vapo]、[DAPO][^dapo]、[DAPO GitHub][^dapo_github]、[UI-TARS][^ui_tars]、[UI-TARS GitHub][^ui_tars_github]、[UI-TARS-2][^ui_tars_2]、[Seed Prover 1.5][^seed_prover]、[Seed1.8][^seed1_8]

字节 Seed 的公开材料适合学习两类东西：reasoning RL 的工程补丁，以及 GUI / prover 这类环境型 agent 后训练。DAPO、VAPO、UI-TARS-2 都不是只给算法名，而是在回答“为什么大规模 rollout 会不稳”。

#### Seed1.5-Thinking：推理模型的基础训练配方

Seed1.5-Thinking 的目标是用 RL 提升数学、代码和复杂推理。它的任务构造仍以可验证题为主：数学看答案，代码看执行，逻辑题看规则 verifier。SFT 阶段先给模型长 CoT 冷启动，RL 阶段再通过 outcome reward 放大可验证能力。这个模式和 DeepSeek-R1、Qwen3 类似，但 Seed 系列后续报告更强调训练系统、采样和 advantage 处理。

#### DAPO：GRPO 的四项稳定性改进

DAPO 的 motivation 是开源社区复现大规模 reasoning RL 时，常见失败不是因为 GRPO 公式不会写，而是因为样本、clip、长度和梯度归一化细节没处理好。它在 GRPO 上加四个关键组件。

Dynamic Sampling 处理“没有学习信号”的 prompt。如果同一题采样的所有答案全对或全错，组内 reward 方差接近零，advantage 没意义。DAPO 会持续采样或过滤，直到 batch 里保留有非零 advantage 的组，把算力放在边界题上。Clip-Higher 处理探索被 PPO clip 压住的问题：长 CoT 中某些低概率 token 可能打开新解法，如果上界太紧，正确但罕见的推理路径无法被充分强化；因此它把 `eps_clip_high` 设得高于下界，例如常见配置是 low 0.2、high 0.28。

Token-Level Policy Gradient 处理长响应被样本级平均稀释的问题。普通 sequence-level loss 会让长 CoT 的关键 token 和无关 token 一起平均，信号变弱；DAPO 改成按 token 聚合，让长推理链中每个生成 token 都更直接参与优化。Overlong Reward Shaping 处理超过长度限制的噪声样本：如果模型写满上下文还没完成，不能简单把截断文本当正常失败样本，否则 reward 噪声很大；需要对过长响应做分段惩罚、遮罩或单独 shaping。

最小复现路径是：用 Qwen2.5-7B/32B base、AIME/MATH 类可验证题、每题 8-16 个 rollout；先跑普通 GRPO 作为 baseline；再依次加入 dynamic sampling、clip-higher、token-level loss、overlong shaping；记录有效 prompt 比例、entropy、平均长度、AIME pass@1 和训练崩溃次数。DAPO 的价值正是在这种 ablation 中体现。

#### VAPO：面向长 CoT 的价值模型与优势估计

VAPO 研究的是 value-model-based RL。长 CoT 下，GAE 容易把最终稀疏 reward 衰减到前面 token，长短响应的 advantage 尺度也不一致。报告的 ablation 很有信息量：移除 decoupled GAE 会导致奖励信号指数衰减并大幅掉点；Length-Adaptive GAE 根据序列长度调节 GAE 参数，让短响应和长响应都能收到合适 credit；token-level policy gradient 给长响应更合理权重；positive-example LM loss 用正样本的语言模型损失稳住策略；group sampling 用较少 prompt、更多 repetition 提高组内比较质量。报告中还给出类似 `epsilon_low=0.2`、`epsilon_high=0.28`、positive LM loss weight 0.1、512 prompts 每个 16 samples 这类可复现实验级参数。

VAPO 的小型复现不必先训练大 value model，可以先做一个简化实验：同一批数学题分别用 GRPO 和带 value baseline 的 PPO/VAPO 风格训练，比较长答案任务上 advantage 方差、reward 延迟衰减和最终正确率。重点不是追求 SOTA，而是观察长序列 RL 里 credit assignment 如何影响训练稳定性。

#### UI-TARS：通过轨迹与偏好学习 GUI 操作

UI-TARS 的输入是截图/界面状态、历史动作和任务目标，输出是点击、输入、滚动等 GUI action。它面临的数据问题是高质量 action trace 极少。公开资料里可学习的方法是用大量虚拟机探索真实软件任务，从构造指令出发生成轨迹，再做规则过滤、VLM 评分和人工复核。Reflection tuning 把错误恢复也纳入训练：标注员指出轨迹中哪一步错了，并给出纠正动作或恢复步骤，再用 DPO 这类偏好优化让模型偏向能修正错误的策略。

#### UI-TARS-2：多轮 RL、混合环境与数据飞轮

UI-TARS-2 的 motivation 是 GUI-only agent 不足以完成真实任务：很多工作流还需要文件系统、终端、下载文件、读取本地数据。它引入 hybrid GUI environment，把 GUI、file system、terminal 放在统一 sandbox 里，并用大规模 rollout 平台支持多轮 RL。数据 flywheel 的逻辑是：模型生成新轨迹，高质量轨迹进入 SFT，低质量但有学习价值的数据进入 continual pre-training 或后续探索；每轮模型变强后，再产生更难、更长的轨迹。

这类系统的 reward 需要多层：最终任务是否完成，界面状态是否达到目标，文件是否生成，终端命令是否成功，动作是否无效或越界，回合数是否过多，是否违反安全边界。最小复现可以用 MiniWoB / BrowserGym / OSWorld 子集：定义统一 action schema；每个任务提供 reset、observe、step、success check；用 200 条人工或强模型轨迹做 SFT；再用多轮 rollout + success reward 做 RL；额外收集失败轨迹训练 reflection。

#### Seed Prover 1.5：形式化证明环境中的 Agentic RL

形式化数学的环境是 theorem prover，而不是浏览器。动作是选择 tactic、生成 lemma、调用搜索器；reward 是 proof 是否通过、证明长度、搜索步数和中间 lemma 是否复用。它给 agent RL 的启发是：只要环境能验证，就可以把复杂任务变成可训练 episode。Seed1.8 则把 reasoning、multimodal、tools 和 generalized agent 能力放进同一模型卡，说明后训练目标正在从“题库正确率”扩展到“多环境任务执行”。

#### DeepSeek

> **资料入口**：[DeepSeekMath][^deepseek_math]、[DeepSeek-R1][^deepseek_r1]、[DeepSeek-V3.2][^deepseek_v3_2]

DeepSeek 的公开材料是理解 GRPO / RLVR 的主线之一。DeepSeekMath 先给出 critic-free 的组内相对优势，R1 再证明纯规则 reward 可以诱发长思考，V3.2 则把可验证任务推进到 agentic task synthesis。

#### DeepSeekMath：GRPO 的最小可复现方案

DeepSeekMath-RL 从 DeepSeekMath-Instruct 7B 出发，使用约 144K 个与 GSM8K、MATH 相关的 CoT 问题做 RL。每个问题采样一组输出，用 reward model / 规则正确性打分，再用组内均值和标准差归一化形成 advantage。这样省掉 PPO 的 critic/value model，显存和训练复杂度更低。报告中的典型设置包括 policy learning rate 1e-6、KL coefficient 0.04、每题 64 个输出、max length 1024、batch size 1024，并在每轮 exploration 后做一次 policy update。

GRPO 的直觉是：模型不需要知道“这个答案绝对值是多少”，只需要知道“同一题的这一组答案里哪个更好”。数学题天然适合，因为同题多采样会产生正确、错误、格式错、半对等候选。最小复现可以用 7B 数学 SFT 模型、MATH 子集、每题 8-16 个 rollout、答案 parser 打分；先用组内 reward 标准化，再加 KL 到参考模型，观察 GSM8K/MATH 提升和通用能力损失。DeepSeekMath 的经验是，SFT 后已经很强的模型，仍能通过 RL 获得 out-of-domain reasoning 提升。

#### DeepSeek-R1-Zero：从基础模型直接进行规则 RL

R1-Zero 的 motivation 是验证长 CoT 是否必须来自人工 SFT。它从 base model 直接做 RL，奖励主要是 accuracy reward 和 format reward：数学/代码题看最终答案或执行结果，格式 reward 保证模型按约定输出。训练后出现反思、回溯、自我验证、延长思考等行为，说明一部分 reasoning pattern 可以由 outcome reward pressure 诱发。

R1-Zero 的局限也很重要：可读性差、语言混杂、输出格式不稳定。这说明“纯 RL 能探索能力上限”，但不等于产品 recipe。复现实验时，应把 R1-Zero 当成研究实验：从 base model 出发，只用可验证数学/代码题，避免开放问答；评估除了正确率，还要看格式失败、重复、语言混杂和平均长度。

#### DeepSeek-R1：冷启动、推理 RL 与拒绝采样

R1 正式版回到更工程化的四段式。第一步用少量高质量 cold-start 数据修正格式、可读性和基本长思考结构；第二步做 reasoning-oriented RL，继续在数学、代码、逻辑等可验证任务上强化；第三步用训练后的模型做 rejection sampling，生成更多 SFT 数据，同时混入写作、事实问答、角色扮演等通用数据，避免模型只会解题；第四步做最终 RL，同时优化 helpfulness、harmlessness 和 reasoning。

这条流水线的核心教训是“能力和体验分开塑造，再合并”。可验证 RL 推高数学/代码，但会带来啰嗦、风格漂移和通用聊天退化；rejection sampling 和 final RL 是能力回填。最小复现可以用 1K cold-start 长 CoT、20K 可验证题 RL、再从 RL 模型采样并筛 10K 通用 SFT 数据，最后混合安全/偏好数据做一轮 DPO 或 GRPO。评估要同时看 math/code、普通指令、拒答、平均长度和格式稳定性。

#### DeepSeek-V3.2：从答案验证器到 Agentic Verifier

V3.2 的方向是让模型在工具环境中合成和完成 agentic tasks。这里的训练样本不是单个答案，而是带工具调用、环境观察、失败恢复和最终交付的 episode。reward 不只看最终文本，还看工具调用是否成功、是否找到证据、代码是否通过测试、任务是否在环境里真的完成。它和 MiniMax M2.1、UI-TARS-2、LongCat 属于同一趋势：RLVR 正从 math verifier 扩展到 software / browser / GUI / tool verifier。

#### 智谱 Z.ai / GLM

> **资料入口**：[GLM-4.5][^glm_4_5]、[GLM-5][^glm_5]

GLM 的主线是 ARC：Agentic、Reasoning、Coding。GLM-4.5 先证明这三类能力可以放在一个 MoE 模型里共同优化；GLM-5 则把后训练 recipe 讲得更清楚：multi-task SFT 之后，按 Reasoning RL、Agentic RL、General RL 顺序推进，并用异步 RL 基础设施提高长程交互训练效率。

#### GLM-4.5：混合推理与专家模型迭代

GLM-4.5 支持 thinking 和 direct response 两种模式。这个设计的 motivation 和 Qwen3 类似：复杂题需要长思考，普通助手场景不能每次都展开冗长 CoT。后训练阶段通过 expert model iteration 和 RL 同时提升 agentic、reasoning、coding。expert iteration 可以理解为“先让专项强模型产生或筛选高质量数据，再回流训练统一模型”；RL 则在数学、代码、工具和 agent benchmark 上继续放大可验证能力。

#### GLM-5：Reasoning RL、Agentic RL 与 General RL

GLM-5 的公开报告明确写出 progressive alignment：先 multi-task SFT，引入 interleaved thinking modes；再做 reasoning RL；再做 agentic RL；最后 general RL 做人类风格对齐。Reasoning RL 主要处理数学、逻辑、代码这类 outcome verifier 高置信任务，先把长链推理和自检能力推上去。Agentic RL 把模型接入多轮工具环境、文件系统、代码库和软件工程任务，让模型学习“观察 -> 行动 -> 环境反馈 -> 修正”。General RL 最后回填普通聊天、简洁性、安全、指令跟随和风格，减少前两段带来的啰嗦和能力偏置。

#### 异步 Agent RL：解耦生成与训练

GLM-5 引入新的 asynchronous RL infrastructure，并使用 slime 的可定制 rollout interface。长程 agent rollout 的耗时差异很大：有的任务只需一轮，有的要跑测试、调用工具、等待环境。同步 PPO/GRPO 会让训练端等最慢 episode。GLM-5 的思路是把 rollout generation、环境交互、verifier 分支和训练解耦，让不同任务的经验可以持续进入训练队列。slime 的 server-based rollout execution 允许为不同任务写 multi-turn loop、tool invocation、environment feedback handling、verifier-guided branching，而不改底层训练栈。

#### On-Policy Cross-Stage Distillation：缓解阶段间遗忘

分阶段 RL 的风险是：Reasoning RL 学到的能力在 Agentic RL 或 General RL 中被冲掉。GLM-5 用 on-policy cross-stage distillation 在后续阶段保留前阶段强能力：当前策略在线生成数据，前一阶段能力作为蒸馏目标或筛选信号参与训练。最小复现可以做三段小实验：MATH/代码 GRPO 得到 reasoning 模型；SWE-bench Lite 或工具任务 RL 得到 agentic 模型；最后混入通用指令做 DPO/GRPO，同时用第一阶段模型在数学题上的输出做 distillation，观察数学是否回退。

#### 腾讯混元 Hunyuan

> **资料入口**：[Hunyuan-T1][^hunyuan_t1]、[Hunyuan-A13B][^hunyuan_a13b]、[Hunyuan-A13B-Instruct Model Card][^hunyuan_a13b_instruct]

混元公开资料可以拆成 T1 的 reasoning RL 和 A13B 的 fast/slow thinking instruct model。A13B 技术报告细节披露不如 MiniMax/MiMo 那么完整，但 T1 页面给出了若干训练稳定性线索，足够作为 reasoning RL 系统设计参考。

#### Hunyuan-T1：以 RL 为主的后训练算力分配

T1 明确说 post-training 阶段 96.7% 的算力投入在 reinforcement learning，目标是提升纯推理和人类偏好对齐。任务来源覆盖数学、逻辑推理、科学、代码等 world science and reasoning problems，并结合 ground-truth feedback。这里的 reward 主体仍是可验证信号：数学答案、逻辑题规则、代码执行、科学题标准答案或 judge。公开资料没有给出完整 reward 公式，但说明它不是单一聊天偏好 RL，而是 reasoning-heavy RL。

#### 课程学习、上下文扩展与 Token 效率

T1 的训练计划使用课程学习：逐步提高数据难度，并逐步扩展上下文长度，让模型既提升推理能力，也学会更有效地使用 token。这个设计和 MiniMax-M1 / Qwen3 的长思考训练一致：不能一开始就把最大长度、最难题和复杂 reward 全部打开，否则训练早期会被噪声和超长输出拖垮。复现时可以把数学/代码题按难度分三段，先短 CoT + 中等题，再长 CoT + 难题，最后加入长度/效率评价。

#### 数据回放、策略重置与统一奖励

T1 公开材料提到参考 data replay 和 periodic policy resetting，使长期训练稳定性提升超过 50%。这说明混元在处理长时间 RL 的策略漂移：data replay 防止模型忘掉早期能力；policy reset 则在策略偏离过大或出现退化时，把模型拉回更稳定的 checkpoint / reference。偏好对齐阶段采用 self-reward + reward model：早期 T1-preview 作为 self-reward 评估器，对输出进行综合评分，再叠加 reward model，引导模型自我改进。可复现时可以维护一个 replay buffer，混入旧阶段高质量样本；每 N 步评估 entropy、格式错、平均长度和通用 eval，一旦退化就回滚或重置 reference。

#### Hunyuan-A13B：Fast 与 Slow Thinking

A13B-Instruct 模型卡展示了 slow-thinking 默认开启，也可以通过 `enable_thinking=False` 关闭 CoT。这说明后训练数据要同时包含两类响应：带 `<think>` 的慢思考轨迹，以及直接回答的快思考轨迹。否则模型要么每次都慢想，要么复杂问题推理不足。最小复现可以在同一批 prompt 上构造两份标签：复杂题保留思考过程，简单问答只给短答；SFT 后在 RL / DPO 中加入“是否需要 thinking”的偏好，评估复杂题正确率和简单题平均长度。

#### 百度 ERNIE

> **资料入口**：[ERNIE 4.5 Technical Report][^ernie_4_5]、[ERNIE 5.0][^ernie_5_0]

ERNIE 4.5 的后训练披露很有结构：LLM post-training 是 SFT + RL，RL 阶段用 Progressive RL 和 Unified Preference Optimization；VLM post-training 则是三段 SFT + 一段 reasoning RL。ERNIE 的价值在于把多任务、多 reward、多模态后训练的“兼容性”问题讲得比较清楚。

#### SFT：覆盖任务域后进入 RL

ERNIE 4.5 的 SFT 覆盖通用指令、逻辑、数学、代码、专业任务、安全和多模态理解。这里的关键不是数据量，而是每个能力域要有可评测目标。对 LLM 来说，SFT 让模型学会基本回答格式和任务能力；对 VLM 来说，三段 SFT 分别强化视觉感知、复杂视觉推理，以及 thinking / non-thinking 数据混合。多模态模型如果直接进 RL，很容易把视觉识别错误和推理错误混在一起，所以先用 SFT 把 perception 打稳。

#### Unified Rewarding System：统一多类奖励信号

ERNIE 4.5 图示里列出 rule-based reward、RLLM、sandbox、RDRM、checklist-aware verifier、GRM、DRM 等组件。它要解决的是 reward 来源异构：数学题可能是规则答案；代码题是 sandbox 执行；开放问答是 generative reward model；安全/清单任务是 checklist verifier；偏好任务是 discriminative reward model。若不做 domain normalization，不同 reward 尺度会互相压制。ERNIE 的统一奖励系统可以抽象成三步：先按任务域选择 verifier / RM；再把 reward 归一化到可比较尺度；最后按训练阶段控制不同任务权重。

#### Progressive RL：从 Logic RL 到 General RL

ERNIE 4.5 把 LLM RL 分成 Stage 1 Logic RL、Stage 2 Reasoning RL、Stage 3 General RL。Logic RL 用更干净、规则性更强的任务稳定推理格式；Reasoning RL 扩展到数学、代码、复杂推理；General RL 回填普通指令、人类偏好和安全。这个顺序和 GLM-5 / Qwen3 的“先能力、再泛化”一致。最小复现可以按这个顺序组织数据：先 2K 逻辑/符号题，后 10K 数学/代码题，最后 10K 通用偏好题；每阶段单独评估前一阶段能力是否被覆盖。

#### UPO：多任务 RL 的尺度与稳定性

Unified Preference Optimization 的动机是混合 reasoning tasks 和 non-reasoning tasks 时，reward-format、domain normalization、informative prompt filtering 都会影响训练。数学/代码 0-1 reward、偏好分数、安全分数不能直接相加。UPO 的复现思路是：为每个任务域维护 reward normalization；过滤没有信息量的 prompt；对不同 reward source 做分域权重；训练时记录各域 reward 均值和方差，避免某一类任务主导更新。

#### ERNIE 5.0：统一多模态后训练

ERNIE 5.0 继续面向文本、图像、视频、语音统一模型。这里最大的难点是 reward 可比性和模态平衡：图像理解 reward、视频时序 reward、文本偏好 reward、语音任务 reward 的错误来源完全不同。复现时不要把多模态题简单拼成文本 JSON，而要为每种模态准备感知 eval、推理 eval 和偏好 eval，再统一做阶段式 SFT/RL。

#### 阶跃星辰 StepFun

> **资料入口**：[Step3][^step3]、[STEP3-VL-10B][^step3_vl_10b]、[Step-DeepResearch][^step_deepresearch]

StepFun 的公开资料覆盖多模态 reasoning 和 deep research agent。STEP3-VL-10B 披露了一个紧凑 10B VLM 怎么靠 scaled post-training 追近更大模型；Step-DeepResearch 则属于研究型 agent 训练。

#### STEP3-VL-10B：全参数预训练后的视觉推理 RL

这份报告的 motivation 是小模型也能在多模态复杂推理上接近大模型，但需要把视觉语言协同和后训练一起设计。模型先在 1.2T multimodal tokens 上做统一、完全解冻的预训练，把 perception encoder 和 Qwen3-8B decoder 对齐；post-training 阶段再做超过 1K iterations 的 reinforcement learning。这里的关键是：VLM 的 RL 不是只训文本答案，而是让视觉证据、文本推理和答案生成一起被 reward 约束。

#### RLVR 与 RLHF：分开处理视觉验证与开放偏好

视觉数学、OCR 后计算、图表读数、选择题、几何/空间题可以做 RLVR：答案能规则验证，或能由程序/标准答案检查。开放式图片描述、复杂审美、视觉安全和解释质量更适合 RLHF / judge reward。复现时可以把数据分成两桶：第一桶用 MathVista、ChartQA、OCR-VQA、几何题做 exact / numeric verifier；第二桶用多模态 judge 给 helpfulness、faithfulness、detail、safety 打分。不要把两类 reward 直接相加，先分域归一化。

#### PaCoRe：并行视觉假设与答案协调

Parallel Coordinated Reasoning 的目标是扩展 test-time compute。多模态任务里，错误常来自“看错图”而不是“不会推理”。PaCoRe 让模型探索多个视觉假设或推理路径，再合成更可靠的答案。训练上对应两个信号：候选路径要多样且有证据，最终整合要正确且不幻觉。小型复现可以做 self-consistency 的多模态版：同一张图采样多条证据链，用 verifier / judge 选出正确链，再 SFT 模型学习“列出候选视觉证据 -> 交叉检查 -> 给答案”。

#### Step-DeepResearch：训练完整研究过程

Deep research agent 的任务包括搜索、浏览、证据抽取、冲突比较、引用和长文组织。SFT 阶段应使用高质量 research trajectories，教模型如何规划 query、如何读来源、如何记录证据；RL 阶段的 reward 则要拆成 answer correctness、citation existence、evidence support、source coverage、redundant search penalty、final report structure。复现时可以用 300 个多来源问题、一个搜索 API、一个浏览器提取器和 citation checker，先训练轨迹格式，再对最终答案和引用证据做 episode-level reward。

#### 美团 LongCat

> **资料入口**：[LongCat-Flash-Thinking-2601][^longcat_flash]

LongCat-Flash-Thinking-2601 是一份很像“agent RL 工程系统设计说明”的报告。它的核心不是某个 reward 公式，而是环境扩展、强化学习扩展、噪声鲁棒训练和 heavy thinking。

#### 环境扩展：从领域定义生成可解工具环境

LongCat 的 motivation 是真实 agent 场景太多，手工适配 prompt、工具链和环境接口成本极高。它构建覆盖 20+ 领域、上万情境的环境生成系统：输入领域定义，自动合成 60+ 工具、数据库 schema、工具调用接口和验证逻辑。覆盖的场景包括文件管理、数据分析、电商零售、电信服务等。这个设计把“训练数据”变成“可交互环境图谱”。

环境生成最难的是一致性。一个复杂环境可能有几十个数据库和工具参数依赖，若随机生成任务，容易出现“看似可解、实际无解”。LongCat 使用可解路径优先：先随机采样一条长工具调用链作为黄金工具链；围绕这条链构造任务和数据库状态；再用 BFS 受控扩展环境子图，保证新工具的前序依赖已存在；根据环境复杂度和剩余工具动态加入新黄金链；如果工具数不足 20，就从全局工具库补一条中等规模可用链。这个方法的可复现重点是：先保证至少一条成功路径存在，再扩环境，而不是先造环境再祈祷任务可解。

#### 冷启动数据：真实轨迹与双路合成

LongCat 在 RL 前把预训练/微调目标重新定义为“给 RL 提供冷启动策略”。有真实数据的领域，如数学和编码，通过质量控制和可执行验证筛选高质量轨迹；缺真实数据的领域，如搜索和工具使用，使用文本驱动合成和环境锚定合成。文本驱动合成从任务描述出发生成轨迹；环境锚定合成从已有工具链和数据库状态出发生成任务，保证任务能被环境验证。复现时可以先做一个 5 个工具的小环境：订单查询、退款、库存、用户信息、日志；先采样黄金链，再让模型生成任务和轨迹。

#### DORA：全异步流式 RL

Agent rollout 耗时差异巨大，同步训练会浪费大量 GPU。DORA 支持多版本模型并行探索，不同版本产生的经验随产随收进样本队列；训练器无需等待所有任务完成。调度上拆成轻量 Rollout Manager 和多个 Rollout Controller，后者各自管理虚拟 rollout 组，通过数据并行处理环境交互。环境部署通过扩展 PyTorch RPC，把环境实例化到 CPU 空闲机器上。

为适配 5600 亿参数 MoE，DORA 还做 Prefill-Decode 解耦和 KV-cache 交换。PD 解耦把长上下文 prefill 和 decode 放到不同设备组，避免多轮交互中 prefill 阻塞 decode；KV-cache 以 chunk 级聚合、异步传输、计算重叠和 CPU 驻留方式动态交换，减少重复计算。资源分配上做双层平衡：整体按环境难度调 rollout 配额，批内保证任务域多样性。报告称这种系统达到传统同步训练 2-4 倍效率，并支持千步以上稳定训练。

#### 噪声鲁棒训练：在训练中注入真实扰动

LongCat 主动注入工具超时、工具报错、返回缺字段、数据库不一致、指令歧义、需求变更等扰动，让模型学习恢复。reward 不应只看最终成功，也要奖励错误检测、重新计划、换工具、向用户澄清。最小复现可以在工具环境里随机让 10%-30% 调用失败或返回部分字段，训练模型根据错误码重试、改参数或走备用链；评估 clean success rate 和 noisy success rate 的差距。

#### Heavy Thinking：同时扩展推理宽度与深度

LongCat 的重思考模式不是只把单条 CoT 拉长，而是先生成多条推理/行动路径，再用总结模型分析、筛选和整合。它适合复杂 agent 任务，因为单一路径一旦早期工具选择错，后面会越走越偏。小型复现可以让模型对同一工具任务采样 3-5 条计划，用 verifier / judge 选最佳计划或合并计划，再执行。训练时把“候选路径 -> 比较 -> 最终计划”的轨迹回流 SFT / RL。

#### 蚂蚁 Ling / Ring

> **资料入口**：[Ling-1T][^ling_1t]、[Ring-1T][^ring_1t]

#### 公开资料的边界

Ling / Ring 的公开材料更偏模型发布和推理效率，没有像 DeepSeek-R1、Qwen3 或 MiniMax M2.1 那样展开完整后训练流水线。因此这一节更适合作为“产业信号”阅读，而不是直接复刻训练配方。

#### Deep Thinking 与推理效率

能明确学习到的是两点：第一，trillion-scale MoE 模型也会把 deep thinking / insight 类能力作为 post-training 目标，而不是只做聊天对齐；第二，长序列推理和高效推理部署要一起考虑。

#### 最小复现：Fast 与 Slow Thinking 数据分桶

复现时主要借鉴 fast/slow thinking 数据设计：为复杂数学、代码、分析题保留长思考轨迹，为普通问答保留短答；用偏好数据惩罚无意义长思考，并用 eval 同时看正确率和 token cost。

#### 华为 Pangu

> **资料入口**：[Pangu Ultra][^pangu_ultra]、[Pangu Pro MoE][^pangu_pro_moe]、[盘古开源新闻][^pangu_news]

#### 公开资料的边界

Pangu 公开信息的重点在昇腾原生训练、MoE 稀疏效率和开源模型体系，后训练细节没有像 R1/Qwen/MiniMax 那样展开。

#### 部署硬件对后训练方案的约束

可学习点是硬件和训练 recipe 的耦合：如果模型要部署在 Ascend NPU 上，post-training 不能只看算法，还要考虑 MoE 路由、长上下文显存、推理吞吐和快慢思考的成本。

#### 最小复现：将成本指标纳入评测

复现层面可以把它作为“工程约束型后训练”案例：同一个 reasoning 模型同时评估正确率、激活专家数、平均输出长度、吞吐和部署成本。

#### 01.AI Yi

> **资料入口**：[Yi-Lightning][^yi_lightning]

#### 产品级 RLHF 路线

Yi-Lightning 披露的是传统产品级 LLM 后训练线：pre-training 之后做 SFT 和 RLHF，并强调 multi-stage training、synthetic data construction、reward modeling，以及 RAISE 安全框架贯穿 pre-training、post-training 和 serving。它不像 agent 报告那样提供工具环境 recipe，但适合学习“聊天模型如何被人类偏好拉齐”。

#### 最小复现：SFT、RM 与 PPO/DPO

可复现时可以做三段：用高质量中文/英文指令做 SFT；为同一 prompt 采样多个回答，人工或 judge 排序训练 reward model；再做 PPO/DPO，并单独评估 Chinese、Math、Coding、Hard Prompts 和 safety。

#### 评测范围：静态基准与真实场景

Yi-Lightning 还提醒一点：静态 benchmark 和真实人类偏好会有差距，后训练指标不能只看题库。

#### InternLM / 上海 AI Lab

> **资料入口**：[InternLM2][^internlm2]

#### RLHF 工程化：数据治理与算法实现

InternLM2 是开源社区理解传统 RLHF 工程化的重要参考。它的重点不是长 CoT RLVR，而是数据治理、SFT、reward modeling 和 online RLHF。

#### COOL：条件化偏好优化

COOL（Conditional Online RLHF）的动机是偏好优化会让模型在不同任务域上漂移：某些用户喜欢简洁，某些任务需要详细，安全场景又要保守。条件化训练让模型根据任务条件、偏好条件或数据域调整优化目标，而不是把所有偏好压成一个平均人。

#### 最小复现：为偏好数据增加领域、风格与安全条件

最小复现可以这样做：为每条偏好数据标注 domain / style / safety 条件；训练 reward model 时把条件作为输入；在线 RLHF 时按条件采样 prompt 和 reward；评估时分域看 helpfulness、harmlessness、verbosity 和中文能力。

**学习点**：即使没有可执行 verifier，偏好 RL 也要做数据分桶和条件控制，否则模型容易向单一风格塌缩。

#### 百川 Baichuan 与 360 智脑

> **资料入口**：[Baichuan 2][^baichuan2]、[360Zhinao][^zhinao]

#### Baichuan2：中文开源模型的 SFT、RM 与 PPO

Baichuan2 是国内较早公开 SFT -> RM -> PPO 经典对齐流程的报告。SFT 阶段先让 base model 学会对话和指令；RM 阶段收集偏好比较，训练 reward model；PPO 阶段用 RM 分数优化策略并加 KL 约束。它适合放在课程里作为 InstructGPT 路线的中文/开源对照：在没有大规模可验证 RLVR 时，SFT/RM/PPO 仍是完整后训练闭环。

#### 360Zhinao：奖励模型与数据治理

360Zhinao 的公开材料强调数据质量和数据治理。RM 不只是 PPO 的奖励器，也可以做 judge、过滤器和数据重标工具：对候选回答打分，筛掉低质量样本，发现重复模式，再回流 SFT。

#### 最小复现：拒绝采样 SFT 与 DPO

可复现实验可以把同一批中文指令采样 4 个回答，用 judge/RM 打分，保留 top-1 做 rejection sampling SFT，再用 bottom/top pair 做 DPO。这个流程虽然不如 agent RL 酷，但非常接近大量真实产品模型的日常后训练。

#### 昆仑万维 Skywork 与 小米 MiMo

> **资料入口**：[Skywork-OR1][^skywork_or1]、[MiMo][^mimo]、[MiMo-VL-Miloco][^mimo_vl]

Skywork-OR1 和 MiMo 都适合学习“小模型 / 蒸馏模型继续做 RL”的问题。它们不像 frontier lab 只堆规模，而是关注 entropy collapse、数据难度、reward 稀疏和训练稳定性。

#### Skywork-OR1：持续 RL 与熵坍缩风险

Skywork-OR1 建在 DeepSeek-R1-Distill 系列之上。蒸馏模型已经会长 CoT，但继续 RL 时很容易过早收敛到少数表达和解题模式，entropy 下降后探索消失。报告的主线就是通过训练 pipeline 和 ablation 找出影响 entropy dynamics 的因素，并证明缓解 premature entropy collapse 对测试性能关键。公开结果显示 32B 平均准确率从 57.8% 到 72.8%，7B 从 43.6% 到 57.5%，并开源权重、代码和数据。

复现重点是监控 entropy，而不是只看 reward。用 R1-Distill-7B 做数学/代码 RL；每步记录 token entropy、response length、pass@1、重复 n-gram、格式错误率；尝试调整采样温度、KL、clip、数据难度和动态采样。如果 reward 上升但 entropy 快速塌缩，后期往往泛化差。

#### MiMo：7B 推理模型的可验证数据

MiMo-7B 在 post-training 阶段构造 130K verifiable mathematics and programming problems 做 RL。数学题用答案 verifier；编程题用测试执行。它还提出 test-difficulty-driven code reward，缓解代码 reward 稀疏：不是所有测试通过/失败都等价，能通过更难测试或更多隐藏测试应有更细粒度奖励。Strategic data resampling 则用来稳定训练，把算力集中在既有挑战又可学习的样本上。

MiMo 的最小复现非常清楚：准备 80K 数学题和 50K 编程题，或更小的 5K/2K 版本；数学用 Math-Verify / parser 判答案；代码题为每题准备 easy/medium/hard tests，reward 按通过测试难度加权；每轮 RL 后统计哪些题全对、全错、半对，对全对/全错下采样，对半对题提高采样。这个方法对 7B 尤其重要，因为小模型训练预算有限，不能把 rollout 浪费在没有学习信号的样本上。

#### MiMo-VL-Miloco：小模型的多模态推理扩展

MiMo-VL 延续了“小模型 + 高质量可验证数据 + 稳定 RL”的路线，但对象变成视觉语言。可学习点和 STEP3-VL 类似：视觉题要区分 perception 错误和 reasoning 错误；reward 需要同时覆盖答案正确性、视觉证据引用和输出格式。复现时可把数学图表/OCR/几何题作为 RLVR 数据，再混入开放图片描述偏好数据做回填。

#### 快手、商汤、讯飞

> **资料入口**：[Kwai Keye-VL][^keye_vl]、[SenseNova U1][^sensenova_u1]、[Spark X1][^spark_x1]

#### 产业动态资料

这三家公司公开了多模态后训练（快手）、原生理解生成（商汤）和深度推理（讯飞）的动态，但缺乏完整的训练 recipe 报告。课程里可以把它们作为“方向覆盖”保留，避免误写成可复现的详细训练配方。

**阅读重点**：看它们分别押注哪些能力面，例如 VLM、多模态生成、深度推理、中文场景和端到端产品体验；不要从发布材料反推出未披露的 SFT/RL 细节。

---

### 国外团队的公开案例

国外团队的公开材料更多覆盖通用偏好、安全对齐、端侧部署和企业工具。阅读方法仍然相同：先确定目标能力和成功标准，再看数据、奖励、训练方法与评测怎样连接。

#### OpenAI

> **资料入口**：[InstructGPT][^instructgpt]、[GPT-4][^gpt4]、[o1][^o1]、[o3/o4-mini][^o3_o4_mini]、[o3 Operator][^o3_operator]、[GPT-4.5][^gpt4_5]、[GPT-5][^gpt5]、[GPT-5.1][^gpt5_1]、[GPT-5.4 Thinking][^gpt5_4]、[GPT-5.5][^gpt5_5]、[GPT-5.5 Instant][^gpt5_5_instant]、[GPT-5-Codex][^gpt5_codex]、[GPT-5.1-Codex-Max][^gpt5_1_codex_max]、[GPT-5.2-Codex][^gpt5_2_codex]

OpenAI 的公开资料横跨三代后训练：InstructGPT 的经典 RLHF、o-series 的 reasoning / deliberation、安全系统卡中的 deliberative alignment，以及 Codex / Operator 类 agent 模型。闭源系统卡不会披露完整 recipe，但方法边界很清楚。

#### InstructGPT：RLHF 的最小闭环

InstructGPT 的流程可以直接复现成教学实验。第一步是 demonstration SFT：标注员写高质量回答，让 base model 先学会按指令完成任务。第二步是 reward modeling：对同一 prompt 采样多个回答，标注员排序，训练 reward model 预测人类偏好。第三步是 PPO：用 reward model 给 policy 输出打分，并用 KL penalty 约束 policy 不要偏离 SFT 模型太远。这里的关键是三类数据不同：SFT 数据是“好答案”，RM 数据是“偏好比较”，PPO 数据是“prompt + on-policy samples”。

最小复现可以用 5K 指令做 SFT；为 1K prompt 各采样 4 个回答，做 pairwise preference 训练 RM；最后用 PPO / DPO / IPO 任选一种偏好优化。评估不能只看 reward model 分数，还要人工或 LLM judge 看 helpfulness、truthfulness、toxicity、verbosity 和 instruction following，因为 RM 很容易被策略钻空子。

#### GPT-4 到 o-series：推理后训练扩展动作空间

GPT-4 技术报告只高层描述 post-training 和 safety；o1/o3/o4-mini 系统卡更明确：模型通过强化学习学会在回答前进行更长 deliberation，并在需要时使用工具。这里的变化是 action 不再只是“下一个 token”，还包括何时写代码、何时浏览、何时调用图像/文件工具、何时停止、何时拒答。reward 也从人类偏好扩展到最终答案、工具结果、策略合规、安全边界和用户体验。

这种能力的可复现抽象是：选择一个有工具的任务族，例如代码执行数学题；模型输出思考和工具调用；环境返回 execution result；reward 同时检查最终答案、工具格式、调用次数和安全策略。先 SFT 少量成功工具轨迹，再做 RL。这样能复现 o-series 的方法形态，而不是复刻闭源配方。

#### Deliberative Alignment：将安全约束纳入推理

OpenAI 系统卡中反复出现的一个方向是让模型在困难安全问题上先推理策略，再决定回答方式。早期安全对齐容易变成拒答模板；deliberative alignment 更像把 policy spec、边界案例和安全评测做成训练任务：模型要识别请求类型，判断能否安全完成，必要时转换成安全替代方案。复现时可以构造安全 prompt、策略条款和正确处理示例，用 SFT 教模型引用策略，再用 preference/RL 奖励“安全完成”而不是盲拒。

#### Operator 与 Codex：真实环境中的 Agent 后训练

Operator 和 Codex 类模型把后训练扩展到 browser / software engineering episode。coding agent 的环境要包含仓库状态、测试命令、patch verifier、lint、用户指令层级和失败恢复；browser agent 的环境要包含页面状态、可点击元素、任务成功检查和安全沙箱。GPT-5-Codex 的系统卡明确说它通过真实软件工程任务上的 RL 训练，学习贴近人类代码风格和 PR 偏好、严格遵循指令、反复运行测试直到通过；GPT-5.1-Codex-Max 继续把训练对象扩展到跨多个上下文窗口的长程 agentic coding，通过 compaction 在百万 token 级任务中保持连贯；GPT-5.2-Codex 则强调 SWE-Bench Pro、Terminal-Bench 2.0、Windows 原生环境、长上下文理解和可靠工具调用。最小复现可以用 SWE-bench Lite：checkout 仓库，给 issue，模型编辑文件，运行 tests，reward 为测试通过 + patch 合理性；也可以用 MiniWoB / BrowserGym：模型观察 DOM/截图，点击输入，reward 为任务完成和动作合法。

#### Anthropic

> **资料入口**：[Constitutional AI][^constitutional_ai]、[Anthropic CAI overview][^anthropic_cai]、[Claude 4 System Card][^claude4]、[Claude Sonnet 4.5][^claude_sonnet_4_5]、[Claude Opus 4.5][^claude_opus_4_5]、[Claude Opus 4.6][^claude_opus_4_6]

Anthropic 最值得学的是 Constitutional AI 和系统化安全评测。它的公开材料不提供 Claude 4 的完整训练 recipe，但 Constitutional AI 是可以复现的方法。

#### Constitutional AI：用原则与 AI Feedback 扩展安全数据

传统 RLHF 需要大量人工比较。Constitutional AI 先定义一组原则，即 constitution；supervised phase 中，模型先生成回答，再根据 constitution 自我批改并重写，形成更安全的 SFT 样本。preference phase 中，AI 根据 constitution 比较两个回答，生成偏好数据，训练 preference model，最后用 RL 优化 policy。这就是 RLAIF：把人工从逐条偏好判断中部分移到原则设计和质量审计上。

最小复现路径是：写 20-50 条安全/诚实/隐私/无害原则；为有风险 prompt 采样回答；让强模型按原则指出问题并重写；用重写数据做 SFT；再让强模型对两条回答按原则排序，训练 DPO 或 reward model。评估时要单独看 over-refusal，因为安全原则过强会让模型拒绝正常请求。

#### Claude 系统卡：联合设计后训练与评测

Claude 4 系列系统卡关注 reward hacking、sabotage、sycophancy、alignment faking、hidden objectives、jailbreak、extended thinking 下的策略遵循等。这里的学习点不是某个 RL 公式，而是安全后训练必须有 adversarial evaluation。模型在训练 reward 下表现好，不代表不会在长上下文、工具调用、角色扮演或高压提示里偏离。

#### Extended Thinking 的安全风险

当模型有更长思考和工具能力时，安全训练不再只是“输出拒答”。模型可能在思考中制定规避策略，或在工具环境中完成不该完成的步骤。因此安全 reward 要覆盖策略遵循、工具限制、信息泄露、隐私、欺骗和拒答质量。复现时可以把工具任务和安全规则结合：例如要求模型处理文件，但禁止读取无关敏感文件；reward 同时检查任务成功和越权行为。

#### Google DeepMind

> **资料入口**：[Gemini 1.5][^gemini_1_5]、[Gemini 2.5][^gemini_2_5]、[Gemini 2.5 Deep Think][^gemini_2_5_deep_think]、[Gemini 2.5 Computer Use][^gemini_2_5_computer_use]、[Gemini 3.1 Pro][^gemini_3_1_pro]、[Gemma 3][^gemma_3]

Google DeepMind 的公开资料披露粒度不如开放论文，但方向非常明确：多模态、长上下文、工具、reasoning 和安全评测一起做。Gemini / Gemma 这条线适合学习“统一多模态模型怎么设计后训练任务”。

#### Gemini 1.5 与 2.5：长上下文中的证据对齐

长上下文模型不是只把 context window 扩大。后训练要让模型在几十万 token、图片、视频、文档中定位证据，避免把不相关片段混进答案。任务构造应包含 needle-in-haystack、长文档 QA、多文档冲突、视频事件定位和跨模态引用。reward 不能只看最终答案，还要看证据位置是否正确、引用是否支持结论、是否忽略了无关干扰。

#### Deep Think：并行探索与答案汇总

Gemini 2.5 Deep Think 展示的是 test-time compute scaling 的另一种形态：生成多条候选思路，比较并整合。它和 LongCat heavy thinking、self-consistency 属于同族。训练上需要 reward 区分“有用的多样性”和“无意义发散”：候选路径要覆盖不同假设，整合答案要比单一路径更正确。小型复现可以对数学/视觉题采样 5 条推理，verifier 选正确路径，再训练模型输出“候选分析 + 最终合并答案”。

#### Computer Use：GUI 环境中的安全动作学习

Gemini Computer Use 面向屏幕状态和动作序列：观察网页/桌面，输出点击、输入、滚动等动作，再根据环境反馈继续。reward 至少包含任务完成、动作有效、回合数、是否误点敏感控件、是否泄露信息、是否违反用户授权。复现时可用 BrowserGym / OSWorld：每个 task 提供 reset、observe、step、success check 和安全检查；先 SFT 成功轨迹，再 RL 学长期策略和错误恢复。

#### Gemma：小模型蒸馏与定向后训练

Gemma 系列提供更接近开源社区的路径：用强教师蒸馏和高质量数据过滤提升小模型，再针对数学、指令、多语言、安全做专项 post-training。它的意义是：不一定复刻 frontier 级 RL 系统，小模型也能通过数据质量、教师选择、能力分桶和 targeted preference optimization 得到实用能力。

#### Meta Llama

> **资料入口**：[The Llama 3 Herd of Models][^llama3_herd]

Llama 3 Herd 是开放模型里最适合作为产品级 chat model 后训练流水线的参考之一。它的价值不是某个单独算法，而是完整的数据治理、SFT、reward model、rejection sampling、preference optimization、安全对齐和评测闭环。

#### 按能力域配比 SFT 数据

Llama 的 SFT 不应理解成“堆 instruction JSON”。数据需要覆盖通用问答、代码、数学、多语言、工具、安全和长上下文，并为每个域设置独立 eval。工程上通常要做去重、质量过滤、格式统一、拒答边界清洗、过长/过短样本控制。复现时先用小规模能力分桶，而不是一个混杂数据池。

#### Rejection Sampling：将筛选结果回流 SFT

对同一个 prompt 采样多个回答，用 reward model、规则 verifier 或 judge 选出最好的，加入下一轮 SFT。它介于 SFT 和 RL 之间：不直接更新策略梯度，但能把模型自己的高质量输出蒸馏回模型。对数学/代码可以用 verifier；对聊天/安全可以用 RM / judge。最小复现时，每个 prompt 采样 4-8 个回答，保留 top-1，同时保留 top/bottom pair 做 DPO。

#### 贯穿全流程的偏好优化与安全训练

Llama 的安全不是最后加拒答样本，而是在数据过滤、SFT、安全 RM、红队、发布阈值中持续出现。Preference optimization 进一步拉开好坏答案概率差，但也可能牺牲多样性和诚实性，所以需要 truthfulness、安全、拒答、helpfulness 同时评估。它是一条适合开源团队复现的基础线：即使没有 agent 环境，也能把 SFT、RS、DPO/RLHF、安全评测做完整。

#### Microsoft Phi

> **资料入口**：[Phi-4][^phi_4]、[Phi-4-reasoning][^phi_4_reasoning]

Phi-4-reasoning 的重点是小模型 reasoning。它不是靠巨大参数硬推，而是靠高质量合成数据、teachable prompts 和一段 outcome-based RL，把 14B 级模型推到较强推理水平。

#### 数据先行：构造可学习的提示

小模型容量有限，后训练数据必须可学、干净、有清晰监督。数学、科学、代码和逻辑题要按难度组织，过难题全错没有 RL 信号，过易题全对浪费 rollout。复现时应先构造 5K-20K 可教题集，保证 SFT 后模型能达到一定成功率，再进入 RL。

#### SFT 建立行为，短程 RL 修正确率与长度

Phi-4-reasoning 的思路可以抽象为：先用高质量 synthetic reasoning traces 做 SFT，让模型会展开推理；再用 outcome reward 对可验证题做 RL，强化正确路径并控制无效长思考。小模型尤其要监控 average response length，因为一点 RL 就可能让输出变长但正确率不升。最小复现实验是用 Phi/Qwen 7B-14B、MATH/GPQA 子集、强教师生成 CoT，SFT 后每题采样 8 个，用答案 verifier 做 GRPO，并加入长度统计。

#### NVIDIA Nemotron

> **资料入口**：[Nemotron-4 340B][^nemotron_4]、[Llama-Nemotron][^llama_nemotron]、[Llama Nemotron Ultra][^nemotron_ultra]、[Nemotron Agent Blog][^nemotron_agents]、[Nemotron-H][^nemotron_h]、[Nemotron 3][^nemotron_3]

NVIDIA Nemotron 的特点是把后训练做成可复用资产：模型、数据、reward、部署栈一起发布或产品化。Nemotron-4 340B 配套 synthetic data、preference data、reward model；Llama Nemotron 把 reasoning、tool use、RAG、instruction following 和企业部署放在一起。

#### Nemotron-4：对齐数据与模型资产

它不是只发布 instruct 权重，而是把 synthetic data、preference data、reward model 和评测组件作为训练资产。方法是用强模型和规则生成候选数据，通过质量过滤和偏好标注训练 RM，再做 RLHF / preference optimization。可复现时可以把 RM 当作独立产物维护：它既用于 PPO/DPO，也用于 rejection sampling、数据过滤和自动评测。

#### Llama Nemotron：剪枝蒸馏后的推理与 Agent 训练

NVIDIA 的公开博客描述了三段式：从 Llama 底座出发，先剪枝提升效率，再蒸馏改善能力，然后用 post-training 数据和 RL 强化 reasoning、instruction following、function calling 和 chat。Llama-Nemotron-Post-Training Dataset 覆盖 math、coding、general reasoning、instruction following；OpenCodeReasoning 等数据强化代码推理。Ultra 还支持 reasoning on/off，说明它也要处理长思考成本和普通交互体验。

#### RLVR 与企业 Agent

NVIDIA 强调蒸馏能搬运老师能力，但要进一步提升，需要 curriculum-driven RLVR。企业 agent 场景的 reward 来自工具调用正确性、RAG 忠实性、function calling schema、代码执行和用户意图对齐。公开资料还提到使用 REINFORCE 和 heuristic based verifiers 做 instruction following / function calling 增强，再用 HelpSteer2 等偏好数据做 RLHF。复现时可以做两桶 RL：一桶 math/code verifier，一桶 function calling verifier；最后混入 chat/RAG 偏好回填。

#### 将部署约束纳入后训练目标

Nemotron 的输出不只是权重，还有 NIM、NeMo Gym、企业推理栈。post-training 评估应包含 latency、throughput、function calling 成功率、RAG 引用忠实性和推理开销。企业模型如果只看 AIME，会忽略上线最常见的失败点。

#### Mistral

> **资料入口**：[Magistral][^magistral]

Magistral 的公开摘要很值得放在 reasoning RL 章节：Mistral 明确说它使用自家的 scalable RL pipeline，不依赖已有实现或从其他模型蒸馏来的 RL traces，而是 ground-up 做 pure RL。

#### Pure RL：减少教师轨迹对能力上限的约束

蒸馏能快速得到长 CoT 格式，但也会继承老师的风格和错误。Magistral 的方向是从自家 checkpoint 出发，用 RL 自己探索推理能力。公开资料还提到 Magistral Medium 基于 Mistral Medium 3 只用 RL 训练 reasoning，Magistral Small 则包含来自 Medium 的 cold-start 数据。这对应两种复现路径：大模型直接 RL 探索；小模型先蒸馏大模型 cold-start，再 RL。

#### 推理语言约束

多语言模型做 reasoning RL 时可能出现推理语言混杂。Magistral 提到一种 simple method to force reasoning language，说明后训练不仅要管答案正确，还要管推理语言和输出风格。复现时可以在 prompt / template 中显式指定 reasoning language，并用 format reward 检查推理段语言、答案段语言和结构。

#### 文本 RL 的跨能力影响

Magistral 的一个有意思结论是，只在 text data 上做 RL 仍能保持或提升 multimodal understanding、instruction following 和 function calling。这说明 RL 不一定必然破坏通用能力，但需要持续评估。最小复现时，做完数学/文本 RL 后，要同时跑函数调用、普通指令、多语言和视觉文本化任务，确认没有被 reasoning 数据拉偏。

#### Apple

> **资料入口**：[Apple Foundation Models 2024][^apple_fm]、[Apple Foundation Models 2025][^apple_fm_2025]

Apple 的 foundation model 报告把后训练和部署约束绑得很紧：一个约 3B 的端侧模型要在 Apple silicon 上运行，server model 要在 Private Cloud Compute 上服务，模型还要支持多语言、多模态和 tool calls。

#### 异步 SFT 与 RL 的端侧约束

2025 报告明确说模型在大规模多语言、多模态、合成与授权数据上训练后，会通过 supervised fine-tuning 和 reinforcement learning 在新异步平台上进一步优化。这里的后训练目标不是“benchmark 最大化”这么简单，还要考虑 guided generation、constrained tool calling、LoRA adapter fine-tuning、隐私和低延迟。端侧模型尤其不能靠无限长 CoT 提升体验，reward 应包含正确率、简洁性、延迟、内存和能耗。

#### 多源奖励：偏好、规则与工具约束

Apple 报告强调 Responsible AI 和 locale-specific evaluation。消费级模型要处理不同地区语言、安全规范和产品体验。可复现时可以构造三类 reward：文本/图文偏好 RM 用于回答质量；数学/STEM 规则 verifier 用于可验证推理；tool calling schema checker 用于约束工具调用。最后按设备类型设置不同目标：端侧模型重简洁和隐私，云端模型可承担更复杂推理。

#### 后训练与系统接口的联合设计

Apple 的 Foundation Models framework 暴露 guided generation、constrained tool calling 和 LoRA。它说明产品级后训练不能脱离 API：如果接口支持 constrained decoding，训练时就应加入 JSON/schema/tool 数据；如果允许 LoRA 个性化，基础模型后训练要保持可适配性。最小复现可以在小模型上训练工具调用 JSON schema，并用 constrained decoder 评估 schema success rate。

#### xAI Grok

> **资料入口**：[Grok-1][^grok_1]、[Grok 4][^grok_4]、[Grok 4.1][^grok_4_1]、[Grok 4.1 Model Card][^grok_4_1_card]

#### 公开资料的边界

xAI 的公开材料更侧重模型卡和发布说明，没有给出完整 post-training recipe，但反复强调 RL scaling、truthfulness、personality、style 和 emotional intelligence。这类目标不属于传统“答题正确率”，但对 C 端产品非常关键。

#### 将产品人格拆解为奖励目标

可学习点是把产品人格拆成可评测 reward，而不是只写 system prompt。personality reward 可以评估幽默感、直接性、不过度迎合；truthfulness reward 检查事实和不确定性表达；emotional intelligence reward 检查是否识别用户情绪、是否给出合适语气；safety reward 检查风险边界。

#### 最小复现：多目标偏好优化

复现时可以用偏好数据训练多个 reward head 或多个 judge rubric，再做 multi-objective DPO/RL。风险是人格 reward 容易推动 sycophancy，所以必须单独评估“是否为了讨好用户而承认错误事实”。

#### IBM Granite

> **资料入口**：[Granite 3.3][^granite_3_3]、[Granite 4.0][^granite_4_0]、[Granite 4.1][^granite_4_1]

#### 企业小模型：RAG、工具、安全与低成本推理

IBM Granite 的后训练重点是企业小模型：RAG、工具调用、安全、低成本推理和可切换 thinking。Granite 3.3/4.x 的公开材料显示，GRPO/TPO 这类 reasoning 后训练已经进入企业级小模型，而不是只存在于前沿大模型。

#### 最小复现：面向企业任务的小模型后训练

复现时可以把 Granite 路线抽象成“企业任务小模型后训练”：先用企业问答、RAG 引用、工具 schema、数学/逻辑题做 SFT；再用 GRPO 强化可验证推理；用偏好数据优化 RAG 忠实性和拒答；最后用 model merging 或 adapter merging 融合领域专家。

**评估指标**要包括 RAG citation faithfulness、function calling success、refusal correctness、latency、cost 和 thinking on/off 的差异。

#### Salesforce xLAM / SFR-RL

> **资料入口**：[Salesforce xLAM][^xlam]、[Salesforce SFR-RL][^sfr_rl]

Salesforce 的 xLAM / SFR-RL 是工具调用和 agentic RL 基础设施的代表。xLAM 关注 action model：给定用户意图和 API 文档，模型要选择正确工具、填对参数、按顺序调用。SFR-RL 则回答大规模 agent rollout 怎么训得动。

#### xLAM：结构化工具调用奖励

API agent 的错误通常不是“回答不好听”，而是工具选错、参数缺失、调用顺序错误、schema 不合法、结果没用上。训练数据应包含 API schema、用户请求、工具调用序列、环境返回和最终回答。reward 可以拆成 schema validity、tool selection accuracy、argument exact match、execution success、final answer groundedness。最小复现可以用 50 个 mock API，自动生成用户请求和正确调用链，用函数执行结果做 verifier。

#### SFR-RL：流水线同步 RL

Agentic rollouts 长且不稳定，纯同步会等慢任务，纯异步又牺牲 on-policy 性。SFR-RL 的方案是 pipelined synchronous：rollout phase 和 training phase 交替，每个阶段都使用整个 GPU 集群。rollout 时卸载训练模型，把 policy 加载到弹性推理引擎上并发生成；training 时释放推理引擎，重新加载训练模型做 on-policy update。跨 batch 做流水线管理，保证 GPU 不空转，同时维持数据组成和 on-policy 保证。

#### 故障恢复与本地优先的工具执行

长程 agent rollout 中，一个 inference engine crash 或工具卡死都可能拖住 batch。SFR-RL 的 inference gateway 自动检测失败、重建 engine actor、恢复权重、重排 in-flight work。它还强调 scalable local-first tool execution 和 Expert Parallelism 支持。复现时即使用小集群，也应实现 timeout、retry、任务重排和失败标记，否则 agent RL 数据会被系统错误污染。

#### Amazon Nova

> **资料入口**：[Amazon Nova][^nova]、[Nova Family Technical Report][^nova_report]、[Nova Premier][^nova_premier]、[Nova Forge][^nova_forge]

#### Nova Forge：后训练平台化

Amazon Nova 的技术报告偏模型卡，内部后训练 recipe 没有展开到论文级细节。Nova Forge 更直接展示了“后训练平台化”的方向。传统 fine-tuning 是企业给一批数据，模型做 SFT；Nova Forge 允许企业从 pre-training、mid-training 或 post-training checkpoint 进入，把私有数据和 Nova-curated 数据混合，再用 RL 阶段对齐企业任务。

#### Remote Reward Functions：接入企业验证器

最重要的方法是远程 reward functions。企业的 reward 往往不在训练集里，而在内部系统：代码是否通过私有 CI，机器人动作是否通过仿真，客服回答是否符合业务流程，工具调用是否在真实 API 中成功。Nova Forge 把这些系统通过 API 接进 RL，对模型 rollout 结果打分。这个模式可以叫 Reward as a Service。

#### 最小复现：用本地验证器模拟企业奖励

最小复现可以用一个私有 API verifier 模拟企业 reward：模型生成 SQL、代码补丁或客服动作；本地服务执行并返回 pass/fail 和 rubric 分；训练器只通过 HTTP 调 reward。这样能复现 Nova Forge 的关键抽象：模型厂提供 checkpoint、训练基础设施和 reward 接口，企业提供私有环境与验证器。

#### Cohere Command A

> **资料入口**：[Cohere Research][^cohere_research]、[Command A][^command_a]

#### Decentralized Pipeline：分开训练不同能力

Command A 的报告展示了企业模型如何避免“所有能力串行训练互相覆盖”。它的后训练不是一条线跑到底，而是 decentralized pipeline。先训练一个核心模型掌握基本 instruction following，然后为 code、safety、RAG、math、multilingual、long-context 等能力分别训练 expert track。每个 expert track 可以用自己的数据配方、偏好目标和评测标准。

#### Expert Soup：合并专项能力

接下来用参数合并把专家能力汇总。报告里有 SFT Expert Models、SFT Soup Model、RL Expert Models、RL Soup Model 和 Polished Model。六类 expert track 可以分别服务 long-context、safety、instruction、RAG & agents、multilingual、code/reasoning 等能力；RL expert 使用 pairwise comparisons 或 verifiable rewards。合并后再做 polishing：对 RL Soup model 先做 best-of-N supervised training，再在 offline preference 和 online RL 之间 ping-pong，直到人类偏好表现平台化。

#### 最小复现：Expert Track、Soup 与最终优化

Command A 的可复现模板是：训练一个基础 instruct 模型；复制成 3-6 个 expert，各自用不同数据和 DPO/RLVR 优化；用 model soup / task arithmetic 合并；最后用一小段通用偏好数据 polish。这个思路尤其适合企业模型，因为安全、RAG、代码和销售文案的优化目标经常互相冲突，分专家训练能降低 loss 互相打架。

### Databricks, AI21, Cursor, LG, NAVER, AI2 Tulu 3

> **资料入口**：[DBRX Instruct][^dbrx]、[Jamba 1.5a][^jamba_1_5a]、[Jamba 1.5a Whitepaper][^jamba_whitepaper]、[Cursor Composer 2][^cursor_composer_2]、[EXAONE 4.0][^exaone_4_0]、[K-EXAONE][^k_exaone]、[HyperCLOVA X][^hyperclova_x]、[HyperCLOVA X THINK][^hyperclova_x_think]、[Tulu 3][^tulu_3]、[Tulu 3 Blog][^tulu_3_blog]、[RL Post-Training Survey][^rl_survey]

这组资料不必每个都单独开大节，但它们补齐了几类重要实践。

#### Databricks DBRX：企业开源指令模型

DBRX Instruct 代表传统 enterprise instruct：强调数据质量、指令跟随、代码、RAG 和部署效率。它的可借鉴点是把后训练和企业场景 eval 绑定，而不是只看聊天榜单。

#### AI21 Jamba 1.5a：后训练后的安全对齐

Jamba 1.5a 的主题是企业 code of conduct 写入模型。方法上更像二次对齐：在已有 instruct model 之后，用合成安全偏好数据和企业原则修正行为。复现时可以给模型一组公司政策，生成违反/遵守政策的回答对，用 DPO 或 RLAIF 调整。

#### Cursor Composer 2：真实仓库中的代码 Agent 训练

Cursor 的 coding agent 目标不是写单文件函数，而是在代码库里理解上下文、编辑、多文件修改、运行测试和恢复失败。训练环境应包含 repo state、issue、编辑器动作、终端、测试和 patch verifier。它和 GPT-5-Codex、Qwen3-Coder、MiniMax SWE Scaling 是同一类。

#### LG EXAONE 与 NAVER HyperCLOVA X THINK：本土化与 Thinking 模式

韩国大厂的资料提醒一个容易忽略的问题：后训练不仅是英文 benchmark，还要处理本土语言、文化、安全规范和业务风格。thinking / non-thinking 模式在本土语言中也要单独评估，不能假设英文 CoT recipe 直接迁移。

#### AI2 Tulu 3：开放后训练方案

Tulu 3 完整开源数据、代码和训练 recipe，主题就是 multi-stage post-training：SFT、preference learning、RLVR。它的价值在于透明：可以看到 prompt 数据、偏好数据、verifiable rewards、训练参数和评测如何组织。复现现代后训练时，Tulu 3 应作为开源 baseline，再把 MiniMax/Qwen/DeepSeek/Seed 的特定技巧加进去。

---

## 从案例回到训练闭环

看完二十多个团队的实践，再回头看六个阶段，会发现几条清晰的规律：

1. **奖励从“人喜欢哪个回答”变成“任务过程是否真的完成”。** 早期 RLHF 看 preference pair；R1、Qwen、Seed、Mistral 看答案可验证；MiniMax、Kimi、LongCat、Tongyi 看工具轨迹、环境状态和最终交付。
2. **数据从静态样本变成可生成、可验证、可回放的环境。** GitHub PR、Docker、Playwright、浏览器、数据库、工具图谱、搜索网页都变成后训练数据的一部分。
3. **后训练顺序越来越分段。** 常见顺序是 cold-start SFT、reasoning RL、agentic RL、general preference / safety 回填；顺序错了就容易出现长 CoT 过度、聊天退化、工具滥用或安全漂移。
4. **训练系统正在成为竞争力。** 异步 rollout、PD 解耦、KV-cache 交换、环境调度、失败恢复、reward service、LLM-as-judge 和可执行 verifier，都是“后训练实践”的一部分，而不是外围工程。

如果把上面的公司实践抽象成一个可复现的小型项目，可以按下面的顺序做：

1. **先选一个可验证任务族。** 数学题最简单，代码题次之，网页/GUI/研究 agent 最难。开放聊天任务的 reward 较难定义，不适合作为第一版 RL 实验。
2. **把任务封装成环境。** 数学环境需要 answer parser 和 verifier；代码环境需要仓库 checkout、依赖安装、测试命令和 patch 检查；网页环境需要浏览器、状态记录和证据抽取；GUI 环境需要截图、动作空间和可复位 sandbox。
3. **先做 SFT 冷启动。** 收集或生成成功轨迹，让模型学会输出格式、工具协议、思考结构和停止条件。没有冷启动就直接 RL，容易先在格式和工具调用上乱掉。
4. **再做采样和筛选。** 对每个 prompt 采样多个输出，用 verifier / judge / reward model 选出正确、简洁、过程合理的轨迹。这个阶段就是 Qwen、DeepSeek、Kimi、MiniMax 都在反复做的 rejection sampling / self-improvement。
5. **最后做 RL。** 简单题可以用 GRPO / DAPO 风格的组内相对优势；需要 value model 的任务可以参考 PPO / VAPO；长程 agent 要额外处理异步 rollout、失败恢复、工具噪声和 token-level credit assignment。
6. **训练后做能力回填。** 用通用指令、安全、短回答、风格和非思考模式数据再对齐一次，防止模型被 reasoning RL 训练得又长又慢。

一个很小但完整的练习可以是：用 5K 数学题或 1K 代码修复题做 SFT，采样 8 个候选，用规则 verifier 过滤，再用 GRPO 训练一轮，最后评估正确率、平均输出长度、格式错误率和通用聊天退化。这样的闭环比只停留在算法名称层面更能揭示后训练的真实难点。

## 本节小结

工业后训练从目标能力开始，把数据、环境、SFT、RL、评测和失败样本回流接成多轮闭环。算法只是其中一个环节；环境是否可重复、奖励是否可信、评测能否发现真实失败，共同决定下一轮训练能否继续提高能力。

[18.3 训练稳定性](./modern-industrial-practice) 将沿着这条闭环检查训练信号：哪些曲线说明模型正在学习，哪些变化意味着奖励、优化或系统已经出现问题。

---

## 参考资料

### 国内公司与实验室

#### MiniMax

[^minimax_m2_1]: [MiniMax M2.1: Post-Training Experience and Insights for Agent Models](https://www.minimax.io/news/post-training-experience-and-insights-for-agent-models)

[^minimax_m1]: [MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention](https://arxiv.org/abs/2506.13585)

[^minimax_webexplorer]: [WebExplorer: Explore and Evolve for Training Long-Horizon Web Agents](https://arxiv.org/abs/2509.06501)

#### 阿里 Qwen / 通义

[^qwen2_5]: [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115)

[^qwen2_5_math]: [Qwen2.5-Math Technical Report: Toward Mathematical Expert Model via Self-Improvement](https://arxiv.org/abs/2409.12122)

[^qwq_32b]: [QwQ-32B: Embracing the Power of Reinforcement Learning](https://qwenlm.github.io/blog/qwq-32b/)

[^qwen3]: [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)

[^qwen3_coder]: [Qwen3-Coder: Agentic Coding in the World](https://qwenlm.github.io/blog/qwen3-coder/)

[^qwen3_coder_next]: [Qwen3-Coder-Next Technical Report](https://arxiv.org/abs/2603.00729)

[^tongyi_dr]: [Tongyi DeepResearch Technical Report](https://arxiv.org/abs/2510.24701)

#### Moonshot Kimi

[^kimi_k1_5]: [Kimi k1.5: Scaling Reinforcement Learning with LLMs](https://arxiv.org/abs/2501.12599)

[^kimi_k2]: [Kimi K2: Open Agentic Intelligence](https://arxiv.org/abs/2507.20534)

[^kimi_researcher]: [Kimi-Researcher: End-to-End RL Training for Emerging Agentic Capabilities](https://moonshotai.github.io/Kimi-Researcher/)

#### 字节 Seed / Doubao

[^seed1_5_thinking]: [Seed1.5-Thinking: Advancing Superb Reasoning Models with Reinforcement Learning](https://arxiv.org/abs/2504.13914)

[^vapo]: [VAPO: Efficient and Reliable Reinforcement Learning for Advanced Reasoning Tasks](https://arxiv.org/abs/2504.05118)

[^dapo]: [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://seed.bytedance.com/en/public_papers/dapo-an-open-source-llm-reinforcement-learning-system-at-scale)

[^dapo_github]: [DAPO GitHub Repository](https://github.com/BytedTsinghua-SIA/DAPO)

[^seed1_5_vl]: [Seed1.5-VL Technical Report](https://arxiv.org/abs/2505.07062)

[^ui_tars]: [UI-TARS: Pioneering Automated GUI Interaction with Native Agents](https://arxiv.org/abs/2501.12326)

[^ui_tars_github]: [UI-TARS GitHub Repository](https://github.com/bytedance/ui-tars)

[^ui_tars_2]: [UI-TARS-2 Technical Report: Advancing GUI Agent with Multi-Turn Reinforcement Learning](https://huggingface.co/papers/2509.02544)

[^seed_prover]: [Seed Prover 1.5: Advanced Mathematical Reasoning through a Novel Agentic Architecture](https://seed.bytedance.com/en/blog/seed-prover-1-5-advanced-mathematical-reasoning-through-a-novel-agentic-architecture)

[^seed1_8]: [Official Release of Seed1.8: A Generalized Agentic Model](https://seed.bytedance.com/en/blog/official-release-of-seed1-8-a-generalized-agentic-model)

#### DeepSeek

[^deepseek_math]: [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)

[^deepseek_r1]: [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)

[^deepseek_v3_2]: [DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models](https://arxiv.org/abs/2512.02556)

#### 智谱 Z.ai / GLM

[^glm_4_5]: [GLM-4.5: Agentic, Reasoning, and Coding Foundation Models](https://arxiv.org/abs/2508.06471)

[^glm_5]: [GLM-5: from Vibe Coding to Agentic Engineering](https://arxiv.org/html/2602.15763v1)

#### 腾讯混元 Hunyuan

[^hunyuan_t1]: [Hunyuan-T1](https://tencent.github.io/llm.hunyuan.T1/README_EN.html)

[^hunyuan_a13b_instruct]: [Hunyuan-A13B-Instruct Model Card](https://huggingface.co/tencent/Hunyuan-A13B-Instruct)

[^hunyuan_a13b]: [Hunyuan-A13B Technical Report](https://github.com/Tencent-Hunyuan/Hunyuan-A13B/blob/main/report/Hunyuan_A13B_Technical_Report.pdf)

#### 百度 ERNIE

[^ernie_4_5_family]: [ERNIE 4.5 Model Family](https://ernie.baidu.com/blog/posts/ernie4.5/)

[^ernie_4_5]: [ERNIE 4.5 Technical Report](https://ernie.baidu.com/blog/publication/ERNIE_Technical_Report.pdf)

[^ernie_5_0]: [ERNIE 5.0 Technical Report](https://arxiv.org/abs/2602.04705)

#### 阶跃星辰 StepFun

[^step3]: [Step3: Cost-Effective Multimodal Intelligence](https://stepfun.ai/research/en/step3)

[^step3_vl_10b]: [STEP3-VL-10B Technical Report](https://huggingface.co/papers/2601.09668)

[^step_deepresearch]: [Step-DeepResearch Technical Report](https://arxiv.org/abs/2512.20491)

#### 美团 LongCat

[^longcat_flash]: [LongCat-Flash-Thinking-2601 技术报告](https://tech.meituan.com/2026/02/02/LongCat-Flash-Thinking-2601-techreport.html)

#### 蚂蚁 Ling / Ring

[^ling_1t]: [Ling-1T Model](https://ant-ling.medium.com/deep-insight-efficient-inference-introducing-the-trillion-parameter-ling-1t-model-77d6170e5e8e)

[^ring_1t]: [Ring-1T](https://ant-ling.medium.com/ring-1t-release-the-flow-state-of-insight-born-of-epiphany-c20e8e32817c)

#### 华为 Pangu

[^pangu_ultra]: [Pangu Ultra](https://github.com/pangu-tech/pangu-ultra)

[^pangu_pro_moe]: [Pangu Pro MoE: Mixture of Grouped Experts for Efficient Sparsity](https://arxiv.org/abs/2505.21411)

[^pangu_news]: [华为宣布开源盘古 7B 稠密和 72B 混合专家模型](https://www.huawei.com/cn/news/2025/7/pangu-opensource)

#### 01.AI Yi

[^yi_lightning]: [Yi-Lightning Technical Report](https://arxiv.org/abs/2412.01253)

#### InternLM / 上海 AI Lab

[^internlm2]: [InternLM2 Technical Report](https://arxiv.org/abs/2403.17297)

#### 百川 Baichuan 与 360 智脑

[^baichuan2]: [Baichuan 2: Open Large-scale Language Models](https://arxiv.org/abs/2309.10305)

[^zhinao]: [360Zhinao Technical Report](https://arxiv.org/abs/2405.13386)

#### 昆仑万维 Skywork 与小米 MiMo

[^skywork_or1]: [Skywork Open Reasoner 1 Technical Report](https://huggingface.co/papers/2505.22312)

[^skywork_or1_github]: [Skywork-OR1 GitHub Repository](https://github.com/SkyworkAI/Skywork-OR1)

[^mimo]: [MiMo: Unlocking the Reasoning Potential of Language Model -- From Pretraining to Posttraining](https://arxiv.org/abs/2505.07608)

[^mimo_github]: [Xiaomi MiMo GitHub Repository](https://github.com/XiaomiMiMo/MiMo)

[^mimo_vl]: [Xiaomi MiMo-VL-Miloco Technical Report](https://arxiv.org/abs/2512.17436)

#### 快手、商汤、讯飞

[^keye_vl]: [Kwai Keye-VL Technical Report](https://arxiv.org/abs/2507.01949)

[^sensenova_u1]: [SenseNova U1](https://www.sensetime.com/en/news/51170629)

[^spark_x1]: [Spark X1 deep reasoning model](https://news.cgtn.com/news/2025-01-15/China-releases-Spark-X1-deep-reasoning-model-that-packs-a-punch-1AbIq8PzzEI/index.html)

### 国外公司与实验室

#### OpenAI

[^instructgpt]: [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)

[^gpt4]: [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774)

[^o1]: [OpenAI o1 System Card](https://openai.com/index/openai-o1-system-card/)

[^o3_o4_mini]: [OpenAI o3 and o4-mini System Card](https://openai.com/index/o3-o4-mini-system-card/)

[^o3_operator]: [Addendum to o3 and o4-mini system card: OpenAI o3 Operator](https://openai.com/index/o3-o4-mini-system-card-addendum-operator-o3/)

[^gpt4_5]: [OpenAI GPT-4.5 System Card](https://openai.com/index/gpt-4-5-system-card/)

[^gpt5]: [OpenAI GPT-5 System Card](https://openai.com/index/gpt-5-system-card/)

[^gpt5_1]: [Addendum to GPT-5 system card: GPT-5.1](https://openai.com/index/gpt-5-system-card-addendum-gpt-5-1/)

[^gpt5_4]: [OpenAI GPT-5.4 Thinking System Card](https://openai.com/index/gpt-5-4-thinking-system-card/)

[^gpt5_5]: [OpenAI GPT-5.5 System Card](https://openai.com/index/gpt-5-5-system-card/)

[^gpt5_5_instant]: [OpenAI GPT-5.5 Instant System Card](https://openai.com/index/gpt-5-5-instant-system-card/)

[^gpt5_codex]: [Addendum to GPT-5 system card: GPT-5-Codex](https://openai.com/index/gpt-5-system-card-addendum-gpt-5-codex/)

[^gpt5_1_codex_max]: [GPT-5.1-Codex-Max System Card](https://openai.com/index/gpt-5-1-codex-max-system-card/)

[^gpt5_2_codex]: [Introducing GPT-5.2-Codex](https://openai.com/index/introducing-gpt-5-2-codex/)

#### Anthropic

[^constitutional_ai]: [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)

[^anthropic_cai]: [Anthropic Constitutional AI overview](https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback)

[^claude4]: [System Card: Claude Opus 4 & Claude Sonnet 4](https://www.anthropic.com/claude-4-system-card)

[^claude_sonnet_4_5]: [Claude Sonnet 4.5 System Card](https://www.anthropic.com/claude-sonnet-4-5-system-card)

[^claude_opus_4_5]: [Claude Opus 4.5 System Card](https://www.anthropic.com/claude-opus-4-5-system-card)

[^claude_opus_4_6]: [Claude Opus 4.6 System Card](https://www-cdn.anthropic.com/0dd865075ad3132672ee0ab40b05a53f14cf5288.pdf)

#### Google DeepMind

[^gemini_1_5]: [Gemini 1.5 Technical Report](https://arxiv.org/abs/2403.05530)

[^gemini_2_5]: [Gemini 2.5 Technical Report](https://arxiv.org/abs/2507.06261)

[^gemini_2_5_deep_think]: [Gemini 2.5 Deep Think](https://blog.google/products/gemini/gemini-2-5-deep-think)

[^gemini_2_5_computer_use]: [Gemini 2.5 Computer Use Model](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-computer-use-model/)

[^gemini_3_1_pro]: [Gemini 3.1 Pro Model Card](https://deepmind.google/models/model-cards/gemini-3-1-pro/)

[^gemma_3]: [Gemma 3 Technical Report](https://arxiv.org/abs/2503.19786)

#### Meta Llama

[^llama3_herd]: [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)

#### Microsoft Phi

[^phi_4]: [Phi-4 Technical Report](https://arxiv.org/abs/2412.08905)

[^phi_4_reasoning]: [Phi-4-reasoning Technical Report](https://arxiv.org/abs/2504.21318)

#### NVIDIA Nemotron

[^nemotron_4]: [Nemotron-4 340B Technical Report](https://arxiv.org/abs/2406.11704)

[^llama_nemotron]: [Llama-Nemotron: Efficient Reasoning Models](https://arxiv.org/abs/2505.00949)

[^nemotron_ultra]: [NVIDIA Llama Nemotron Ultra Open Model](https://developer.nvidia.com/blog/nvidia-llama-nemotron-ultra-open-model-delivers-groundbreaking-reasoning-accuracy/)

[^nemotron_agents]: [Build Enterprise AI Agents with NVIDIA Llama Nemotron Reasoning Models](https://developer.nvidia.com/blog/build-enterprise-ai-agents-with-advanced-open-nvidia-llama-nemotron-reasoning-models/)

[^nemotron_h]: [Nemotron-H Reasoning Model Family](https://developer.nvidia.com/blog/nemotron-h-reasoning-enabling-throughput-gains-with-no-compromises/)

[^nemotron_3]: [Inside NVIDIA Nemotron 3](https://developer.nvidia.com/blog/inside-nvidia-nemotron-3-techniques-tools-and-data-that-make-it-efficient-and-accurate/)

#### Mistral

[^magistral]: [Magistral](https://arxiv.org/abs/2506.10910)

#### Apple

[^apple_fm]: [Apple Intelligence Foundation Language Models](https://machinelearning.apple.com/research/apple-intelligence-foundation-language-models)

[^apple_fm_2025]: [Apple Intelligence Foundation Language Models Tech Report 2025](https://machinelearning.apple.com/research/apple-foundation-models-tech-report-2025)

#### xAI Grok

[^grok_1]: [xAI Grok-1 Model Card](https://x.ai/news/grok/model-card)

[^grok_4]: [xAI Grok 4](https://x.ai/news/grok-4)

[^grok_4_1]: [xAI Grok 4.1](https://x.ai/news/grok-4-1/)

[^grok_4_1_card]: [xAI Grok 4.1 Model Card](https://data.x.ai/2025-11-17-grok-4-1-model-card.pdf)

#### IBM Granite

[^granite_3_3]: [IBM Granite 3.3](https://www.ibm.com/new/announcements/ibm-granite-3-3-speech-recognition-refined-reasoning-rag-loras)

[^granite_4_0]: [IBM Granite 4.0](https://www.ibm.com/new/announcements/ibm-granite-4-0-hyper-efficient-high-performance-hybrid-models)

[^granite_4_1]: [IBM Granite 4.1 Build Notes](https://huggingface.co/blog/ibm-granite/granite-4-1)

#### Salesforce xLAM / SFR-RL

[^xlam]: [Salesforce xLAM](https://www.salesforce.com/blog/large-action-model-ai-agent/)

[^sfr_rl]: [Salesforce SFR-RL](https://www.salesforce.com/blog/efficient-rl-training-agentic-era/)

#### Amazon Nova

[^nova]: [Amazon Nova](https://aws.amazon.com/nova/)

[^nova_report]: [The Amazon Nova Family of Models: Technical Report and Model Card](https://www.isi.edu/results/publications/31887/the-amazon-nova-family-of-models-technical-report-and-model-card/)

[^nova_premier]: [Amazon Nova Premier: Technical report and model card](https://www.amazon.science/publications/amazon-nova-premier-technical-report-and-model-card)

[^nova_forge]: [Amazon Nova Forge](https://aws.amazon.com/nova/forge/)

#### Cohere Command A

[^cohere_research]: [Cohere Research](https://cohere.com/research)

[^command_a]: [Command A: An Enterprise-Ready Large Language Model](https://cohere.com/research/papers/command-a-technical-report.pdf)

#### Databricks

[^dbrx]: [DBRX Instruct](https://huggingface.co/databricks/dbrx-instruct)

#### AI21

[^jamba_1_5a]: [Jamba 1.5a: Enhancing AI Safety Through Post-Post-Training Alignment](https://www.ai21.com/research/jamba-1-5a/)

[^jamba_whitepaper]: [Jamba 1.5a Whitepaper](https://lp.ai21.com/hubfs/resources/Jamba-1-5a-Whitepaper.pdf)

#### Cursor

[^cursor_composer_2]: [Cursor Composer 2 Technical Report](https://cursor.com/blog/composer-2-technical-report)

#### LG EXAONE

[^exaone_4_0]: [EXAONE 4.0 Technical Report](https://www.lgresearch.ai/data/cdn/upload/EXAONE_4_0.pdf)

[^k_exaone]: [K-EXAONE Technical Report](https://www.lgresearch.ai/data/cdn/upload/K-EXAONE_Technical_Report.pdf)

#### NAVER HyperCLOVA X

[^hyperclova_x]: [HyperCLOVA X Technical Report](https://arxiv.org/abs/2404.01954)

[^hyperclova_x_think]: [HyperCLOVA X THINK Technical Report](https://huggingface.co/papers/2506.22403)

### 开源基线与综述

#### AI2 Tulu / Survey

[^tulu_3]: [Tulu 3: Pushing Frontiers in Open Language Model Post-Training](https://openreview.net/forum?id=i1uGbfHHpH)

[^tulu_3_blog]: [Tulu 3 Technical Blog](https://allenai.org/blog/tulu-3-technical)

[^rl_survey]: [Reinforcement Learning for LLM Post-Training: A Survey](https://openreview.net/forum?id=UdsXTNzzvg)
