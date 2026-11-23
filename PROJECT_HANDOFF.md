# 多智能体协作搜索系统 · 项目交接说明（v2）

> 交接日期：2026-11-23（v2：并入 W7 重复实验与 R06 稳定性实验）  
> 代码仓库：`C:\Users\LEGION\PycharmProjects\multi-agent-search`  
> 审阅基准提交：`2dfe25d`（其基于 `ae05260` 交接文档 v1，再往下是 `7c2eaf2` 信任层改造与 `5bfc922` W7 对比实验）  
> 上一版文档见 git 历史中的 `ae05260:PROJECT_HANDOFF.md`  
> 适用对象：下一位负责代码审阅、继续开发或课程答辩准备的 AI/开发者

---

## 0. 30 秒交接摘要

- **系统状态**：W1–W7 全部完成，信任层（结构化 Fact 审计）已落地并通过离线回归；**W8 Streamlit 前端尚未开始**，这是当前唯一的功能缺口。
- **能做什么**：中文提问 → Planner 路由 → DuckDB 真实查询 / 挖掘工具 / 本地知识库 → Synthesizer 生成事实引用草稿 → Critic 程序级硬校验 + 语义审查 → 输出或阻断。
- **可信度现状**：多 Agent 串行/并行在核心题上严格通过 **25/27（92.6%）**；单 Agent **15/27（55.6%）**，差在能力覆盖与硬校验机制。
- **最该记住的一条**：单次运行结果不可当结论。3 次重复后，串行与并行通过数**打平**（25 vs 25），原来"并行 11/13 比串行 10/13 更准"是采样波动。
- **下一步**：先做 W8 最小前端（必须展示证据链），再补 `requirements.txt` 与三个硬校验启发式缺口。

三条自检命令（在项目根目录、用项目虚拟环境 `dsp`）：

```powershell
dsp\Scripts\python.exe -m unittest discover -s tests          # 信任层 10 用例，秒级
dsp\Scripts\python.exe experiment_w7_repeat.py --smoke        # 单次冒烟，约 10s
dsp\Scripts\python.exe regression_w6.py                       # 10 题回归，约 3 分钟
```

---

## 1. 项目要实现什么

本项目是《大数据挖掘实例》课程的长线项目，目标不是做一个只会聊天的 LLM Demo，而是构建一个**面向非技术用户的 MovieLens 数据挖掘问答与分析系统**。

用户用中文提出分析问题后，系统应完成：

1. **任务理解**：Planner 判断需要 SQL、数据挖掘、RAG 中的哪些能力。
2. **真实检索**：通过 DuckDB 查询 MovieLens 评分数据，禁止凭模型记忆编造。
3. **数据挖掘**：调用 W3/W4 已完成的 K-Means、Apriori 和决策树结果。
4. **知识解释**：通过本地项目知识库解释方法、参数和指标。
5. **证据综合**：Synthesizer 只基于专家证据生成中文答案。
6. **质量审查**：Critic 检查结论、切题性、夸大和数字证据。
7. **性能验证**：比较多 Agent 串行与物理并行，以及单 Agent 与多 Agent。
8. **最终演示**：用 Streamlit 把系统包装成可现场操作的课程展示项目（**未完成**）。

一句话定位：

> 让普通用户用自然语言调用真实数据库、数据挖掘算法和项目知识库，同时保留可追溯证据链与程序级校验。

---

## 2. 当前总体状态

| 阶段 | 状态 | 说明 |
|---|---|---|
| W1 数据理解与 EDA | 已完成 | MovieLens 数据规模、分布、稀疏度等已分析 |
| W2 预处理与 DuckDB | 已完成 | 数据清洗、DuckDB 建库、检索 Agent v1 |
| W3 用户聚类 | 已完成 | K-Means，K=2，轮廓系数 0.4077 |
| W4 关联规则与分类 | 已完成 | Apriori、分组规则、决策树分类 |
| W5 多 Agent 编排 | 已完成 | LangGraph 六角色链路跑通 |
| W6 可信性与并行 | 已完成 | async 物理并行、Critic 硬校验、hybrid RAG、10 题回归 |
| W7 对比实验 | 已完成 | 单 Agent / 串行 / 并行，13 题单次全量（§6.1） |
| 信任层改造 | 已完成 | 结构化 Fact 审计、四状态工具、三层覆盖、阻断防泄露（`7c2eaf2`） |
| 重复实验（12.2） | 已完成 2026-11-23 | R05–R13 × 3 模式 × 3 次 = 81 次（§6.2）；R06 单独 8 次 = 24 次（§6.3） |
| W8 前端与最终演示 | **未开始** | Streamlit 界面尚未实现，当前只能命令行/PyCharm 运行 |
| PPT | 当前暂停 | 用户要求先落地项目，暂不继续制作周报 PPT |

一句话状态：**主流程可运行、实验产物完整、数字可靠性有程序级保证；缺前端、缺依赖锁定、部分题的样本量仍偏小。**

---

## 3. 系统架构

核心文件：`multi_agent.py`

```text
START
  │
  ▼
Planner
  │  输出 routes: sql / mining / rag，可多选
  ▼
条件扇出
  ├─ SQL Agent       → run_sql → DuckDB
  ├─ Mining Agent    → cluster_profiles / association_rules / decision_tree_summary
  └─ RAG Agent       → 查询重写 → 本地知识库 hybrid 检索
  │
  ▼
Synthesizer
  │  只基于原始工具证据，用 [F###] 声明式引用 Fact，生成 200 字以内中文回答
  ▼
Critic
  ├─ 程序级硬校验（出处 / 对象关系 / 格式声明）
  ├─ PASS    → END，输出渲染后的最终答案
  ├─ REVISE  → 回 Synthesizer，最多修订 1 次
  └─ FAIL    → 只输出固定阻断文案（不泄露草稿数字）
```

### State 关键字段

- `question`：用户问题。
- `routes`：Planner 路由结果。
- `evidences`：专家摘要证据，仅供 Synthesizer 参考，**不作事实源头**。
- `tool_evidences`：原始工具证据，Synthesizer 与审计统一以此为准。
- `tool_records`：工具结构化结果（status + facts），由信任层生成 Fact 目录。
- `draft` / `rendered_draft` / `final`：草稿（含 `[F###]`）、渲染后草稿、最终回答。
- `block_reason`：阻断原因（仅诊断字段）。
- `verdict` / `critique` / `hard_check`：Critic 结果。
- `revision_count`：修订次数，上限 1。
- `trace`：执行轨迹，reducer 累加，支持并行分支写回。

### 信任层（`src/trust_layer.py`，commit `7c2eaf2`）

不调用 LLM，可离线回归（`tests/test_trust_layer.py`，10 用例全过）：

1. **工具四状态**：`success / empty / error / not_called`；error 内容不进入事实证据。
2. **结构化 Fact**：工具返回 `facts`（subject/metric/value/unit/display/aliases），程序分配 Fact ID；Synthesizer 用 `[F003]`、`[F003:round:2]`、`[F003:percent:1]` 引用，数字由程序渲染。
3. **三层审计**（`audit_answer`）：出处校验 → 对象关系校验（subject/metric 绑定）→ 格式声明校验（舍入/百分比必须显式声明；已是百分数的 Fact 禁止再叠加 percent）。
4. **阻断防泄露**：硬校验 FAIL 且修订用尽时，`final` 只含固定文案。
5. **覆盖三层口径**：`planned_routes` / `actual_routes` / `successful_routes`，能力覆盖只认 successful。
6. **数字清洗**：剥离行首有序列表编号、`Top-10` 连字符标签、Markdown 标记。

已知边界（§7 P1-7 有实测证据）：`_local_context` 是启发式切分；括号内与跨句引用只做保守检查；`Top 10`（空格）不被清洗；SQL 结果行的 subject 是"结果第 N 行"，一旦草稿把它当"轻度/重度用户"引用就会判为对象错配；派生数字（如 532+78=610）一律拦截。

### 并行方式

专家节点是 `async` 的。Planner 同时路由多个专家时，LangGraph 在同一 super-step 并发执行；阻塞的本地工具调用通过 `asyncio.to_thread` 放入线程。**单分支题的串行与并行执行的是同一段代码**，因此这类题的耗时差只反映 API 抖动（见 §6.2、§6.4）。

---

## 4. 数据与算法成果

### 4.1 数据基础

- 数据集：MovieLens `ml-latest-small`
- 评分：100,836 条；用户：610 人；电影表：9,742 部；有评分记录的电影：9,724 部
- 稀疏度：98.30%
- 数据库：`data/movielens.duckdb`

### 4.2 K-Means 用户分群

- K 从 2 到 8 扫描，按轮廓系数选择；最优 **K=2，轮廓系数 0.4077**。
- 轻度用户 532 人（87.2%），重度用户 78 人（12.8%）。
- 分群主要反映活跃度差异，而不是平均评分高低。
- 产物：`outputs/w3/`

### 4.3 Apriori 关联规则

- 事务定义：每位用户高评分电影中最偏好的 top-3 类型（避免"看过一次就算偏好"）。
- 规则按 lift 排序，而不只看 confidence。
- 代表规则：`Romance → Comedy + Drama`、`Sci-Fi → Action`、`Adventure → Action`。
- 产物：`outputs/w4/`

### 4.4 决策树分类

- 任务：预测宽容型/严格型评分者；5 折交叉验证准确率 **70.3% ± 1.7%**。
- 最重要特征 `std_rating`，重要性约 0.58。
- 产物：`outputs/w4/`

---

## 5. 开发进程

| 周次 | 内容 | 关键结果 |
|---|---|---|
| W1 | 数据理解与 EDA | 规模、分布、长尾、稀疏度 |
| W2 | 预处理、DuckDB、检索 Agent v1 | 清洗、建库、初版检索 |
| W3 | K-Means 用户分群 | K=2，轮廓系数 0.4077 |
| W4 | Apriori 与决策树 | 规则挖掘、分组规则、分类器 |
| W5 | LangGraph 六角色编排 | 链路跑通，四道演示题 |
| W6 | 物理并行、硬校验、hybrid RAG、回归 | 专家 async 化；10 题回归（**旧口径**）10/10 |
| W7 | 单/多 Agent 对比实验 | 13 题 × 3 模式；引入原始工具证据审计 |
| W7+ | 信任层改造 | 结构化 Fact、四状态工具、三层覆盖、阻断防泄露 |
| W7++ | 重复实验（2026-11-23） | R05–R13 × 3 次 = 81 次；R06 × 8 次 = 24 次 |

W6 的 10/10 与 W7 的严格原始证据审计**不是同一口径**：信任层改造后重跑 W6 回归为 9/10（R06 为 Critic 语义审查两次 REVISE 用尽，属设计内阻断）。

---

## 6. 实验结果

> 口径说明：`elapsed_seconds` 只计模式自身执行时间，不含外部评审；`overall_pass = 能力覆盖 + 硬校验 + 关键事实 + 外部评审 PASS`。  
> 外部评审（DeepSeek）只做语义复核，不参与作答，也不是事实标准。

### 6.0 实验集与统一约束

13 道题（`regression_w6.CASES` + `experiment_w7.EXTRA_CASES`）：

| 题号 | 问题（简） | 期望路由 | 关键事实 |
|---|---|---|---|
| R01 | 评分 >4.0 且 ≥100 人评价的电影有哪些 | sql | 47 |
| R02 | 评价人数最多的 10 部 | sql | — |
| R03 | Comedy 中平均分最高且 ≥50 人的 5 部 | sql | — |
| R04 | 用户分群 + 科幻用户还喜欢什么 | mining | 0.4077 |
| R05 | 决策树准确率与最重要特征 | mining | 70.3%、std_rating |
| R06 | 轻/重度用户关联规则差异与跨组稳定规则 | mining | 重度、轻度 |
| R07 | 为什么 K=2、PCA 为何不参与训练 | rag | 0.4077、PCA |
| R08 | lift / support / confidence 怎么理解 | rag | lift、confidence |
| R09 | LangGraph 并行写 State 为何冲突、reducer 怎么解 | rag | reducer |
| R10 | MovieLens 用户偏好简报 + 挖掘方法 | mining+rag | 532、78 |
| R11 | 47 部 + ratings 表字段 | sql+rag | 47、userId |
| R12 | 分群差异 + 轮廓系数选 K | mining+rag | 532、78、0.4077 |
| R13 | 数据集概况 + 用数据库核实总数 | sql+rag | 100836、610、9742 |

三种模式的统一约束（公平性所在，改动前务必保持）：

1. 同一 DeepSeek 模型、`temperature=0`；
2. 同一套 SQL、挖掘、知识库工具（单 Agent 额外拿到同一个 `retrieve_knowledge`）；
3. 同一关键事实检查与同一 `audit_answer` 程序级硬校验，且统一对照**原始工具证据**，不用专家摘要；
4. 同一外部评审，评审不参与作答；
5. `elapsed_seconds` 只计模式自身执行时间，不含外部评审。

### 6.1 单次全量（13 题 × 3 模式，`outputs/w7/`）

| 模式 | 严格通过 | 能力覆盖 | 数字硬校验 | 外部语义评审 | 平均耗时 |
|---|---:|---:|---:|---:|---:|
| 单 Agent | 5/13（38.46%） | 11/13 | 6/13 | 13/13 | 2.766s |
| 多 Agent 串行 | 10/13（76.92%） | 13/13 | 10/13 | 13/13 | 9.796s |
| 多 Agent 并行 | 11/13（84.62%） | 13/13 | 11/13 | 13/13 | 8.523s |

单 Agent 硬校验失败是**设计使然**：没有 Fact 引用与声明式舍入，派生数字必然不满足审计标准。

### 6.2 核心题 3 次重复（R05–R13，81 次运行，`outputs/w7_repeat/`）

| 模式 | 运行 | 严格通过 | 每轮通过题数（中位，区间） | 耗时中位 | 耗时区间 | 硬校验通过 | 外部评审通过 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 单 Agent | 27 | 15（55.56%） | 5/9（4–6） | 3.111s | 1.572–5.576s | 18/27 | 27/27 |
| 多 Agent 串行 | 27 | 25（92.59%） | 8/9（8–9） | 10.989s | 5.029–21.554s | 26/27 | 26/27 |
| 多 Agent 并行 | 27 | 25（92.59%） | 8/9（8–9） | 8.969s | 4.570–21.764s | 25/27 | 27/27 |

每题通过次数（单 Agent / 串行 / 并行）：

| 题号 | R05 | R06 | R07 | R08 | R09 | R10 | R11 | R12 | R13 |
|---|---|---|---|---|---|---|---|---|---|
| 单 Agent | 3 | 0 | 3 | 2 | 3 | 1 | 0 | 3 | 0 |
| 多 Agent 串行 | 3 | 3 | 3 | 3 | 3 | 2 | 3 | 3 | 2 |
| 多 Agent 并行 | 3 | 1 | 3 | 3 | 3 | 3 | 3 | 3 | 3 |

并行 vs 串行（按轮配对）：

- 配对样本 27 个，并行更快 20 个；题级耗时比中位 1.12x、均值 1.18x。
- 全部 9 题聚合耗时下降：中位 **20.06%**（3.53%~22.05%）。
- 预期多分支题 R10–R13：聚合下降中位 **19.09%**（9.86%~31.34%），三轮分别 2/4、4/4、4/4 题更快。
- 单分支题（R05–R09）串行/并行同代码路径，第 1 轮 9 题口径只有 3.53%，纯噪声，不能当收益证据。

**三条必须修正的单次结论**：

1. 串行与并行通过数打平（25 vs 25），"并行更准"不成立。
2. 单 Agent 的 15/27 主要输在能力覆盖（R11/R13 未调用 RAG，capability 21/27）与硬校验（18/27），其外部评审仍 27/27 PASS。
3. R06 是唯一明显不稳定题（9 次仅 4 次通过），且失败全在硬校验，不是并行缺陷。

### 6.3 R06 单独 8 次（24 次运行，`outputs/w7_repeat_r06/`）

| 模式 | 通过 | 通过率 | 95% Wilson | 硬校验 | 外部评审 | 耗时均值±标准差 | CV | 中位 | 区间 |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| 单 Agent | 0/8 | 0.0% | 0.0%~32.4% | 0/8 | 8/8 | 2.67±0.17s | 6.4% | 2.617s | 2.504–3.024s |
| 多 Agent 串行 | 5/8 | 62.5% | 30.6%~86.3% | 6/8 | 7/8 | 16.23±6.77s | 41.7% | 13.285s | 11.801–31.842s |
| 多 Agent 并行 | 6/8 | 75.0% | 40.9%~92.8% | 6/8 | 8/8 | 13.07±3.19s | 24.4% | 12.141s | 9.838–18.794s |

- 串行 5/8 vs 并行 6/8：**Fisher 精确检验双侧 p = 1.0**，8 次无法区分 62.5% 与 75%。
- 单 Agent 0/8 vs 串行 5/8：p = 0.0256，但差异来自机制（无 Fact 渲染，硬校验 0/8），是设计对照。
- 耗时：串行被 31.8s 离群拉高（CV 41.7%），并行更稳（CV 24.4%）；5/8 轮并行更快，逐轮耗时比中位 1.265x（离散 ±0.805），中位之比仅 1.094x。
- 单 Agent 最快最稳（CV 6.4%）却 0 通过：**快 ≠ 对**。
- 多 Agent 的 5 条失败记录中，4 条硬校验失败，涉及派生数字、Top 10 标签或 SQL 行号与用户分组的对象错配。第 2 轮串行硬校验通过，外部评审判定置信度数值与组别引用错误。
- 本机展示入口为 `dsp\Scripts\python.exe showcase.py`，浏览器打开 `http://127.0.0.1:8765`。回放读取上述 24 条历史记录，实时问答调用当前多 Agent 系统。
- 口径注记：本节耗时比仅基于 R06 的 8 轮配对样本，与 §6.2 的 27 个配对样本（中位 1.12x）口径不同，两处数字不可混用。

### 6.4 报告写作口径（能说 / 不能说）

可以说：

- 多 Agent 显著提高复杂问题的**能力覆盖**（11/13 → 13/13）与**数字可靠性**（硬校验 6/13 → 11/13）。
- 多分支题并行后聚合耗时下降**约 20%**（3 次中位数 19.1%，区间 9.9%~31.3%）。
- 程序级硬校验能拦住语义评审漏掉的数字错误（R13 的 2660.0% 案例）。

不能说：

- "多 Agent 的数字正确率必然更高"（串并行 25 vs 25；R06 5/8 vs 6/8，p=1.0）。
- "外部 LLM 评审 PASS 就等于事实可靠"（单 Agent 27/27 PASS 但硬校验 18/27；R06 第 2 轮评审把正确数字判错）。
- "单次耗时实验具有统计显著性"；也不要混用 W6 benchmark 加速比与 W7 全链路耗时。
- 不要用"全部 9 题"口径当并行收益证据；也不要只报 3 次重复的通过率而不给区间。

---

## 7. 已知问题

### 已修复（commit `7c2eaf2`，均有离线回归）

- ~~P0：专家摘要可能"洗数字"~~ → 工具返回结构化 facts，Synthesizer 只能引用 Fact ID，数字由程序渲染；审计以原始工具证据为准。
- ~~P0：W6 回归口径需升级~~ → 回归与 W7 共用 `audit_answer`，覆盖判定改用 `successful_routes`；重跑 9/10（R06 为设计内阻断）。
- ~~P1：硬校验过于字面化~~ → 声明式舍入/百分比引用，未声明的派生数字仍拦截。
- ~~P1：SQL 失败文本混入证据 / 覆盖统计口径~~ → 四状态工具 + planned/invoked/successful 三层覆盖。
- 新增防护：双重百分比、Fact 引用旁重复写数字、阻断输出泄露草稿数字。

### P1-1：外部评审与内部 Critic 都有随机性

- 内部 Critic 修订上限只有 1 次；R06 在 W6 回归中曾两次 REVISE 用尽而阻断。
- 外部评审会误判：R06 8 次实验第 2 轮串行，答案数字与证据一致，评审却判"置信度严重不符"（数值 1.0000 vs 0.8105 实际一致）。
- 结论：评审只能作为语义复核，最终防线必须是程序级硬校验。

### P1-2：还没有前端（W8）

Streamlit 界面未开始，当前只能命令行/PyCharm 运行。现场展示前至少需要：证据链展示（原始工具证据、Fact 引用、硬校验状态、阻断原因）+ 历史日志回放兜底。

### P1-3：样本量仍不足

核心题 3 次、仅 R06 做到 8 次；其余题没有置信区间。3 次只能看方向，8 次仍无法区分 62.5% 与 75%。若要写进正式报告，建议统一到 5 次，或把 R05/R11/R13 也做到 8~10 次。

### P1-4：外部 LLM 评审不是事实标准

见 P1-1 的两个实例。R13 的 2660.0% 案例也证明：语义评审与旧版硬校验都会漏，只有程序级校验能兜住。

### P1-5：依赖环境缺少统一锁定文件

没有 `requirements.txt` / 锁文件；`dsp/` 是本机虚拟环境，不能提交。换机器无法精确复现 langchain / duckdb / sentence-transformers 版本。

### P1-6：本地 BGE 模型未入库

`models/bge-small-zh-v1.5/` 已在 `.gitignore` 排除。换机器需重新下载，否则 hybrid 检索可能回退或尝试联网。

### P1-7：硬校验启发式边界（有实测证据）

1. `Top-10` 会被剥离，但 `Top 10`（空格）不会，导致"10"被判为无出处数字。
2. 派生数字一律拦截：`532+78=610` 这种可由证据相加得到的数字也被判失败（R06 多次命中）。
3. Planner 路由漂移时（R06/R13 偶尔把 rag 换成 sql），SQL 行的 subject 是"结果第 N 行"，草稿若写成"轻度/重度用户"就会触发对象错配（R06 第 8 轮并行，19 处）。
4. 这三项是"宁可错杀"的设计代价，但也解释了 R06 的 3/8 失败。建议后续：补 `Top 10|前 10` 清洗；为 SQL 结果提供别名 subject；或允许显式"计算声明"（如 `[F001+F002]`）。

### P2-1：仓库卫生

工作区有大量与本任务无关的历史修改与未跟踪文件（notebook、旧输出、IDE 配置）。**未经用户确认不要删除**，建议单独开一次仓库整理任务。

### P2-2：课程结论边界

MovieLens `ml-latest-small` 规模小，关联规则与分类结果适合课程实验，不应包装成工业级推荐系统结论。

---

## 8. 关键文件与产物

| 文件/目录 | 作用 |
|---|---|
| `multi_agent.py` | 多 Agent 主系统：六角色、并行、结构化工具结果、内部 Critic |
| `src/trust_layer.py` | 信任层：Fact 目录、声明式渲染、三层审计、四状态与覆盖统计（无 LLM） |
| `tests/test_trust_layer.py` | 信任层离线回归（10 用例，秒级） |
| `experiment_w7.py` | W7 单/多 Agent 对比实验，13 题全量入口 |
| `experiment_w7_repeat.py` | 重复实验入口（`--cases` / `--repeats` / `--out` / `--smoke`），复用同一 `evaluate_case` |
| `analyze_w7_repeat.py` | 重复实验离线复算：9 题 / R10–R13 / 动态多分支三种口径（无 LLM） |
| `analyze_r06_stability.py` | R06 8 次：Wilson 区间、Fisher 检验、耗时标准差（无 LLM） |
| `regression_w6.py` | W6 十题回归（与正式问答同一套审计） |
| `src/rag/knowledge_base.py` | 项目知识库，支持 TF-IDF / semantic / hybrid |
| `src/rag/evaluate_retrieval.py` | 检索后端对比评估 |
| `src/mining/mining_tools.py` | 聚类、关联规则、决策树的统一工具接口 |
| `clean.py` / `build.py` / `cluster_users.py` | 数据预处理 / 建库 / 聚类脚本 |
| `outputs/w3/`、`outputs/w4/` | 聚类、关联规则、决策树结果与图 |
| `outputs/w5/`、`outputs/w6/` | W5 日志、W6 回归与 benchmark |
| `outputs/w7/` | 13 题单次全量结果 |
| `outputs/w7_repeat/` | 81 次重复：`repeat_results.csv`、`case_mode_stats.csv`、`repeat_summary.json`、`repeat_report.md`、`repeat_report_addendum.md`、`repeat_multiroute_stats.json` |
| `outputs/w7_repeat_r06/` | R06 8 次：`repeat_results.csv`、`r06_stability.md/json`、`console.log` |
| `.env` | DeepSeek API Key，禁止提交 |
| `models/`、`data/`、`dsp/` | 模型、数据、虚拟环境，禁止提交 |

---

## 9. 运行方法

默认在项目根目录执行：

```powershell
cd C:\Users\LEGION\PycharmProjects\multi-agent-search
```

环境要求：Windows；虚拟环境 `dsp`；`.env` 内 `DEEPSEEK_API_KEY=...`（不要写进代码、日志、PPT、Git）。

### 9.1 主程序与回归

```powershell
dsp\Scripts\python.exe multi_agent.py                 # 四道演示题 + RAG/并行 benchmark（约 2 分钟）
dsp\Scripts\python.exe regression_w6.py               # 10 题回归（约 3 分钟）
dsp\Scripts\python.exe -m unittest discover -s tests  # 信任层 10 用例（秒级，改信任层后必跑）
```

### 9.2 W7 全量对比（13 题 × 3 模式）

```powershell
dsp\Scripts\python.exe experiment_w7.py                 # 全量，约 5 分钟
dsp\Scripts\python.exe experiment_w7.py --cases R10     # 单题
```

### 9.3 重复实验（本文档 §6.2 / §6.3 的数据来源）

```powershell
# 冒烟：1 题 1 模式 1 次，约 10s，验证环境与写盘
dsp\Scripts\python.exe experiment_w7_repeat.py --smoke

# 核心题 3 次重复（81 次运行，实测 774.958s ≈ 13 分钟）
dsp\Scripts\python.exe experiment_w7_repeat.py

# R06 单独 8 次（24 次运行，实测约 286s ≈ 5 分钟）
dsp\Scripts\python.exe experiment_w7_repeat.py --cases R06 --repeats 8 --out outputs/w7_repeat_r06

# 离线复算（不调用 LLM，秒级）
dsp\Scripts\python.exe analyze_w7_repeat.py
dsp\Scripts\python.exe analyze_r06_stability.py
```

`--out` 会切换产物目录，避免覆盖已有结果；重复实验每题每模式立即落盘，网络中断不丢已完成样本。

---

## 10. 给下一位 AI 的审阅重点

1. **信任层启发式边界**：`_local_context` 语义段切分、括号对收缩、别名长度加权都是启发式。是否升级为依存分析，还是在课程场景保持"宁可错杀"？先跑 `tests/test_trust_layer.py` 再改。
2. **实验公平性**：单 Agent 无 Fact 引用导致硬校验天然吃亏，这是量化信任层价值的对照组，但请确认提示词、工具、验收指标没有其它系统性偏袒。
3. **Planner 路由稳定性**：R06/R13 偶尔多选或换选（mining→sql），会同时影响覆盖、审计与耗时。建议给"只用挖掘结果"的题加硬约束，或让 SQL 结果提供 subject 别名。
4. **Critic 收敛度**：修订上限是否从 1 提到 2？外部评审误判（R06 第 2 轮）是否要加一致性检查？
5. **W8 前端**：必须展示证据链与审查状态，不能只显示最终答案；保留历史日志回放兜底。
6. **统计口径**：任何通过率都要带 n 与区间；任何耗时都要说明是否含外部评审、是否只算多分支题。

---

## 11. 不应做的事

- 不要读取、打印、提交 `.env` 中的真实 API Key。
- 不要把 `models/`、`data/`、`dsp/` 提交进 Git。
- 不要删除用户保留的历史 notebook、输出文件和未跟踪实验文件，除非用户明确确认。
- 不要把外部 LLM 评审 PASS 当作事实正确性证明。
- 不要把 W6 的 10/10 与 W7 严格原始证据审计结果混为一谈。
- 不要把 W6 benchmark 的加速比与 W7 全链路耗时直接比较。
- 不要为了演示好看删除失败案例：R06 的偶发失败与 R13 的 2660% 案例是最有价值的研究发现。
- 不要把 3 次重复的数字当作统计显著结论。

---

## 12. 建议的下一步顺序

1. **W8 Streamlit 最小界面**（P1-2）：问题输入 → 答案 + 证据链（原始工具证据、Fact 引用、硬校验状态、阻断原因）→ 历史日志回放；先做最小可用，不追求样式。
2. **补硬校验三个缺口**（P1-7）：`Top 10|前 10` 清洗、SQL 结果 subject 别名、派生数字的显式声明方案；每项配 `tests/test_trust_layer.py` 用例。
3. **补 `requirements.txt`**（P1-5）：锁定 langchain / langgraph / duckdb / sentence-transformers / pandas 版本；附 BGE 模型下载说明。
4. **统一样本量**（P1-3）：把 R05/R11/R13 也各做 8~10 次，或核心题统一到 5 次，报告中位数 + 标准差。
5. **现场演示脚本**：用已验证题 R01/R04/R05/R10，预设失败案例（R06 硬校验阻断、R13 的双重百分比）讲解信任层价值。

> 执行注记：若延续用户"先别急着做前端"的当前意图，可先执行第 2、3 项（硬校验三个缺口与 `requirements.txt`，均不依赖前端、可独立交付），第 1 项（W8 前端）在演示前完成即可。

---

## 13. Git 状态（2026-11-23）

最新提交：

```text
ae05260 docs: 交接文档更新至信任层改造后状态(7c2eaf2)
7c2eaf2 信任层: 结构化Fact审计+四状态工具+三层覆盖统计+阻断防泄露
5bfc922 W7: 单/多 Agent 对比实验与原始证据审计
eb61340 W6: 物理并行(async fan-out) + Critic 硬校验 + hybrid 语义检索 + 10 题回归全过
```

工作区状态（本次交接提交后）：

- **本次已提交**：`PROJECT_HANDOFF.md`（本 v2 版）、`experiment_w7_repeat.py`、`analyze_w7_repeat.py`、`analyze_r06_stability.py`、`outputs/w7_repeat/`、`outputs/w7_repeat_r06/`。
- 仍保留未提交：`.idea/*`、`dsp.ipynb`、`outputs/w5/run_log.txt`、`prepare_transactions.py`、`rule.py` 等历史改动，以及用户自己的 notebook、旧输出、IDE 配置等。**留待用户确认后再整理，不要代删。**

---

## 14. 移交结论

系统已经完成从数据层、算法层、多 Agent 编排层、对比实验层到程序级信任层的建设，能够：

- 查询真实 MovieLens 数据，调用课程数据挖掘结果，检索项目知识库；
- 生成结构化中文分析，数字由程序按 Fact 声明渲染；
- 对数字做出处、对象关系、格式声明三层硬校验，不通过即阻断且不泄露；
- 记录执行轨迹与原始工具证据；
- 对单 Agent、多 Agent 串行、并行做定量比较，并用 81 + 24 次重复实验给出中位数、标准差与区间。

仍不能称为最终完成版，因为：

1. **前端演示界面尚未实现**（W8）。
2. **部分题样本量仍小**，只有 R06 做到 n=8。
3. **依赖未锁定，模型与数据未入库**，换机器不能一键复现。
4. 硬校验还有三个已知启发式缺口（P1-7）。

建议下一位 AI 从 **W8 最小前端** 开始；信任层与重复实验产物已足以支撑现场演示的证据链展示与统计口径。
