# 多智能体协作搜索系统 · 项目交接说明

> 交接日期：2026-11-23  
> 代码仓库：`C:\Users\LEGION\PycharmProjects\multi-agent-search`  
> 审阅基准提交：`7c2eaf2 信任层: 结构化Fact审计+四状态工具+三层覆盖统计+阻断防泄露`（在 `5bfc922` 之上）  
> 适用对象：下一位负责代码审阅、继续开发或课程答辩准备的 AI/开发者

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
8. **最终演示**：后续用 Streamlit 把系统包装成可现场操作的课程展示项目。

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
| W7 对比实验 | 已完成 | 单 Agent / 多 Agent 串行 / 多 Agent 并行，13 题实验 |
| 信任层改造 | 已完成 | 结构化 Fact 审计、四状态工具、三层覆盖统计、阻断防泄露（commit 7c2eaf2） |
| W8 前端与最终演示 | 未开始 | Streamlit 界面尚未实现 |
| PPT | 当前暂停 | 用户要求先落地项目，暂不继续制作周报 PPT |

当前最重要的结论：**系统主流程可运行，实验产物完整；W7 审阅指出的四项 P0/P1 问题（专家摘要洗数字、数字错配、SQL 状态、覆盖统计口径）已全部按结构化 Fact 方案修复，并经全量回归验证。**

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
  │  合并专家证据，生成 200 字以内中文回答
  ▼
Critic
  ├─ PASS    → END，输出最终答案
  ├─ REVISE  → 回 Synthesizer，最多修订 1 次
  └─ FAIL    → 阻断输出
```

### State 关键字段

- `question`：用户问题。
- `routes`：Planner 路由结果。
- `evidences`：专家生成的摘要证据，仅供 Synthesizer 参考，不作事实源头。
- `tool_evidences`：原始工具证据，Synthesizer 与审计统一以此为准，避免摘要“洗数字”。
- `tool_records`：工具结构化结果（status + facts），由信任层生成 Fact 目录。
- `draft` / `rendered_draft` / `final`：草稿（含 `[F###]` 引用）、渲染后草稿、最终回答。
- `block_reason`：阻断原因（仅诊断字段，FAIL 时的 `final` 只含固定文案，不泄露草稿数字）。
- `verdict` / `critique` / `hard_check`：Critic 审查结果。
- `revision_count`：修订次数，上限 1 次。
- `trace`：执行轨迹，使用 reducer 累加，支持并行分支写回。

### 信任层（`src/trust_layer.py`，W7 审阅后新增）

针对审阅 AI 的四项发现实现，不调用 LLM，可离线回归（`tests/test_trust_layer.py` 10 用例）：

1. **工具四状态**：`success / empty / error / not_called`；error 工具的内容不进入事实证据。
2. **结构化 Fact**：工具返回 `facts`（subject/metric/value/unit/display/aliases），程序分配 Fact ID；Synthesizer 用 `[F003]`、`[F003:round:2]`、`[F003:percent:1]` 声明式引用，最终数字由程序渲染。
3. **三层审计**（`audit_answer`）：出处校验（数字必须来自原始证据或 Fact 渲染）→ 对象关系校验（subject/metric 绑定，语义段切分 + 括号对收缩 + 别名长度加权距离）→ 格式声明校验（舍入/百分比必须显式声明，且禁止对已是百分数的 Fact 叠加 percent）。
4. **阻断防泄露**：硬校验 FAIL 且修订用尽时，`final` 只输出固定阻断文案，草稿数字与缺失明细只进诊断字段。
5. **覆盖三层口径**：`planned_routes`（Planner 计划）/ `actual_routes`（实际调用）/ `successful_routes`（成功取证），能力覆盖只认 successful。
6. **数字提取清洗**：剥离行首有序列表编号、`Top-10` 连字符标签数字与 Markdown 标记。

已知边界：括号内引用与跨句引用只做保守检查（宁可跳过不判，不误伤）；`_local_context` 是启发式切分，不是通用 NLP。

### 并行方式

W6 已将专家节点改为 `async`。当 Planner 同时路由多个专家时，LangGraph 在同一 super-step 中并发执行；阻塞的本地工具调用通过 `asyncio.to_thread` 放入线程。

---

## 4. 数据与算法成果

### 4.1 数据基础

- 数据集：MovieLens `ml-latest-small`
- 评分：100,836 条
- 用户：610 人
- 电影表：9,742 部
- 有评分记录的电影：9,724 部
- 稀疏度：98.30%
- 数据库：`data/movielens.duckdb`

### 4.2 K-Means 用户分群

- K 从 2 到 8 扫描，按轮廓系数选择。
- 最优：K=2，轮廓系数 **0.4077**。
- 轻度用户：532 人，占 87.2%。
- 重度用户：78 人，占 12.8%。
- 分群主要反映活跃度差异，而不是平均评分高低。

主要产物：`outputs/w3/`

### 4.3 Apriori 关联规则

- 事务定义：每位用户高评分电影中最偏好的 top-3 类型。
- 目的：避免“看过一次就算偏好”导致 Drama 等主流类型 support 过高、失去区分度。
- 规则按 lift 排序，而不是只看 confidence。
- 代表规则：
  - `Romance → Comedy + Drama`
  - `Sci-Fi → Action`
  - `Adventure → Action`

主要产物：`outputs/w4/`

### 4.4 决策树分类

- 任务：预测宽容型/严格型评分者。
- 5 折交叉验证准确率：约 **70.3% ± 1.7%**。
- 最重要特征：`std_rating`，重要性约 0.58。
- 结果说明用户评分严格度可以由行为特征部分预测，但提升有限。

---

## 5. 每周已完成工作

### W1：数据理解与 EDA

完成 MovieLens 数据加载、规模统计、评分分布、长尾分布和稀疏度分析，为后续数据挖掘建立基础。

### W2：预处理、DuckDB 与检索 Agent v1

完成：

- 时间字段转换与特征处理。
- genres 多热编码。
- 用户特征表构建。
- DuckDB 建表与验收 SQL。
- 初版 `retriever_agent.py`。

### W3：K-Means 用户分群

完成用户特征构建、标准化、K 扫描、轮廓系数评估和分群画像。保留 PCA 图用于观察，不把 PCA 后的坐标用于训练，避免把“可视化压缩”误当成真实聚类空间。

### W4：Apriori 与决策树

完成：

- 关联规则挖掘模块：`src/mining/mine_rules.py`
- 分组规则对比：`src/mining/group_rules.py`
- 决策树分类：`src/mining/decision_tree.py`
- 统一工具接口：`src/mining/mining_tools.py`

### W5：LangGraph 六角色编排

完成 Planner、SQL Agent、Mining Agent、RAG Agent、Synthesizer、Critic 的完整链路，四道演示题跑通。此时还是逻辑扇出，未做严格物理并行实验。

### W6：物理并行、硬校验、语义检索与回归

完成：

- 专家节点 async 化，支持多专家物理并行。
- Critic 增加程序级数字硬校验。
- `run_sql` 返回完整结果行数，修复 W5 曾把“展示 15 行”误认为“总共 15 部”的问题；正确结果是 47 部。
- 本地 BGE 中文向量模型 + TF-IDF 组成 hybrid 检索。
- 建立 10 题端到端回归。
- W6 旧口径回归结果：10/10 通过。

注意：W6 的 10/10 是在当时的证据层级下得到的结果，不能直接当作 W7 原始证据审计口径下的最终质量结论。

### W7：单 Agent / 多 Agent 对比实验

新增 `experiment_w7.py`，构建 13 题实验集，其中 R10–R13 是预设双分支题。

三种模式：

1. 单 Agent：一个 LLM 使用全部工具。
2. 多 Agent 串行：Planner + 专家顺序执行 + Synthesizer + Critic。
3. 多 Agent 并行：多个专家在同一轮并发执行。

统一约束：

- 同一 DeepSeek 模型，`temperature=0`。
- 同一 SQL、挖掘和知识库工具。
- 同一关键事实检查。
- 同一数字硬校验。
- 同一外部语义评审。
- 外部评审不参与作答。
- W7 最终口径统一对照**原始工具证据**，不使用专家摘要作为事实源头。

---

## 6. W7 实验结果

实验产物：`outputs/w7/`

以下数据为信任层改造后（commit 7c2eaf2）的最终全量运行（13 题 × 3 模式，LLM 有随机性，单次运行结果在 ±1~2 题间波动，结论口径不变）。

### 6.1 总体结果

| 模式 | 严格通过 | 能力覆盖 | 数字硬校验 | 外部语义评审 | 平均耗时 |
|---|---:|---:|---:|---:|---:|
| 单 Agent | 5/13（38.46%） | 11/13 | 6/13 | 13/13 | 2.766s |
| 多 Agent 串行 | 10/13（76.92%） | 13/13 | 10/13 | 13/13 | 9.796s |
| 多 Agent 并行 | 11/13（84.62%） | 13/13 | 11/13 | 13/13 | 8.523s |

单 Agent 硬校验失败是**设计使然**：单 Agent 没有 Fact 引用与声明式舍入机制，派生数字（舍入/百分比）必然不满足审计标准，这正好量化了多 Agent 信任层的价值。

### 6.2 并行结果

全部 13 题：

- 并行快于串行：9/13。
- 聚合耗时下降：13.0%。
- 平均耗时比：1.21x。

预设多分支题 R05–R13：

- 并行快于串行：5/6。
- 聚合耗时下降：**17.4%**。

历史上两次全量运行该指标在 17%~29% 间波动，报告中应使用“约 20%~30%”或附多次中位数。

### 6.3 结果解释

这组结果支持：

- 多 Agent 提高了复杂问题的能力覆盖（11/13 → 13/13）。
- 多专家物理并行在多分支题中有明确效率收益。
- 程序级硬校验能发现语义评审发现不了的数字问题（实例：R13 中“26.60% 被声明式引用二次放大成 2660.0%”，外部 LLM 评审与旧版硬校验双双漏过，双重百分比防护加入后被程序拦截并修订通过）。

这组结果不支持：

- “多 Agent 的数字正确率必然更高”（单次运行 10/13 vs 11/13 属随机波动）。
- “外部 LLM 评审 PASS 就等于事实可靠”。
- “单次耗时实验具有统计显著性”。

---

## 7. 目前已知问题

### 已修复（commit 7c2eaf2，均有离线回归用例）

- ~~P0：专家摘要可能“洗数字”~~ → 工具返回结构化 facts，Synthesizer 只能引用 Fact ID，数字由程序渲染；审计以原始工具证据为准。
- ~~P0：W6 回归口径需要升级~~ → 回归与 W7 实验共用 `audit_answer`，覆盖判定改用 `successful_routes`，重跑 9/10（R06 为 Critic 语义审查两次 REVISE 用尽，属设计内阻断）。
- ~~P1：硬校验过于字面化~~ → 声明式舍入/百分比引用（`[F001:round:2]` / `[F001:percent:1]`），未声明的派生数字仍拦截。
- ~~P1：SQL 失败文本混入证据 / 覆盖统计口径~~ → 四状态工具 + planned/invoked/successful 三层覆盖。
- 新增防护：双重百分比（已是百分数的 Fact 禁止叠加 percent）；Fact 引用旁重复写数字；阻断输出泄露草稿数字。

### P1：内部 Critic 语义审查偏严且有随机性

W6 回归中 R06 两次运行一次 PASS 一次 FAIL（FAIL 为合法阻断：Critic 提出 5 条详细修改意见，1 次修订上限用尽）。修订上限是否从 1 提到 2、Critic 提示词是否收敛“必须修改”的判定标准，值得评估。

### P1：还没有前端

W8 的 Streamlit 展示界面尚未开始。当前只能通过 PyCharm/命令行运行。课程现场展示前需要完成最小可用界面和历史日志回放兜底。

### P1：实验样本和稳定性不足

W7 是每题每模式运行一次，API 延迟会波动。当前结果适合课程演示和趋势分析，不适合声称统计显著。若要更严谨，应对核心题至少重复 3 次并报告中位数或置信区间。

### P1：外部 LLM 评审不是最终事实标准

LLM 评审适合看切题性和明显矛盾，不适合单独证明数字可靠；R13 的 2660.0% 案例证明程序级校验仍是最后防线。

### P1：依赖环境缺少统一锁定文件

当前仓库没有完整的 `requirements.txt` 或环境锁文件。`dsp/` 是本机虚拟环境，不适合提交。下一个开发者拿到仓库后，无法仅根据仓库精确复现依赖版本。

### P1：本地 BGE 模型未入库

`models/bge-small-zh-v1.5/` 已在 `.gitignore` 中排除。换机器后需要重新下载模型，否则 hybrid 检索可能回退或尝试联网下载。

### P2：仓库卫生需要整理

仓库中存在与本任务无关的历史修改、未跟踪 notebook、临时输出和旧实验文件。它们可能是用户自己的过程材料，**不要在未确认前删除**。建议另开一次专门的仓库整理任务，并由用户确认哪些文件保留或提交。

### P2：课程结论边界

MovieLens `ml-latest-small` 规模较小，关联规则和分类结果适合作为课程实验，不应包装成工业级推荐系统结论。

---

## 8. 关键文件说明

| 文件/目录 | 作用 |
|---|---|
| `multi_agent.py` | 当前多 Agent 主系统，六角色、并行、结构化工具结果与内部 Critic |
| `src/trust_layer.py` | 信任层：Fact 目录、声明式渲染、三层审计、四状态与覆盖统计（无 LLM 依赖） |
| `tests/test_trust_layer.py` | 信任层离线回归（10 用例，秒级） |
| `experiment_w7.py` | W7 单/多 Agent 对比实验，最终实验入口 |
| `regression_w6.py` | W6 十题回归（与正式问答同一套审计） |
| `src/rag/knowledge_base.py` | 项目知识库，支持 TF-IDF、semantic、hybrid |
| `src/rag/evaluate_retrieval.py` | 检索后端对比评估 |
| `src/mining/mining_tools.py` | 聚类、关联规则、决策树的统一工具接口 |
| `src/mining/mine_rules.py` | Apriori 关联规则模块 |
| `src/mining/group_rules.py` | 轻/重度用户分组规则对比 |
| `src/mining/decision_tree.py` | 评分者分类决策树 |
| `clean.py` | W2 数据预处理 |
| `build.py` | DuckDB 建库 |
| `cluster_users.py` | W3 用户聚类脚本 |
| `outputs/w3/` | 聚类结果与图 |
| `outputs/w4/` | 关联规则、决策树结果与图 |
| `outputs/w5/` | W5 运行日志 |
| `outputs/w6/` | W6 回归、检索评估、并行 benchmark |
| `outputs/w7/` | W7 最终对比实验结果 |
| `.env` | DeepSeek API Key，本地使用，禁止提交 |
| `models/` | 本地向量模型目录，已排除 Git |

---

## 9. 运行方法

以下命令默认在项目根目录执行：

```powershell
cd C:\Users\LEGION\PycharmProjects\multi-agent-search
```

### 9.1 环境要求

- Windows
- 项目虚拟环境：`dsp`
- DeepSeek API Key 放在 `.env`：

```text
DEEPSEEK_API_KEY=你的密钥
```

不要把真实密钥写入代码、日志、PPT 或 Git。

### 9.2 运行多 Agent 主程序

```powershell
dsp\Scripts\python.exe multi_agent.py
```

会执行：

1. 打印 LangGraph ASCII 图。
2. 打印 RAG 后端。
3. 运行 mining + rag 的串行/并行 benchmark。
4. 运行四道演示题。
5. 输出日志到 `outputs/w6/run_log.txt`。

### 9.3 运行 W6 回归

```powershell
dsp\Scripts\python.exe regression_w6.py
```

输出：

- `outputs/w6/regression_results.csv`
- `outputs/w6/regression_summary.json`
- `outputs/w6/regression_log.txt`

### 9.3.1 运行信任层离线测试

```powershell
dsp\Scripts\python.exe -m unittest discover -s tests
```

10 个用例，不调用 LLM，秒级完成；改信任层后必须先跑这套再跑在线实验。

### 9.4 运行 W7 完整对比实验

```powershell
dsp\Scripts\python.exe experiment_w7.py
```

输出：

- `outputs/w7/experiment_results.csv`
- `outputs/w7/mode_summary.csv`
- `outputs/w7/paired_timings.csv`
- `outputs/w7/experiment_summary.json`
- `outputs/w7/experiment_log.txt`

### 9.5 运行单题实验

```powershell
dsp\Scripts\python.exe experiment_w7.py --cases R10
```

可指定多题：

```powershell
dsp\Scripts\python.exe experiment_w7.py --cases R01 R10 R13
```

---

## 10. 给下一位 AI 的审阅重点

上一轮的审阅发现（专家摘要洗数字、数字/对象错配、SQL 状态、覆盖口径）已按 `src/trust_layer.py` 的结构化 Fact 方案修复并有离线回归。请优先审阅以下新问题：

1. **审阅 `src/trust_layer.py` 的启发式边界**  
   `_local_context` 语义段切分、括号对收缩、别名长度加权都是启发式：能挡住已发现的误报，但可能漏掉更复杂的跨句绑定错误。评估是否需要升级为依存分析或更强的 NLP，还是课程场景下保持保守即可。

2. **检查 `experiment_w7.py` 的公平性**  
   单 Agent 无 Fact 引用机制导致其硬校验天然吃亏——这是量化信任层价值的对照组，但也请确认提示词、工具、验收指标没有其它系统性偏袒。

3. **评估 Critic 语义审查的收敛度**  
   R06 类“合法但偶发”的阻断是否可接受？修订上限 1 次是否过少？

4. **准备 W8 最小前端**  
   前端必须展示证据和审查状态（原始工具证据、Fact 引用、硬校验状态、阻断原因），而不是只显示最终答案。

5. **重复实验取中位数**  
   对 R05–R13 每题每模式 ≥3 次，替换单次运行数字后再写进报告。

---

## 11. 不应做的事

- 不要读取、打印、提交 `.env` 中的真实 API Key。
- 不要把 `models/`、`data/`、`dsp/` 提交进 Git。
- 不要删除用户保留的历史 notebook、输出文件和未跟踪实验文件，除非用户明确确认。
- 不要把外部 LLM 评审 PASS 当作事实正确性证明。
- 不要把 W6 的 10/10 与 W7 的严格原始证据审计结果混为一谈。
- 不要把 W6 benchmark 的加速比与 W7 全链路耗时直接比较，二者测量范围不同。
- 不要为了演示好看而删除失败案例；W7 的失败案例正是当前最有价值的研究发现。

---

## 12. 建议的下一步顺序

1. **（可选）调 Critic 修订上限与审查收敛度**：R06 类“合法但偶发”的语义阻断是否接受，需用户决定。
2. **补充核心题重复实验**：R05–R13 每题每模式 ≥3 次，报告中位数与波动区间，替换单次运行数字。
3. **开始 W8 Streamlit 最小界面**：必须展示证据链（原始工具证据、Fact 引用、硬校验状态、阻断原因），不是只显示最终答案；保留历史日志回放兜底。
4. **补 requirements.txt**：锁定 langchain / duckdb / sentence-transformers 等关键版本。
5. **准备现场演示**：使用已验证问题（R01/R04/R05/R10），预设失败案例讲解信任层价值。

---

## 13. 审阅基准与当前 Git 状态

最新提交：

```text
7c2eaf2 信任层: 结构化Fact审计+四状态工具+三层覆盖统计+阻断防泄露
5bfc922 W7: 单/多 Agent 对比实验与原始证据审计
```

前一阶段提交：

```text
eb61340 W6: 物理并行(async fan-out) + Critic 硬校验 + hybrid 语义检索 + 10 题回归全过
```

当前工作区仍存在与 W7 无关的历史修改和未跟踪文件，例如 notebook、旧实验输出、IDE 配置等。它们不属于本次 W7 提交，但不代表没有价值。审阅或整理前请先与用户确认。

---

## 14. 移交结论

项目已经完成了从数据层、算法层、多 Agent 编排层、对比实验层到程序级信任层的完整建设。当前系统已经能够：

- 查询真实 MovieLens 数据。
- 调用课程数据挖掘结果。
- 检索项目知识库。
- 生成结构化中文分析，数字由程序按 Fact 声明渲染。
- 对数字做出处、对象关系、格式声明三层硬校验，不通过即阻断且不留泄露。
- 记录执行轨迹和原始工具证据。
- 对单 Agent、多 Agent 串行、多 Agent 并行进行定量比较。

但项目还不能称为最终完成版，因为：

1. 前端演示界面尚未实现。
2. 实验重复次数不足，单次运行数字有波动。
3. 依赖未锁定，模型与数据未入库。

建议下一位 AI 从“重复实验取中位数”和“W8 前端”继续，信任层已可支撑现场演示的证据链展示。
