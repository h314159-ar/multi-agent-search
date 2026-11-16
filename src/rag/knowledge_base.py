r"""W6 · RAG 知识库与可切换检索器

知识来源：W1-W5 已验证的项目结论，每条文档都对应一次真实实验。

检索后端：
- tfidf：TF-IDF + 字符 n-gram，W5 基线。
- semantic：本地 BGE 中文向量（BAAI/bge-small-zh-v1.5）。
- hybrid：语义得分与 TF-IDF 得分加权融合，兼顾语义与专有名词精确匹配。

模型首次加载需联网下载，之后使用本地缓存；加载失败会明确记录原因并
回退 TF-IDF，不会把回退结果伪装成语义检索。

运行：dsp\Scripts\python src/rag/knowledge_base.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
LOCAL_MODEL_DIR = ROOT / "models" / "bge-small-zh-v1.5"
# 已从 hf-mirror 预下载到本地时直接离线加载，避免运行时依赖网络
MODEL_NAME = str(LOCAL_MODEL_DIR) if (LOCAL_MODEL_DIR / "model.safetensors").exists() \
    else DEFAULT_MODEL
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

# ── 知识库：每条 = (文档ID, 主题, 正文)。正文全部来自实测结论 ──
DOCUMENTS: list[tuple[str, str, str]] = [
    ("dataset_overview", "数据集概况",
     "MovieLens ml-latest-small 数据集：100,836 条评分，610 名用户，9,742 部电影，"
     "评分范围 0.5-5.0 分。用户-电影评分矩阵稀疏度 98.30%。"
     "评分时间跨度 1996-03-29 至 2018-09-24，共 22 年真实行为数据。"
     "数据干净无缺失值，仅 34 部电影标记为 (no genres listed)。"),
    ("schema_ratings", "ratings 表数据字典",
     "DuckDB 数据库 data/movielens.duckdb 中 ratings 表有 6 列："
     "userId 用户ID，movieId 电影ID，rating 评分 0.5-5.0，timestamp Unix 时间戳，"
     "rated_at 评分时间 DATETIME，rated_year 评分年份整数。共 100,836 行。"),
    ("schema_movies", "movies 表数据字典",
     "movies 表有 movieId、title（英文电影名含年份）和 20 个类型 0/1 标记列："
     "Action Adventure Animation Children Comedy Crime Documentary Drama Fantasy "
     "Film-Noir Horror IMAX Musical Mystery Romance Sci-Fi Thriller War Western "
     "以及 (no genres listed)。多热编码由 genres 字段按竖线拆分得到。共 9,742 行。"),
    ("eda_longtail", "EDA 长尾分布结论",
     "W1 EDA 发现：用户评分行为呈明显长尾分布。人均评分 165 条，中位数 70.5 条，"
     "最高 2698 条。少数重度用户贡献大量评分，这是后续 K-Means 用户分群的动机。"
     "高分经典片霸榜：肖申克的救赎平均 4.43 分（317 人评价）居首。"),
    ("kmeans_result", "K-Means 用户分群结果",
     "W3 K-Means 用户分群：最佳 K=2，轮廓系数 0.4077，在 K=2 到 8 中最高。"
     "轻度用户 532 人占 87.2%，平均评分 106.9 条、活跃 1.1 年、平均评分 3.67；"
     "重度用户 78 人占 12.8%，平均评分 563.9 条、活跃 5.6 年、平均评分 3.58。"
     "两组平均分接近，说明分群识别的是活跃度差异而非高分低分用户。"
     "约 13% 的重度用户贡献约 44% 的评分，定量复现二八结构。"),
    ("kmeans_method", "K-Means 特征工程方法",
     "聚类特征 4 个：log_n_ratings（评分数量 log1p 压缩长尾）、mean_rating、"
     "std_rating、log_active_years（活跃年数 log1p）。必须 StandardScaler 标准化，"
     "因为 K-Means 基于欧氏距离，量纲差异会主导距离计算。"
     "PCA 只用于把 4 维特征投影到 2 维可视化，不参与训练：先降维再聚类会丢失"
     "低方差但有区分度的维度，且聚类中心失去业务可解释性。"
     "K 的下界 2 因为 K=1 无聚类意义且轮廓系数无法定义；上界 8 考虑业务可解释性。"),
    ("apriori_result", "Apriori 关联规则结果",
     "W4 Apriori 关联规则：min_support=0.05 时挖出 13 条规则（lift>1 且 confidence>=0.5）。"
     "典型规则：Romance 推出 Comedy+Drama，confidence 0.78 lift 1.79，83 人支撑；"
     "Sci-Fi 推出 Action，confidence 0.80 lift 1.68，40 人支撑；"
     "Crime 推出 Drama，confidence 0.94 lift 1.21。情感片三角与科幻动作绑定是主要模式。"),
    ("apriori_method", "Apriori 事务构建方法",
     "关联规则事务定义：每位用户取评分>=4 的电影中类型出现次数最多的前 3 个类型，"
     "形成偏好签名，共 609 条事务（1 名用户 userId=442 最高评分仅 2.5 被过滤）。"
     "v1 定义「看过至少 1 部 4 分片的所有类型」失败：人均 80 部 4 分片几乎覆盖全部类型，"
     "主流类型 support 超 95% 无区分度。改为 top-3 偏好后 support 梯度拉开。"
     "规则按 lift 而非 confidence 排序：lift=confidence/P(B)，大于 1 才是正相关，"
     "高 confidence 可能只是后项本身热门。支撑人数=support 乘以事务数 609。"),
    ("group_rules", "分组关联规则对比结果",
     "W4 按 K-Means 分群分组挖掘：重度组 78 人 13 条规则，轻度组 531 人 13 条规则。"
     "重度用户信号更纯：Romance 推出 Comedy 在重度组 confidence=1.000，轻度组 0.811。"
     "组独有规则：重度组有 Crime 复合规则族，轻度组有 Thriller 系规则。"
     "核心规则（情感三角、Sci-Fi 推 Action）跨组稳定，验证全量规则稳健性。"
     "注意重度组每条规则仅 5-14 人支撑，属小样本探索性发现。"),
    ("decision_tree", "决策树分类结果",
     "W4 决策树分类任务：预测用户是否宽容型评分者（平均评分>=3.5）。"
     "5 折交叉验证准确率 70.3% 正负 1.7%，多数类基线 65.2%，真实提升约 5 个百分点。"
     "测试集准确率 66.4%（n=122）。特征重要性：std_rating 0.58 主导，"
     "log_n_ratings 0.30，first_year 0.11。规律：评分波动大（爱憎分明）的用户"
     "平均分更低更严格，且更活跃。局限：纯行为特征对评分严格度解释力有限，"
     "行为数据中没有品味维度，这是引入 LLM 语义层的动机。"),
    ("leakage_guard", "防标签泄漏方法",
     "决策树防泄漏设计：mean_rating 只用于构造目标 y（>=3.5 为宽容型），"
     "绝不进入特征 X，否则准确率虚高构成循环论证。特征 5 个：log_n_ratings、"
     "std_rating、log_active_years、first_year、rating_freq。"
     "树模型按分裂点工作对量纲不敏感，不需要标准化，与 K-Means 形成对照。"
     "使用 stratify 分层划分防止类别失衡，max_depth=4 限制深度防过拟合。"),
    ("architecture", "多智能体系统架构",
     "系统六个 Agent：Planner 规划路由、数据检索 Agent 执行 DuckDB SQL、"
     "挖掘 Agent 调用 K-Means/Apriori/决策树工具、RAG Agent 检索知识库、"
     "Synthesizer 综合多源证据生成报告、Critic 审查数据支撑与逻辑。"
     "LangGraph 中 State 是共享字典，Node 是 Agent，条件边是路由规则。"
     "工具调用结果必须包成带 tool_call_id 的 ToolMessage 才能配对回 assistant。"),
    ("w5_orchestration", "W5 编排层实现与踩坑",
     "W5 用 LangGraph 把六角色组成完整系统。Planner 输出 JSON 路由，条件边按路由扇出；"
     "多个专家写同一个 State 字段会触发 InvalidUpdateError，因此 evidences 使用字典合并 "
     "reducer，trace 使用 operator.add 列表累加，每个分支只返回增量。"
     "RAG 先做查询重写，再把综合长问题拆成 1 到 3 个子查询分别检索，按 doc_id 合并最高分。"
     "Synthesizer 只能根据专家证据写作，禁止新增数字；Critic 不通过则带意见回流，"
     "最多修订 1 次以防无限循环。W5 的扇出是逻辑并行，同步 SDK 下物理上仍串行。"),
]


class KnowledgeBase:
    """可切换 TF-IDF / 本地语义向量 / 混合检索；向量只编码一次。"""

    def __init__(self, documents: list[tuple[str, str, str]] = DOCUMENTS,
                 mode: str = "hybrid", model_name: str = MODEL_NAME,
                 embedder=None, allow_fallback: bool = True):
        if mode not in ("tfidf", "semantic", "hybrid"):
            raise ValueError(f"未知检索模式: {mode}")
        self.ids = [d[0] for d in documents]
        self.topics = [d[1] for d in documents]
        self.texts = [f"{d[1]}。{d[2]}" for d in documents]
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        self.matrix = self.vectorizer.fit_transform(self.texts)
        self.requested_mode = mode
        self.mode = "tfidf"
        self.model_name = model_name
        self.embedder = None
        self.doc_embeddings = None
        self.fallback_reason = ""

        if mode in ("semantic", "hybrid"):
            try:
                if embedder is None:
                    from sentence_transformers import SentenceTransformer
                    embedder = SentenceTransformer(model_name)
                self.embedder = embedder
                self.doc_embeddings = np.asarray(embedder.encode(
                    self.texts, normalize_embeddings=True, show_progress_bar=False))
                self.mode = mode
            except Exception as exc:
                self.fallback_reason = f"{type(exc).__name__}: {exc}"
                if not allow_fallback:
                    raise

    def retrieve(self, query: str, top_k: int = 3) -> pd.DataFrame:
        """返回 top_k 条文档：doc_id / topic / score / text。"""
        q = self.vectorizer.transform([query])
        lexical = cosine_similarity(q, self.matrix).ravel()
        if self.mode in ("semantic", "hybrid"):
            query_embedding = np.asarray(self.embedder.encode(
                [QUERY_INSTRUCTION + query], normalize_embeddings=True,
                show_progress_bar=False))
            semantic = cosine_similarity(query_embedding, self.doc_embeddings).ravel()
            if self.mode == "semantic":
                scores = semantic
            else:
                # 语义为主，保留字面得分以照顾算法名、字段名等精确匹配。
                scores = 0.35 * lexical + 0.65 * semantic
        else:
            scores = lexical
        order = scores.argsort()[::-1][:top_k]
        return pd.DataFrame({
            "doc_id": [self.ids[i] for i in order],
            "topic": [self.topics[i] for i in order],
            "score": scores[order].round(4),
            "text": [self.texts[i] for i in order],
        })


if __name__ == "__main__":
    kb = KnowledgeBase()
    print(f"知识库：{len(kb.ids)} 条文档 | 请求模式：{kb.requested_mode} | "
          f"实际模式：{kb.mode}")
    if kb.fallback_reason:
        print(f"语义模型回退原因：{kb.fallback_reason}")
    print(f"TF-IDF 特征维度：{kb.matrix.shape[1]}")
    for q in ["为什么 K-Means 选 K=2？",
              "关联规则怎么构建事务？",
              "决策树的准确率是多少？",
              "ratings 表有哪些字段？",
              "并行分支写 State 冲突怎么解决？"]:
        print(f"\n问：{q}")
        hits = kb.retrieve(q, top_k=2)
        for _, r in hits.iterrows():
            print(f"  [{r['score']:.3f}] {r['doc_id']} · {r['topic']}")
