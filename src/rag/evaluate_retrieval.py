r"""W6 · RAG 检索后端公平对比：TF-IDF vs BGE 语义向量 vs 混合检索。

指标：Top-1 命中率、Recall@3、MRR。相关文档由人工按问题主题标注。
运行：dsp\Scripts\python src/rag/evaluate_retrieval.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "rag"))
from knowledge_base import KnowledgeBase, DEFAULT_MODEL

OUT = ROOT / "outputs" / "w6"
OUT.mkdir(parents=True, exist_ok=True)

EVAL_CASES = [
    ("Q1", "为什么 K-Means 选 K=2？", ["kmeans_result", "kmeans_method"]),
    ("Q2", "关联规则怎么构建事务？", ["apriori_method"]),
    ("Q3", "决策树的准确率是多少？", ["decision_tree"]),
    ("Q4", "ratings 表有哪些字段？", ["schema_ratings"]),
    ("Q5", "并行分支同时写 State 冲突怎么解决？", ["w5_orchestration", "architecture"]),
    ("Q6", "给我一份用户偏好模式简报，并说明挖掘方法。",
     ["kmeans_result", "apriori_result", "decision_tree", "w5_orchestration"]),
    ("Q7", "评分严格度能否由用户行为预测？", ["decision_tree", "leakage_guard"]),
    ("Q8", "为什么 evidences 需要 reducer？", ["w5_orchestration", "architecture"]),
]


def evaluate(kb: KnowledgeBase, label: str) -> tuple[list[dict], dict]:
    rows = []
    for case_id, query, relevant in EVAL_CASES:
        hits = kb.retrieve(query, top_k=3)
        retrieved = hits["doc_id"].tolist()
        ranks = {doc_id: i + 1 for i, doc_id in enumerate(retrieved)}
        relevant_set = set(relevant)
        first_rank = min((ranks[d] for d in relevant_set if d in ranks), default=None)
        rows.append({
            "case_id": case_id,
            "mode": label,
            "actual_backend": kb.mode,
            "question": query,
            "relevant_docs": "|".join(relevant),
            "top1": retrieved[0],
            "top3": "|".join(retrieved),
            "top1_hit": int(retrieved[0] in relevant_set),
            "recall_at_3": len(relevant_set & set(retrieved)) / len(relevant_set),
            "mrr": 1 / first_rank if first_rank else 0.0,
        })
    frame = pd.DataFrame(rows)
    summary = {
        "mode": label,
        "actual_backend": kb.mode,
        "top1_accuracy": round(float(frame["top1_hit"].mean()), 4),
        "recall_at_3": round(float(frame["recall_at_3"].mean()), 4),
        "mrr": round(float(frame["mrr"].mean()), 4),
    }
    return rows, summary


def main():
    all_rows, summaries = [], []
    tfidf = KnowledgeBase(mode="tfidf")
    rows, summary = evaluate(tfidf, "tfidf")
    all_rows += rows
    summaries.append(summary)

    semantic_error = ""
    try:
        semantic = KnowledgeBase(mode="semantic", allow_fallback=False)
        for label, kb in [("semantic", semantic),
                          ("hybrid", KnowledgeBase(mode="hybrid",
                                                   embedder=semantic.embedder))]:
            rows, summary = evaluate(kb, label)
            all_rows += rows
            summaries.append(summary)
    except Exception as exc:
        semantic_error = f"{type(exc).__name__}: {exc}"
        print(f"语义模型不可用：{semantic_error}")

    detail = pd.DataFrame(all_rows)
    detail.to_csv(OUT / "retrieval_evaluation.csv", index=False, encoding="utf-8-sig")
    report = {
        "model": DEFAULT_MODEL,
        "document_count": len(tfidf.ids),
        "query_count": len(EVAL_CASES),
        "semantic_error": semantic_error,
        "summary": summaries,
    }
    with (OUT / "retrieval_summary.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"知识文档 {len(tfidf.ids)} 条 | 测试查询 {len(EVAL_CASES)} 条 | 模型 {DEFAULT_MODEL}")
    print(pd.DataFrame(summaries)[["mode", "top1_accuracy", "recall_at_3", "mrr"]]
          .to_string(index=False))
    for row in all_rows:
        if row["case_id"] == "Q6":
            print(f"综合题 {row['mode']:8s} Top3：{row['top3']}")
    print(f"评估结果已保存：{OUT / 'retrieval_summary.json'}")


if __name__ == "__main__":
    main()
