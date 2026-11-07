r"""W5 · 挖掘工具统一接口（W4 任务 4 收尾）

设计原则：K-Means / Apriori / 决策树的训练已在 W3/W4 完成并落盘，
本模块只做「读取 + 格式化」，签名统一为 (参数) -> str，
供挖掘 Agent 作为 LangChain 工具直接调用，秒级返回、结果可复现。

运行：dsp\Scripts\python src/mining/mining_tools.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
W3 = ROOT / "outputs" / "w3"
W4 = ROOT / "outputs" / "w4"


def get_cluster_profiles() -> str:
    """返回 K-Means 用户分群画像（W3 实测结果）。"""
    profile = pd.read_csv(W3 / "cluster_profiles.csv")
    metrics = pd.read_csv(W3 / "k_metrics.csv")
    best_k = int(metrics.loc[metrics["silhouette"].idxmax(), "k"])
    best_sil = metrics["silhouette"].max()
    cols = ["cluster_name", "user_count", "user_ratio", "avg_ratings",
            "avg_active_years", "avg_rating"]
    head = (f"K-Means 分群结果：K={best_k}，轮廓系数={best_sil:.4f}\n"
            f"K 扫描范围 2~8，{best_k} 为轮廓系数最高点\n")
    return head + profile[cols].round(3).to_string(index=False)


def get_association_rules(top_n: int = 5, group: str = "all") -> str:
    """返回 Apriori 关联规则 Top-N。group: all / heavy / light。"""
    if group == "all":
        rules = pd.read_csv(W4 / "top_rules.csv")
        title = "全量用户关联规则（min_support=0.05，609 条事务）"
    else:
        rules = pd.read_csv(W4 / f"group_rules_{group}.csv")
        title = f"{'重度' if group == 'heavy' else '轻度'}用户组关联规则"
    cols = ["antecedents", "consequents", "support", "confidence", "lift"]
    return title + "\n" + rules[cols].head(top_n).round(4).to_string(index=False)


def get_decision_tree_summary() -> str:
    """返回决策树分类任务结论（W4 实测结果）。"""
    return (
        "决策树任务：预测用户是否宽容型评分者（mean_rating>=3.5）\n"
        "5 折交叉验证准确率：70.3% ± 1.7%（多数类基线 65.2%，提升约 5pp）\n"
        "测试集准确率 66.4%（n=122），树深 4，叶节点 14\n"
        "特征重要性：std_rating 0.58 > log_n_ratings 0.30 > first_year 0.11\n"
        "规律：评分波动大（爱憎分明）的用户更严格、更活跃\n"
        "局限：纯行为特征对评分严格度解释力有限，行为数据中没有「品味」维度"
    )


def get_group_rule_comparison() -> str:
    """返回重度/轻度两组规则的交叉对比要点。"""
    summary = pd.read_csv(W4 / "group_rules_summary.csv")
    return (f"分组规则交叉对比（{len(summary)} 条规则 × 组）\n"
            + summary.head(12).to_string(index=False))


if __name__ == "__main__":
    print("=== 工具 1：用户分群 ===")
    print(get_cluster_profiles())
    print("\n=== 工具 2：关联规则（全量 Top5）===")
    print(get_association_rules())
    print("\n=== 工具 3：关联规则（重度组 Top3）===")
    print(get_association_rules(top_n=3, group="heavy"))
    print("\n=== 工具 4：决策树摘要 ===")
    print(get_decision_tree_summary())
