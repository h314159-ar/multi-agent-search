# src/mining/decision_tree.py
"""
W4 任务 3 · 决策树分类：什么样的用户更爱打高分？

目标变量（独立于 W3 聚类标签，避免循环论证）:
    y = 用户平均评分 >= 3.5 → "宽容型"，否则 "严格型"
特征（5 个；mean_rating 只做 y、绝不进 X —— 防标签泄漏）:
    log_n_ratings / std_rating / log_active_years / first_year / rating_freq

树模型按分裂点工作、对量纲不敏感，故不做标准化（方法论点，PPT 可讲）。

输入: outputs/w3/user_clusters.csv（W3 产出，含特征原字段）
输出: outputs/w4/tree_model.png / feature_importance.png
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, classification_report

ROOT = Path(r"C:\Users\LEGION\PycharmProjects\multi-agent-search")
SRC = ROOT / "outputs" / "w3" / "user_clusters.csv"
OUT = ROOT / "outputs" / "w4"

THRESHOLD = 3.5        # 宽容型判定线（MovieLens 半分制的中间档）
MAX_DEPTH = 4          # 限制深度防过拟合（610 样本 / 5 特征）
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV = 5
CLASS_NAMES = ["严格型(<3.5)", "宽容型(>=3.5)"]
FEATURES = ["log_n_ratings", "std_rating", "log_active_years",
            "first_year", "rating_freq"]


def build_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """特征矩阵 X（5 列，全部可由行为数据计算）与目标 y。"""
    assert "mean_rating" not in FEATURES          # 防泄漏自查
    X = pd.DataFrame({
        "log_n_ratings": np.log1p(df["n_ratings"]),
        "std_rating": df["std_rating"],
        "log_active_years": np.log1p(df["active_years"]),
        "first_year": df["first_year"],
        "rating_freq": df["n_ratings"] / df["active_years"].clip(lower=1),
    })
    y = (df["mean_rating"] >= THRESHOLD).astype(int)   # 1=宽容型
    return X, y


def main():
    df = pd.read_csv(SRC)
    X, y = build_xy(df)

    # ── 类别分布与基线 ──
    counts = y.value_counts().sort_index()
    baseline = counts.max() / len(y)
    print("=== 类别分布 ===")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {counts[i]} 人 ({counts[i]/len(y):.1%})")
    print(f"多数类基线: {baseline:.1%}（模型准确率须显著高于它才有意义）")

    # ── 5 折交叉验证（全量数据，分层抽样由 sklearn 默认保证）──
    clf = DecisionTreeClassifier(max_depth=MAX_DEPTH, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(clf, X, y, cv=CV)
    print(f"\n=== 5 折交叉验证 ===\n  各折: {np.round(cv_scores, 4).tolist()}")
    print(f"  均值 ± 标准差: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── 留出法评估（混淆矩阵 + 分类报告）──
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)

    print(f"\n=== 测试集评估（n={len(y_te)}）===")
    cm = confusion_matrix(y_te, y_pred)
    print("混淆矩阵（行=真实，列=预测）:")
    print(pd.DataFrame(cm, index=[f"真·{n}" for n in CLASS_NAMES],
                       columns=[f"预·{n}" for n in CLASS_NAMES]).to_string())
    print(classification_report(y_te, y_pred, target_names=CLASS_NAMES, digits=3))
    print(f"树深: {clf.get_depth()} | 叶节点: {clf.get_n_leaves()}")

    # ── 特征重要性 ──
    imp = pd.Series(clf.feature_importances_, index=X.columns) \
            .sort_values(ascending=True)
    print("\n=== 特征重要性（基尼）===")
    print(imp.sort_values(ascending=False).round(4).to_string())

    # ── 出图 ──
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    OUT.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 8))
    plot_tree(clf, feature_names=list(X.columns), class_names=CLASS_NAMES,
              filled=True, rounded=True, fontsize=9, ax=ax)
    ax.set_title(f"决策树（max_depth={MAX_DEPTH}，测试集准确率 "
                 f"{(y_pred == y_te).mean():.1%}）", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "tree_model.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(imp.index, imp.values, color="#1D9E75")
    ax.set_xlabel("基尼重要性")
    ax.set_title("特征重要性排序（越高越能区分宽容/严格型评分者）")
    fig.tight_layout()
    fig.savefig(OUT / "feature_importance.png", dpi=180)
    plt.close(fig)

    print(f"\n已落盘: {OUT/'tree_model.png'} | {OUT/'feature_importance.png'}")


if __name__ == "__main__":
    main()
