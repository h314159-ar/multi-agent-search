# src/mining/group_rules.py
"""
W4 任务 2 · 分组关联规则对比（本周 PPT 亮点）。

复用 mine_rules.py 工具模块：按 W3 聚类标签（重度 78 / 轻度 532）切分用户，
各自构建偏好事务并挖掘关联规则，并排对比两组 Top 规则。

方法论要点（PPT 素材）：
  - min_support=0.05 是相对阈值（事务占比），对 78 人与 532 人同样公平：
    重度组 ≈ ≥4 人支撑，轻度组 ≈ ≥27 人支撑。
  - 但重度组 4 人支撑的规则统计上较弱，解读时需注明（小样本警示）。

输入: data/ml-latest-small/*.csv + outputs/w3/user_clusters.csv
输出: outputs/w4/group_rules_heavy.csv / group_rules_light.csv
      outputs/w4/group_rules_compare.png（双面板对比图）
"""
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))          # 同目录导入工具模块
from mine_rules import (load_data, build_transactions, mine_rules,
                        format_rules, FINAL_MS)

ROOT = Path(r"C:\Users\LEGION\PycharmProjects\multi-agent-search")
OUT = ROOT / "outputs" / "w4"
CLUSTERS = ROOT / "outputs" / "w3" / "user_clusters.csv"
TOP_SHOW = 8          # 每组展示/画图的前 N 条规则


def mine_group(ratings: pd.DataFrame, movies: pd.DataFrame,
               user_ids: pd.Index, label: str) -> pd.DataFrame:
    """对一组用户：过滤评分 → 构建事务 → 挖掘规则，打印摘要并返回规则表。"""
    sub = ratings[ratings["userId"].isin(user_ids)]
    transactions = build_transactions(sub, movies)
    rules = mine_rules(transactions, min_support=FINAL_MS)

    print(f"\n=== {label} | {len(transactions)} 条事务 | {len(rules)} 条规则 ===")
    if rules.empty:
        print("  （无满足条件的规则）")
    else:
        print(format_rules(rules.head(TOP_SHOW))
              [["antecedents", "consequents", "support", "confidence", "lift"]]
              .round(3).to_string(index=False))
    return rules


def compare(groups: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """交叉对比：每条规则在哪些组出现、各组 lift 差异。返回汇总表。"""
    frames = []
    for label, rules in groups.items():
        if rules.empty:
            continue
        f = format_rules(rules).copy()
        f["rule"] = f["antecedents"] + " → " + f["consequents"]
        f["组"] = label
        frames.append(f[["rule", "组", "support", "confidence", "lift"]])
    if not frames:
        return pd.DataFrame()

    all_rules = pd.concat(frames, ignore_index=True)
    pivot = all_rules.pivot_table(index="rule", columns="组",
                                  values="lift", aggfunc="first")
    counts = all_rules.groupby("rule")["组"].apply(list)
    pivot["出现于"] = counts.apply(lambda g: "+".join(sorted(g)))

    summary = (all_rules.groupby("rule")
               .agg(最高lift=("lift", "max"), 总出现次数=("组", "size")))
    summary = summary.join(pivot["出现于"]).sort_values("最高lift", ascending=False)

    print(f"\n=== 交叉对比（规则 × 组，按最高 lift 排序）===")
    print(summary.round(3).to_string())
    return summary


def plot_compare(groups: dict[str, pd.DataFrame], out_png: Path) -> None:
    """双面板：两组 Top 规则按 lift 的条形图（PPT 主图）。"""
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    colors = {"重度用户(78人)": "#D9534F", "轻度用户(532人)": "#378ADD"}

    fig, axes = plt.subplots(1, len(groups), figsize=(7 * len(groups), 5.5))
    axes = axes if len(groups) > 1 else [axes]

    for ax, (label, rules) in zip(axes, groups.items()):
        if rules.empty:
            ax.text(0.5, 0.5, "无规则", ha="center", va="center",
                    transform=ax.transAxes); ax.set_title(label); continue
        top = format_rules(rules.head(TOP_SHOW))
        labels = [f"{a} → {c}" for a, c in
                  zip(top["antecedents"], top["consequents"])][::-1]
        ax.barh(labels, top["lift"][::-1], color=colors.get(label, "#888888"))
        ax.axvline(1.0, color="grey", linestyle="--", linewidth=1)
        ax.set_xlabel("lift"); ax.set_title(f"{label} · Top{TOP_SHOW} 规则")

    fig.suptitle(f"分群关联规则对比（min_support={FINAL_MS}，灰色虚线 lift=1）",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=180); plt.close(fig)


def main():
    ratings, movies = load_data()
    clusters = pd.read_csv(CLUSTERS)
    # cluster_name 已含"重度/轻度"字样；加人数后缀便于图题与打印对齐
    name_map = (clusters.groupby(["cluster", "cluster_name"]).size()
                .reset_index(name="n"))
    name_map["label"] = (name_map["cluster_name"] + "(" + name_map["n"].astype(str) + "人)")

    groups: dict[str, pd.DataFrame] = {}
    for _, row in name_map.sort_values("cluster", ascending=False).iterrows():
        ids = clusters.loc[clusters["cluster"] == row["cluster"], "userId"]
        label = row["label"]
        groups[label] = mine_group(ratings, movies, ids, label)
        if not groups[label].empty:
            out_csv = OUT / f"group_rules_{'heavy' if row['cluster'] == 1 else 'light'}.csv"
            from mine_rules import save_rules
            save_rules(groups[label], OUT, out_csv.name)
            print(f"  已落盘: {out_csv.name}")

    summary = compare(groups)
    if not summary.empty:
        summary.round(4).to_csv(OUT / "group_rules_summary.csv",
                                index=False, encoding="utf-8-sig")

    png = OUT / "group_rules_compare.png"
    plot_compare(groups, png)
    print(f"\n已落盘: group_rules_summary.csv | {png.name}")


if __name__ == "__main__":
    main()
