# -*- coding: utf-8 -*-
"""W4 任务 1 · Step 1-2：用户偏好事务（top-3 偏好签名）+ TransactionEncoder 编码

事务定义 v2（方法论决策，PPT 方法论页素材）：
  v1「看过」= 用户评过 >=1 部 4 分电影的类型
     → 人均 80 部 4 分电影几乎覆盖全部类型，主流类型 support 全部 >95%，无区分度
  v2「最偏好」= 每人 >=4 评分部数最多的前 3 个类型（偏好签名）
     → support 梯度拉开（Drama 0.77 / Comedy 0.54 / Action 0.48 ...），规则才有意义
  同计数并列时按类型字母序取，保证结果可复现。
"""
from pathlib import Path

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

DATA = Path(r"C:\Users\LEGION\PycharmProjects\multi-agent-search\data")

# ── Step 1：用户级偏好事务 ──
ratings = pd.read_csv(DATA / "ml-latest-small/ratings.csv")
movies = pd.read_csv(DATA / "ml-latest-small/movies.csv")

pref = ratings[ratings["rating"] >= 4.0]                     # ① 只留"偏好"：>=4 分
pref = pref.merge(movies[["movieId", "genres"]], on="movieId")
exp = pref.assign(genres=pref["genres"].str.split("|")).explode("genres")
exp = exp[exp["genres"] != "(no genres listed)"]             # ② 剔除无类型占位符

counts = exp.groupby(["userId", "genres"]).size().reset_index(name="n")
top3 = (counts.sort_values(["userId", "n", "genres"], ascending=[True, False, True])
        .groupby("userId").head(3))                          # ③ 每人取部数最多的前 3 个类型
transactions = top3.groupby("userId")["genres"].agg(list)

print(f"{len(transactions)} 条事务 | {pref.shape[0]} 条偏好评分 | 类型展开后 {exp.shape[0]} 行")
print("事务长度分布:", transactions.str.len().value_counts().to_dict())

# ── Step 2：TransactionEncoder 编码 → 609×19 布尔矩阵 ──
te = TransactionEncoder()
df = pd.DataFrame(te.fit(transactions).transform(transactions),
                  columns=te.columns_)
print(f"编码矩阵: {df.shape[0]} 行 × {df.shape[1]} 列")
print("类型偏好率（support 基线）:")
print(df.mean().sort_values(ascending=False).round(3).to_string())

for ms in [0.03, 0.05, 0.08, 0.10, 0.15]:   # 0.03 ≈ 18 人起步
    freq = apriori(df, min_support=ms, use_colnames=True, max_len=3)
    rules = association_rules(freq, metric="confidence", min_threshold=0.5)
    rules = rules[rules["lift"] > 1]
    print(ms, "项集:", len(freq), "规则:", len(rules))

# ── Step 4：最终挖掘（min_support=0.05，≥30 人）+ Top 规则落盘 + 出图 ──
# 选型依据：0.03 的榜首规则仅 20 人支撑（样本偏小）；0.05 全部规则 ≥30 人，
# 且模式成体系（Romance↔Comedy↔Drama / Sci-Fi→Action / Crime→Drama）
FINAL_MS = 0.05
freq = apriori(df, min_support=FINAL_MS, use_colnames=True, max_len=3)
rules = association_rules(freq, metric="confidence", min_threshold=0.5)
rules = rules[rules["lift"] > 1].sort_values("lift", ascending=False).reset_index(drop=True)

top = rules.head(10).copy()
top["antecedents"] = top["antecedents"].apply(lambda x: "+".join(sorted(x)))
top["consequents"] = top["consequents"].apply(lambda x: "+".join(sorted(x)))
print(f"\n=== Top 规则（min_support={FINAL_MS}，共 {len(rules)} 条，按 lift 排序）===")
print(top[["antecedents", "consequents", "support", "confidence", "lift"]]
      .round(3).to_string(index=False))

out = Path(r"C:\Users\LEGION\PycharmProjects\multi-agent-search\outputs\w4")
out.mkdir(parents=True, exist_ok=True)
rules.assign(antecedents=rules["antecedents"].apply(lambda x: "+".join(sorted(x))),
             consequents=rules["consequents"].apply(lambda x: "+".join(sorted(x)))
             )[["antecedents", "consequents", "support", "confidence", "lift"]] \
    .round(4).to_csv(out / "top_rules.csv", index=False, encoding="utf-8-sig")

import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 图 1：support-confidence 散点，气泡大小 = lift
fig, ax = plt.subplots(figsize=(7, 5))
sc = ax.scatter(rules["support"], rules["confidence"],
                s=rules["lift"] * 120, alpha=0.55,
                c=rules["lift"], cmap="viridis", edgecolors="grey", linewidths=0.5)
ax.set_xlabel("support（支持度）"); ax.set_ylabel("confidence（置信度）")
ax.set_title(f"关联规则分布（min_support={FINAL_MS}，气泡大小/颜色 = lift）")
fig.colorbar(sc, label="lift")
fig.tight_layout(); fig.savefig(out / "rules_scatter.png", dpi=180); plt.close(fig)

# 图 2：Top10 规则按 lift 横向条形图
fig, ax = plt.subplots(figsize=(8, 5))
labels = [f"{a} → {c}" for a, c in zip(top["antecedents"], top["consequents"])][::-1]
ax.barh(labels, top["lift"][::-1], color="#378ADD")
ax.axvline(1.0, color="red", linestyle="--", linewidth=1, label="lift=1（无关联）")
ax.set_xlabel("lift（提升度）"); ax.set_title("Top10 关联规则（按 lift 排序）")
ax.legend()
fig.tight_layout(); fig.savefig(out / "top_rules_bar.png", dpi=180); plt.close(fig)

print(f"\n已落盘: {out / 'top_rules.csv'} | rules_scatter.png | top_rules_bar.png")

