# src/mining/mine_rules.py
"""
MovieLens 类型关联规则挖掘。

输入: ml-latest-small/ratings.csv + movies.csv
输出: outputs/w4/top_rules.csv + 两张图
"""
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# ── 常量区：所有魔法值集中，main 里不再出现数字 ──
DATA = Path(r"C:\Users\LEGION\PycharmProjects\multi-agent-search\data")
OUT = Path(r"C:\Users\LEGION\PycharmProjects\multi-agent-search\outputs\w4")
RATING_THRESHOLD = 4.0
TOP_K_GENRES = 3
NO_GENRE = "(no genres listed)"
FINAL_MS = 0.05
MIN_CONF = 0.5
MAX_LEN = 3
SCAN_MS = [0.03, 0.05, 0.08, 0.10, 0.15]


# ── Step 1a：加载 ──
def load_data(data_dir: Path = DATA) -> tuple[pd.DataFrame, pd.DataFrame]:
    ratings = pd.read_csv(data_dir / "ml-latest-small/ratings.csv")
    movies = pd.read_csv(data_dir / "ml-latest-small/movies.csv")
    return ratings, movies


# ── Step 1b：构建事务（保留原脚本"每人取前 3 个最多类型"的逻辑）──
def build_transactions(ratings: pd.DataFrame, movies: pd.DataFrame,
                       threshold: float = RATING_THRESHOLD,
                       top_k: int = TOP_K_GENRES) -> pd.Series:
    """
    返回 Series[userId → list[str]]，每个用户只保留出现次数最多的前 top_k 个类型。
    """
    pref = ratings[ratings["rating"] >= threshold]
    pref = pref.merge(movies[["movieId", "genres"]], on="movieId")

    exp = pref.assign(genres=pref["genres"].str.split("|")).explode("genres")
    exp = exp[exp["genres"] != NO_GENRE]

    counts = exp.groupby(["userId", "genres"]).size().reset_index(name="n")
    topk = (counts.sort_values(["userId", "n", "genres"],
                               ascending=[True, False, True])
                  .groupby("userId").head(top_k))

    return topk.groupby("userId")["genres"].agg(list)


# ── Step 2：编码 ──
def encode_transactions(transactions: pd.Series) -> pd.DataFrame:
    """list[str] → 0/1 布尔矩阵，行索引保留 userId。"""
    te = TransactionEncoder()
    arr = te.fit(transactions).transform(transactions)
    return pd.DataFrame(arr, columns=te.columns_, index=transactions.index)


# ── Step 3+4：挖掘（一个函数管"编码→频繁项集→规则"）──
def mine_rules(transactions: pd.Series, min_support: float = FINAL_MS,
               min_confidence: float = MIN_CONF,
               max_len: int = MAX_LEN) -> pd.DataFrame:
    """
    返回按 lift 降序、lift>1 的规则表；无规则时返回空 DataFrame。
    """
    if len(transactions) == 0:
        return pd.DataFrame()

    df = encode_transactions(transactions)
    freq = apriori(df, min_support=min_support, use_colnames=True, max_len=max_len)
    if freq.empty:
        return pd.DataFrame()

    rules = association_rules(freq, metric="confidence",
                              min_threshold=min_confidence)
    rules = rules[rules["lift"] > 1]
    return rules.sort_values("lift", ascending=False).reset_index(drop=True)


# ── 辅助：frozenset → "A+B"，避免重复写 lambda ──
def format_rules(rules: pd.DataFrame) -> pd.DataFrame:
    out = rules.copy()
    out["antecedents"] = out["antecedents"].apply(lambda x: "+".join(sorted(x)))
    out["consequents"] = out["consequents"].apply(lambda x: "+".join(sorted(x)))
    return out


# ── 诊断：support 扫描 ──
def scan_support(transactions: pd.Series, supports=SCAN_MS) -> None:
    for ms in supports:
        rules = mine_rules(transactions, min_support=ms)
        print(f"  min_support={ms:.2f} → {len(rules)} 条规则")


# ── 落盘 ──
def save_rules(rules: pd.DataFrame, out_dir: Path = OUT,
               name: str = "top_rules.csv") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    (format_rules(rules)
     [["antecedents", "consequents", "support", "confidence", "lift"]]
     .round(4).to_csv(path, index=False, encoding="utf-8-sig"))
    return path


# ── 出图 ──
def plot_rules(rules: pd.DataFrame, out_dir: Path = OUT) -> None:
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    out_dir.mkdir(parents=True, exist_ok=True)

    # 图 1：散点
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(rules["support"], rules["confidence"],
                    s=rules["lift"] * 120, alpha=0.55,
                    c=rules["lift"], cmap="viridis",
                    edgecolors="grey", linewidths=0.5)
    ax.set_xlabel("support（支持度）"); ax.set_ylabel("confidence（置信度）")
    ax.set_title("关联规则分布（气泡大小/颜色 = lift）")
    fig.colorbar(sc, label="lift")
    fig.tight_layout(); fig.savefig(out_dir / "rules_scatter.png", dpi=180); plt.close(fig)

    # 图 2：Top10 条形
    top = format_rules(rules.head(10))
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [f"{a} → {c}" for a, c in zip(top["antecedents"], top["consequents"])][::-1]
    ax.barh(labels, top["lift"][::-1], color="#378ADD")
    ax.axvline(1.0, color="red", linestyle="--", linewidth=1, label="lift=1（无关联）")
    ax.set_xlabel("lift（提升度）"); ax.set_title("Top10 关联规则（按 lift 排序）")
    ax.legend()
    fig.tight_layout(); fig.savefig(out_dir / "top_rules_bar.png", dpi=180); plt.close(fig)


# ── 主流程 ──
def main():
    ratings, movies = load_data()
    transactions = build_transactions(ratings, movies)
    print(f"{len(transactions)} 条事务 | 事务长度分布:",
          transactions.str.len().value_counts().to_dict())

    df = encode_transactions(transactions)
    print(f"编码矩阵: {df.shape[0]} 行 × {df.shape[1]} 列")
    print("类型偏好率（support 基线）:")
    print(df.mean().sort_values(ascending=False).round(3).to_string())

    print("\nmin_support 扫描:")
    scan_support(transactions)

    rules = mine_rules(transactions, min_support=FINAL_MS)
    print(f"\n=== Top 规则（min_support={FINAL_MS}，共 {len(rules)} 条，按 lift 排序）===")
    print(format_rules(rules.head(10))
          [["antecedents", "consequents", "support", "confidence", "lift"]]
          .round(3).to_string(index=False))

    save_rules(rules)
    plot_rules(rules)
    print(f"\n已落盘: {OUT}/top_rules.csv | rules_scatter.png | top_rules_bar.png")


if __name__ == "__main__":
    main()
