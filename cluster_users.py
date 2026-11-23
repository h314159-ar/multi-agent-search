r"""W3：基于用户特征的 K-Means 分群。

输入：data/processed/user_feat.parquet（W2 已生成）
输出：outputs/w3/ 下的分群表、评估指标和三张汇报图。
运行：dsp\Scripts\python cluster_users.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "data" / "processed" / "user_feat.parquet"
OUTPUT = ROOT / "outputs" / "w3"
OUTPUT.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    # 1. 读取 W2 用户特征，并构造适合距离计算的聚类特征。
    users = pd.read_parquet(INPUT).reset_index()
    users["active_years"] = users["last_year"] - users["first_year"] + 1

    features = pd.DataFrame({
        "log_n_ratings": np.log1p(users["n_ratings"]),
        "mean_rating": users["mean_rating"],
        "std_rating": users["std_rating"],
        "log_active_years": np.log1p(users["active_years"]),
    })
    print("=== ① 聚类输入 ===")
    print(f"用户数：{len(users)}；原始特征：5；聚类特征：{features.shape[1]}")
    print("处理：n_ratings、active_years 做 log1p 压缩长尾，再进行 Z-score 标准化")
    print(f"缺失值：{int(features.isna().sum().sum())}")

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(features)

    # 2. 同时观察 inertia（肘部法则）和 silhouette（轮廓系数），选择 K。
    metrics = []
    for k in range(2, 9):
        model = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
        labels = model.fit_predict(x_scaled)
        metrics.append({
            "k": k,
            "inertia": model.inertia_,
            "silhouette": silhouette_score(x_scaled, labels),
        })
    metrics_df = pd.DataFrame(metrics)
    best_k = int(metrics_df.loc[metrics_df["silhouette"].idxmax(), "k"])
    print("\n=== ② K 值评估（轮廓系数越大越好） ===")
    print(metrics_df.round({"inertia": 2, "silhouette": 4}).to_string(index=False))
    print(f"选择 K={best_k}：在 2~8 中轮廓系数最高")

    # 3. 用最佳 K 重新训练，保存每位用户的分群编号。
    final_model = KMeans(n_clusters=best_k, n_init=20, random_state=RANDOM_STATE)
    users["cluster"] = final_model.fit_predict(x_scaled)

    # K=2 时按平均评分数量命名，避免把簇编号误当作有固定含义的标签。
    if best_k == 2:
        activity_order = (
            users.groupby("cluster")["n_ratings"].mean().sort_values().index.tolist()
        )
        name_map = {activity_order[0]: "轻度用户", activity_order[1]: "重度用户"}
        users["cluster_name"] = users["cluster"].map(name_map)
    else:
        users["cluster_name"] = "群体 " + users["cluster"].astype(str)

    # 4. 用原始量纲生成画像，不能用标准化后的数值解释业务含义。
    profile = users.groupby(["cluster", "cluster_name"]).agg(
        user_count=("userId", "size"),
        avg_ratings=("n_ratings", "mean"),
        median_ratings=("n_ratings", "median"),
        avg_rating=("mean_rating", "mean"),
        avg_rating_std=("std_rating", "mean"),
        avg_active_years=("active_years", "mean"),
    ).reset_index()
    profile["user_ratio"] = profile["user_count"] / len(users)

    print("\n=== ③ 分群画像（原始量纲） ===")
    print(profile.round({
        "avg_ratings": 1,
        "median_ratings": 1,
        "avg_rating": 2,
        "avg_rating_std": 2,
        "avg_active_years": 1,
        "user_ratio": 3,
    }).to_string(index=False))

    # 图 1：K 选择依据。
    fig, ax1 = plt.subplots(figsize=(8, 4.8))
    ax1.plot(metrics_df["k"], metrics_df["inertia"], "o-", color="#378ADD", label="Inertia")
    ax1.set_xlabel("聚类数 K")
    ax1.set_ylabel("Inertia（越小越紧凑）", color="#1F5F99")
    ax1.set_xticks(metrics_df["k"])
    ax2 = ax1.twinx()
    ax2.plot(metrics_df["k"], metrics_df["silhouette"], "s--", color="#EF9F27", label="Silhouette")
    ax2.set_ylabel("轮廓系数（越大越好）", color="#9A5B00")
    ax2.axvline(best_k, color="#888888", linestyle=":", linewidth=1.2)
    ax1.set_title(f"K-Means 参数选择：最佳 K={best_k}")
    fig.tight_layout()
    fig.savefig(OUTPUT / "k_selection.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 图 2：PCA 仅用于把四维标准化特征投影到二维展示，不参与聚类训练。
    coords = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(x_scaled)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for cluster_id in sorted(users["cluster"].unique()):
        mask = users["cluster"] == cluster_id
        group_name = users.loc[mask, "cluster_name"].iloc[0]
        ax.scatter(
            coords[mask, 0], coords[mask, 1], s=25, alpha=0.7,
            label=f"{group_name}（n={int(mask.sum())}）",
        )
    ax.set_xlabel("主成分 1")
    ax.set_ylabel("主成分 2")
    ax.set_title(f"K-Means 用户分群（K={best_k}，PCA 二维投影）")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "user_clusters_pca.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 图 3：各簇在标准化特征上的中心，便于比较相对高低。
    centers = pd.DataFrame(
        scaler.inverse_transform(final_model.cluster_centers_),
        columns=features.columns,
    )
    centers.insert(0, "cluster_name", [name_map[i] for i in range(best_k)] if best_k == 2 else [f"群体 {i}" for i in range(best_k)])
    display_centers = centers.set_index("cluster_name").T
    fig, ax = plt.subplots(figsize=(8, 4.8))
    display_centers.plot(kind="bar", ax=ax, width=0.72)
    ax.set_title("各用户群的特征中心（已还原为对数/原始量纲）")
    ax.set_xlabel("聚类特征")
    ax.set_ylabel("特征中心值")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="用户群")
    fig.tight_layout()
    fig.savefig(OUTPUT / "cluster_feature_centers.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 5. 落盘：结果可被 W5 的挖掘 Agent 作为工具直接调用。
    users.to_csv(OUTPUT / "user_clusters.csv", index=False, encoding="utf-8-sig")
    profile.to_csv(OUTPUT / "cluster_profiles.csv", index=False, encoding="utf-8-sig")
    metrics_df.to_csv(OUTPUT / "k_metrics.csv", index=False, encoding="utf-8-sig")
    features.to_csv(OUTPUT / "cluster_input_features.csv", index=False, encoding="utf-8-sig")

    print("\n=== ④ 输出完成 ===")
    for path in sorted(OUTPUT.iterdir()):
        print(f"{path.name}: {path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
