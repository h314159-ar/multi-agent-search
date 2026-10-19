import pandas as pd
from pathlib import Path

DATA = Path(r"C:\Users\LEGION\PycharmProjects\multi-agent-search\data")
ratings = pd.read_csv(DATA / "ml-latest-small/ratings.csv")
movies  = pd.read_csv(DATA / "ml-latest-small/movies.csv")

# ① 清洗检查（预期全部通过，写进 PPT 对比页）
assert ratings.isnull().sum().sum() == 0
n_bad_genre = (movies["genres"] == "(no genres listed)").sum()   # 唯一"脏"点

# ② 时间特征：时间戳 → 真实日期
ratings["rated_at"] = pd.to_datetime(ratings["timestamp"], unit="s")
ratings["rated_year"] = ratings["rated_at"].dt.year

# ③ genres 多热编码（W4 关联规则直接用）
genre_dummies = movies["genres"].str.get_dummies(sep="|")       # 9712 × 20 的 0/1 矩阵
movies_feat = pd.concat([movies[["movieId", "title"]], genre_dummies], axis=1)

# ④ 用户特征聚合（W3 聚类直接用）
user_feat = ratings.groupby("userId").agg(
    n_ratings=("rating", "size"),
    mean_rating=("rating", "mean"),
    std_rating=("rating", "std"),
    first_year=("rated_year", "min"),
    last_year=("rated_year", "max"),
)

# ⑤ 落盘
out = DATA / "processed"; out.mkdir(exist_ok=True)
ratings.to_parquet(out / "ratings_clean.parquet")
movies_feat.to_parquet(out / "movies_feat.parquet")
user_feat.to_parquet(out / "user_feat.parquet")

# ========== 运行报告（截图素材） ==========
print("=" * 60)
print("① 清洗检查")
print(f"   ratings 缺失值总数 : {ratings.isnull().sum().sum()}（预期 0）")
print(f"   无类型电影数       : {n_bad_genre} 部（唯一脏点，保留原样）")

print("=" * 60)
print("② 时间特征：评分时间跨度")
print(f"   {ratings['rated_at'].min()}  →  {ratings['rated_at'].max()}")

print("=" * 60)
print("③ 预处理前后对比（前 3 行）")
print("--- before: 原始 ratings 列 ---")
print(list(pd.read_csv(DATA / 'ml-latest-small/ratings.csv', nrows=0).columns))
print("--- after: 处理后新增列 ---")
print(list(ratings.columns))

print("=" * 60)
print("④ genres 多热编码矩阵")
print(f"   形状: {movies_feat.shape[0]} 部电影 × {movies_feat.shape[1] - 2} 个类型列")
print(movies_feat.iloc[:3, :8].to_string(index=False))

print("=" * 60)
print("⑤ 用户特征表（W3 聚类输入）")
print(user_feat.describe().round(2).to_string())

print("=" * 60)
print("⑥ 落盘确认 → data/processed/")
for f in sorted(out.iterdir()):
    print(f"   {f.name}  ({f.stat().st_size / 1024:.0f} KB)")
print("全部完成 ✓")
