import pandas as pd
from pathlib import Path

DATA = Path(r"C:\Users\LEGION\PycharmProjects\multi-agent-search\data")
ratings = pd.read_csv(DATA / "ml-latest-small/ratings.csv")
movies  = pd.read_csv(DATA / "ml-latest-small/movies.csv")

pref = ratings[ratings["rating"] >= 4.0]                     # ① 只留"偏好"：≥4 分
pref = pref.merge(movies[["movieId", "genres"]], on="movieId")
genres = pref["genres"].str.split("|").explode()
genres = genres[genres != "(no genres listed)"]              # ② 剔除 34 部无类型电影的占位符
transactions = (pref.groupby("userId")["genres"]
                .agg(lambda s: sorted({g for gs in s for g in gs.split("|")
                                       if g != "(no genres listed)"})))
print(len(transactions), "条事务")                            # 预期 610

