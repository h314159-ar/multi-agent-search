import duckdb

# ── 建库建表：从预处理产物导入（data/movielens.duckdb 自动生成） ──
con = duckdb.connect("data/movielens.duckdb")
con.execute("CREATE OR REPLACE TABLE ratings AS SELECT * FROM 'data/processed/ratings_clean.parquet'")
con.execute("CREATE OR REPLACE TABLE movies  AS SELECT * FROM 'data/processed/movies_feat.parquet'")

print("=== 建表完成，库中表清单 ===")
print(con.sql("SHOW TABLES").df().to_string(index=False))

# ── 验收 1：评分最高的 10 部电影（至少 100 人评过） ──
print("\n=== 验收 1：评分最高 Top10（≥100 人评） ===")
print(con.sql("""
    SELECT m.title, ROUND(AVG(r.rating), 2) AS avg_score, COUNT(*) AS n
    FROM ratings r JOIN movies m USING (movieId)
    GROUP BY m.title
    HAVING COUNT(*) >= 100
    ORDER BY avg_score DESC
    LIMIT 10
""").df().to_string(index=False))

# ── 验收 2：每年的评分数量（看 22 年的时间趋势） ──
print("\n=== 验收 2：每年评分数 ===")
print(con.sql("""
    SELECT rated_year AS 年份, COUNT(*) AS 评分数
    FROM ratings
    GROUP BY rated_year
    ORDER BY 年份
""").df().to_string(index=False))

# ── 验收 3：Drama 类型电影数（验证多热编码列可用） ──
print("\n=== 验收 3：Drama 电影数 ===")
print(con.sql("SELECT COUNT(*) AS drama_movies FROM movies WHERE Drama = 1").df().to_string(index=False))

con.close()
print("\n全部验收通过，DuckDB 就绪")
