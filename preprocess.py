import pandas as pd

# ------------ Load Anime Dataset ------------
anime_df = pd.read_csv("data/anime.csv")
anime_ratings = pd.read_csv("data/rating.csv")

# ✅ Clean anime dataset
anime_df = anime_df.rename(columns={"genres": "genre"})  # fix genre column
anime_df = anime_df[["anime_id", "title", "genre", "type"]]
anime_ratings = anime_ratings[anime_ratings["rating"] > 0]  # remove -1 ratings

# ✅ Merge anime info + ratings
anime_merged = anime_ratings.merge(anime_df, on="anime_id")
anime_merged["item_id"] = anime_merged["anime_id"]
anime_merged["source"] = "Anime"

# ------------ Load MovieLens Dataset ------------
mov_ratings = pd.read_csv("data/u.data", sep="\t", names=["user_id", "movie_id", "rating", "timestamp"])
movie_df = pd.read_csv("data/u.item", sep="|", encoding="latin-1", header=None, usecols=[0, 1], names=["movie_id", "title"])

# ✅ Merge movie info + ratings
mov_merged = mov_ratings.merge(movie_df, on="movie_id")
mov_merged["item_id"] = mov_merged["movie_id"]
mov_merged["genre"] = ""  # placeholder
mov_merged["type"] = "Movie"
mov_merged["source"] = "Movie"

# ------------ Combine Anime + Movie Ratings ------------
columns = ["user_id", "item_id", "title", "type", "genre", "rating", "source"]
combined_df = pd.concat([
    anime_merged[columns],
    mov_merged[columns]
], ignore_index=True)

# ✅ Save to CSV
combined_df.to_csv("data/combined_ratings.csv", index=False)
print("✅ Preprocessing complete! Saved to data/combined_ratings.csv")
