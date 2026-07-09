import pandas as pd

GENRE_COLUMNS = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]

def load_ratings(path: str = "data/raw/ml-100k/u.data") -> pd.DataFrame:
    columns = ["user_id", "item_id", "rating", "timestamp"]
    return pd.read_csv(path, sep="\t", names=columns)

def load_items(path: str = "data/raw/ml-100k/u.item") -> pd.DataFrame:
    columns = ["item_id", "title", "release_date", "video_release_date", "imdb_url"] + GENRE_COLUMNS
    return pd.read_csv(path, sep="|", names=columns, encoding="latin-1")

def build_user_features(ratings: pd.DataFrame) -> pd.DataFrame:
    """One row per user: rating count, mean, std."""
    agg = ratings.groupby("user_id")["rating"].agg(
        num_ratings="count",
        avg_rating="mean",
        rating_std="std"
    ).reset_index()
    agg["rating_std"] = agg["rating_std"].fillna(0)
    return agg

def build_item_features(ratings: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    """One row per item: rating count, mean, plus genre flags."""
    agg = ratings.groupby("item_id")["rating"].agg(
        num_ratings="count",
        avg_rating="mean"
    ).reset_index()
    item_genres = items[["item_id"] + GENRE_COLUMNS]
    return agg.merge(item_genres, on="item_id", how="left")


if __name__ == "__main__":
    ratings = load_ratings()
    items = load_items()

    user_features = build_user_features(ratings)
    item_features = build_item_features(ratings, items)

    user_features.to_csv("data/processed/user_features.csv", index=False)
    item_features.to_csv("data/processed/item_features.csv", index=False)

    print(f"User features: {user_features.shape}")
    print(user_features.head())
    print(f"\nItem features: {item_features.shape}")
    print(item_features.head())