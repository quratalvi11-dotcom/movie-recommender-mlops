import pickle
import numpy as np
import scipy.sparse as sp
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Movie Recommender API")

# Load ratings once at startup to rebuild the user-item matrix
# (In a production system this would come from a feature store / database instead)
RATINGS_PATH = "data/raw/ml-100k/u.data"
ITEMS_PATH = "data/raw/ml-100k/u.item"
MODEL_PATH = "model.pkl"

def load_ratings():
    columns = ["user_id", "item_id", "rating", "timestamp"]
    return pd.read_csv(RATINGS_PATH, sep="\t", names=columns)

def load_items():
    columns = ["item_id", "title", "release_date", "video_release_date", "imdb_url"] + [f"genre_{i}" for i in range(19)]
    return pd.read_csv(ITEMS_PATH, sep="|", names=columns, encoding="latin-1")

ratings = load_ratings()
items = load_items()
n_users = ratings["user_id"].max()
n_items = ratings["item_id"].max()

rows = ratings["user_id"].values - 1
cols = ratings["item_id"].values - 1
vals = ratings["rating"].values.astype(np.float32)
user_item_matrix = sp.csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

item_titles = items.set_index("item_id")["title"].to_dict()


class Recommendation(BaseModel):
    item_id: int
    title: str
    score: float

class RecommendResponse(BaseModel):
    user_id: int
    recommendations: list[Recommendation]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/recommend/{user_id}", response_model=RecommendResponse)
def recommend(user_id: int, k: int = 10):
    if user_id < 1 or user_id > n_users:
        raise HTTPException(status_code=404, detail=f"user_id must be between 1 and {n_users}")

    user_idx = user_id - 1
    item_indices, scores = model.recommend(
        user_idx, user_item_matrix[user_idx], N=k, filter_already_liked_items=True
    )

    recs = [
        Recommendation(
            item_id=int(idx) + 1,
            title=item_titles.get(int(idx) + 1, "Unknown"),
            score=float(score),
        )
        for idx, score in zip(item_indices, scores)
    ]

    return RecommendResponse(user_id=user_id, recommendations=recs)