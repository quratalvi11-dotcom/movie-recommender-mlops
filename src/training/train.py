import pandas as pd
import numpy as np
import scipy.sparse as sp
import mlflow
import mlflow.sklearn
from implicit.als import AlternatingLeastSquares
from sklearn.model_selection import train_test_split

def load_ratings(path: str = "data/raw/ml-100k/u.data") -> pd.DataFrame:
    columns = ["user_id", "item_id", "rating", "timestamp"]
    return pd.read_csv(path, sep="\t", names=columns)

def build_sparse_matrix(df: pd.DataFrame, n_users: int, n_items: int):
    """Build a user-item sparse matrix from ratings."""
    rows = df["user_id"].values - 1  # 0-indexed
    cols = df["item_id"].values - 1
    vals = df["rating"].values.astype(np.float32)
    return sp.csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))

def precision_at_k(model, train_matrix, test_df, k=10):
    """Simple precision@k: fraction of top-k recs that appear in test set."""
    test_by_user = test_df.groupby("user_id")["item_id"].apply(set).to_dict()
    hits, total = 0, 0

    for user_id, true_items in test_by_user.items():
        user_idx = user_id - 1
        if user_idx >= train_matrix.shape[0]:
            continue
        recommended = model.recommend(
            user_idx, train_matrix[user_idx], N=k, filter_already_liked_items=True
        )
        rec_item_ids = {idx + 1 for idx, _ in zip(*recommended)} if isinstance(recommended, tuple) else {r[0] + 1 for r in recommended}
        hits += len(rec_item_ids & true_items)
        total += k

    return hits / total if total > 0 else 0.0

def main():
    mlflow.set_experiment("movie_recommender")

    ratings = load_ratings()
    n_users = ratings["user_id"].max()
    n_items = ratings["item_id"].max()

    train_df, test_df = train_test_split(ratings, test_size=0.2, random_state=42)
    train_matrix = build_sparse_matrix(train_df, n_users, n_items)

    factors = 50
    regularization = 0.01
    iterations = 20

    with mlflow.start_run():
        mlflow.log_param("factors", factors)
        mlflow.log_param("regularization", regularization)
        mlflow.log_param("iterations", iterations)
        mlflow.log_param("train_size", len(train_df))
        mlflow.log_param("test_size", len(test_df))

        model = AlternatingLeastSquares(
            factors=factors, regularization=regularization, iterations=iterations
        )
        model.fit(train_matrix)

        precision = precision_at_k(model, train_matrix, test_df, k=10)
        mlflow.log_metric("precision_at_10", precision)

        print(f"Trained model: factors={factors}, precision@10={precision:.4f}")
        print("Run logged to MLflow.")

if __name__ == "__main__":
    main()