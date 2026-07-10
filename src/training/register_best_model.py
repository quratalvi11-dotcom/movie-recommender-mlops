import pandas as pd
import numpy as np
import scipy.sparse as sp
import mlflow
import mlflow.pyfunc
import pickle
from implicit.als import AlternatingLeastSquares
from sklearn.model_selection import train_test_split
from mlflow.tracking import MlflowClient

def load_ratings(path: str = "data/raw/ml-100k/u.data") -> pd.DataFrame:
    columns = ["user_id", "item_id", "rating", "timestamp"]
    return pd.read_csv(path, sep="\t", names=columns)

def build_sparse_matrix(df: pd.DataFrame, n_users: int, n_items: int):
    rows = df["user_id"].values - 1
    cols = df["item_id"].values - 1
    vals = df["rating"].values.astype(np.float32)
    return sp.csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))

def precision_at_k(model, train_matrix, test_df, k=10):
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

class ALSWrapper(mlflow.pyfunc.PythonModel):
    """Wraps an implicit ALS model so MLflow can log/register/serve it."""
    def load_context(self, context):
        with open(context.artifacts["model_path"], "rb") as f:
            self.model = pickle.load(f)

    def predict(self, context, model_input, params=None):
        # model_input expected: DataFrame with a 'user_id' column
        user_ids = model_input["user_id"].values
        return [self.model.similar_users(uid, N=1) for uid in user_ids]

BEST_PARAMS = {"factors": 13, "regularization": 0.0024074503698876615, "iterations": 15}
MODEL_NAME = "movie_recommender_als"

def main():
    mlflow.set_experiment("movie_recommender")

    ratings = load_ratings()
    n_users = ratings["user_id"].max()
    n_items = ratings["item_id"].max()
    train_df, test_df = train_test_split(ratings, test_size=0.2, random_state=42)
    train_matrix = build_sparse_matrix(train_df, n_users, n_items)

    with mlflow.start_run(run_name="best_model_registration") as run:
        mlflow.log_params(BEST_PARAMS)
        mlflow.set_tag("source", "optuna_best")

        model = AlternatingLeastSquares(**BEST_PARAMS)
        model.fit(train_matrix)

        precision = precision_at_k(model, train_matrix, test_df, k=10)
        mlflow.log_metric("precision_at_10", precision)

        with open("model.pkl", "wb") as f:
            pickle.dump(model, f)

        mlflow.pyfunc.log_model(
            name="model",
            python_model=ALSWrapper(),
            artifacts={"model_path": "model.pkl"},
        )

        run_id = run.info.run_id
        print(f"Run ID: {run_id}")
        print(f"Precision@10: {precision:.4f}")

    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
    print(f"Registered as: {MODEL_NAME}, version {result.version}")

    client = MlflowClient()
    client.set_registered_model_alias(MODEL_NAME, "production", result.version)
    print(f"Alias 'production' set on version {result.version}")

if __name__ == "__main__":
    main()