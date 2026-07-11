import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from src.data.validate import load_ratings, validate_ratings
from src.features.build_features import build_user_features, build_item_features, load_items

def test_ratings_load():
    df = load_ratings()
    assert len(df) == 100000
    assert set(df.columns) == {"user_id", "item_id", "rating", "timestamp"}

def test_ratings_valid_range():
    df = load_ratings()
    assert df["rating"].min() >= 1
    assert df["rating"].max() <= 5

def test_data_validation_passes():
    df = load_ratings()
    assert validate_ratings(df) is True

def test_user_features_shape():
    ratings = load_ratings()
    user_features = build_user_features(ratings)
    assert "user_id" in user_features.columns
    assert "avg_rating" in user_features.columns
    assert user_features["avg_rating"].between(1, 5).all()

def test_item_features_shape():
    ratings = load_ratings()
    items = load_items()
    item_features = build_item_features(ratings, items)
    assert "item_id" in item_features.columns
    assert len(item_features) > 0