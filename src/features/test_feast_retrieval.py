from feast import FeatureStore

store = FeatureStore(repo_path="src/features/feature_repo")

# Fetch historical features for a few users (simulates what training would pull)
entity_df = __import__("pandas").DataFrame({
    "user_id": [1, 2, 3],
    "event_timestamp": __import__("pandas").Timestamp.now()
})

features = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "user_features:num_ratings",
        "user_features:avg_rating",
        "user_features:rating_std",
    ],
).to_df()

print(features)