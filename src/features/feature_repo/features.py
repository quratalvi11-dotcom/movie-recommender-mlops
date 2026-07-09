from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Int64, Float32

user = Entity(name="user_id", join_keys=["user_id"])
item = Entity(name="item_id", join_keys=["item_id"])

user_source = FileSource(
    path="../../../data/processed/user_features_ts.parquet",
    timestamp_field="event_timestamp",
)

item_source = FileSource(
    path="../../../data/processed/item_features_ts.parquet",
    timestamp_field="event_timestamp",
)

user_features_view = FeatureView(
    name="user_features",
    entities=[user],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="num_ratings", dtype=Int64),
        Field(name="avg_rating", dtype=Float32),
        Field(name="rating_std", dtype=Float32),
    ],
    source=user_source,
)

item_features_view = FeatureView(
    name="item_features",
    entities=[item],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="num_ratings", dtype=Int64),
        Field(name="avg_rating", dtype=Float32),
    ],
    source=item_source,
)