import pandas as pd
import great_expectations as gx

def load_ratings(path: str = "data/raw/ml-100k/u.data") -> pd.DataFrame:
    """Load the raw MovieLens ratings file."""
    columns = ["user_id", "item_id", "rating", "timestamp"]
    return pd.read_csv(path, sep="\t", names=columns)

def validate_ratings(df: pd.DataFrame) -> bool:
    """Run data quality checks on the ratings dataframe."""
    context = gx.get_context()
    data_source = context.data_sources.add_pandas(name="ratings_source")
    data_asset = data_source.add_dataframe_asset(name="ratings_asset")
    batch_definition = data_asset.add_batch_definition_whole_dataframe("ratings_batch")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    results = []

    # Ratings must be between 1 and 5
    results.append(batch.validate(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="rating", min_value=1, max_value=5
        )
    ))

    # No missing user_id or item_id
    results.append(batch.validate(
        gx.expectations.ExpectColumnValuesToNotBeNull(column="user_id")
    ))
    results.append(batch.validate(
        gx.expectations.ExpectColumnValuesToNotBeNull(column="item_id")
    ))

    # Expect roughly 100,000 rows (MovieLens 100K)
    results.append(batch.validate(
        gx.expectations.ExpectTableRowCountToBeBetween(
            min_value=95000, max_value=105000
        )
    ))

    all_passed = all(r.success for r in results)

    print("\n=== Data Validation Results ===")
    for r in results:
        status = "PASS" if r.success else "FAIL"
        print(f"[{status}] {r.expectation_config.type}")

    print(f"\nOverall: {'PASSED' if all_passed else 'FAILED'}")
    return all_passed


if __name__ == "__main__":
    df = load_ratings()
    print(f"Loaded {len(df)} ratings")
    validate_ratings(df)