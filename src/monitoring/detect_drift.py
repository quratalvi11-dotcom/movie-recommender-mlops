import pandas as pd
import numpy as np
from evidently import Report
from evidently.presets import DataDriftPreset

def load_ratings(path: str = "data/raw/ml-100k/u.data") -> pd.DataFrame:
    columns = ["user_id", "item_id", "rating", "timestamp"]
    return pd.read_csv(path, sep="\t", names=columns)

def simulate_new_batch(reference: pd.DataFrame, shift: bool = True) -> pd.DataFrame:
    """Simulate a new batch of incoming ratings, optionally with drift."""
    new_batch = reference.sample(n=2000, random_state=123).copy()
    if shift:
        # Simulate a ratings-inflation drift: bump ratings up, clip at 5
        new_batch["rating"] = (new_batch["rating"] + 1).clip(upper=5)
    return new_batch

def detect_drift(reference: pd.DataFrame, current: pd.DataFrame) -> bool:
    report = Report(metrics=[DataDriftPreset()])
    result = report.run(reference_data=reference[["rating"]], current_data=current[["rating"]])

    result.save_html("drift_report.html")

    result_dict = result.dict()

    drift_share = None
    for metric in result_dict.get("metrics", []):
        if metric.get("metric_name", "").startswith("DriftedColumnsCount"):
            drift_share = metric["value"]["share"]
            break

    drift_detected = drift_share is not None and drift_share > 0

    print(f"Drift share: {drift_share}")
    print(f"Drift detected: {drift_detected}")
    print("Full report saved to drift_report.html")
    return drift_detected

if __name__ == "__main__":
    import os
    ratings = load_ratings()
    new_batch = simulate_new_batch(ratings, shift=True)
    drift_found = detect_drift(ratings, new_batch)

    if drift_found:
        print("ALERT: Significant drift detected. Retraining recommended.")
    else:
        print("No significant drift detected.")

    # Write result for GitHub Actions to read (no-op if not running in CI)
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"drift_detected={str(drift_found).lower()}\n")