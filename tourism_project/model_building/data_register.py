"""
data_register.py
----------------
Registers the raw tourism dataset that lives inside the GitHub repository.

Steps:
1. Load the raw CSV from tourism_project/data/.
2. Validate the schema (expected columns, binary target, non-empty file).
3. Print a summary (shape, dtypes, missing values, duplicates, target balance).
4. Write a small registry record (row/column counts + MD5 fingerprint) so every
   pipeline run leaves a trace of exactly which version of the data was used.
"""
import hashlib
import json
import os
from datetime import datetime, timezone

import pandas as pd

RAW_PATH = "tourism_project/data/tourism.csv"
REGISTRY_PATH = "tourism_project/data/data_registry.json"
TARGET = "ProdTaken"

EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]


def file_md5(path: str) -> str:
    """Return the MD5 hash of a file (used as a data-version fingerprint)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    # 1. Load -----------------------------------------------------------------
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"Raw dataset not found at {RAW_PATH}")
    df = pd.read_csv(RAW_PATH)

    # 2. Validate -------------------------------------------------------------
    if df.empty:
        raise ValueError("The dataset is empty.")

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")

    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]

    if not set(df[TARGET].dropna().unique()).issubset({0, 1}):
        raise ValueError(f"Target '{TARGET}' must be binary (0/1).")

    # 3. Summary --------------------------------------------------------------
    print("=" * 60)
    print("DATASET REGISTERED SUCCESSFULLY")
    print("=" * 60)
    print(f"Source file        : {RAW_PATH}")
    print(f"Rows x Columns     : {df.shape[0]} x {df.shape[1]}")
    print(f"Expected columns   : all {len(EXPECTED_COLUMNS)} present")
    print(f"Extra columns      : {extra if extra else 'none'} (removed during data prep)")
    print(f"Missing values     : {int(df.isna().sum().sum())}")
    print(f"Duplicate CustomerID: {int(df['CustomerID'].duplicated().sum())}")
    print("\nColumn data types:")
    print(df[EXPECTED_COLUMNS].dtypes.to_string())
    print(f"\n{TARGET} distribution (counts):")
    print(df[TARGET].value_counts().to_string())
    print(f"\n{TARGET} distribution (%):")
    print((df[TARGET].value_counts(normalize=True) * 100).round(2).to_string())

    # 4. Registry record --------------------------------------------------------
    record = {
        "dataset": "tourism.csv",
        "path": RAW_PATH,
        "registered_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "md5": file_md5(RAW_PATH),
        "target_positive_rate": round(float(df[TARGET].mean()), 4),
    }
    with open(REGISTRY_PATH, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\nRegistry record written to {REGISTRY_PATH}:")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
