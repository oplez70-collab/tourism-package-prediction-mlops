"""
prep.py
-------
Data preparation step of the MLOps pipeline.

1. Load the registered dataset directly from the repository data folder.
2. Clean it:
   - drop columns with no predictive value (index column, CustomerID),
   - fix inconsistent category labels ('Fe Male' -> 'Female'),
   - trim stray whitespace in text columns,
   - impute any missing values (median for numeric, mode for categorical) so the
     pipeline stays robust when new data with gaps is pushed,
   - remove exact duplicate customer records.
3. Split into stratified train/test sets (80/20) and save them as CSV files
   in the working directory. GitHub Actions uploads them as a workflow artifact
   for the training job.

Categorical columns are deliberately kept as raw strings: they are one-hot
encoded inside the model pipeline, so training and the Streamlit app use the
exact same representation.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "tourism_project/data/tourism.csv"
TARGET = "ProdTaken"
TEST_SIZE = 0.20
RANDOM_STATE = 42

# Columns that carry no predictive information
DROP_COLS = ["Unnamed: 0", "CustomerID"]

NUMERIC_FEATURES = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips", "Passport",
    "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting", "MonthlyIncome",
]
CATEGORICAL_FEATURES = [
    "TypeofContact", "Occupation", "Gender", "ProductPitched",
    "MaritalStatus", "Designation",
]


def main() -> None:
    # 1. Load ---------------------------------------------------------------
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {DATA_PATH}: {df.shape[0]} rows x {df.shape[1]} columns")

    # 2. Clean --------------------------------------------------------------
    # 2a. Remove unnecessary columns (only those that actually exist)
    to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=to_drop)
    print(f"Dropped columns: {to_drop}")

    # 2b. Standardise text categories
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str).str.strip()
    df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

    # 2c. Impute missing values (defensive: future data may contain gaps)
    n_missing = int(df.isna().sum().sum())
    for col in NUMERIC_FEATURES:
        df[col] = df[col].fillna(df[col].median())
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna(df[col].mode()[0])
    df = df.dropna(subset=[TARGET])
    print(f"Missing values imputed: {n_missing}")

    # 2d. Remove exact duplicate records (avoids leakage between train and test)
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Duplicate rows removed: {before - len(df)}")

    # 2e. Enforce data types
    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].astype(float)
    df[TARGET] = df[TARGET].astype(int)

    # 3. Split --------------------------------------------------------------
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    # stratify keeps the (imbalanced) purchase ratio identical in both splits
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    Xtrain.to_csv("Xtrain.csv", index=False)
    Xtest.to_csv("Xtest.csv", index=False)
    ytrain.to_csv("ytrain.csv", index=False)
    ytest.to_csv("ytest.csv", index=False)

    print(f"\nClean dataset: {df.shape[0]} rows x {X.shape[1]} features")
    print(f"Train: {Xtrain.shape}, Test: {Xtest.shape}")
    print("ProdTaken rate -> train: {:.3f} | test: {:.3f}".format(ytrain.mean(), ytest.mean()))
    print("Files written: Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv")


if __name__ == "__main__":
    main()
