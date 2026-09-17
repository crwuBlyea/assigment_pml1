"""STAGE 1 – Data Engineering.

Loads the raw data, cleans it (imputes missing values, removes outliers) and
splits it into train/test files saved under data/processed/.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "diabetes.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
TRAIN_PATH = PROCESSED_DIR / "train.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"

TARGET = "Outcome"
# Columns in which a value of 0 is physiologically impossible → missing value
ZERO_MEANS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_data(path: Path = RAW_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at '{path}'. "
            "Run `python -m code.datasets.download_data` first."
        )
    df = pd.read_csv(path)
    print(f"[stage1] loaded {len(df)} rows, {len(df.columns)} columns from {path}")
    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Replace impossible zeros with NaN, then impute with the column median."""
    df = df.copy()
    n_missing = int(df[ZERO_MEANS_MISSING].eq(0).sum().sum())
    df[ZERO_MEANS_MISSING] = df[ZERO_MEANS_MISSING].replace(0, float("nan"))
    medians = df[ZERO_MEANS_MISSING].median()
    df[ZERO_MEANS_MISSING] = df[ZERO_MEANS_MISSING].fillna(medians)
    print(f"[stage1] imputed {n_missing} missing values (impossible zeros) with column medians")
    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows outside 1.5 * IQR in any feature column."""
    feature_cols = [c for c in df.columns if c != TARGET]
    mask = pd.Series(True, index=df.index)
    for col in feature_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        col_mask = df[col].between(lower, upper)
        n_new = int((~col_mask & mask).sum())
        mask &= col_mask
        if n_new:
            print(f"[stage1] {n_new} outliers in '{col}' (outside [{lower:.2f}, {upper:.2f}])")
    df = df[mask].reset_index(drop=True)
    print(f"[stage1] removed {int((~mask).sum())} outlier rows in total, {len(df)} rows remain")
    return df


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, stratify=df[TARGET], random_state=RANDOM_STATE
    )
    print(f"[stage1] stratified split: {len(train_df)} train / {len(test_df)} test rows")
    return train_df, test_df


def main() -> None:
    df = load_data()
    df = impute_missing(df)
    df = remove_outliers(df)
    train_df, test_df = split_data(df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)
    print(f"[stage1] saved {TRAIN_PATH.name} and {TEST_PATH.name} to {PROCESSED_DIR}")


if __name__ == "__main__":
    main()