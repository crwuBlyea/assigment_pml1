"""Feature engineering helpers for Stage 2."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
TARGET = "Outcome"


def load_processed_data(processed_dir: Path = PROCESSED_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = processed_dir / "train.csv"
    test_path = processed_dir / "test.csv"
    for p in (train_path, test_path):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found – run Stage 1 first "
                                    "(`python -m code.datasets.prepare_data`).")
    return pd.read_csv(train_path), pd.read_csv(test_path)


def separate_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """X = all feature columns, y = the target column."""
    return df.drop(columns=[TARGET]), df[TARGET]