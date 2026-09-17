"""Downloads the raw dataset (Pima Indians Diabetes) into data/raw/.

If the download fails (e.g. no internet), a small synthetic dataset with the
same schema is generated so the pipeline can still be demonstrated.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
RAW_PATH = RAW_DIR / "diabetes.csv"

URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome",
]


def _synthetic(n: int = 768, seed: int = 42) -> pd.DataFrame:
    """Offline fallback: synthetic data with the same schema."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "Pregnancies": rng.integers(0, 15, n),
        "Glucose": np.clip(rng.normal(120, 30, n), 40, 250).astype(int),
        "BloodPressure": np.clip(rng.normal(70, 12, n), 40, 120).astype(int),
        "SkinThickness": np.clip(rng.normal(20, 9, n), 5, 60).astype(int),
        "Insulin": np.clip(rng.normal(80, 60, n), 0, 400).astype(int),
        "BMI": np.round(np.clip(rng.normal(32, 7, n), 15, 60), 1),
        "DiabetesPedigreeFunction": np.round(np.clip(rng.lognormal(-1.0, 0.5, n), 0.05, 2.5), 3),
        "Age": rng.integers(21, 81, n),
    })
    risk = ((df["Glucose"] - 120) / 25 + (df["BMI"] - 32) / 6
            + (df["Age"] - 35) / 15 + rng.normal(0, 0.5, n))
    df["Outcome"] = (risk > 0).astype(int)
    return df[COLUMNS]


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_PATH.exists():
        print(f"[download] {RAW_PATH} already exists – skipping download.")
        return
    try:
        df = pd.read_csv(URL, header=None, names=COLUMNS)
        print(f"[download] downloaded {len(df)} rows from {URL}")
    except Exception as exc:
        print(f"[download] download failed ({exc}) – generating synthetic data instead.")
        df = _synthetic()
    df.to_csv(RAW_PATH, index=False)
    print(f"[download] raw data saved to {RAW_PATH}")


if __name__ == "__main__":
    main()