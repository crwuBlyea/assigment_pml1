"""Model API (Stage 3) – FastAPI service exposing the trained model."""
import json
import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "/app/models/model.joblib"))

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
LABELS = {0: "not diabetic", 1: "diabetic"}


class PatientFeatures(BaseModel):
    """Input schema – one patient described by the 8 model features."""
    Pregnancies: float = Field(..., ge=0, le=25, description="Number of times pregnant")
    Glucose: float = Field(..., ge=0, le=300, description="Plasma glucose concentration")
    BloodPressure: float = Field(..., ge=0, le=200, description="Diastolic blood pressure (mm Hg)")
    SkinThickness: float = Field(..., ge=0, le=120, description="Triceps skin fold thickness (mm)")
    Insulin: float = Field(..., ge=0, le=1000, description="2-hour serum insulin (mu U/ml)")
    BMI: float = Field(..., ge=0, le=100, description="Body mass index")
    DiabetesPedigreeFunction: float = Field(..., ge=0, le=5, description="Diabetes pedigree function")
    Age: float = Field(..., ge=1, le=120, description="Age (years)")


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    probability_diabetes: float


class ModelCache:
    """Loads the model on first use and hot-reloads it whenever the file changes on disk."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._model = None
        self._mtime = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def get_model(self):
        if not self._path.exists():
            raise FileNotFoundError(f"No model file at {self._path} – run the pipeline first.")
        mtime = self._path.stat().st_mtime
        if self._model is None or mtime != self._mtime:
            self._model = joblib.load(self._path)
            self._mtime = mtime
        return self._model


cache = ModelCache(MODEL_PATH)

app = FastAPI(
    title="Diabetes Prediction API",
    description="Serves the model trained by Stage 2 of the PMLDL pipeline.",
    version="1.0.0",
)


@app.get("/")
def root() -> dict:
    return {"service": "diabetes-prediction-api", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": cache.is_loaded, "model_path": str(MODEL_PATH)}


@app.get("/model_info")
def model_info() -> dict:
    """Returns the testing metrics of the latest training run."""
    metrics_path = MODEL_PATH.parent / "metrics.json"
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="metrics.json not found – run the pipeline first.")
    return json.loads(metrics_path.read_text())


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientFeatures) -> PredictionResponse:
    try:
        model = cache.get_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    X = pd.DataFrame([patient.model_dump()], columns=FEATURES)
    prediction = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0][1])
    return PredictionResponse(
        prediction=prediction,
        label=LABELS[prediction],
        probability_diabetes=round(probability, 4),
    )