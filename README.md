PMLDL Assignment 1 – Deployment
An automated, three-stage MLOps pipeline that runs every 5 minutes:

Stage	What happens	Tools
1 · Data engineering	load → clean (impute missing values, remove outliers) → stratified 80/20 split	pandas, scikit-learn
2 · Model engineering	feature scaling, training of 2 candidate models, evaluation, logging, packaging	scikit-learn, MLflow, joblib
3 · Deployment	build & run a model API + web app in separate Docker containers	Docker, FastAPI, Streamlit
Dataset: Pima Indians Diabetes (UCI) — 768 patients, 8 features, binary target Outcome.

Architecture
                 ┌──────────────────────────────────────────────┐                 │  scheduler.py  (every 5 min, APScheduler)    │                 └───────────────────────┬──────────────────────┘                                         ▼       ┌────────────────────────── run_pipeline.py ──────────────────────────┐       ▼                                                                      ▼┌──────────────────┐   ┌──────────────────────┐   ┌────────────────────────────────┐│ Stage 1: data    │   │ Stage 2: model       │   │ Stage 3: deployment            ││ prepare_data.py  │ → │ train_model.py       │ → │ docker compose up -d --build   ││ load/clean/split │   │ features/train/eval/ │   │  • api  (FastAPI)   :8000      │└──────────────────┘   │ package  + MLflow    │   │  • app  (Streamlit) :8501      │                       └──────────────────────┘   └────────────────────────────────┘data/raw/diabetes.csv → data/processed/{train,test}.csv → models/{model.joblib, metrics.json}                                                            │ bind mount + hot-reload                                                            ▼                                             app ── POST /predict ──► api ──► model
Repository structure
.├── README.md├── requirements.txt          # host-side dependencies (Stages 1–2, scheduler)├── run_pipeline.py           # runs all three stages once├── scheduler.py              # runs the pipeline every 5 minutes├── code/│   ├── __init__.py           # ⚠ must exist (empty!) — see Troubleshooting│   ├── datasets/│   │   ├── __init__.py       # ⚠ must exist (empty!)│   │   ├── download_data.py  # Stage 0: fetch raw dataset (synthetic fallback offline)│   │   └── prepare_data.py   # Stage 1│   ├── models/│   │   ├── __init__.py       # ⚠ must exist (empty!)│   │   ├── build_features.py│   │   └── train_model.py    # Stage 2│   └── deployment/│       ├── docker-compose.yml        # Stage 3│       ├── api/│       │   ├── Dockerfile│       │   ├── requirements.txt│       │   └── main.py               # FastAPI model API│       └── app/│           ├── Dockerfile│           ├── requirements.txt│           └── streamlit_app.py      # web application├── data/│   ├── raw/                  # diabetes.csv (auto-downloaded by Stage 0)│   └── processed/            # train.csv / test.csv (Stage 1 output, gitignored)├── models/                   # model.joblib + metrics.json (Stage 2 output, gitignored)├── mlruns/                   # MLflow tracking data (gitignored)└── notebooks/
Prerequisites
Requirement	Notes
Python 3.9–3.12	check with python --version
Docker Desktop	installed and running (green icon in tray) — required for Stage 3
Quickstart
Run all commands from the repository root.

Clone and set up the environment:
git clone <your-repo-url>cd <repo>python -m venv venvvenv\Scripts\activate            # Windows (cmd / PowerShell)# source venv/bin/activate       # Linux / macOSpip install -r requirements.txt
Run the complete pipeline once (raw data is downloaded automatically if missing):
python run_pipeline.py
Automate — run the complete pipeline every 5 minutes:
python scheduler.py
Where to look after launch
Service	URL
Streamlit app (input fields → Predict → prediction)	http://localhost:8501
Model API (Swagger docs)	http://localhost:8000/docs
Model API health	http://localhost:8000/health
Latest training metrics	http://localhost:8000/model_info
Running containers	docker ps (diabetes-api, diabetes-app)
MLflow UI (optional; on MLflow ≥ 3.1 the env variable is required — see MLflow 3.x notes):

:: Windows (cmd)set MLFLOW_ALLOW_FILE_STORE=truemlflow ui --port 5000
# Windows (PowerShell) $env:MLFLOW_ALLOW_FILE_STORE = "true"mlflow ui --port 5000
# Linux / macOSexport MLFLOW_ALLOW_FILE_STORE=truemlflow ui --port 5000
Then open http://localhost:5000.

Stop everything: Ctrl+C (scheduler) and

docker compose -f code/deployment/docker-compose.yml down
Running the stages individually
python -m code.datasets.download_datapython -m code.datasets.prepare_datapython -m code.models.train_modeldocker compose -f code/deployment/docker-compose.yml up -d --build
run_pipeline.py runs exactly these commands in this order.

Pipeline stages
Stage 0 – Data acquisition (code/datasets/download_data.py)
Downloads the raw CSV to data/raw/diabetes.csv. If there is no internet access, a synthetic dataset with the same schema is generated, so the pipeline can always be demonstrated.

Stage 1 – Data engineering (code/datasets/prepare_data.py)
loads data/raw/diabetes.csv;
cleaning: zeros in Glucose, BloodPressure, SkinThickness, Insulin, BMI are physiologically impossible → replaced with NaN and imputed with the column median; rows outside 1.5·IQR in any feature are removed as outliers;
stratified 80/20 split → data/processed/train.csv, data/processed/test.csv.
Note: applying the IQR rule to all 8 features removes a substantial share of rows (the Pima dataset has heavy tails in Insulin and DiabetesPedigreeFunction) — this is acceptable for this assignment.

Stage 2 – Model engineering (code/models/)
feature engineering: StandardScaler inside a sklearn Pipeline fitted on the training data only (no leakage) and reused for evaluation and serving;
trains LogisticRegression and RandomForestClassifier; evaluates accuracy / precision / recall / F1 / ROC-AUC on the test set;
logs params, metrics and each model to MLflow (mlruns/);
packages the best model (by F1) into models/model.joblib (atomic file replace) and writes the test metrics to models/metrics.json.
Stage 3 – Deployment (code/deployment/)
docker-compose.yml builds and starts two separate containers:

api (FastAPI): POST /predict, GET /health, GET /model_info.The host folder models/ is bind-mounted into the container, and the API hot-reloads the model whenever model.joblib changes — every 5-minute retrain is picked up without restarting the container.
app (Streamlit): input fields for the 8 features, a Predict button, and a prediction area showing the label + probability returned by the API (calls http://api:8000/predict over the compose network).
API example
curl -X POST http://localhost:8000/predict \  -H "Content-Type: application/json" \  -d '{"Pregnancies": 6, "Glucose": 148, "BloodPressure": 72, "SkinThickness": 35,       "Insulin": 0, "BMI": 33.6, "DiabetesPedigreeFunction": 0.627, "Age": 50}'# → {"prediction": 1, "label": "diabetic", "probability_diabetes": 0.83}
On Windows, the easiest way to test the API is the Swagger UI: http://localhost:8000/docs → POST /predict → Try it out.

Automation
scheduler.py (APScheduler, interval trigger, max_instances=1):

runs the full pipeline immediately on startup and then every 5 minutes;
a failing run is logged but does not stop the scheduler;
overlapping runs are prevented (max_instances=1).
How to verify automation: every 5 minutes new runs appear in the MLflow UI, models/metrics.json gets a fresh trained_at timestamp, and docker ps keeps showing both containers.

Alternative: cron — */5 * * * * cd /path/to/repo && ./venv/bin/python run_pipeline.py >> pipeline.log 2>&1

MLflow 3.x compatibility notes
The code works with both MLflow 2.x and 3.x. Two 3.x specifics are already handled inside code/models/train_model.py:

Filesystem tracking backend (./mlruns) is blocked by default in MLflow ≥ 3.1.train_model.py sets os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true") at the top, so the pipeline works out of the box. When starting the MLflow UI, set the same variable yourself (commands above).
skops security check. MLflow 3.x saves sklearn models via skops, which blocks sklearn.tree._tree.Tree (RandomForest internals) as an "untrusted type". Since the model is trained locally by this very pipeline, the code passes skops_trusted_types=["sklearn.tree._tree.Tree"] to mlflow.sklearn.log_model.
Troubleshooting
Error message	Fix
No module named 'code.datasets'; 'code' is not a package	__init__.py files are missing — without them Python imports the stdlib code module instead of the code/ folder. Create three empty files (commands below).
ModuleNotFoundError: No module named 'pandas' (or sklearn / mlflow)	venv not activated or dependencies not installed: activate venv → pip install -r requirements.txt
MlflowException: ... filesystem tracking backend ... is in maintenance mode	Already fixed in train_model.py; for mlflow ui set MLFLOW_ALLOW_FILE_STORE=true
Untrusted types found ... ['sklearn.tree._tree.Tree']	Already fixed in train_model.py (skops_trusted_types)
RuntimeError: Docker Compose not found	Install Docker Desktop and make sure it is running
Bind for 0.0.0.0:8000 failed: port is already allocated	Ports 8000 / 8501 are busy → change the left side of the mapping in code/deployment/docker-compose.yml (e.g. "8080:8000"); API_URL needs no change (internal compose network)
...running scripts is disabled (venv activation in PowerShell)	Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
can't open file 'run_pipeline.py': [Errno 2]	Wrong folder (run from repo root) or the file is actually run_pipeline.py.txt (enable extension display in Explorer)
Creating the __init__.py files:

# Linux / macOStouch code/__init__.py code/datasets/__init__.py code/models/__init__.py
:: Windows (cmd)type nul > code\__init__.pytype nul > code\datasets\__init__.pytype nul > code\models\__init__.py
Or one cross-platform command from the repo root:

python -c "from pathlib import Path; [Path(p).touch() for p in ['code/__init__.py','code/datasets/__init__.py','code/models/__init__.py']]"
Notes
The dataset (Pima Indians Diabetes) is allowed — it is neither CelebFaces nor the smoking-status dataset.
data/processed/, models/ and mlruns/ are generated by the pipeline and gitignored. The raw CSV is tiny (~25 KB) and can be committed; otherwise Stage 0 downloads it.
The model served by the API is models/model.joblib; the copies inside mlruns/ are for experiment tracking only.
If the API logs scikit-learn unpickling warnings, align the scikit-learn version in code/deployment/api/requirements.txt with your local one.
Grading checklist
Criterion	Where it is implemented
Data engineering stage works (1 pt)	code/datasets/prepare_data.py → data/processed/{train,test}.csv
Model engineering stage works (1 pt)	code/models/train_model.py → MLflow logs, models/model.joblib, models/metrics.json
Deployment: API + app in separate Docker containers, app displays predictions from the API (1 pt)	code/deployment/ — FastAPI API + Streamlit app via docker-compose.yml
Complete pipeline automated, runs every 5 min (1 pt)	run_pipeline.py + scheduler.py
Logical repository structure (1 pt)	follows the recommended layout