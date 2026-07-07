# End-to-End MLOps Project — Self-Healing Customer Churn Prediction Service

> **Goal:** A production-grade ML system with automated training, deployment, monitoring,
> and drift-triggered retraining — built 100% on free tools, zero heavy compute on the local laptop.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [High-Level Design (HLD)](#2-high-level-design-hld)
3. [Low-Level Design (LLD)](#3-low-level-design-lld)
4. [Tools & Free Tier Summary](#4-tools--free-tier-summary)
5. [Tool-by-Tool Setup & Configuration](#5-tool-by-tool-setup--configuration)
6. [End-to-End Task Breakdown (Step by Step)](#6-end-to-end-task-breakdown-step-by-step)
7. [CI/CD Workflows in Detail](#7-cicd-workflows-in-detail)
8. [Monitoring & Auto-Retraining Loop](#8-monitoring--auto-retraining-loop)
9. [Testing Strategy](#9-testing-strategy)
10. [Interview Talking Points](#10-interview-talking-points)

---

## 1. Project Overview

| Item | Detail |
|---|---|
| **Problem** | Predict which telecom customers will churn (binary classification) |
| **Dataset** | Telco Customer Churn (Kaggle, ~7,000 rows, tabular) — small on purpose so CPU training on free runners works |
| **Model** | XGBoost / scikit-learn (Logistic Regression baseline → XGBoost champion) |
| **Serving** | REST API (FastAPI) in Docker on Hugging Face Spaces |
| **Key differentiator** | Closed loop: drift detection **automatically triggers retraining**, and a challenger model replaces the champion only if it wins |

### Success Criteria
- Every experiment reproducible (code + data + params versioned together)
- `git push` → tests → train → evaluate → conditional deploy, with **zero manual steps**
- Live public API endpoint anyone can `curl`
- Weekly scheduled drift check; drift above threshold auto-opens a retraining run
- All green badges on the README (CI, tests, deploy)

---

## 2. High-Level Design (HLD)

### 2.1 Architecture Diagram

```
                        ┌─────────────────────────────────────────────┐
                        │                 GITHUB REPO                 │
                        │  code + DVC pointers + workflows + configs  │
                        └───────┬──────────────────────┬──────────────┘
                                │ push / PR            │ cron (weekly)
                                ▼                      ▼
              ┌─────────────────────────┐   ┌──────────────────────────┐
              │   CI PIPELINE (GH       │   │  MONITORING PIPELINE     │
              │   Actions)              │   │  (GH Actions, scheduled) │
              │  lint → unit tests →    │   │  pull prediction logs →  │
              │  data validation        │   │  Evidently drift report  │
              └───────────┬─────────────┘   └───────────┬──────────────┘
                          │ merge to main               │ drift > threshold?
                          ▼                             │ yes → dispatch
              ┌─────────────────────────┐               │ training workflow
              │  TRAINING PIPELINE      │◄──────────────┘
              │  (GH Actions runner)    │
              │  dvc pull → preprocess  │
              │  → train → evaluate     │
              └───────────┬─────────────┘
                          │ log run + model
                          ▼
              ┌─────────────────────────┐
              │  DAGSHUB (free hosted)  │
              │  • MLflow tracking      │
              │  • MLflow registry      │
              │  • DVC remote storage   │
              └───────────┬─────────────┘
                          │ challenger beats champion?
                          │ yes → promote + deploy
                          ▼
              ┌─────────────────────────┐        ┌─────────────────────┐
              │  HUGGING FACE SPACES    │◄───────│  Docker image build │
              │  FastAPI + Docker       │        │  (in GH Actions)    │
              │  /predict /health       │        └─────────────────────┘
              └───────────┬─────────────┘
                          │ every request logged (inputs + prediction)
                          ▼
              ┌─────────────────────────┐
              │  PREDICTION LOGS        │
              │  (HF Space persistent   │
              │   storage / DagsHub)    │──────► feeds MONITORING PIPELINE
              └─────────────────────────┘
```

### 2.2 Components

| # | Component | Responsibility | Tool |
|---|---|---|---|
| C1 | Source control | Code, configs, pipeline definitions | GitHub |
| C2 | Data & model versioning | Version datasets/artifacts outside git | DVC |
| C3 | Artifact storage | Remote store for DVC + MLflow artifacts | DagsHub (free 10 GB) |
| C4 | Experiment tracking | Params, metrics, artifacts per run | MLflow (hosted on DagsHub) |
| C5 | Model registry | Versioned models with stage aliases (champion/challenger) | MLflow Registry |
| C6 | Orchestration & CI/CD | Run pipelines on push/schedule/dispatch | GitHub Actions |
| C7 | Model serving | REST inference API | FastAPI + Uvicorn |
| C8 | Containerization | Reproducible runtime | Docker |
| C9 | Hosting | Free public deployment | Hugging Face Spaces |
| C10 | Monitoring | Data drift + prediction drift reports | Evidently AI |
| C11 | Data validation | Schema & quality gates | Pandera |
| C12 | Config management | Single source of truth for params | YAML (`params.yaml`) + Pydantic Settings |

### 2.3 Data Flow (happy path)

1. Raw CSV versioned with DVC → pushed to DagsHub storage.
2. `preprocess` stage: clean, encode, split → processed data (DVC-tracked).
3. `train` stage: fit model, log everything to MLflow.
4. `evaluate` stage: metrics on held-out test set → `metrics.json`.
5. `promote` step: compare challenger F1/AUC vs current champion in registry → promote if better.
6. `deploy` step: build Docker image, push to HF Space; API loads champion model at startup.
7. API logs every request/response → weekly Evidently job compares live inputs vs training reference → drift triggers step 2 again with fresh data.

### 2.4 Design Decisions (defend these in interviews)

- **GitHub Actions as orchestrator** instead of Airflow: free, serverless, sufficient for batch cadence; Airflow would need an always-on server (costs money / laptop load).
- **Champion/Challenger promotion** instead of blind auto-deploy: prevents a bad retrain from degrading production.
- **Tabular problem on CPU** instead of deep learning: the MLOps engineering is the product here; training must be fast/free to allow frequent full-pipeline runs.
- **DagsHub** instead of self-hosting MLflow: a self-hosted MLflow server needs a VM; DagsHub gives hosted MLflow + DVC remote + it's the industry-standard MLflow API (skills transfer 1:1).

---

## 3. Low-Level Design (LLD)

### 3.1 Repository Structure

```
churn-mlops/
├── .github/
│   └── workflows/
│       ├── ci.yml                 # lint + tests + data validation (on PR/push)
│       ├── train.yml              # full training pipeline (on merge / dispatch)
│       ├── deploy.yml             # build & push to HF Space (on model promotion)
│       └── monitor.yml            # weekly drift check (cron)
├── data/
│   ├── raw/                       # DVC-tracked (only .dvc pointer in git)
│   └── processed/                 # DVC-tracked
├── src/
│   └── churn/
│       ├── __init__.py
│       ├── config.py              # Pydantic Settings — loads params.yaml + env vars
│       ├── data/
│       │   ├── ingest.py          # download / load raw data
│       │   ├── validate.py        # Pandera schema checks
│       │   └── preprocess.py      # cleaning, encoding, train/test split
│       ├── features/
│       │   └── build_features.py  # feature engineering (tenure buckets, charges ratios…)
│       ├── models/
│       │   ├── train.py           # fit + MLflow logging
│       │   ├── evaluate.py        # metrics, confusion matrix, SHAP plot artifacts
│       │   └── promote.py         # champion vs challenger comparison + registry update
│       └── monitoring/
│           ├── log_predictions.py # request/response logging helpers
│           └── drift_check.py     # Evidently report + threshold decision
├── app/
│   ├── main.py                    # FastAPI app
│   ├── schemas.py                 # Pydantic request/response models
│   └── model_loader.py            # pull champion model from MLflow registry
├── tests/
│   ├── test_preprocess.py
│   ├── test_features.py
│   ├── test_api.py                # FastAPI TestClient
│   └── test_promote.py
├── notebooks/                     # EDA only — nothing production lives here
│   └── 01_eda.ipynb
├── dvc.yaml                       # pipeline stage definitions
├── params.yaml                    # ALL hyperparameters & thresholds
├── dvc.lock                       # auto-generated
├── Dockerfile
├── requirements.txt
├── requirements-dev.txt           # pytest, ruff, pandera, evidently
├── pyproject.toml                 # ruff config, package metadata
├── .dvc/config                    # DVC remote = DagsHub
├── .env.example                   # names of secrets (never real values)
└── README.md                      # architecture diagram, badges, demo GIF
```

### 3.2 `params.yaml` (single source of truth)

```yaml
data:
  raw_path: data/raw/telco_churn.csv
  test_size: 0.2
  random_state: 42
  target: Churn

features:
  drop_cols: [customerID]
  tenure_bins: [0, 12, 24, 48, 72]

train:
  model_type: xgboost          # or "logreg" for baseline
  n_estimators: 300
  max_depth: 5
  learning_rate: 0.05
  scale_pos_weight: 2.8        # class imbalance handling

evaluate:
  primary_metric: f1           # promotion decided on this
  min_improvement: 0.002       # challenger must beat champion by this margin

monitoring:
  drift_threshold: 0.3         # share of drifted columns that triggers retrain
  reference_window_days: 30
```

### 3.3 `dvc.yaml` (pipeline definition)

```yaml
stages:
  preprocess:
    cmd: python -m churn.data.preprocess
    deps: [data/raw/telco_churn.csv, src/churn/data/preprocess.py]
    params: [data]
    outs: [data/processed/train.parquet, data/processed/test.parquet]

  train:
    cmd: python -m churn.models.train
    deps: [data/processed/train.parquet, src/churn/models/train.py]
    params: [train]
    outs: [models/model.pkl]

  evaluate:
    cmd: python -m churn.models.evaluate
    deps: [models/model.pkl, data/processed/test.parquet]
    params: [evaluate]
    metrics:
      - metrics.json: {cache: false}
```

### 3.4 Key Module Contracts

**`config.py`**
```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    mlflow_tracking_uri: str      # from env: MLFLOW_TRACKING_URI
    mlflow_tracking_username: str # DagsHub user
    mlflow_tracking_password: str # DagsHub token
    model_name: str = "churn-xgb"

    class Config:
        env_file = ".env"
```

**`validate.py` (Pandera schema — the CI data-quality gate)**
```python
import pandera as pa

schema = pa.DataFrameSchema({
    "tenure":         pa.Column(int, pa.Check.ge(0)),
    "MonthlyCharges": pa.Column(float, pa.Check.in_range(0, 500)),
    "TotalCharges":   pa.Column(float, nullable=True, coerce=True),
    "Churn":          pa.Column(str, pa.Check.isin(["Yes", "No"])),
    # ... every column, exhaustively
})
```

**`train.py` (MLflow logging contract)**
```python
import mlflow

mlflow.set_experiment("churn-prediction")
with mlflow.start_run():
    mlflow.log_params(params["train"])
    # fit model ...
    mlflow.log_metrics({"train_f1": f1, "train_auc": auc})
    mlflow.sklearn.log_model(model, "model",
                             registered_model_name="churn-xgb")
```

**`promote.py` (champion/challenger logic)**
```python
from mlflow import MlflowClient

client = MlflowClient()
challenger = get_latest_version("churn-xgb")
champion   = get_version_by_alias("churn-xgb", "champion")  # may be None on first run

if champion is None or challenger_f1 > champion_f1 + min_improvement:
    client.set_registered_model_alias("churn-xgb", "champion", challenger.version)
    print("PROMOTED=true")   # captured by the workflow to trigger deploy
else:
    print("PROMOTED=false")
```

**`app/main.py` (serving contract)**
```python
from fastapi import FastAPI
from app.schemas import CustomerFeatures, PredictionOut
from app.model_loader import load_champion

app = FastAPI(title="Churn Prediction API")
model = load_champion()          # models:/churn-xgb@champion

@app.get("/health")
def health(): return {"status": "ok", "model_version": model.version}

@app.post("/predict", response_model=PredictionOut)
def predict(payload: CustomerFeatures):
    proba = model.predict_proba(payload.to_frame())[0, 1]
    log_prediction(payload, proba)          # for drift monitoring
    return {"churn_probability": round(float(proba), 4),
            "churn": proba >= 0.5}
```

### 3.5 API Specification

| Method | Path | Request | Response | Notes |
|---|---|---|---|---|
| GET | `/health` | — | `{status, model_version}` | Used by HF Space healthcheck |
| POST | `/predict` | `CustomerFeatures` JSON (all 19 features, Pydantic-validated) | `{churn_probability, churn}` | 422 on invalid input |
| GET | `/model-info` | — | model version, training date, metrics | transparency endpoint |

### 3.6 Dockerfile

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ app/
COPY src/ src/
ENV PORT=7860
# HF Spaces expects port 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
```

### 3.7 Secrets Inventory

| Secret name | Where set | Used by |
|---|---|---|
| `DAGSHUB_TOKEN` | GitHub repo secrets + HF Space secrets | DVC pull/push, MLflow auth |
| `MLFLOW_TRACKING_URI` | GitHub secrets + HF Space | all MLflow calls |
| `MLFLOW_TRACKING_USERNAME` | same | MLflow auth |
| `MLFLOW_TRACKING_PASSWORD` | same (= DagsHub token) | MLflow auth |
| `HF_TOKEN` | GitHub secrets only | pushing to HF Space from deploy workflow |

---

## 4. Tools & Free Tier Summary

| Tool | Purpose | Free tier limit | Enough? |
|---|---|---|---|
| GitHub | code + Actions CI/CD | 2,000 Action-min/month (private), unlimited public | ✅ use a public repo → generous |
| DagsHub | hosted MLflow + DVC remote | 10 GB storage, unlimited public repos | ✅ dataset ~1 MB |
| DVC | data versioning | open source | ✅ |
| MLflow | tracking + registry | open source (hosted free via DagsHub) | ✅ |
| Pandera | data validation | open source | ✅ |
| XGBoost / scikit-learn | modeling | open source | ✅ CPU-friendly |
| FastAPI + Uvicorn | serving | open source | ✅ |
| Docker | container | free | built in CI, **not** on laptop |
| Hugging Face Spaces | hosting | free CPU Space (2 vCPU, 16 GB) | ✅ |
| Evidently AI | drift detection | open source | ✅ |
| Kaggle | dataset + free notebooks | free | ✅ |
| ruff + pytest | lint + tests | open source | ✅ |

**Laptop requirements:** just Git, Python, and a code editor. All training, Docker builds, and serving happen on GitHub Actions runners and HF Spaces. (Even Docker Desktop is optional locally.)

---

## 5. Tool-by-Tool Setup & Configuration

### 5.1 Local Environment (lightweight)

```bash
# 1. Install Python 3.11 (python.org) and Git
# 2. Create project
mkdir churn-mlops && cd churn-mlops
git init
python -m venv .venv
source .venv/Scripts/activate        # Git Bash on Windows

# 3. Install only dev-time deps locally (no training locally!)
pip install dvc[s3] mlflow pandera fastapi uvicorn pytest ruff pydantic-settings
pip freeze > requirements-dev.txt
```

> You'll write and unit-test code locally on tiny data samples; full training runs in CI.

### 5.2 GitHub

1. Create a **public** repo `churn-mlops` (public = unlimited free Actions minutes).
2. Push the initial structure.
3. `Settings → Secrets and variables → Actions → New repository secret` — add every secret from §3.7.
4. `Settings → Actions → General → Workflow permissions` → "Read and write" (needed for auto-commits of `dvc.lock`).

### 5.3 DagsHub (MLflow + DVC remote in one)

1. Sign up at dagshub.com **with your GitHub account**.
2. `Create → New Repository → Connect a repository → GitHub` → select `churn-mlops`. DagsHub now mirrors it and provisions:
   - MLflow server: `https://dagshub.com/<user>/churn-mlops.mlflow`
   - DVC remote: `https://dagshub.com/<user>/churn-mlops.dvc`
3. Get your token: DagsHub → your avatar → `Your Settings → Tokens → Generate`. This one token is used for both DVC and MLflow auth.
4. Locally create `.env` (git-ignored):
   ```
   MLFLOW_TRACKING_URI=https://dagshub.com/<user>/churn-mlops.mlflow
   MLFLOW_TRACKING_USERNAME=<user>
   MLFLOW_TRACKING_PASSWORD=<token>
   ```

### 5.4 DVC

```bash
dvc init
dvc remote add origin https://dagshub.com/<user>/churn-mlops.dvc
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <user>
dvc remote modify origin --local password <token>
dvc remote default origin

# track data
dvc add data/raw/telco_churn.csv
git add data/raw/telco_churn.csv.dvc data/raw/.gitignore .dvc/config
git commit -m "Track raw data with DVC"
dvc push          # uploads data to DagsHub storage
```

> `--local` keeps credentials in `.dvc/config.local` (git-ignored). In CI, auth comes from env vars instead.

### 5.5 MLflow (client side — server is DagsHub)

Nothing to install server-side. In code, the tracking URI + credentials come from env vars (§5.3). Verify once:

```bash
python -c "import mlflow, os; mlflow.set_experiment('smoke'); \
           mlflow.start_run(); mlflow.log_metric('ok', 1); mlflow.end_run()"
# → open the DagsHub repo → 'Experiments' tab → the run appears
```

### 5.6 Kaggle (dataset)

1. kaggle.com → Account → `Create New API Token` → downloads `kaggle.json`.
2. Place at `~/.kaggle/kaggle.json` (or set `KAGGLE_USERNAME`/`KAGGLE_KEY` env vars — do this in GitHub secrets too if CI needs to re-download).
3. ```bash
   pip install kaggle
   kaggle datasets download -d blastchar/telco-customer-churn -p data/raw --unzip
   ```

### 5.7 Hugging Face Spaces (serving host)

1. Sign up at huggingface.co → `Settings → Access Tokens → New token` (**write** scope) → save as `HF_TOKEN` in GitHub secrets.
2. `New Space` → name `churn-api` → SDK: **Docker** → hardware: CPU basic (free).
3. Space `Settings → Variables and secrets` → add `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD` (so the container can pull the champion model at startup).
4. Deployment = `git push` of the Dockerfile + `app/` to the Space repo — the deploy workflow (§7.3) does this automatically.
5. Live URL: `https://<user>-churn-api.hf.space` → `POST /predict`.

### 5.8 Evidently (drift)

```bash
pip install evidently
```
No account needed — reports are generated inside the scheduled workflow (§8) and uploaded as workflow artifacts / committed HTML.

### 5.9 ruff + pytest (quality gates)

`pyproject.toml`:
```toml
[tool.ruff]
line-length = 100
src = ["src", "app", "tests"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

---

## 6. End-to-End Task Breakdown (Step by Step)

Work in this exact order. Each phase ends with a working, committed, pushed state.

### Phase 0 — Foundations (Day 1–2)
- [ ] 0.1 Create GitHub repo (public) with README, `.gitignore` (python + `.env` + `data/`)
- [ ] 0.2 Local venv, install dev deps, `pyproject.toml` with ruff/pytest config
- [ ] 0.3 Create the full folder skeleton from §3.1 with empty modules
- [ ] 0.4 Connect DagsHub, generate token, set all GitHub secrets
- [ ] 0.5 First CI workflow (`ci.yml`) running ruff + a placeholder test — get a green badge

### Phase 1 — Data Layer (Day 3–5)
- [ ] 1.1 Download Telco dataset via Kaggle API into `data/raw/`
- [ ] 1.2 `dvc init`, add remote, `dvc add` + `dvc push`
- [ ] 1.3 Quick EDA notebook (Kaggle/Colab — not laptop): class balance, nulls, dtypes
- [ ] 1.4 Write exhaustive Pandera schema in `validate.py`
- [ ] 1.5 Write `preprocess.py`: fix `TotalCharges` blanks, encode categoricals, stratified split, save parquet
- [ ] 1.6 Unit tests for preprocessing edge cases (blank charges, unseen category)
- [ ] 1.7 Add `preprocess` stage to `dvc.yaml`; run `dvc repro`; commit `dvc.lock`

### Phase 2 — Training & Tracking (Day 6–9)
- [ ] 2.1 `train.py`: baseline LogisticRegression, full MLflow logging
- [ ] 2.2 `evaluate.py`: F1, ROC-AUC, precision/recall, confusion-matrix PNG logged as MLflow artifact; write `metrics.json`
- [ ] 2.3 Switch to XGBoost via `params.yaml` (`model_type`), verify it beats baseline in MLflow UI
- [ ] 2.4 Add `train` + `evaluate` stages to `dvc.yaml`
- [ ] 2.5 Register model to MLflow Registry as `churn-xgb`
- [ ] 2.6 `promote.py` with champion/challenger alias logic + `min_improvement` margin
- [ ] 2.7 Unit test promotion logic with mocked registry

### Phase 3 — Training Pipeline in CI (Day 10–12)
- [ ] 3.1 `train.yml` workflow: checkout → setup python → `dvc pull` → `dvc repro` → `promote.py`
- [ ] 3.2 Auth via env vars from secrets (DVC + MLflow)
- [ ] 3.3 Auto-commit updated `dvc.lock` + `metrics.json` back to main
- [ ] 3.4 Expose `PROMOTED` as a workflow output
- [ ] 3.5 Trigger: `workflow_dispatch` + push to `main` touching `src/` or `params.yaml`
- [ ] 3.6 Run it end-to-end twice; verify second run with same params does NOT promote (margin logic works)

### Phase 4 — Serving (Day 13–16)
- [ ] 4.1 Pydantic request/response schemas mirroring the 19 features
- [ ] 4.2 `model_loader.py`: load `models:/churn-xgb@champion` from registry at startup
- [ ] 4.3 FastAPI endpoints `/health`, `/predict`, `/model-info`
- [ ] 4.4 Prediction logging: append request features + prediction + timestamp to a log file / DagsHub
- [ ] 4.5 API tests with `TestClient` (valid input, missing field → 422, out-of-range value)
- [ ] 4.6 Dockerfile (port 7860); create HF Space, set its secrets
- [ ] 4.7 `deploy.yml`: push `app/ + src/ + Dockerfile` to the Space repo using `HF_TOKEN`; runs only when `PROMOTED=true` (or manual dispatch)
- [ ] 4.8 Smoke test the live URL with `curl` from the workflow itself

### Phase 5 — Monitoring & Self-Healing (Day 17–20)
- [ ] 5.1 `drift_check.py`: load reference (training data) + current (prediction logs), run Evidently `DataDriftPreset`, save HTML report, print drifted-column share
- [ ] 5.2 `monitor.yml`: weekly cron → run drift check → upload HTML report as artifact
- [ ] 5.3 If drift share > `drift_threshold` → `workflow_dispatch` the training pipeline via `gh api` (this is the self-healing loop)
- [ ] 5.4 Demo the loop: write a script that sends 200 deliberately-shifted requests to the live API, then manually run `monitor.yml` and watch it trigger retraining. **Record this as a GIF for the README.**

### Phase 6 — Polish (Day 21–23)
- [ ] 6.1 README: architecture diagram, badges (CI, train, deploy), live API link, demo GIF, "How it works" section
- [ ] 6.2 `/model-info` returns training date + metrics for transparency
- [ ] 6.3 Blog post (dev.to / Medium): "I built a self-healing ML system for $0"
- [ ] 6.4 Pin the repo on your GitHub profile; add it to LinkedIn Featured

---

## 7. CI/CD Workflows in Detail

### 7.1 `ci.yml` — every PR and push

```yaml
name: CI
on: [push, pull_request]
jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.11"}
      - run: pip install -r requirements-dev.txt
      - run: ruff check .
      - run: pytest -q
```

### 7.2 `train.yml` — the training pipeline

```yaml
name: Train
on:
  workflow_dispatch:
  push:
    branches: [main]
    paths: ["src/**", "params.yaml", "dvc.yaml"]
jobs:
  train:
    runs-on: ubuntu-latest
    outputs:
      promoted: ${{ steps.promote.outputs.promoted }}
    env:
      MLFLOW_TRACKING_URI:      ${{ secrets.MLFLOW_TRACKING_URI }}
      MLFLOW_TRACKING_USERNAME: ${{ secrets.MLFLOW_TRACKING_USERNAME }}
      MLFLOW_TRACKING_PASSWORD: ${{ secrets.DAGSHUB_TOKEN }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.11"}
      - run: pip install -r requirements.txt -r requirements-dev.txt
      - name: Pull data
        run: |
          dvc remote modify origin --local auth basic
          dvc remote modify origin --local user ${{ secrets.MLFLOW_TRACKING_USERNAME }}
          dvc remote modify origin --local password ${{ secrets.DAGSHUB_TOKEN }}
          dvc pull
      - name: Run pipeline
        run: dvc repro
      - name: Promote if challenger wins
        id: promote
        run: |
          python -m churn.models.promote | tee promote.log
          echo "promoted=$(grep -oP 'PROMOTED=\K\w+' promote.log)" >> "$GITHUB_OUTPUT"
      - name: Commit lockfile & metrics
        run: |
          git config user.name "github-actions"
          git config user.email "actions@github.com"
          git add dvc.lock metrics.json && git commit -m "ci: update pipeline outputs" || true
          git push
  deploy:
    needs: train
    if: needs.train.outputs.promoted == 'true'
    uses: ./.github/workflows/deploy.yml
    secrets: inherit
```

### 7.3 `deploy.yml` — push to Hugging Face Space

```yaml
name: Deploy
on:
  workflow_call:
  workflow_dispatch:
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Push to HF Space
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          pip install huggingface_hub
          python - <<'EOF'
          from huggingface_hub import HfApi
          api = HfApi()
          api.upload_folder(folder_path=".", repo_id="<user>/churn-api",
                            repo_type="space",
                            allow_patterns=["app/**","src/**","Dockerfile","requirements.txt"])
          EOF
      - name: Smoke test
        run: |
          sleep 120   # wait for Space rebuild
          curl -sf https://<user>-churn-api.hf.space/health
```

### 7.4 `monitor.yml` — weekly drift check + self-healing trigger

```yaml
name: Monitor
on:
  schedule:
    - cron: "23 6 * * 1"     # Mondays 06:23 UTC
  workflow_dispatch:
jobs:
  drift:
    runs-on: ubuntu-latest
    env:
      MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
      MLFLOW_TRACKING_USERNAME: ${{ secrets.MLFLOW_TRACKING_USERNAME }}
      MLFLOW_TRACKING_PASSWORD: ${{ secrets.DAGSHUB_TOKEN }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.11"}
      - run: pip install -r requirements.txt evidently
      - name: Run drift check
        id: drift
        run: |
          python -m churn.monitoring.drift_check | tee drift.log
          echo "drifted=$(grep -oP 'DRIFTED=\K\w+' drift.log)" >> "$GITHUB_OUTPUT"
      - uses: actions/upload-artifact@v4
        with: {name: drift-report, path: reports/drift_report.html}
      - name: Trigger retraining
        if: steps.drift.outputs.drifted == 'true'
        env: {GH_TOKEN: '${{ secrets.GITHUB_TOKEN }}'}
        run: gh workflow run train.yml --ref main
```

---

## 8. Monitoring & Auto-Retraining Loop

### 8.1 `drift_check.py` core logic

```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=train_df, current_data=recent_requests_df)
report.save_html("reports/drift_report.html")

result = report.as_dict()
share = result["metrics"][0]["result"]["share_of_drifted_columns"]
print(f"DRIFTED={'true' if share > params['monitoring']['drift_threshold'] else 'false'}")
```

### 8.2 The full self-healing loop

```
live traffic → prediction logs → weekly Evidently comparison vs training reference
     → share_of_drifted_columns > 0.3 ?
         no  → upload report, done
         yes → gh workflow run train.yml
                 → dvc pull (optionally with newly labeled data)
                 → retrain → evaluate → challenger vs champion
                     challenger wins → promote alias → deploy.yml → new model live
                     challenger loses → keep champion, alert only
```

### 8.3 What "prediction logs" means concretely

The API appends one JSON line per request: `{timestamp, features..., churn_probability}`. Options for persistence (pick 1, simplest first):
1. **HF Space persistent storage** (free small disk) + an endpoint `/logs/export` the monitor workflow pulls.
2. Push a log file to the DagsHub repo daily via a background task.

---

## 9. Testing Strategy

| Layer | Tests | Tool |
|---|---|---|
| Data | Schema conformance, no target leakage, split ratios | Pandera + pytest |
| Features | Deterministic output for fixed input, unseen-category handling | pytest |
| Model | Trained model beats a dummy classifier (sanity floor); reproducible with fixed seed | pytest (small fixture data) |
| Promotion | Promotes only when margin exceeded; handles "no champion yet" | pytest + mocked MLflow client |
| API | 200 on valid payload, 422 on bad payload, `/health` reports version | FastAPI TestClient |
| Pipeline | `dvc repro` completes on a 100-row sample in CI | GitHub Actions |
| Deployment | Post-deploy `curl /health` smoke test | deploy workflow |

---

## 10. Interview Talking Points

Prepare a 2-minute story for each:

1. **"Walk me through your architecture"** → draw §2.1 from memory.
2. **"How do you prevent a bad model from reaching production?"** → CI gates + champion/challenger with a minimum-improvement margin + post-deploy smoke test + registry rollback (re-alias champion to previous version — one command).
3. **"How do you know the model is degrading?"** → no live labels for churn (they arrive months later), so monitor **input drift** as a proxy with Evidently; explain the threshold trade-off (too low = retrain churn/noise, too high = stale model).
4. **"How is retraining triggered?"** → scheduled drift job dispatches the training workflow; fully automated, human only reviews the drift report artifact.
5. **"How do you ensure reproducibility?"** → git (code) + DVC (data, `dvc.lock` pins exact data hashes) + MLflow (params/metrics/artifacts) + Docker (runtime) + fixed seeds.
6. **"Why not Airflow/Kubernetes/SageMaker?"** → right-sizing: batch cadence + small model = serverless CI runners suffice; name what you'd change at scale (Airflow/Prefect for complex DAGs, K8s + KServe for high-QPS serving, feature store for online features).
7. **"What would you improve?"** → A/B or shadow deployment, online feature store (Feast), model performance monitoring once labels arrive, alerting to Slack.

---

*Total cost of this project: $0. Total laptop load: a code editor and git.*
