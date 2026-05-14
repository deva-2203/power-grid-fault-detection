#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_HOST="${FRONTEND_HOST:-127.0.0.1}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
API_URL="http://${BACKEND_HOST}:${BACKEND_PORT}/api"

BACKEND_PID=""
FRONTEND_PID=""
CLEANED_UP=0

cleanup() {
  if [[ "$CLEANED_UP" -eq 1 ]]; then
    return
  fi
  CLEANED_UP=1

  echo
  echo "[run] Stopping servers..."
  if [[ -n "$FRONTEND_PID" ]] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
    kill "$FRONTEND_PID" 2>/dev/null || true
  fi
  if [[ -n "$BACKEND_PID" ]] && kill -0 "$BACKEND_PID" 2>/dev/null; then
    kill "$BACKEND_PID" 2>/dev/null || true
  fi
  wait "$FRONTEND_PID" 2>/dev/null || true
  wait "$BACKEND_PID" 2>/dev/null || true
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[run] Missing required command: $1"
    exit 1
  fi
}

ensure_python_env() {
  require_command "$PYTHON_BIN"

  if [[ ! -x ".venv/bin/python" ]]; then
    echo "[setup] Creating Python virtual environment in .venv..."
    "$PYTHON_BIN" -m venv .venv
  fi

  if ! .venv/bin/python -c "import fastapi, uvicorn, pandas, sklearn, xgboost, imblearn, networkx, joblib, matplotlib, seaborn" >/dev/null 2>&1; then
    echo "[setup] Installing Python dependencies..."
    .venv/bin/python -m pip install -r requirements.txt
  fi
}

ensure_frontend_env() {
  require_command npm

  if [[ ! -d "frontend/node_modules" ]]; then
    echo "[setup] Installing frontend dependencies..."
    (cd frontend && npm install)
  fi
}

ensure_model_artifacts() {
  local required=(
    "outputs/day1/splits/X_test_raw.csv"
    "outputs/day1/splits/y_test.csv"
    "outputs/day2/splits/selected_features.csv"
    "outputs/day2/standard_scaler.joblib"
    "outputs/learning_algorithms/models/rf_classifier.joblib"
    "outputs/learning_algorithms/models/xgboost_classifier.joblib"
    "outputs/learning_algorithms/models/pca.joblib"
    "outputs/learning_algorithms/models/isolation_forest.joblib"
    "outputs/learning_algorithms/models/kmeans.joblib"
    "outputs/learning_algorithms/models/ensemble_meta.joblib"
  )
  local missing=()

  for path in "${required[@]}"; do
    if [[ ! -f "$path" ]]; then
      missing+=("$path")
    fi
  done

  if (( ${#missing[@]} > 0 )); then
    echo "[setup] Missing trained artifacts. Rebuilding the ML pipeline now."
    echo "[setup] This can take a while on the first run."
    .venv/bin/python eda.py
    .venv/bin/python feature_engineering.py
    .venv/bin/python learning_algorithms.py
  fi
}

start_backend() {
  echo "[run] Starting backend at http://${BACKEND_HOST}:${BACKEND_PORT}"
  .venv/bin/python -m uvicorn api:app --host "$BACKEND_HOST" --port "$BACKEND_PORT" &
  BACKEND_PID=$!
  sleep 2

  if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    echo "[run] Backend failed to start."
    wait "$BACKEND_PID"
    exit 1
  fi
}

start_frontend() {
  echo "[run] Starting frontend at http://${FRONTEND_HOST}:${FRONTEND_PORT}"
  (
    cd frontend
    VITE_API_URL="$API_URL" npm run dev -- --host "$FRONTEND_HOST" --port "$FRONTEND_PORT"
  ) &
  FRONTEND_PID=$!
}

trap cleanup INT TERM EXIT

ensure_python_env
ensure_frontend_env
ensure_model_artifacts

echo
echo "[run] Dashboard URL: http://${FRONTEND_HOST}:${FRONTEND_PORT}"
echo "[run] API URL:       ${API_URL}"
echo "[run] Press Ctrl+C to stop both servers."
echo

start_backend
start_frontend

wait -n "$BACKEND_PID" "$FRONTEND_PID"
