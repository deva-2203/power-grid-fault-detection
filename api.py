import os
import io
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from pathlib import Path
import warnings
import pickle

import inference as _inf

app = FastAPI(title="Grid Fault Monitor API")

# Allow frontend to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT = Path(__file__).parent
_RAW_FEATS = ["tau1","tau2","tau3","tau4","p1","p2","p3","p4","g1","g2","g3","g4"]

@app.get("/api/stats")
def get_stats():
    """Returns stable statistics for computing node deviations."""
    try:
        return _inf._load_raw_stable_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/scenarios")
def get_scenarios():
    """Returns available preset scenarios."""
    try:
        return _inf.PRESET_SCENARIOS
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/feed/dataset")
def get_dataset():
    """Precomputes or loads the dataset replay feed."""
    cache_path = ROOT / "outputs" / "precomputed_feed.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, list) and data and "_sorted" in data[0]:
            return data

    try:
        X = pd.read_csv(ROOT / "outputs/day1/splits/X_test_raw.csv")[_RAW_FEATS]
        y = pd.read_csv(ROOT / "outputs/day1/splits/y_test.csv")["stabf"].astype(int)
        rows = X.to_dict("records")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = _inf.predict_sequence(rows)
        for i, (r, row) in enumerate(zip(results, rows)):
            r["raw_input"] = row
            r["_sorted"]   = True
            r["_true_label"] = int(y.iloc[i])
            # We don't need to return counterfactuals to the live UI
            if "counterfactuals" in r:
                del r["counterfactuals"]
        # Sort: stable (low prob) first, fault (high prob) last
        results.sort(key=lambda r: r["ensemble_prob"])
        
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(results, f)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/feed/scenario/{name}")
def get_scenario(name: str):
    """Runs a specific preset scenario and returns the feed."""
    if name not in _inf.PRESET_SCENARIOS:
        raise HTTPException(status_code=404, detail="Scenario not found")
        
    try:
        preset   = _inf.PRESET_SCENARIOS[name]
        clearly  = _inf.load_clearly_stable_rows()
        base_row = clearly[0]["row"] if clearly else _inf.stable_defaults()

        def _resolve(events, row):
            out = []
            for ev in events:
                sv = row[ev["feature"]] if ev["start_val"] is None else float(ev["start_val"])
                ev_end = ev["end_val"]
                if isinstance(ev_end, str) and ev_end.endswith("x"):
                    ev_end = sv * float(ev_end[:-1])
                elif ev_end == "0":
                    ev_end = 0.0
                else:
                    ev_end = float(ev_end)
                out.append({**ev, "start_val": float(sv), "end_val": ev_end})
            return out

        resolved = _resolve(preset["events"], base_row)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = _inf.run_scenario(base_row, resolved, n_steps=40)
            
        # Clean up counterfactuals to save bandwidth
        for r in results:
            if "counterfactuals" in r:
                del r["counterfactuals"]
                
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict/upload")
async def predict_upload(file: UploadFile = File(...)):
    """
    Accept a CSV file with the 12 raw features, run batch inference,
    and return per-row predictions + summary statistics.
    """
    # ── Validate file type ────────────────────────────────────────────────
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    # ── Validate columns ──────────────────────────────────────────────────
    missing = [c for c in _RAW_FEATS if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"CSV is missing required columns: {', '.join(missing)}. "
                   f"Required: {', '.join(_RAW_FEATS)}"
        )

    # Keep only the 12 raw feature columns (ignore extras like stabf, stab, etc.)
    df = df[_RAW_FEATS].copy()

    # Drop rows with NaN in any feature column
    n_before = len(df)
    df = df.dropna().reset_index(drop=True)
    n_dropped = n_before - len(df)

    if len(df) == 0:
        raise HTTPException(status_code=400, detail="CSV has no valid rows after dropping NaNs.")

    # ── Run inference ─────────────────────────────────────────────────────
    rows = df.to_dict("records")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = _inf.predict_sequence(rows)

    # Enrich each result with row index and raw input
    for i, (r, row) in enumerate(zip(results, rows)):
        r["row_index"] = i
        r["raw_input"] = row
        # Remove counterfactuals (expensive, not needed for batch)
        if "counterfactuals" in r:
            del r["counterfactuals"]

    # ── Build summary ─────────────────────────────────────────────────────
    probs   = [r["ensemble_prob"] for r in results]
    labels  = [r["ensemble_label"] for r in results]
    tiers   = [r["risk_tier"] for r in results]

    fault_count  = sum(labels)
    stable_count = len(labels) - fault_count

    summary = {
        "total_rows":       len(results),
        "rows_dropped":     n_dropped,
        "fault_count":      fault_count,
        "stable_count":     stable_count,
        "fault_pct":        round(100 * fault_count / len(results), 1),
        "mean_prob":        round(sum(probs) / len(probs), 4),
        "max_prob":         round(max(probs), 4),
        "min_prob":         round(min(probs), 4),
        "tier_green":       tiers.count("Green"),
        "tier_amber":       tiers.count("Amber"),
        "tier_red":         tiers.count("Red"),
    }

    return {"results": results, "summary": summary}
