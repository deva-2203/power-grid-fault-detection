import os
from fastapi import FastAPI, HTTPException
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
