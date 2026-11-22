"""
api.py – backend for the Phoenix web app. The trained model runs here, behind the scenes.

Start:   uvicorn api:app --host 0.0.0.0 --port 8000
Docs:    http://localhost:8000/docs   (try every endpoint in the browser)

Endpoints used by the UI
  GET  /api/health                        -> model loaded? trained on what?
  POST /api/run    (multipart files)      -> runs VFM + PTA on uploaded raw data, returns JSON for the result page
        high_freq      (required)  high-frequency minute data, CSV/XLSX
        choke_events   (required)  choke settings
        well_completion, reservoir_pvt (optional – default copies in data/ are used if not uploaded)
        well_tests     (optional)  NEW well tests of the uploaded period. Used twice:
                                   1) to RECALIBRATE the VFM causally (each day only uses tests that ended before it),
                                   2) to SCORE it honestly (R², MAPE, WAPE, RMSE: v2 recalibrated vs v1 frozen vs baseline)
  GET  /api/model                         -> model card: parameters per well + validation metrics (R², MAPE, WAPE, RMSE)
  GET  /api/files/{run_id}/{path}         -> any result file / PNG plot of that run
  GET  /api/download/{run_id}             -> ZIP with all results (Excel, CSV, plots)
  POST /api/train  (multipart files)      -> train page: retrain on uploaded data incl. well_tests, saves a new .pkl
"""
from __future__ import annotations

import io
import os
import shutil
import uuid
import zipfile
from typing import Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

import vfm_pipeline as vp

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "models", "vfm_model.pkl")
DEFAULTS = {"well_completion": os.path.join(HERE, "data", "well_completion.csv"),
            "reservoir_pvt": os.path.join(HERE, "data", "reservoir_pvt.csv")}
RUNS = os.path.join(HERE, "runs")

app = FastAPI(title="Phoenix – VFM + PTA backend")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
MODEL = vp.load_model(MODEL_PATH)


def _file(up: Optional[UploadFile], default: str | None = None):
    """Uploaded file -> file-like object with a .name (CSV or Excel); falls back to a default file."""
    if up is None:
        if default is None:
            return None
        return default
    buf = io.BytesIO(up.file.read()); buf.name = up.filename or "upload.csv"
    return buf


def _records(df: pd.DataFrame) -> list:
    import json
    return json.loads(df.to_json(orient="records", date_format="iso"))


@app.get("/api/health")
def health():
    return {"status": "ok", "model": type(MODEL.model).__name__, "version": getattr(MODEL, "version", "1"),
            "trained_on": MODEL.trained_on,
            "n_tests": MODEL.n_tests, "wells": MODEL.wells}


def _metrics_table(rows: list) -> list:
    """Validation records -> rounded rows: rate, (who), R2, MAPE_%, WAPE_%, RMSE."""
    return [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for r in rows]


@app.get("/api/model")
def model_card():
    m = MODEL.model
    params = [{"well_name": w, "J_stbd_per_psi": round(float(m.J[w]), 2),
               "reservoir_pressure_2025_01_01_psia": round(m.pr_trend[w][0], 1),
               "pressure_decline_psi_per_month": round(-m.pr_trend[w][1] * 30.44, 2),
               "water_cut_2025_01_01": round(m.wc_model[w][0], 3), "GOR_scf_stb": round(float(m.gor[w]), 1)}
              for w in MODEL.wells]
    v = MODEL.validation or {}
    return {"model": type(m).__name__, "description": "Liquid = J x (Pr(t) - Pwf); oil/water by water-cut trend; gas = oil x GOR",
            "trained_on": MODEL.trained_on, "n_tests": MODEL.n_tests, "parameters": params,
            "half_life_days": getattr(MODEL, "half_life_days", None),
            "validation": {"leave_one_out": _metrics_table(v.get("leave_one_out", [])),
                           "latest_3_tests_hidden_vs_baseline": _metrics_table(v.get("rolling_3_latest", [])),
                           "rehearsal_train_jan_jun_upload_jul_sep": _metrics_table(
                               v.get("rehearsal_train_jan_jun_upload_jul_sep", []))}}


@app.post("/api/run")
def run(high_freq: UploadFile = File(...), choke_events: UploadFile = File(...),
        well_completion: Optional[UploadFile] = File(None), reservoir_pvt: Optional[UploadFile] = File(None),
        well_tests: Optional[UploadFile] = File(None)):
    try:
        data = vp.load_inputs(_file(high_freq), _file(choke_events),
                              _file(well_completion, DEFAULTS["well_completion"]),
                              _file(reservoir_pvt, DEFAULTS["reservoir_pvt"]), _file(well_tests))
        result = vp.run_all(data, MODEL)
    except Exception as e:                                         # bad / incomplete upload -> readable message
        raise HTTPException(status_code=400, detail=f"Could not process the uploaded data: {e}")

    run_id = uuid.uuid4().hex[:8]
    out = os.path.join(RUNS, run_id)
    vp.export_all(result, out)
    base = f"/api/files/{run_id}"

    pta = result["pta"]
    pta_cols = ["event_id", "well_name", "t_shut", "duration_hr", "q_oil_stbd", "horner_slope_psi_per_cycle",
                "permeability_md", "skin", "delta_p_skin_psi", "flow_efficiency", "p_star_psia",
                "flow_regimes", "condition", "recommendation", "summary", "confidence", "rate_source",
                "fit_from_hr", "fit_to_hr", "horner_r2"]
    events = []
    for r in (_records(pta[pta_cols].round(3)) if len(pta) else []):
        r["plots"] = {k: f"{base}/pta_plots/{r['event_id']}_{k}.png"
                      for k in ["1_history", "2_loglog", "3_horner", "4_mdh"]}
        events.append(r)

    pred = result["predictions"]
    hf = result["hf_clean"]
    sensors_daily = (hf.assign(date=hf.timestamp.dt.normalize())
                       .groupby(["well_name", "date"])[vp.SENSORS + ["choke_opening_pct"]].mean().round(2).reset_index())
    scores = None
    if result["test_scores"] is not None:
        det = result["test_detail"].assign(start=result["test_detail"].start.dt.strftime("%Y-%m-%d %H:%M")).round(2)
        scores = {"n_tests": int(len(det)), "metrics": _metrics_table(result["test_scores"].to_dict("records")),
                  "per_test": _records(det)}
    downloads = {"vfm_excel": f"{base}/VFM_predictions.xlsx", "vfm_csv": f"{base}/VFM_predictions.csv",
                 "pta_excel": f"{base}/PTA_results.xlsx", "pta_csv": f"{base}/PTA_results.csv",
                 "all_zip": f"/api/download/{run_id}"}
    if scores:
        downloads["vfm_accuracy_excel"] = f"{base}/VFM_accuracy_on_uploaded_tests.xlsx"
    return {
        "run_id": run_id,
        "model": {"name": type(MODEL.model).__name__, "version": getattr(MODEL, "version", "1"),
                  "trained_on": MODEL.trained_on},
        "period": {"start": str(pred.date.min().date()), "end": str(pred.date.max().date())},
        "wells": sorted(pred.well_name.unique()),
        "vfm_mode": result["vfm_mode"],                         # recalibrated / forecast / training period
        "vfm_recalibration_steps": result["vfm_log"],
        "data_quality": _records(result["issues"]),
        "model_metrics": model_card()["validation"],          # R², MAPE, WAPE, RMSE from training validation
        "metrics_on_uploaded_tests": scores,                   # v2 recalibrated vs v1 frozen vs baseline (if tests uploaded)
        "predictions": _records(pred.assign(date=pred.date.dt.strftime("%Y-%m-%d")).round(2)),
        "sensors_daily": _records(sensors_daily.assign(date=sensors_daily.date.dt.strftime("%Y-%m-%d"))),
        "pta": events,
        "charts": {f[:-4]: f"{base}/charts/{f}" for f in sorted(os.listdir(os.path.join(out, "charts")))},
        "downloads": downloads,
    }


@app.get("/api/files/{run_id}/{path:path}")
def files(run_id: str, path: str):
    full = os.path.realpath(os.path.join(RUNS, run_id, path))
    if not full.startswith(os.path.realpath(RUNS) + os.sep) or not os.path.isfile(full):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(full)


@app.get("/api/download/{run_id}")
def download(run_id: str):
    folder = os.path.realpath(os.path.join(RUNS, run_id))
    if not folder.startswith(os.path.realpath(RUNS) + os.sep) or not os.path.isdir(folder):
        raise HTTPException(status_code=404, detail="Run not found")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, fs in os.walk(folder):
            for f in fs:
                p = os.path.join(root, f); z.write(p, os.path.relpath(p, folder))
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/zip",
                             headers={"Content-Disposition": f"attachment; filename=phoenix_results_{run_id}.zip"})


@app.post("/api/train")
def train(high_freq: UploadFile = File(...), choke_events: UploadFile = File(...), well_tests: UploadFile = File(...),
          well_completion: Optional[UploadFile] = File(None), reservoir_pvt: Optional[UploadFile] = File(None)):
    """Train page: retrain HybridVFM on uploaded data, validate, save as the active model."""
    global MODEL
    try:
        data = vp.load_inputs(_file(high_freq), _file(choke_events),
                              _file(well_completion, DEFAULTS["well_completion"]),
                              _file(reservoir_pvt, DEFAULTS["reservoir_pvt"]), _file(well_tests))
        bundle, _, tr = vp.train_model(data)
        val, _ = vp.validate(tr, bundle.model.J, bundle.rs)
        loo = vp.validate_loo(tr, bundle.model.J, bundle.rs)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Training failed: {e}")
    old_reh = (MODEL.validation or {}).get("rehearsal_train_jan_jun_upload_jul_sep", [])
    bundle.validation = {"rolling_3_latest": val.to_dict("records"), "leave_one_out": loo.to_dict("records"),
                         "rehearsal_train_jan_jun_upload_jul_sep": old_reh}
    if os.path.exists(MODEL_PATH):
        shutil.copy(MODEL_PATH, MODEL_PATH.replace(".pkl", "_previous.pkl"))
    vp.save_model(bundle, MODEL_PATH)
    MODEL = bundle
    return {"status": "trained", "trained_on": bundle.trained_on, "n_tests": bundle.n_tests,
            "J_stbd_per_psi": {k: round(float(v), 2) for k, v in bundle.model.J.items()},
            "validation_leave_one_out": _records(loo.round(3)),
            "validation_latest_3_hidden": _records(val.round(3))}
