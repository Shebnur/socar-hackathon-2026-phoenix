"""
build_model.py – reproduce every Phoenix v2 result with one command.

    python build_model.py                       # data files in this folder
    python build_model.py --data path/to/data   # or somewhere else

Steps
  1. Train HybridVFM on all data (Jan–Sep 2025, 85 well tests) → models/vfm_model.pkl
  2. Validate: leave-one-test-out, 3 latest tests per well hidden, and the DEMO-DAY REHEARSAL:
     train on Jan–Jun only, then upload Jul–Sep exactly as the jury will upload Oct–Dec
     (high-frequency + choke changes + well tests of the new period only) and score the uploaded tests.
  3. Run the full app pipeline (VFM → shut-ins → PTA) and write outputs/ (Excel, CSV, plots)
     + models/pta_results.pkl (PTA results, plot curves and settings).
"""
import argparse
import json
import os
import pickle
import time

import pandas as pd

import vfm_pipeline as vp

HERE = os.path.dirname(os.path.abspath(__file__))


def main(data_dir: str, out_dir: str):
    t0 = time.time()
    f = lambda n: os.path.join(data_dir, n)
    hf_file = f("high_freq_data.csv") if os.path.exists(f("high_freq_data.csv")) else f("high_freq_data.csv.gz")
    data = vp.load_inputs(hf_file, f("choke_events.csv"), f("well_completion.csv"),
                          f("reservoir_pvt.csv"), f("well_tests.csv"))
    print(f"[{time.time() - t0:4.0f}s] data loaded: {len(data.hf):,} minute rows, {len(data.well_tests)} well tests")

    # ── 1. train on everything
    bundle, hf_v, tr = vp.train_model(data)
    print(f"[{time.time() - t0:4.0f}s] trained HybridVFM on {bundle.n_tests} tests")

    # ── 2a. classic validation (J fixed)
    val, holdout = vp.validate(tr, bundle.model.J, bundle.rs)
    loo = vp.validate_loo(tr, bundle.model.J, bundle.rs)

    # ── 2b. demo-day rehearsal: train Jan–Jun, upload Jul–Sep (new-period files only)
    cut = pd.Timestamp("2025-07-01")
    hf, ch, wt = data.hf, data.choke, data.well_tests
    train_part = vp.Inputs(hf[hf.timestamp < cut], ch[ch.timestamp < cut], data.completion, data.pvt,
                           wt[wt.test_end < cut])
    b6, _, _ = vp.train_model(train_part)
    upload = vp.Inputs(hf[hf.timestamp >= cut], ch[ch.timestamp >= cut], data.completion, data.pvt,
                       wt[wt.test_start >= cut])
    reh = vp.run_all(upload, b6)
    reh_scores = reh["test_scores"]
    print(f"[{time.time() - t0:4.0f}s] rehearsal Jul–Sep ({len(reh['new_tests'])} new tests):")
    print(reh_scores.pivot(index="rate", columns="who", values="MAPE_%").round(2).to_string())

    bundle.validation = {
        "leave_one_out": loo.to_dict("records"),
        "rolling_3_latest": val.to_dict("records"),
        "rehearsal_train_jan_jun_upload_jul_sep": reh_scores.to_dict("records"),
    }
    model_path = os.path.join(HERE, "models", "vfm_model.pkl")
    vp.save_model(bundle, model_path)
    print(f"[{time.time() - t0:4.0f}s] saved {model_path}")

    # ── 3. full pipeline on the training period (Phase 1 + 2 deliverables)
    result = vp.run_all(vp.Inputs(data.hf, data.choke, data.completion, data.pvt), bundle)
    tests_plot = tr[["well_name", "test_start"] + vp.TARGETS].rename(columns={"test_start": "start"})
    vp.export_all(result, out_dir, tests_for_plots=tests_plot, val=val, holdout_df=holdout)
    with pd.ExcelWriter(os.path.join(out_dir, "VFM_validation.xlsx")) as xw:
        loo.round(3).to_excel(xw, sheet_name="leave_one_test_out", index=False)
        val.round(3).to_excel(xw, sheet_name="latest_3_hidden_vs_baseline", index=False)
        reh_scores.round(3).to_excel(xw, sheet_name="rehearsal_jul_sep", index=False)
        d = reh["test_detail"].copy(); d["start"] = d.start.dt.strftime("%Y-%m-%d %H:%M")
        d.round(2).to_excel(xw, sheet_name="rehearsal_per_test", index=False)
    vp.export_all(reh, os.path.join(out_dir, "rehearsal_jul_sep"))

    # PTA has no trained parameters (it is a fixed engineering calculation), but its results, plot data and
    # settings are saved as a .pkl too, so the Phase 2 analysis can be reloaded without re-running it.
    pta_bundle = {"version": bundle.version, "method": "Horner / Bourdet build-up analysis, single-phase oil, VFM oil rate",
                  "settings": {"radial_flow_rule": "longest stretch with Bourdet derivative flat within ±5 %",
                               "min_log_cycles": 0.5, "bourdet_L": 0.15, "rate": "mean VFM oil, last 3 producing days",
                               "tp": "cumulative oil / q"},
                  "results": result["pta"], "curves": result["curves"], "events": result["events"]}
    with open(os.path.join(HERE, "models", "pta_results.pkl"), "wb") as fh:
        pickle.dump(pta_bundle, fh)
    pta = result["pta"]
    print(f"[{time.time() - t0:4.0f}s] outputs written to {out_dir}")
    print(pta[["event_id", "horner_slope_psi_per_cycle", "permeability_md", "skin", "delta_p_skin_psi",
               "fit_from_hr", "fit_to_hr", "confidence"]].round(2).to_string(index=False))
    summary = {"leave_one_out": loo.round(3).to_dict("records"),
               "rehearsal_MAPE_%": reh_scores.pivot(index="rate", columns="who", values="MAPE_%").round(2).to_dict()}
    with open(os.path.join(out_dir, "validation_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=HERE)
    ap.add_argument("--out", default=os.path.join(HERE, "outputs"))
    a = ap.parse_args()
    main(a.data, a.out)
