# Changelog

## v2.0
**VFM – ready for new data**
- Recalibration: uploaded well tests now update reservoir-pressure, water-cut and GOR trends (recency-weighted,
  90-day half-life, causal: only tests that ended before a given day are used). J stays fixed.
  Rehearsal (train Jan–Jun, upload Jul–Sep): oil MAPE 4.4 % → 3.3 %, water 5.5 % → 4.2 %, gas 7.8 % → 6.4 %.
- Accuracy on uploaded tests is reported for v2 recalibrated, v1 frozen and the reuse-last-test baseline
  (`VFM_accuracy_on_uploaded_tests.xlsx`, API `metrics_on_uploaded_tests`).
- Training-period predictions are unchanged from v1 (identical `VFM_predictions` for Jan–Sep 2025).

**Robustness fixes (demo-day blockers)**
- Choke file listing only changes: the setting at the start of the upload is carried over from training.
  v1 filled the gap with the well's FIRST listed change (e.g. a shut-in → weeks of zero production) and crashed
  for a well with no change in the upload.
- Shut-in in the first days of an upload: PTA rate falls back to the last training rate instead of crashing.
- A PTA event that cannot be analysed is reported in `data_quality` instead of stopping the whole run.
- Cumulative oil for Horner producing time is built from the training-period daily rates (correct for any start date).
- `api.py` now runs on Python 3.9 (v1 used `X | None` annotations that FastAPI cannot evaluate before 3.10).
- Plots use the file-only matplotlib backend: v1 crashed on macOS when the API drew plots in a worker thread.

**PTA**
- Radial-flow window = longest stretch with the derivative flat within ±5 % (v1 always picked the minimum 0.5 cycle).
- Confidence per event (High / Medium / Low); "no clean radial flow" is stated when the derivative never flattens.
- Flow-regime text derived from the data (storage hump, radial flow, late-time drop/rise) instead of fixed text.
- Pwf = median of the last 5 flowing minutes (noise-robust).

**Housekeeping**
- `build_model.py` reproduces every result with one command; stale joblib CLI scripts removed; hard-coded paths removed.
