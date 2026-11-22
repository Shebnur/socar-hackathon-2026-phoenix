# Developer guide – connecting the UI to the Phoenix model (v2)

The model is Python, so the web page cannot load the `.pkl` directly. It runs in a small Python service, `api.py`.
The page only uploads files and shows the JSON, plots and downloads that come back.

```
[ React page ]  --upload files-->  [ api.py (Python) ] --> vfm_pipeline.py --> models/vfm_model.pkl
       ^                                    |
       └------- JSON + PNG plot URLs + Excel┘
```

## 1. Start the backend (once, on the demo laptop)
```bash
cd Phoenix_v2
pip install -r requirements.txt            # Python 3.9 or newer
uvicorn api:app --host 0.0.0.0 --port 8000
```
Open http://localhost:8000/docs – every endpoint can be tested there by hand, no UI needed.

## 2. Demo day: ONE upload, ONE run
The jury uploads **3 new files** for the next 3 months. Completion and PVT stay the same (defaults in `data/`).

| Upload field | Required? | File |
|---|---|---|
| `high_freq` | **yes** | new high-frequency minute data (CSV or Excel) |
| `choke_events` | **yes** | new choke settings (may list only changes – the setting at the period start is carried over automatically) |
| `well_tests` | **strongly recommended** | new well tests – the model **recalibrates** on them and reports its accuracy |
| `well_completion`, `reservoir_pvt` | no | defaults in `data/` are used |

```js
const API = "http://localhost:8000";
const fd = new FormData();
fd.append("high_freq", hfFile);
fd.append("choke_events", chokeFile);
if (testsFile) fd.append("well_tests", testsFile);       // ALWAYS send it when the jury gives it
const res = await fetch(`${API}/api/run`, { method: "POST", body: fd });   // ~30–60 s for 3 months
if (!res.ok) { alert((await res.json()).detail); return; }
const r = await res.json();
// images:   <img src={API + r.pta[0].plots["3_horner"]} />
// download: <a href={API + r.downloads.all_zip}>Download all results</a>
```
Show a loading spinner while it runs.

## 3. What `/api/run` returns → where to show it
| JSON key | Content | Show it in |
|---|---|---|
| `period`, `wells`, `model` | date range, well list, model name + version | header |
| `vfm_mode` | `recalibrated with uploaded well tests` / `forecast from training tests…` / `training period` | VFM tab, badge |
| `data_quality` | what cleaning found and fixed (incl. choke carry-over, skipped PTA events) | **Data** tab |
| `sensors_daily` | daily mean pressures, temperature, gas-lift, choke per well | **Data** tab charts |
| `predictions` | date, well_name, oil_rate_stbd, water_rate_stbd, gas_rate_mscfd | **VFM** tab (table + chart per well) |
| `metrics_on_uploaded_tests` | `metrics`: R², MAPE %, WAPE %, RMSE per rate for `v2_recalibrated`, `v1_frozen`, `baseline_last_test`; `per_test`: measured vs predicted | **VFM tab → "Accuracy on your data"** (null if no tests uploaded) |
| `model_metrics` | training validation incl. Jan–Jun → Jul–Sep rehearsal | **VFM tab → "Model accuracy"** |
| `charts` | PNG URLs: field_production, oil_change_by_well, rates_GL-xx (uploaded tests as dots), skin_by_shutin | VFM / PTA tabs |
| `pta` | one item per shut-in: slope, k, skin, ΔP skin, flow efficiency, p*, **confidence**, flow regimes, condition, recommendation, summary, `plots` (history, log-log, Horner, MDH) | **PTA tab** (event selector → 4 plots + cards; colour the confidence badge) |
| `downloads` | VFM Excel/CSV, PTA Excel/CSV, `vfm_accuracy_excel` (if tests uploaded), ZIP with everything | **Export** tab |

Metric names: `R2`, `MAPE_%`, `WAPE_%`, `RMSE` (STB/d for oil and water, Mscf/d for gas).

## 4. Other endpoints
| Endpoint | Use |
|---|---|
| `GET /api/health` | check the backend is up before enabling the Run button |
| `GET /api/model` | model card: J, reservoir pressure, decline, water cut, GOR per well + validation tables |
| `POST /api/train` | retrain on uploaded data incl. well_tests (backs up the old `.pkl` as `vfm_model_previous.pkl`) |

## 5. Notes
- Do **not** call `/api/train` on demo day: `/api/run` with `well_tests` already recalibrates safely.
- Result files of each run are kept under `runs/<run_id>/`; delete the folder to free space.
