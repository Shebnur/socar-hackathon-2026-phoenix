# Team Phoenix – SOCAR Hackathon 2026 · Intelligent Well Performance Analysis (v2)

**VFM** = HybridVFM (physics-anchored regression, recalibrated with every new well test)
**PTA** = automated Horner / Bourdet build-up analysis on the VFM-predicted oil rate (single-phase oil)

## Reproduce everything (one command)
```bash
pip install -r requirements.txt          # Python 3.9+
python build_model.py                    # uses the 5 data files in this folder; ~2–3 min
```
This trains the model, runs every validation, saves `models/vfm_model.pkl` + `models/pta_results.pkl` and writes `outputs/`.

## Data (included)
`high_freq_data.csv.gz` (per-minute sensor data, gzip-compressed to fit GitHub's 100 MB limit; pandas reads it directly,
or unpack with `gunzip -k high_freq_data.csv.gz`), `choke_events.csv`, `well_tests.csv`, `well_completion.csv`, `reservoir_pvt.csv`.

## Deliverables
| Requirement | Where |
|---|---|
| 1. VFM and PTA results (CSV / Excel) | `outputs/VFM_predictions.csv/.xlsx`, `outputs/PTA_results.csv/.xlsx` |
| 2. Trained model files (.pkl) | `models/vfm_model.pkl`, `models/pta_results.pkl` |
| 3. Notebook, all cells run | `Phoenix_SOCAR_VFM_PTA.ipynb` (section 10 = final results) |
| 4. PTA plots | `outputs/pta_plots/` (6 shut-ins × 4 plots) |
| Application backend | `api.py` – see `DEVELOPER_GUIDE.md` |

## Files
| File | Purpose |
|---|---|
| `build_model.py` | Train → validate (incl. demo-day rehearsal) → export all deliverables |
| `vfm_pipeline.py` | One entry point for the app: load → clean → VFM (recalibrated) → shut-ins → PTA → interpretation → export |
| `vfm_model.py` | `HybridVFM` (final): Q_liq = J·(Pr(t) − Pwf); `ChokeVFM` (comparison); hourly → daily aggregation |
| `pi_estimation.py` | Productivity index J per well from choke steps + stabilised build-ups in the minute data |
| `vfm_clean.py`, `vfm_data.py` | Minute-data cleaning; well-test QC (GL-06 m³/d → Mscf/d, 5 sign errors) + test-window features |
| `api.py` | FastAPI backend used by the web UI (see `DEVELOPER_GUIDE.md`) |
| `Phoenix_SOCAR_VFM_PTA.ipynb` | Full analysis, all cells run: EDA → cleaning → model comparison → VFM → PTA → **section 10: final v2 results** |
| `models/vfm_model.pkl` | Trained VFM model bundle |
| `models/pta_results.pkl` | PTA results, plot curves and settings for every shut-in (PTA is a calculation, not a trained model) |
| `outputs/` | `VFM_predictions.xlsx/.csv`, `PTA_results.xlsx/.csv`, `VFM_validation.xlsx`, `pta_plots/` (4 per shut-in), `charts/`, `rehearsal_jul_sep/` |

## Method
**VFM (Phase 1).** Every well test flows into a low-pressure test separator, so a model fitted only to tests
mis-predicts normal production days. HybridVFM anchors the rate on reservoir inflow:

| Part | Formula | Source |
|---|---|---|
| Liquid rate | Q_liq = J · (Pr(t) − Pwf) | Pwf = downhole gauge, every minute |
| Productivity index J | from ~30 choke steps (ΔlnQ vs ΔPwf) + stabilised build-ups | high-frequency data (independent of tests) |
| Reservoir pressure Pr(t), water cut, GOR | per-well trends fitted to the well tests | well tests |
| Oil / water / gas | Q_liq·(1−WC), Q_liq·WC, oil·GOR | – |

Predictions are hourly (zero while shut in, scaled by on-stream fraction) and averaged to daily rates.

**New in v2 – recalibration for new data.** When the next months of data are uploaded *with* their well tests,
Pr(t), water cut and GOR are refitted with **recency weighting** (90-day half-life) on the training tests plus every
uploaded test that had **already ended**. A day is never predicted using a test from its own future (causal). J is
kept fixed: it is a rock/well property measured in the sensor data. Without new tests, the recency-weighted
training model forecasts forward.

**PTA (Phase 2).** Shut-in = choke to 0 and back. q = mean VFM oil rate of the last 3 producing days;
tp = cumulative oil / q. Bourdet derivative on Horner superposition time; radial-flow window chosen automatically.

| Property | Formula (field units) |
|---|---|
| Permeability | k = 162.6 · q · B · μ / (m · h) |
| Skin | s = 1.151 · [ (P1hr − Pwf)/m − log10( k / (φ μ ct rw²) ) + 3.23 ] |
| ΔP skin | 0.869 · m · s |

v2 window rule: the **longest** stretch where the derivative is flat within ±5 % (true radial flow). If the
derivative never flattens, the flattest 0.5-cycle stretch is used and the result is flagged **Low confidence**.
Flow regimes (storage hump, radial flow, boundary) are now read from the data, not fixed text.

## Validation
| Test | Oil MAPE | Water MAPE | Gas MAPE |
|---|---|---|---|
| Leave-one-test-out, 85 tests (R² ≥ 0.995) | 2.7 % | 4.3 % | 5.9 % |
| **Demo-day rehearsal**: train Jan–Jun, upload Jul–Sep (28 new tests) – **v2 recalibrated** | **3.3 %** | **4.2 %** | **6.4 %** |
| same rehearsal – v1 frozen model | 4.4 % | 5.5 % | 7.8 % |
| same rehearsal – baseline "reuse last test" | 6.7 % | 5.2 % | 11.8 % |

The rehearsal uploads only new-period files (minute data, choke *changes*, well tests), exactly as on demo day.
Note: leave-one-out keeps J fixed and J used all tests, so that figure is mildly optimistic; the rehearsal is the
honest out-of-time number.

## PTA results (6 build-ups)
| Event | m psi/cycle | k md | Skin | ΔP skin psi | Radial window | Confidence |
|---|---|---|---|---|---|---|
| GL-01-SI-01 | 23.1 | 104.7 | +5.85 | +117 | 2.1–6.7 h (not flat) | Low |
| GL-02-SI-01 | 53.2 | 55.3 | −0.57 | −26 | 1.3–4.8 h | Medium |
| GL-04-SI-01 | 146.0 | 54.7 | −1.29 | −163 | 11.6–40.1 h | Medium |
| GL-06-SI-01 | 15.0 | 298.6 | +1.95 | +25 | 1.9–6.8 h | Medium |
| GL-08-SI-01 | 93.7 | 42.8 | −1.68 | −137 | 1.6–59.6 h | High |
| GL-10-SI-01 | 28.9 | 171.7 | +2.00 | +50 | 0.9–2.9 h (not flat) | Low |

GL-01 is the clearest stimulation candidate (skin ≈ +6, flow efficiency ≈ 68 %), but its derivative keeps falling
(early boundary / long storage effect), so a longer, cleaner build-up is recommended before acting.

## Limitations
- J is less certain for GL-05 and GL-07 (small pressure response at their choke steps).
- Without new well tests, Pr(t) and water-cut trends are extrapolated linearly (error grows ~0.5 %/month).
- k is proportional to the VFM oil rate, so the VFM error carries into permeability; slope m and skin depend on pressure only.
- If an upload starts in the middle of a shut-in, the choke state at its start is taken from the end of training.
