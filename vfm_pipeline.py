"""
vfm_pipeline.py – SOCAR Hackathon 2026 · Intelligent Well Performance Analysis
=============================================================================

One module for the whole workflow, used by the notebook (training) and the app (test / results).
Team Phoenix v2:  VFM = HybridVFM (vfm_model.py: choke regression + IPR anchoring, recalibrated on new well tests)
                  PTA = Horner / Bourdet analysis below

    raw files ──► clean ──► features ──► VFM model ──► daily rates ──► shut-ins ──► PTA ──► interpretation ──► export

Typical app use (test/results page):

    import vfm_pipeline as vp
    data   = vp.load_inputs(hf_file, choke_file, completion_file, pvt_file, well_tests_file=None)
    model  = vp.load_model("models/vfm_model.pkl")
    result = vp.run_all(data, model)            # dict with every table the UI needs
    vp.export_all(result, "outputs")            # Excel files + pta_plots/ + charts/

Training (train page / notebook):

    model = vp.train_model(data)                # needs well_tests
    vp.save_model(model, "models/vfm_model.pkl")
"""
from __future__ import annotations

import os
import pickle
import warnings
from dataclasses import dataclass, field

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")   # file-only backend: plots are made inside the web server's worker threads (GUI backends crash there)

from vfm_clean import clean_hf                                   # VFM cleaning (HybridVFM)
from vfm_data import qc_well_tests, test_window_features         # well-test QC + test-window features
from vfm_model import ChokeVFM, HybridVFM, T0, hourly_features, predict_hours, aggregate_daily  # VFM models

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────── constants
SENSORS = ["whp_psia", "whp_downstream_psia", "dh_pressure_psia",
           "dh_temperature_degf", "annulus_pressure_psia", "gas_lift_rate_mscfd"]
MIN_JUMP = {"whp_psia": 30, "whp_downstream_psia": 20, "dh_pressure_psia": 60,
            "dh_temperature_degf": 5, "annulus_pressure_psia": 40, "gas_lift_rate_mscfd": 150}
TARGETS = ["oil_rate_stbd", "water_rate_stbd", "gas_rate_mscfd"]
FEATURES = ["dh_pressure_psia", "dh_temperature_degf", "whp_psia", "whp_downstream_psia",
            "annulus_pressure_psia", "gas_lift_rate_mscfd", "choke_pct", "dp_tubing", "dp_choke",
            "dp_annulus_tubing", "gilbert_term", "orifice_term", "gl_per_area"]
M3_PER_MSCF = 28.3168

# colours shared by every chart (match the UI)
COL = {"oil": "#B5651D", "water": "#1565C0", "gas": "#2E7D32", "pressure": "#1565C0",
       "derivative": "#B5651D", "fit": "#C62828", "window": "#F2C14E", "grey": "#9E9E9E",
       "text": "#1F1E1D", "muted": "#6B6A68", "good": "#2E7D32", "bad": "#C62828"}


# ──────────────────────────────────────────────────────────────────────────── 1. load
@dataclass
class Inputs:
    hf: pd.DataFrame
    choke: pd.DataFrame
    completion: pd.DataFrame
    pvt: pd.DataFrame
    well_tests: pd.DataFrame | None = None
    report: dict = field(default_factory=dict)


def _read(f):
    if f is None:
        return None
    name = getattr(f, "name", str(f)).lower()
    return pd.read_excel(f) if name.endswith((".xlsx", ".xls")) else pd.read_csv(f)


def load_inputs(hf, choke, completion, pvt, well_tests=None) -> Inputs:
    """Accepts file paths or uploaded file objects (CSV or Excel)."""
    hf, choke = _read(hf), _read(choke)
    hf["timestamp"] = pd.to_datetime(hf["timestamp"])
    choke["timestamp"] = pd.to_datetime(choke["timestamp"])
    wt = _read(well_tests)
    if wt is not None:
        wt["test_start"] = pd.to_datetime(wt["test_start"])
        wt["test_end"] = pd.to_datetime(wt["test_end"])
    return Inputs(hf=hf, choke=choke, completion=_read(completion), pvt=_read(pvt), well_tests=wt)


# ──────────────────────────────────────────────────────────────────────────── 2. clean sensors
def _fill_short_gaps(s: pd.Series, max_gap: int = 10) -> pd.Series:
    """Linear fill ONLY for gaps of <= max_gap minutes; longer gaps stay empty."""
    isna = s.isna()
    run_id = (isna != isna.shift()).cumsum()
    run_len = isna.groupby(run_id).transform("size")
    return s.where(~isna | (run_len > max_gap), s.interpolate(limit_area="inside"))


def clean_sensors(hf: pd.DataFrame, choke: pd.DataFrame):
    """Returns (clean minute data, data-quality report dict)."""
    rep = {"rows_raw": len(hf),
           "sentinels": int((hf[SENSORS] <= -999).sum().sum()),
           "duplicates": int(hf.duplicated(["well_name", "timestamp"]).sum())}
    gap = hf.sort_values(["well_name", "timestamp"]).groupby("well_name").timestamp.diff().dt.total_seconds() / 60
    rep["gaps_over_60min"] = int((gap > 60).sum())
    rep["longest_gap_hours"] = float(round(gap.max() / 60, 1)) if len(gap) else 0.0

    df = hf.groupby(["well_name", "timestamp"], as_index=False)[SENSORS].mean()
    out, spikes, missing_before = [], 0, 0
    for well, g in df.groupby("well_name"):
        g = g.set_index("timestamp").asfreq("1min")
        for c in SENSORS:
            s = g[c].where(g[c] > -999)
            missing_before += int(s.isna().sum())
            med = s.rolling(31, center=True, min_periods=5).median()
            mad = (s - med).abs().rolling(31, center=True, min_periods=5).median()
            dev = (s - med).abs()
            spike = (dev > 6 * 1.4826 * mad.clip(lower=1e-6)) & (dev > MIN_JUMP[c])
            spikes += int(spike.sum())
            g[c] = _fill_short_gaps(s.mask(spike))
        g["well_name"] = well
        out.append(g.reset_index())
    hc = pd.concat(out, ignore_index=True).sort_values("timestamp")
    hc = pd.merge_asof(hc, choke.sort_values("timestamp"), on="timestamp", by="well_name")
    hc["shut_in"] = hc["choke_opening_pct"] <= 0
    hc = hc.sort_values(["well_name", "timestamp"]).reset_index(drop=True)
    rep.update(spikes_removed=spikes,
               values_filled=int(missing_before + spikes - hc[SENSORS].isna().sum().sum()),
               still_missing=int(hc[SENSORS].isna().sum().sum()),
               shut_in_minutes=int(hc.shut_in.sum()),
               period_start=str(hc.timestamp.min().date()), period_end=str(hc.timestamp.max().date()),
               wells=sorted(hc.well_name.unique().tolist()))
    return hc, rep


def valid_data_by_month(hf_clean: pd.DataFrame) -> pd.DataFrame:
    """% of minutes with a valid downhole pressure, well × month (for the UI heatmap)."""
    x = hf_clean.assign(month=hf_clean.timestamp.dt.to_period("M").astype(str))
    return (x.groupby(["well_name", "month"]).dh_pressure_psia.apply(lambda s: s.notna().mean() * 100)
             .unstack().round(1))


# ──────────────────────────────────────────────────────────────────────────── 3. well-test QC
def clean_well_tests(wt: pd.DataFrame):
    """Fix only errors the report itself proves (unit, sign). Returns (clean tests, list of fixes)."""
    wt = wt.copy()
    num = wt.select_dtypes("number").columns.tolist()
    ratio = wt.gas_rate_mscfd / (wt.oil_rate_stbd * wt.gor_scf_stb / 1000)
    wt["fix_gas_unit"] = ratio.round(1).eq(28.3)
    wt.loc[wt.fix_gas_unit, ["gas_rate_mscfd", "total_gas_rate_mscfd"]] /= M3_PER_MSCF
    wt["fix_sign"] = (wt[num] < 0).any(axis=1)
    wt[num] = wt[num].abs()
    fixes = []
    for w, g in wt[wt.fix_gas_unit].groupby("well_name"):
        fixes.append({"severity": "HIGH", "issue": f"{w} gas reported in m³/d, not Mscf/d (ratio 28.3)", "count": len(g)})
    if wt.fix_sign.any():
        fixes.append({"severity": "MEDIUM", "issue": "Negative values in well tests (sign errors) – "
                      + ", ".join(sorted(wt[wt.fix_sign].well_name.unique())), "count": int(wt.fix_sign.sum())})
    return wt, fixes


def data_issues(rep: dict, test_fixes: list) -> pd.DataFrame:
    """Data-issue table for the upload page."""
    rows = list(test_fixes) + [
        {"severity": "HIGH", "issue": "Missing-value code −999.2 in sensor columns", "count": rep["sentinels"]},
        {"severity": "MEDIUM", "issue": "Sensor spikes removed (e.g. impossible downhole pressure)", "count": rep["spikes_removed"]},
        {"severity": "LOW", "issue": "Duplicate rows (same well and minute) averaged", "count": rep["duplicates"]},
        {"severity": "LOW", "issue": f"Gaps over 60 min (longest {rep['longest_gap_hours']} h) left empty", "count": rep["gaps_over_60min"]},
        {"severity": "INFO", "issue": "Short gaps and removed spikes (≤10 min) filled by straight line", "count": rep["values_filled"]}]
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────── 4–5. features + VFM model
# Test-window features: vfm_data.test_window_features · Model: vfm_model.HybridVFM (imported above)

HALF_LIFE_DAYS = 90          # recency weighting for forecasting (chosen on a Jan–Jun → Jul–Sep rehearsal)


@dataclass
class ModelBundle:
    """Everything the app needs, saved as ONE .pkl."""
    model: HybridVFM                        # production VFM model (fitted on all training tests)
    trained_on: str
    n_tests: int
    wells: list
    validation: dict                        # R² / MAPE / WAPE / RMSE tables
    cum_oil_end: dict                       # cumulative oil per well at end of training data
    training_end: str
    choke_model: object = None              # ChokeVFM, kept for comparison only
    train_tests: pd.DataFrame | None = None # QC'd training tests + sensor window features (used for recalibration)
    rs: dict = field(default_factory=dict)  # solution GOR at Pb per well
    last_choke: dict = field(default_factory=dict)        # choke % per well at the end of training (carry-over)
    daily_oil: pd.DataFrame | None = None   # training-period daily VFM oil (cumulative oil for PTA)
    half_life_days: float = HALF_LIFE_DAYS
    version: str = "2.0"


def complete_choke(choke: pd.DataFrame, hf: pd.DataFrame, last_choke: dict | None = None):
    """The choke file lists only CHANGES. Every well needs a setting at the start of the uploaded period,
    otherwise the period before its first change is unknown. Carry over the last setting known from training;
    if that is not available, assume the first listed setting (flagged). Returns (choke, notes)."""
    choke = choke.copy()
    start = hf.groupby("well_name").timestamp.min()
    add, carried, assumed = [], [], []
    for w, t0 in start.items():
        c = choke[choke.well_name == w]
        if len(c) and (c.timestamp <= t0).any():
            continue
        if last_choke and w in last_choke:
            v = float(last_choke[w]); carried.append(f"{w} {v:g}%")
        elif len(c):
            v = float(c.sort_values("timestamp").choke_opening_pct.iloc[0]); assumed.append(f"{w} {v:g}%")
        else:
            raise ValueError(f"No choke setting at all for well {w}: the choke file needs at least one row for it")
        add.append({"timestamp": t0, "well_name": w, "choke_opening_pct": v})
    if add:
        choke = pd.concat([pd.DataFrame(add), choke], ignore_index=True)
    notes = []
    if carried:
        notes.append({"severity": "INFO", "count": len(carried),
                      "issue": "Choke setting at period start not in choke file – carried over from training: " + ", ".join(carried)})
    if assumed:
        notes.append({"severity": "MEDIUM", "count": len(assumed),
                      "issue": "Choke setting at period start unknown – assumed equal to first listed change: " + ", ".join(assumed)})
    return choke.sort_values(["well_name", "timestamp"]).reset_index(drop=True), notes


def prepare_tests(hf_v: pd.DataFrame, well_tests: pd.DataFrame, wells=None) -> pd.DataFrame:
    """QC'd well tests + mean sensor values over each test window (drops tests with no sensor data)."""
    tr = test_window_features(hf_v, qc_well_tests(well_tests)).dropna(subset=["dh_pressure_psia"])
    if wells is not None:
        tr = tr[tr.well_name.isin(wells)]
    return tr.sort_values(["well_name", "test_start"])


def train_model(data: Inputs, validation: dict | None = None, half_life_days: float = HALF_LIFE_DAYS):
    """Fit HybridVFM on all well tests. Returns (bundle, VFM-cleaned minute data, test table)."""
    choke, _ = complete_choke(data.choke, data.hf)
    hf_v, _ = clean_hf(data.hf, choke)
    tr = prepare_tests(hf_v, data.well_tests)
    rs = data.completion.set_index("well_name").solution_gor_at_pb_scf_stb.to_dict()
    model = HybridVFM().fit(tr, rs, hf_v, choke)                # J from choke steps + build-ups, Pr(t) from tests
    pred = predict_daily(model, hf_v)
    cum = (data.completion.set_index("well_name").cum_oil_at_2025_01_01_stb
           + pred.groupby("well_name").oil_rate_stbd.sum()).to_dict()
    t0, t1 = hf_v.timestamp.min(), hf_v.timestamp.max()
    last_choke = choke.sort_values("timestamp").groupby("well_name").choke_opening_pct.last().astype(float).to_dict()
    bundle = ModelBundle(model=model, trained_on=f"{t0:%Y-%m-%d} – {t1:%Y-%m-%d}", n_tests=len(tr),
                         wells=sorted(tr.well_name.unique()), validation=validation or {},
                         cum_oil_end={k: float(v) for k, v in cum.items()}, training_end=str(t1),
                         choke_model=ChokeVFM().fit(tr, rs), train_tests=tr, rs=rs, last_choke=last_choke,
                         daily_oil=pred[["date", "well_name", "oil_rate_stbd"]].copy(), half_life_days=half_life_days)
    return bundle, hf_v, tr


def predict_daily(model, hf_v: pd.DataFrame) -> pd.DataFrame:
    """VFM-cleaned minute data -> daily oil / water / gas with ONE fixed model (hourly model, shut-in hours = 0)."""
    d = model.predict_daily(hf_v)
    return d[["date", "well_name"] + TARGETS].sort_values(["well_name", "date"]).reset_index(drop=True)


def forecast_model(bundle: ModelBundle, tests: pd.DataFrame) -> HybridVFM:
    """Recalibrated model: reservoir-pressure trend, water-cut trend and GOR refitted on `tests` with recency
    weights (latest tests count most). J stays fixed – it is a rock/well property measured in the sensor data."""
    tests = tests.drop_duplicates(["well_name", "test_start"], keep="last")
    return HybridVFM(half_life_days=bundle.half_life_days).fit(tests, bundle.rs, J=bundle.model.J)


def predict_period(bundle: ModelBundle, hf_v: pd.DataFrame, new_tests: pd.DataFrame | None = None):
    """Daily rates for ANY uploaded period (the demo-day path).
      • hours inside the training period → the trained model (as in v1);
      • hours after it → recalibrated model, refitted on training tests + every uploaded test that had ENDED
        before that hour. Causal: a day is never predicted with a well test from its own future.
    Returns (daily predictions, log of the recalibration steps)."""
    H = hourly_features(hf_v)
    t_end = pd.Timestamp(bundle.training_end)
    P = predict_hours(bundle.model, H, rows=H.hour <= t_end)
    future = H.hour > t_end
    log = []
    if future.any():
        nt = new_tests if new_tests is not None and len(new_tests) else None
        cuts = sorted(t for t in (nt.test_end.unique() if nt is not None else []) if t > H.hour[future].min())
        edges = [H.hour[future].min()] + cuts + [H.hour.max() + pd.Timedelta("1h")]
        for a, b in zip(edges[:-1], edges[1:]):
            seg = future & (H.hour >= a) & (H.hour < b)
            if not seg.any():
                continue
            used = bundle.train_tests if nt is None else pd.concat([bundle.train_tests, nt[nt.test_end <= a]])
            Pk = predict_hours(forecast_model(bundle, used), H, rows=seg)
            P.loc[seg] = Pk.loc[seg]
            log.append({"from": str(a), "to": str(b), "tests_used": int(len(used)),
                        "new_tests_used": int(0 if nt is None else (nt.test_end <= a).sum())})
    d = aggregate_daily(H, P)
    return d[["date", "well_name"] + TARGETS].sort_values(["well_name", "date"]).reset_index(drop=True), log


def score_uploaded_tests(bundle: ModelBundle, nt: pd.DataFrame):
    """Honest accuracy on uploaded (new) well tests. Each test is predicted
       • v1_frozen         : trained model only (what v1 of the app did),
       • v2_recalibrated   : recalibrated on training tests + uploaded tests that ENDED before this test started,
       • baseline_last_test: reuse the latest earlier test of the well."""
    nt = nt.sort_values("test_start")
    frozen = bundle.model.predict_flowing(nt)[TARGETS]
    rec, base = [], []
    for i, r in nt.iterrows():
        used = pd.concat([bundle.train_tests, nt[nt.test_end <= r.test_start]])
        rec.append(forecast_model(bundle, used).predict_flowing(nt.loc[[i]])[TARGETS])
        last = used[used.well_name == r.well_name].sort_values("test_start").tail(1)
        base.append(pd.DataFrame([last[TARGETS].iloc[0].values], index=[i], columns=TARGETS))
    R, B = pd.concat(rec), pd.concat(base)
    rows = [{"rate": t, "who": who} | _scores(nt[t], Q.loc[nt.index, t]) for t in TARGETS
            for who, Q in [("v2_recalibrated", R), ("v1_frozen", frozen), ("baseline_last_test", B)]]
    detail = (nt[["well_name", "test_start"] + TARGETS].rename(columns={"test_start": "start"})
              .join(R.add_prefix("pred_")).join(frozen.add_prefix("pred_v1_")))
    return pd.DataFrame(rows), detail


def save_model(bundle: ModelBundle, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bundle, f)


def load_model(path_or_file) -> ModelBundle:
    if hasattr(path_or_file, "read"):
        return pickle.load(path_or_file)
    with open(path_or_file, "rb") as f:
        return pickle.load(f)


def _scores(y, p):
    """Metrics: R², MAPE %, WAPE %, RMSE."""
    y, p = pd.Series(np.asarray(y, float)), pd.Series(np.asarray(p, float))
    e = y - p
    ss = ((y - y.mean()) ** 2).sum()
    return {"R2": float(1 - (e ** 2).sum() / ss) if ss > 0 else float("nan"),
            "MAPE_%": float((e.abs() / y).mean() * 100), "WAPE_%": float(e.abs().sum() / y.sum() * 100),
            "RMSE": float(np.sqrt((e ** 2).mean()))}


def validate(tr: pd.DataFrame, J: dict, rs: dict, holdout: int = 3):
    """Rolling-origin: hold out the latest, 2nd, …, n-th latest test of each well; refit on earlier tests only
    (J stays fixed – it comes from the high-frequency data, not from the tests). Baseline = reuse last test."""
    tr = tr.sort_values(["well_name", "test_start"]).reset_index(drop=True)
    rank = tr.groupby("well_name").cumcount(ascending=False)
    preds, base = [], []
    for k in range(holdout):
        a, b = tr[rank > k], tr[rank == k]
        preds.append(HybridVFM().fit(a, rs, J=J).predict_flowing(b)[TARGETS])
        last = a.groupby("well_name")[TARGETS].last()
        base.append(b[["well_name"]].join(last, on="well_name")[TARGETS])
    P, B = pd.concat(preds).sort_index(), pd.concat(base).sort_index()
    y = tr.loc[P.index]
    rows = [{"rate": t, "who": who} | _scores(y[t], Q[t]) for t in TARGETS
            for who, Q in [("model", P), ("baseline_last_test", B)]]
    holdout_df = y[["well_name", "test_start"] + TARGETS].rename(columns={"test_start": "start"}).join(P.add_prefix("pred_"))
    return pd.DataFrame(rows), holdout_df


def validate_loo(tr: pd.DataFrame, J: dict, rs: dict) -> pd.DataFrame:
    """Leave-one-test-out (J fixed – note: J was estimated with all tests, so this is mildly optimistic)."""
    P = pd.concat([HybridVFM().fit(tr.drop(i), rs, J=J).predict_flowing(tr.loc[[i]])[TARGETS] for i in tr.index])
    return pd.DataFrame([{"rate": t} | _scores(tr.loc[P.index, t], P[t]) for t in TARGETS])


# ──────────────────────────────────────────────────────────────────────────── 7. PTA
def detect_shutins(choke: pd.DataFrame, hf_clean: pd.DataFrame, min_hours: float = 2) -> pd.DataFrame:
    """Shut-in = choke goes to 0 from an open setting, inside the uploaded period, lasting ≥ min_hours."""
    ch = choke.sort_values(["well_name", "timestamp"]).copy()
    ch["prev"] = ch.groupby("well_name").choke_opening_pct.shift(fill_value=1)
    ch["t_open"] = ch.groupby("well_name").timestamp.shift(-1)
    ev = ch[(ch.choke_opening_pct <= 0) & (ch.prev > 0)][["well_name", "timestamp", "t_open"]] \
        .rename(columns={"timestamp": "t_shut"}).reset_index(drop=True)
    start = hf_clean.groupby("well_name").timestamp.min()
    end = hf_clean.groupby("well_name").timestamp.max()
    ev["t_open"] = ev.t_open.fillna(ev.well_name.map(end))
    ev = ev[(ev.t_shut > ev.well_name.map(start)) & (ev.t_shut < ev.well_name.map(end))]
    ev["duration_hr"] = (ev.t_open - ev.t_shut).dt.total_seconds() / 3600
    ev = ev[ev.duration_hr >= min_hours].reset_index(drop=True)
    ev["event_id"] = ev.well_name + "-SI-" + (ev.groupby("well_name").cumcount() + 1).astype(str).str.zfill(2)
    return ev


def _bourdet(x, p, L=0.15):
    d = np.full(len(x), np.nan)
    for i in range(len(x)):
        l = np.searchsorted(x, x[i] - L, "right") - 1
        r = np.searchsorted(x, x[i] + L, "left")
        if 0 <= l < i < r < len(x):
            dl, dr = x[i] - x[l], x[r] - x[i]
            d[i] = ((p[i] - p[l]) / dl * dr + (p[r] - p[i]) / dr * dl) / (dl + dr)
    return d


def _radial_window(dt, der, tol=0.05, min_cycles=0.5, min_dt=0.2):
    """Infinite-acting radial flow (IARF) window.
    1st choice: the LONGEST stretch (≥ min_cycles log cycles) where the Bourdet derivative stays within ±tol of
    its median – i.e. a genuinely flat derivative, after wellbore storage and before boundary effects.
    Fallback (derivative never flattens): the flattest min_cycles stretch, flagged as not-flat.
    Returns ((from_hr, to_hr), coefficient of variation, flat?)."""
    ok = ~np.isnan(der) & (der > 0) & (dt >= min_dt)
    t, d = dt[ok], der[ok]
    lt = np.log10(t)
    best, best_key = None, None
    for i in range(len(t)):
        for j in range(len(t) - 1, i + 3, -1):
            span = lt[j] - lt[i]
            if span < min_cycles:
                break
            seg = d[i:j + 1]
            med = np.median(seg)
            if np.abs(seg / med - 1).max() <= tol:
                key = (span, -seg.std() / seg.mean())
                if best_key is None or key > best_key:
                    best, best_key = (i, j), key
                break                                    # longest for this start found
    if best is not None:
        i, j = best
        return (t[i], t[j]), float(d[i:j + 1].std() / d[i:j + 1].mean()), True
    # fallback: flattest window of about min_cycles
    rng, score, cv = (t[0], t[-1]), np.inf, np.nan
    for i in range(len(t)):
        for j in range(i + 4, len(t)):
            if lt[j] - lt[i] >= min_cycles:
                c = d[i:j + 1].std() / d[i:j + 1].mean()
                if c < score:
                    score, rng, cv = c, (t[i], t[j]), c
                break
    return rng, float(cv), False


def analyze_buildup(ev, hf_clean, predictions, pvt, completion, cum_oil_start=None, window=None, q_fallback=None):
    """One shut-in → dict of results + curves. `window=(from_hr, to_hr)` overrides the automatic radial-flow window."""
    w = hf_clean[hf_clean.well_name == ev.well_name]
    rock = pvt.set_index("well_name").loc[ev.well_name]
    p = predictions[(predictions.well_name == ev.well_name) & (predictions.date < ev.t_shut.normalize())]
    q, rate_source = p[p.oil_rate_stbd > 0].oil_rate_stbd.tail(3).mean(), "VFM, last 3 producing days"
    if not np.isfinite(q) and q_fallback and ev.well_name in q_fallback:   # shut-in in the first day of the upload
        q, rate_source = q_fallback[ev.well_name], "VFM, last 3 producing days of training data"
    if not np.isfinite(q) or q <= 0:
        raise ValueError("no producing VFM rate before the shut-in")
    np0 = (cum_oil_start or {}).get(ev.well_name,
                                    completion.set_index("well_name").loc[ev.well_name, "cum_oil_at_2025_01_01_stb"])
    tp = (np0 + p.oil_rate_stbd.sum()) / q * 24

    pre = w[(w.timestamp <= ev.t_shut)].dropna(subset=["dh_pressure_psia"]).tail(5)
    if not len(pre):
        raise ValueError("no downhole pressure before the shut-in")
    pwf = float(pre.dh_pressure_psia.median())
    bu = w[(w.timestamp > ev.t_shut) & (w.timestamp <= ev.t_open)].dropna(subset=["dh_pressure_psia"])
    if len(bu) < 60:
        raise ValueError("less than 1 hour of build-up pressure data")
    dt_raw = (bu.timestamp - ev.t_shut).dt.total_seconds().values / 3600
    edges = np.logspace(np.log10(dt_raw.min()), np.log10(dt_raw.max()), 200)
    g = pd.DataFrame({"b": np.digitize(dt_raw, edges), "dt": dt_raw, "p": bu.dh_pressure_psia.values}).groupby("b").median()
    dt, pws = g.dt.values, g.p.values

    hx = np.log10((tp + dt) / dt)
    der = _bourdet(-np.log((tp + dt) / dt), pws)
    if window is None:
        (a, b), cv, flat = _radial_window(dt, der)
    else:
        (a, b), cv, flat = window, np.nan, True
    sel = (dt >= a) & (dt <= b)
    slope, icpt = np.polyfit(hx[sel], pws[sel], 1)
    m = -slope
    r2 = 1 - ((pws[sel] - (icpt + slope * hx[sel])) ** 2).sum() / ((pws[sel] - pws[sel].mean()) ** 2).sum()
    p1hr = icpt + slope * np.log10(tp + 1)

    B, mu, h = rock.oil_fvf_rb_stb, rock.oil_viscosity_cp, rock.net_pay_thickness_ft
    phi, ct, rw = rock.porosity_frac, rock.total_compressibility_1_psi, rock.wellbore_radius_ft
    k = 162.6 * q * B * mu / (m * h)
    s = 1.151 * ((p1hr - pwf) / m - np.log10(k / (phi * mu * ct * rw ** 2)) + 3.23)
    dps = 0.869 * m * s
    fe = (icpt - pwf - dps) / (icpt - pwf)
    cycles = float(np.log10(b / a))
    confidence = ("High" if flat and cycles >= 1 and ev.duration_hr >= 12 else
                  "Medium" if flat else "Low")

    res = dict(event_id=ev.event_id, well_name=ev.well_name, t_shut=ev.t_shut, duration_hr=ev.duration_hr,
               q_oil_stbd=q, rate_source=rate_source, tp_hr=tp, pwf_psia=pwf, p1hr_psia=p1hr, p_star_psia=icpt,
               p_end_psia=pws[-1], horner_slope_psi_per_cycle=m, permeability_md=k, skin=s, delta_p_skin_psi=dps,
               flow_efficiency=fe, fit_from_hr=a, fit_to_hr=b, fit_log_cycles=cycles, horner_r2=r2,
               derivative_cv=cv, radial_flow_flat=flat, confidence=confidence)
    res.update(interpret(res, dt, pws - pwf, der))
    curves = dict(dt=dt, pws=pws, hx=hx, der=der, sel=sel, slope=slope, icpt=icpt)
    return res, curves


def interpret(res: dict, dt, dp, der) -> dict:
    """Plain-language reading of the build-up, derived from the data: flow regimes + condition + recommendation."""
    a, b = res["fit_from_hr"], res["fit_to_hr"]
    ok = ~np.isnan(der) & (der > 0)
    radial = der[ok & (dt >= a) & (dt <= b)]
    level = np.median(radial) if len(radial) else np.nan
    regimes = []
    # early time: unit slope on ΔP = pure wellbore storage; derivative above the radial level = storage/skin hump
    early = (dt <= 0.1) & (dp > 0)
    if early.sum() >= 3:
        us = np.polyfit(np.log10(dt[early]), np.log10(dp[early]), 1)[0]
        if 0.8 <= us <= 1.2:
            regimes.append(f"Wellbore storage (unit slope, {us:.2f})")
    before = der[ok & (dt < a)]
    if len(before) and np.nanmax(before) > 1.2 * level:
        regimes.append(f"Wellbore storage / skin transition (derivative hump up to {np.nanmax(before) / level:.1f}× radial level)")
    if res["radial_flow_flat"]:
        regimes.append(f"Infinite-acting radial flow {a:.1f}–{b:.1f} h (derivative flat within ±5 %, {res['fit_log_cycles']:.1f} log cycles) – used for Horner")
    else:
        regimes.append(f"No clean radial flow: the derivative never flattens; Horner line fitted on the flattest stretch "
                       f"{a:.1f}–{b:.1f} h – k and skin are indicative only")
    late = der[ok & (dt > b * 1.5)]
    if len(late) >= 3 and late[-3:].mean() < 0.8 * level:
        regimes.append("Late-time derivative drop: boundary / depletion effects (closed or pressure-supported drainage area)")
    elif len(late) >= 3 and late[-3:].mean() > 1.2 * level:
        regimes.append("Late-time derivative rise: sealing fault or lower-mobility zone")
    else:
        regimes.append("No boundary seen within the test duration")
    s, fe = res["skin"], res["flow_efficiency"]
    if s > 3:
        cond, rec = "Damaged", f"Stimulation candidate: removing skin could raise rate up to ~{(1 / fe - 1) * 100:.0f}% at the same drawdown"
    elif s > 1:
        cond, rec = "Slightly damaged", "Monitor; consider stimulation if rate declines faster than neighbours"
    elif s >= -1:
        cond, rec = "Undamaged", "No near-wellbore action needed"
    else:
        cond, rec = "Stimulated", "Near-wellbore zone flows better than the rock; no action needed"
    if res["confidence"] == "Low":
        rec += " (confirm with a longer build-up: radial flow not clearly reached)"
    k = res["permeability_md"]
    rock = "high" if k > 150 else "moderate" if k > 50 else "low"
    return {"flow_regimes": "; ".join(regimes), "condition": cond, "recommendation": rec,
            "summary": f"{cond} well (skin {s:+.1f}, flow efficiency {fe:.0%}) in {rock}-permeability rock "
                       f"({k:.0f} md). Confidence: {res['confidence']}. {rec}."}


def run_pta(events, hf_clean, predictions, pvt, completion, cum_oil_start=None, windows=None, q_fallback=None):
    """All shut-ins. An event that cannot be analysed is reported, not allowed to stop the run."""
    out, skipped = [], []
    for ev in events.itertuples():
        try:
            out.append(analyze_buildup(ev, hf_clean, predictions, pvt, completion, cum_oil_start,
                                       (windows or {}).get(ev.event_id), q_fallback))
        except Exception as e:                                    # noqa: BLE001 – reported to the user
            skipped.append({"severity": "MEDIUM", "count": 1, "issue": f"PTA skipped for {ev.event_id}: {e}"})
    table = pd.DataFrame([r for r, _ in out])
    return table, {r["event_id"]: c for r, c in out}, skipped


# ──────────────────────────────────────────────────────────────────────────── 8. whole run (test/results page)
def run_all(data: Inputs, bundle: ModelBundle, windows: dict | None = None) -> dict:
    """Raw uploaded files (+ optional new well tests) + trained model → every table the results page needs."""
    start = data.hf.timestamp.min()
    t_end = pd.Timestamp(bundle.training_end)
    choke, choke_notes = complete_choke(data.choke, data.hf, bundle.last_choke if start > t_end else None)
    hf_v, _ = clean_hf(data.hf, choke)                              # VFM cleaning (60-min interpolation)

    new_tests, fixes, test_scores, test_detail = None, [], None, None
    if data.well_tests is not None and len(data.well_tests):
        fixes = clean_well_tests(data.well_tests)[1]
        new_tests = prepare_tests(hf_v, data.well_tests, bundle.wells)
        if not len(new_tests):
            new_tests = None
    pred, vfm_log = predict_period(bundle, hf_v, new_tests)         # VFM: daily oil / water / gas
    if new_tests is not None:
        test_scores, test_detail = score_uploaded_tests(bundle, new_tests)

    hc, rep = clean_sensors(data.hf, choke)                         # PTA cleaning (keeps raw build-up shape)
    # cumulative oil before this upload starts (Horner producing time) and fallback rate for early shut-ins
    cum0 = data.completion.set_index("well_name").cum_oil_at_2025_01_01_stb.to_dict()
    q_fb = {}
    if bundle.daily_oil is not None:
        prior = bundle.daily_oil[bundle.daily_oil.date < start.normalize()]
        for w, v in prior.groupby("well_name").oil_rate_stbd.sum().items():
            cum0[w] = cum0.get(w, 0) + v
        q_fb = prior[prior.oil_rate_stbd > 0].groupby("well_name").oil_rate_stbd.apply(lambda s: s.tail(3).mean()).to_dict()
    events = detect_shutins(choke, hc)
    if len(events):
        pta, curves, skipped = run_pta(events, hc, pred, data.pvt, data.completion, cum0, windows, q_fb)
    else:
        pta, curves, skipped = pd.DataFrame(), {}, []
    issues = pd.concat([data_issues(rep, fixes), pd.DataFrame(choke_notes + skipped)], ignore_index=True)
    mode = ("recalibrated with uploaded well tests" if new_tests is not None else
            "forecast from training tests (no new well tests uploaded)") if (hf_v.timestamp > t_end).any() else "training period"
    return {"hf_clean": hc, "quality": rep, "issues": issues,
            "valid_by_month": valid_data_by_month(hc), "predictions": pred, "vfm_mode": mode, "vfm_log": vfm_log,
            "new_tests": new_tests, "test_scores": test_scores, "test_detail": test_detail,
            "events": events, "pta": pta, "curves": curves, "bundle": bundle}


# ──────────────────────────────────────────────────────────────────────────── 9. plots (clean, consistent)
def _style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "font.size": 10, "axes.titlesize": 12,
                         "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": "#E6E6E6",
                         "grid.linewidth": 0.8, "axes.edgecolor": "#BDBDBD", "axes.labelcolor": COL["muted"],
                         "xtick.color": COL["muted"], "ytick.color": COL["muted"], "legend.frameon": False})
    return plt


def _box(ax, text, loc="lower right"):
    from matplotlib.offsetbox import AnchoredText
    at = AnchoredText(text, loc=loc, prop=dict(size=9, color=COL["text"]), frameon=True)
    at.patch.set(edgecolor="#DDDDDD", facecolor="white", alpha=0.95)
    ax.add_artist(at)


def plot_history(r, hf_clean, predictions):
    plt = _style()
    fig, ax = plt.subplots(figsize=(8, 4))
    t0, t1 = r["t_shut"], r["t_shut"] + pd.Timedelta(hours=r["duration_hr"])
    w = hf_clean[(hf_clean.well_name == r["well_name"]) &
                 hf_clean.timestamp.between(t0 - pd.Timedelta("3D"), t1 + pd.Timedelta("1D"))]
    ax.plot(w.timestamp, w.dh_pressure_psia, color=COL["pressure"], lw=1.2, label="Downhole pressure")
    ax.axvspan(t0, t1, color=COL["bad"], alpha=0.07, label="Shut-in")
    ax.set_ylabel("Downhole pressure, psia")
    # rate history as the engineer sees it: VFM rate before, 0 during the shut-in, VFM rate after
    pr = predictions[predictions.well_name == r["well_name"]].set_index("date").oil_rate_stbd
    before = pr[(pr.index >= (t0 - pd.Timedelta("3D")).normalize()) & (pr.index < t0.normalize())]
    after_day = pr[pr.index > t1.normalize()]
    q_after = after_day.iloc[0] if len(after_day) else np.nan
    xs = list(before.index) + [t0, t0, t1, t1, t1 + pd.Timedelta("1D")]
    ys = list(before.values) + [r["q_oil_stbd"], 0, 0, q_after, q_after]
    ax2 = ax.twinx(); ax2.grid(False); ax2.spines["right"].set_visible(True)
    ax2.step(xs, ys, where="post", color=COL["oil"], lw=1.6, label="VFM oil rate")
    ax2.set_ylabel("Oil rate, STB/d", color=COL["oil"]); ax2.set_ylim(0, max(ys) * 1.35)
    ax.set_title(f"{r['event_id']} · Pressure & rate history")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.autofmt_xdate(); fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


def plot_loglog(r, c):
    plt = _style()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.loglog(c["dt"], c["pws"] - r["pwf_psia"], "o", ms=3, color=COL["pressure"], label="ΔP")
    ax.loglog(c["dt"], c["der"], "o", ms=3, color=COL["derivative"], label="Derivative ΔP′")
    ax.axvspan(r["fit_from_hr"], r["fit_to_hr"], color=COL["window"], alpha=0.25, label="Radial flow (used for Horner)")
    ax.set_xlabel("Shut-in time Δt, hours"); ax.set_ylabel("ΔP and ΔP′, psi")
    ax.set_title(f"{r['event_id']} · Log-log diagnostic")
    ax.legend(loc="lower left", fontsize=8)
    import textwrap
    note = "\n".join(textwrap.wrap("Flow regimes: " + r["flow_regimes"].replace("; ", " · "), 120))
    fig.text(0.01, 0.01, note, fontsize=8, color=COL["text"], ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.04 + 0.035 * note.count("\n") + 0.03, 1, 1))
    return fig


def plot_horner(r, c):
    plt = _style()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(c["hx"], c["pws"], "o", ms=3, color=COL["grey"], label="Shut-in pressure")
    ax.plot(c["hx"][c["sel"]], c["pws"][c["sel"]], "o", ms=4, color=COL["pressure"], label="Radial-flow points")
    xl = np.array([0, c["hx"].max()])
    ax.plot(xl, c["icpt"] + c["slope"] * xl, "--", color=COL["fit"], lw=1.4, label="Horner straight line")
    ax.invert_xaxis()                                                    # time runs left → right
    ax.set_xlabel("log₁₀ Horner time  (tp + Δt) / Δt      ← early            late →")
    ax.set_ylabel("Shut-in pressure Pws, psia")
    ax.set_title(f"{r['event_id']} · Horner plot")
    ax.legend(loc="upper left", fontsize=8)
    _box(ax, f"m = {r['horner_slope_psi_per_cycle']:.1f} psi/cycle\nk = {r['permeability_md']:.0f} md\n"
             f"skin = {r['skin']:+.2f}\nΔP skin = {r['delta_p_skin_psi']:.0f} psi\np* = {r['p_star_psia']:.0f} psia\n"
             f"confidence: {r.get('confidence', '–')}")
    fig.tight_layout()
    return fig


def plot_mdh(r, c):
    plt = _style()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.semilogx(c["dt"], c["pws"], "o", ms=3, color=COL["pressure"])
    ax.axvspan(r["fit_from_hr"], r["fit_to_hr"], color=COL["window"], alpha=0.25, label="Radial flow")
    ax.set_xlabel("Shut-in time Δt, hours (log)"); ax.set_ylabel("Shut-in pressure Pws, psia")
    ax.set_title(f"{r['event_id']} · MDH plot")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def plot_well_rates(well, predictions, tests=None):
    """Daily VFM oil / water / gas for one well, with measured well tests as dots."""
    plt = _style()
    p = predictions[predictions.well_name == well]
    fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for a, t, key, unit in zip(ax, TARGETS, ["oil", "water", "gas"], ["STB/d", "STB/d", "Mscf/d"]):
        a.plot(p.date, p[t], color=COL[key], lw=1.3, label=f"VFM {key} (daily)")
        if tests is not None and len(tests):
            tt = tests[tests.well_name == well]
            a.scatter(tt.start, tt[t], color="black", s=22, zorder=3, label="Measured well test")
        a.set_ylabel(f"{key.title()}, {unit}"); a.set_ylim(bottom=0)
        a.legend(loc="lower left", fontsize=8)
    ax[0].set_title(f"{well} · Daily production rates (VFM)")
    fig.autofmt_xdate(); fig.tight_layout()
    return fig


def plot_field_production(predictions):
    plt = _style()
    f = predictions.groupby("date")[TARGETS].sum()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.stackplot(f.index, f.oil_rate_stbd, f.water_rate_stbd, colors=[COL["oil"], COL["water"]], alpha=0.85,
                 labels=["Oil", "Water"])
    ax.set_ylabel("Liquid, STB/d"); ax.set_title("Field production · all wells (VFM)")
    ax2 = ax.twinx(); ax2.grid(False); ax2.spines["right"].set_visible(True)
    ax2.plot(f.index, f.gas_rate_mscfd, color=COL["gas"], lw=1.5, label="Gas")
    ax2.set_ylabel("Gas, Mscf/d", color=COL["gas"]); ax2.set_ylim(bottom=0)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", ncol=3, fontsize=8)
    fig.autofmt_xdate(); fig.tight_layout()
    return fig


def plot_measured_vs_predicted(holdout_df):
    plt = _style()
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
    for a, t, key in zip(ax, TARGETS, ["oil", "water", "gas"]):
        y, p = holdout_df[t], holdout_df["pred_" + t]
        a.scatter(y, p, color=COL[key], s=28, alpha=0.85)
        lim = [0, max(y.max(), p.max()) * 1.05]
        a.plot(lim, lim, "--", color=COL["grey"], lw=1); a.set_xlim(lim); a.set_ylim(lim)
        mape = (abs(y - p) / y).mean() * 100
        a.set_title(f"{key.title()} · error {mape:.1f}%"); a.set_xlabel("Measured"); a.set_ylabel("Predicted")
    fig.suptitle(f"Measured vs predicted on {len(holdout_df)} hidden well tests (perfect = dashed line)", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    return fig


def plot_model_vs_baseline(val):
    plt = _style()
    v = val.pivot(index="rate", columns="who", values="MAPE_%").loc[TARGETS]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(3); wdt = 0.36
    b1 = ax.bar(x - wdt / 2, v.baseline_last_test, wdt, color=COL["grey"], label="Reuse last test (today)")
    b2 = ax.bar(x + wdt / 2, v.model, wdt, color=COL["pressure"], label="Our VFM model")
    ax.bar_label(b1, fmt="%.1f%%", fontsize=8); ax.bar_label(b2, fmt="%.1f%%", fontsize=8)
    ax.set_xticks(x, ["Oil", "Water", "Gas"]); ax.set_ylabel("Average error, %")
    ax.set_title("Prediction error: our model vs reusing the last test"); ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_skin_summary(pta):
    plt = _style()
    t = pta.sort_values("skin")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    cols = [COL["bad"] if s > 1 else COL["good"] if s < -1 else COL["grey"] for s in t.skin]
    bars = ax.barh(t.event_id, t.skin, color=cols)
    ax.bar_label(bars, fmt="%+.1f", fontsize=8, padding=3)
    ax.axvline(0, color="#757575", lw=1)
    ax.set_xlabel("Skin  (positive = damaged, negative = stimulated)")
    ax.set_title("Near-wellbore condition by shut-in (skin)")
    fig.tight_layout()
    return fig


def plot_oil_change(predictions):
    plt = _style()
    p = predictions[predictions.oil_rate_stbd > 0].copy()
    m0, m1 = p.date.min().to_period("M"), p.date.max().to_period("M")
    a = p[p.date.dt.to_period("M") == m0].groupby("well_name").oil_rate_stbd.mean()
    b = p[p.date.dt.to_period("M") == m1].groupby("well_name").oil_rate_stbd.mean()
    ch = ((b / a - 1) * 100).sort_values()
    fig, ax = plt.subplots(figsize=(8, 3.8))
    bars = ax.barh(ch.index, ch.values, color=[COL["pressure"] if v == ch.min() else COL["grey"] for v in ch.values])
    ax.bar_label(bars, fmt="%.0f%%", fontsize=8, padding=4)
    ax.set_xlim(min(ch.min() * 1.18, -1), max(ch.max() * 1.18, 1))
    ax.axvline(0, color="#757575", lw=1); ax.invert_yaxis()
    ax.set_xlabel(f"Change in average oil rate, {m0.strftime('%b')} → {m1.strftime('%b %Y')}")
    ax.set_title("Oil decline by well (VFM)")
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────── 10. export
def export_all(result: dict, out_dir: str = "outputs", tests_for_plots=None, val=None, holdout_df=None):
    """Writes the Phase 1/2 deliverables: Excel files, pta_plots/ (4 per shut-in), charts/ (overview)."""
    import matplotlib.pyplot as plt
    os.makedirs(f"{out_dir}/pta_plots", exist_ok=True); os.makedirs(f"{out_dir}/charts", exist_ok=True)
    pred = result["predictions"].copy()
    if tests_for_plots is None and result.get("test_detail") is not None:
        tests_for_plots = result["test_detail"]                    # uploaded well tests as dots on the rate charts
    if result.get("test_scores") is not None:
        with pd.ExcelWriter(f"{out_dir}/VFM_accuracy_on_uploaded_tests.xlsx") as xw:
            result["test_scores"].round(3).to_excel(xw, sheet_name="metrics", index=False)
            d = result["test_detail"].copy(); d["start"] = d.start.dt.strftime("%Y-%m-%d %H:%M")
            d.round(2).to_excel(xw, sheet_name="per_test", index=False)
    xl = pred.assign(date=pred.date.dt.date).round(2)
    xl.to_excel(f"{out_dir}/VFM_predictions.xlsx", index=False)
    xl.assign(date=pred.date.dt.strftime("%Y-%m-%d")).to_csv(f"{out_dir}/VFM_predictions.csv", index=False)
    pta = result["pta"]
    if len(pta):
        main = ["well_name", "t_shut", "duration_hr", "horner_slope_psi_per_cycle", "permeability_md", "skin",
                "delta_p_skin_psi"]
        with pd.ExcelWriter(f"{out_dir}/PTA_results.xlsx") as xw:
            pta[main].round(3).to_excel(xw, sheet_name="pta_results", index=False)
            pta.round(3).to_excel(xw, sheet_name="details_and_interpretation", index=False)
        # submission format: required fields only, one row per well + shut-in time
        (pta[["well_name", "t_shut", "horner_slope_psi_per_cycle", "permeability_md", "skin", "delta_p_skin_psi"]]
            .rename(columns={"t_shut": "shut_in_time"}).round(3)
            .to_csv(f"{out_dir}/PTA_results.csv", index=False))
        for r in pta.to_dict("records"):
            c = result["curves"][r["event_id"]]
            for name, fig in [("1_history", plot_history(r, result["hf_clean"], pred)), ("2_loglog", plot_loglog(r, c)),
                              ("3_horner", plot_horner(r, c)), ("4_mdh", plot_mdh(r, c))]:
                fig.savefig(f"{out_dir}/pta_plots/{r['event_id']}_{name}.png", bbox_inches="tight"); plt.close(fig)
        fig = plot_skin_summary(pta); fig.savefig(f"{out_dir}/charts/skin_by_shutin.png", bbox_inches="tight"); plt.close(fig)
    for name, fig in [("field_production", plot_field_production(pred)), ("oil_change_by_well", plot_oil_change(pred))]:
        fig.savefig(f"{out_dir}/charts/{name}.png", bbox_inches="tight"); plt.close(fig)
    for w in sorted(pred.well_name.unique()):
        fig = plot_well_rates(w, pred, tests_for_plots)
        fig.savefig(f"{out_dir}/charts/rates_{w}.png", bbox_inches="tight"); plt.close(fig)
    if val is not None:
        fig = plot_model_vs_baseline(val); fig.savefig(f"{out_dir}/charts/model_vs_baseline.png", bbox_inches="tight"); plt.close(fig)
    if holdout_df is not None:
        fig = plot_measured_vs_predicted(holdout_df)
        fig.savefig(f"{out_dir}/charts/measured_vs_predicted.png", bbox_inches="tight"); plt.close(fig)
    return out_dir
