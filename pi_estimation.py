"""Estimate productivity index J (stb/d/psi, liquid) per well from high-frequency data, independent of tests' backpressure.
Two sources:
 1) Choke steps in normal operation: relative rate change from the choke-orifice relation (choke opening term only, i.e.
    dlnQ = b_choke*dln(choke) + b_dp*dln(dp)) vs measured DHP change -> drawdown D = -dDHP / dlnQ ; J = Q/D.
 2) Stabilised build-ups (DHP flat at end of shut-in): D = P_end - Pwf_before.
"""
import numpy as np, pandas as pd

def _win(g, t0, a, b):
    x = g[(g.timestamp >= t0 + pd.Timedelta(a)) & (g.timestamp < t0 + pd.Timedelta(b)) & (g.choke_pct > 0)]
    return x

def drawdown_estimates(hf, choke_events, coefs, min_ddhp=5.0):
    ce = choke_events.copy(); ce['timestamp'] = pd.to_datetime(ce['timestamp'])
    rows = []
    for _, e in ce.iterrows():
        g = hf[hf.well_name == e.well_name]
        if e.timestamp <= g.timestamp.min() + pd.Timedelta('1D'): continue
        a = _win(g, e.timestamp, '-12h', '0h'); b = _win(g, e.timestamp, '6h', '18h')
        if len(a) < 300 or len(b) < 300 or abs(b.choke_pct.mean() - e.choke_opening_pct) > 0.1 or a.choke_pct.nunique() > 1: continue
        if e.choke_opening_pct == 0: continue
        if (g[(g.timestamp >= e.timestamp - pd.Timedelta('24h')) & (g.timestamp < e.timestamp)].choke_pct == 0).any(): continue  # restart transient
        f = lambda x: (x.choke_pct.mean(), (x.whp_psia - x.whp_downstream_psia).mean(), x.gas_lift_rate_mscfd.mean(), x.dh_pressure_psia.mean())
        (c0, dp0, gl0, p0), (c1, dp1, gl1, p1) = f(a), f(b)
        dlnq = coefs['lchoke'] * np.log(c1 / c0) + coefs['ldp'] * np.log(dp1 / dp0) + coefs['lgl'] * np.log(gl1 / gl0) + coefs['dhp'] * (p1 - p0) / 1000
        ddhp = p1 - p0
        if abs(ddhp) < min_ddhp or dlnq * ddhp >= 0: continue
        rows.append(dict(well_name=e.well_name, time=e.timestamp, source='choke step', dDHP=ddhp, dlnQ=dlnq,
                         drawdown_psi=-ddhp / dlnq, pwf_ref=p0))
    return pd.DataFrame(rows)

def buildup_estimates(hf, flat_tol=3.0):
    rows = []
    for w, g in hf.groupby('well_name'):
        g = g.reset_index(drop=True); sh = g.index[g.choke_pct == 0]
        if len(sh) == 0: continue
        # contiguous shut-in blocks
        blocks = np.split(sh, np.where(np.diff(sh) > 1)[0] + 1)
        for bl in blocks:
            p = g.loc[bl, 'dh_pressure_psia']
            last_hour_rise = p.iloc[-1] - p.iloc[-60] if len(p) > 120 else np.inf
            pre = g.loc[max(bl[0] - 720, 0):bl[0] - 1]; pre = pre[pre.choke_pct > 0]
            rows.append(dict(well_name=w, time=g.timestamp[bl[0]], source='build-up', p_end=p.iloc[-1], pwf_ref=pre.dh_pressure_psia.mean(),
                             drawdown_psi=p.iloc[-1] - pre.dh_pressure_psia.mean(), last_hour_rise=last_hour_rise,
                             stabilised=last_hour_rise < flat_tol, duration_h=len(bl) / 60))
    return pd.DataFrame(rows)

def ipr_regression_estimates(tests):
    """Per-well OLS of liquid rate on (time, DHP) over the tests: -coef(DHP) = J. Kept only if physically plausible."""
    from sklearn.linear_model import LinearRegression
    rows = []
    for w, g in tests.groupby('well_name'):
        m = LinearRegression().fit(g[['tday', 'dh_pressure_psia']].values, g.liquid_rate_stbd.values)
        J = -m.coef_[1]
        if 0.3 < J < 60:
            rows.append(dict(well_name=w, time=g.test_start.iloc[len(g) // 2], source='test IPR regression', J_direct=J,
                             drawdown_psi=g.liquid_rate_stbd.mean() / J))
    return pd.DataFrame(rows)

def estimate_J(hf, choke_events, coefs, tests):
    ds = drawdown_estimates(hf, choke_events, coefs)
    bu = buildup_estimates(hf)
    est = pd.concat([ds, bu[bu.stabilised]], ignore_index=True)
    # wells with fewer than 3 independent HF estimates: add the (weak) test-based IPR regression estimate
    few = [w for w in tests.well_name.unique() if (est.well_name == w).sum() < 3]
    ir = ipr_regression_estimates(tests); est = pd.concat([est, ir[ir.well_name.isin(few)]], ignore_index=True)
    # liquid rate at the event time from tests (time interpolation) -> J = Q / drawdown
    out = []
    for w, g in tests.groupby('well_name'):
        g = g.sort_values('tday')
        e = est[est.well_name == w].copy()
        td = (e.time - pd.Timestamp('2025-01-01')).dt.total_seconds() / 86400
        q = np.interp(td, g.tday, g.liquid_rate_stbd)
        e['J'] = q / e.drawdown_psi
        if 'J_direct' in e: e['J'] = e['J_direct'].fillna(e['J'])
        out.append(e)
    est = pd.concat(out, ignore_index=True)
    J = est.groupby('well_name').J.median()
    return J, est, bu
