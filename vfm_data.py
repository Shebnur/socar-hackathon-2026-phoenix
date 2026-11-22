"""Well-test QC and window feature extraction."""
import numpy as np, pandas as pd
from vfm_model import SENS, T0
M3_PER_MSCF = 28.316846592

def qc_well_tests(t):
    t = t.copy(); t['qc_flag'] = ''
    for c in ['test_start', 'test_end']: t[c] = pd.to_datetime(t[c])
    # Unit check: formation gas should equal oil*GOR/1000. Rows ~28.3x too high are in m3/d.
    ratio = t.gas_rate_mscfd.abs() / (t.oil_rate_stbd.abs() * t.gor_scf_stb / 1000)
    m = (ratio - M3_PER_MSCF).abs() < 1.0
    for c in ['gas_rate_mscfd', 'total_gas_rate_mscfd']: t.loc[m, c] = t.loc[m, c] / M3_PER_MSCF
    t.loc[m, 'qc_flag'] += 'gas m3/d->mscf/d;'
    # Sign errors: negative rates (magnitude confirmed by liquid / GOR / gas-balance redundancy)
    for c in ['oil_rate_stbd', 'water_rate_stbd', 'gas_rate_mscfd', 'total_gas_rate_mscfd', 'gas_lift_rate_mscfd', 'liquid_rate_stbd']:
        neg = t[c] < 0; t.loc[neg, c] = t.loc[neg, c].abs(); t.loc[neg, 'qc_flag'] += f'{c} sign fixed;'
    t['chk_liquid'] = (t.oil_rate_stbd + t.water_rate_stbd - t.liquid_rate_stbd).round(2)
    t['chk_gas_balance'] = (t.gas_rate_mscfd + t.gas_lift_rate_mscfd - t.total_gas_rate_mscfd).round(2)
    t['chk_gor'] = (1000 * t.gas_rate_mscfd / t.oil_rate_stbd - t.gor_scf_stb).round(2)
    return t

def test_window_features(hf, tests):
    rows = []
    for i, r in tests.iterrows():
        g = hf[(hf.well_name == r.well_name) & (hf.timestamp >= r.test_start) & (hf.timestamp < r.test_end) & (hf.choke_pct > 0)]
        f = g[SENS + ['choke_pct']].mean(); f.name = i; rows.append(f)
    F = pd.DataFrame(rows)
    out = tests.drop(columns=[c for c in SENS if c in tests.columns]).join(F)
    out['tday'] = ((out.test_start + (out.test_end - out.test_start) / 2) - T0).dt.total_seconds() / 86400
    return out
