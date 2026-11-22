"""
Virtual Flow Meter (VFM).

HybridVFM (production model) = ChokeVFM (data-driven choke-orifice regression) + IPR anchoring:
  * ChokeVFM is fitted on the well tests and used to read the relative rate response at choke steps in the
    high-frequency data; together with stabilised build-ups this gives each well's productivity index J.
  * Liquid rate = J * (Pr(t) - Pwf), Pwf = downhole gauge pressure, Pr(t) = per-well linear trend fitted to the
    tests (Pr_i = Pwf_i + Q_i / J). This transfers test-separator conditions (lower backpressure) to normal
    production conditions through the reservoir inflow, which the tests alone cannot identify.

ChokeVFM — physics-informed regression model.

Liquid rate (choke-orifice relationship, exponents shared by all wells, one calibration constant per well):
    ln(Q_liq) = a_well + b1*ln(choke%) + b2*ln(WHP - WHP_dn) + b3*ln(Q_gl) + b4*DHP/1000
Water cut:  per-well linear trend in time (fitted to well tests)
GOR:        per-well constant = 0.5 * mean(test GOR) + 0.5 * Rs at Pb (completion data)
Oil = Q_liq*(1-WC), Water = Q_liq*WC, Gas = Oil*GOR/1000  (formation gas, excludes lift gas)
"""
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression

T0 = pd.Timestamp('2025-01-01')
SPEC = ('lchoke', 'ldp', 'lgl', 'dhp')
SENS = ['whp_psia','whp_downstream_psia','dh_pressure_psia','dh_temperature_degf','annulus_pressure_psia','gas_lift_rate_mscfd']

def _feat(d, s):
    if s == 'lchoke': return np.log(d['choke_pct'].clip(lower=0.1).values)
    if s == 'ldp':    return np.log((d['whp_psia'] - d['whp_downstream_psia']).clip(lower=1).values)
    if s == 'lgl':    return np.log(d['gas_lift_rate_mscfd'].clip(lower=1).values)
    if s == 'dhp':    return d['dh_pressure_psia'].values / 1000
    if s == 'tday':   return d['tday'].values / 100
    raise KeyError(s)

def recency_weights(tday, half_life_days=None):
    """Weight of each well test: 1 for the latest test, halving every `half_life_days` back in time (None = all equal)."""
    tday = np.asarray(tday, dtype=float)
    if not half_life_days:
        return np.ones_like(tday)
    return 0.5 ** ((tday.max() - tday) / half_life_days)


class ChokeVFM:
    def __init__(self, spec=SPEC, gor_weight=0.5, half_life_days=None):
        self.spec = tuple(spec); self.gor_weight = gor_weight; self.half_life_days = half_life_days

    def _X(self, d):
        X = np.column_stack([_feat(d, s) for s in self.spec])
        W = np.column_stack([(d['well_name'] == w).values.astype(float) for w in self.wells])
        return np.hstack([X, W])

    def fit(self, tests, rs_by_well):
        """tests: QC'd well tests with HF window-mean features, columns liquid_rate_stbd, water_rate_stbd, gor_scf_stb, tday."""
        self.wells = sorted(tests.well_name.unique())
        self.liq = LinearRegression(fit_intercept=False).fit(self._X(tests), np.log(tests.liquid_rate_stbd.values))
        self.wc_model, self.gor = {}, {}
        for w, g in tests.groupby('well_name'):
            wt = recency_weights(g.tday.values, self.half_life_days)
            wc = g.water_rate_stbd / g.liquid_rate_stbd
            m = LinearRegression().fit(g[['tday']].values, wc.values, sample_weight=wt)
            self.wc_model[w] = (float(m.intercept_), float(m.coef_[0]))
            self.gor[w] = self.gor_weight * np.average(g.gor_scf_stb, weights=wt) + (1 - self.gor_weight) * rs_by_well[w]
        return self

    @property
    def coefficients(self):
        c = dict(zip(self.spec, self.liq.coef_[:len(self.spec)]))
        c.update({f'a_{w}': v for w, v in zip(self.wells, self.liq.coef_[len(self.spec):])})
        return c

    def predict_flowing(self, d):
        """Rates while flowing for rows with window-mean sensor features + tday. Returns DataFrame oil/water/gas."""
        liq = np.exp(self.liq.predict(self._X(d)))
        wc = np.array([self.wc_model[w][0] + self.wc_model[w][1] * t for w, t in zip(d['well_name'], d['tday'])])
        wc = np.clip(wc, 0, 0.99)
        gor = d['well_name'].map(self.gor).values
        oil = liq * (1 - wc)
        return pd.DataFrame({'oil_rate_stbd': oil, 'water_rate_stbd': liq * wc,
                             'gas_rate_mscfd': oil * gor / 1000, 'liquid_rate_stbd': liq, 'water_cut': wc}, index=d.index)

    def predict_daily(self, hf_clean):
        """hf_clean: cleaned per-minute data with choke_pct. Predicts on hourly means, zero when shut in,
        then averages the 24 hours -> daily rate (stb/d, mscf/d)."""
        H = hourly_features(hf_clean)
        return aggregate_daily(H, predict_hours(self, H))


def hourly_features(hf_clean):
    """Per-minute data -> one row per well-hour: sensor means while flowing, on-stream fraction, tday."""
    h = hf_clean.copy()
    h['flowing'] = (h['choke_pct'] > 0).astype(float)
    h['hour'] = h['timestamp'].dt.floor('h')
    fl = h[h.flowing > 0]
    agg = fl.groupby(['well_name', 'hour'])[SENS + ['choke_pct']].mean()
    on = h.groupby(['well_name', 'hour'])['flowing'].mean().rename('onstream_frac')
    H = on.to_frame().join(agg).reset_index()
    parts = []
    for w, g in H.groupby('well_name'):
        g = g.set_index('hour').sort_index()
        g['sensor_gap'] = g['whp_psia'].isna() & (g.onstream_frac > 0)
        g[SENS] = g[SENS].interpolate(method='time', limit_direction='both')
        g['choke_pct'] = g['choke_pct'].fillna(0)
        parts.append(g.reset_index())
    H = pd.concat(parts, ignore_index=True)
    H['tday'] = (H['hour'] - T0).dt.total_seconds() / 86400
    return H


def predict_hours(model, H, rows=None):
    """Hourly rates (zero when shut in, scaled by on-stream fraction). `rows` = boolean mask of hours to predict."""
    P = pd.DataFrame(0.0, index=H.index, columns=['oil_rate_stbd', 'water_rate_stbd', 'gas_rate_mscfd'])
    flow = (H.onstream_frac > 0) & H['dh_pressure_psia'].notna()
    if rows is not None:
        flow &= rows
    if flow.any():
        pf = model.predict_flowing(H[flow])
        for c in P.columns:
            P.loc[flow, c] = pf[c] * H.loc[flow, 'onstream_frac']
    return P


def aggregate_daily(H, P):
    """Hourly rates -> daily average rate per well (= daily volume / 1 day)."""
    H = pd.concat([H, P], axis=1)
    H['date'] = H['hour'].dt.normalize()
    return H.groupby(['date', 'well_name']).agg(
        oil_rate_stbd=('oil_rate_stbd', 'mean'), water_rate_stbd=('water_rate_stbd', 'mean'),
        gas_rate_mscfd=('gas_rate_mscfd', 'mean'), onstream_frac=('onstream_frac', 'mean'),
        sensor_gap_hours=('sensor_gap', 'sum')).reset_index()


class HybridVFM(ChokeVFM):
    """IPR-anchored VFM. Inherits water-cut, GOR and daily aggregation logic from ChokeVFM."""
    def __init__(self, gor_weight=0.5, half_life_days=None):
        super().__init__(SPEC, gor_weight, half_life_days)

    def fit(self, tests, rs_by_well, hf=None, choke_events=None, J=None):
        super().fit(tests, rs_by_well)                     # choke regression + WC + GOR
        if J is None:
            from pi_estimation import estimate_J
            J, self.pi_estimates, self.buildups = estimate_J(hf, choke_events, self.coefficients, tests)
            # unstabilised build-ups give a lower bound on drawdown -> upper bound on J
            for _, b in self.buildups[~self.buildups.stabilised].iterrows():
                g = tests[tests.well_name == b.well_name].sort_values('tday')
                td = (b.time - T0).total_seconds() / 86400
                jmax = np.interp(td, g.tday, g.liquid_rate_stbd) / b.drawdown_psi
                if b.well_name in J and J[b.well_name] > jmax: J[b.well_name] = jmax
        self.J = dict(J)
        self.pr_trend = {}
        for w, g in tests.groupby('well_name'):
            pr = g.dh_pressure_psia + g.liquid_rate_stbd / self.J[w]
            m = LinearRegression().fit(g[['tday']].values, pr.values,
                                       sample_weight=recency_weights(g.tday.values, self.half_life_days))
            self.pr_trend[w] = (float(m.intercept_), float(m.coef_[0]))
        return self

    def reservoir_pressure(self, well, tday):
        a, b = self.pr_trend[well]; return a + b * np.asarray(tday)

    def predict_flowing(self, d):
        J = d['well_name'].map(self.J).values
        pr = np.array([self.reservoir_pressure(w, t) for w, t in zip(d['well_name'], d['tday'])])
        liq = np.clip(J * (pr - d['dh_pressure_psia'].values), 0, None)
        wc = np.array([self.wc_model[w][0] + self.wc_model[w][1] * t for w, t in zip(d['well_name'], d['tday'])])
        wc = np.clip(wc, 0, 0.99)
        gor = d['well_name'].map(self.gor).values
        oil = liq * (1 - wc)
        return pd.DataFrame({'oil_rate_stbd': oil, 'water_rate_stbd': liq * wc, 'gas_rate_mscfd': oil * gor / 1000,
                             'liquid_rate_stbd': liq, 'water_cut': wc}, index=d.index)
