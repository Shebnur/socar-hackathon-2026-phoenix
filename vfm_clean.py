"""Step 1: clean raw per-minute high-frequency data."""
import pandas as pd, numpy as np, sys
REL_MIN={'dh_temperature_degf':0.01}
SENS=['whp_psia','whp_downstream_psia','dh_pressure_psia','dh_temperature_degf','annulus_pressure_psia','gas_lift_rate_mscfd']

def clean_hf(raw, choke_events, interp_limit=60, win=15, rel_tol=0.2, abs_tol=5.0, report=None):
    raw=raw.copy(); raw['timestamp']=pd.to_datetime(raw['timestamp'])
    rep=[]
    out=[]
    for w,g in raw.groupby('well_name'):
        n0=len(g)
        g=g.drop_duplicates('timestamp').sort_values('timestamp').set_index('timestamp')
        ndup=n0-len(g)
        full=pd.date_range(g.index.min().normalize(), g.index.max().normalize()+pd.Timedelta('1D')-pd.Timedelta('1min'), freq='1min')
        g=g.reindex(full); nmiss=g[SENS[0]].isna().sum()
        r={'well_name':w,'raw_rows':n0,'duplicates':ndup,'missing_minutes':int(nmiss)}
        for c in SENS:
            s=g[c].astype(float)
            sent=(s<=-900)|(s<0); r[f'{c}_sentinel']=int(sent.sum()); s=s.mask(sent)
            med=s.rolling(win,center=True,min_periods=3).median()
            res=(s-med).abs()
            mad=res.rolling(121,center=True,min_periods=10).median()*1.4826
            spike=(res>np.maximum(8*mad,abs_tol))&(res>REL_MIN.get(c,0.03)*med.abs())
            spike|=res>(rel_tol*med.abs()+abs_tol)
            r[f'{c}_spikes']=int(spike.sum()); s=s.mask(spike)
            s=s.interpolate(method='time',limit=interp_limit,limit_area='inside')
            g[c]=s
        # choke from events (step function, forward fill)
        ce=choke_events[choke_events.well_name==w].copy(); ce['timestamp']=pd.to_datetime(ce['timestamp'])
        ch=ce.set_index('timestamp')['choke_opening_pct'].sort_index()
        ch=ch[~ch.index.duplicated(keep='last')]
        g['choke_pct']=ch.reindex(g.index.union(ch.index)).ffill().reindex(g.index).fillna(ch.iloc[0])
        g['well_name']=w
        g.index.name='timestamp'
        out.append(g.reset_index()); rep.append(r)
    return pd.concat(out,ignore_index=True), pd.DataFrame(rep)

if __name__=='__main__':
    # usage: python vfm_clean.py high_freq_data.csv choke_events.csv  -> hf_clean.pkl + cleaning_report.csv
    raw=pd.read_csv(sys.argv[1]); ce=pd.read_csv(sys.argv[2])
    hf,rep=clean_hf(raw,ce)
    hf.to_pickle('hf_clean.pkl'); rep.to_csv('cleaning_report.csv',index=False)
    print(rep.T)
