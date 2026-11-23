#!/usr/bin/env python3
"""Compare raw-vs-endogenous model rankings with donor/gene Fisher-z aggregation."""
import argparse, pandas as pd, json, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from leakage_vso.metrics import pairwise_ranking_inversion
p=argparse.ArgumentParser(); p.add_argument('--runs',required=True); p.add_argument('--out',required=True); a=p.parse_args(); rows=[]
eps=1e-6
for rd in sorted(Path(a.runs).iterdir()):
    rb=rd/'eval'/'metrics_raw_by_donor.csv'; cb=rd/'eval'/'metrics_clean_by_donor.csv'; r0=rd/'eval'/'metrics_raw.csv'; c0=rd/'eval'/'metrics_clean.csv'
    if rb.exists() and cb.exists():
        r=pd.read_csv(rb)[['donor_id','gene','pcc']].rename(columns={'pcc':'raw_pcc'}); c=pd.read_csv(cb)[['donor_id','gene','pcc']].rename(columns={'pcc':'clean_pcc'}); q=r.merge(c,on=['donor_id','gene']).dropna()
        q['raw_z']=np.arctanh(q.raw_pcc.clip(-1+eps,1-eps)); q['clean_z']=np.arctanh(q.clean_pcc.clip(-1+eps,1-eps)); bydon=q.groupby('donor_id')[['raw_z','clean_z']].mean()
        rr=float(bydon.raw_z.mean()); cc=float(bydon.clean_z.mean()); nd=len(bydon); ng=q.gene.nunique()
    elif r0.exists() and c0.exists():
        r=pd.read_csv(r0)[['gene','pcc']].rename(columns={'pcc':'raw_pcc'}); c=pd.read_csv(c0)[['gene','pcc']].rename(columns={'pcc':'clean_pcc'}); q=r.merge(c,on='gene').dropna(); rr=float(np.arctanh(q.raw_pcc.clip(-1+eps,1-eps)).mean()); cc=float(np.arctanh(q.clean_pcc.clip(-1+eps,1-eps)).mean()); nd=np.nan; ng=len(q)
    else: continue
    rows.append({'model':rd.name,'raw_fisher_z':rr,'clean_fisher_z':cc,'raw_equiv_r':float(np.tanh(rr)),'clean_equiv_r':float(np.tanh(cc)),'n_donors':nd,'n_genes':ng})
d=pd.DataFrame(rows); Path(a.out).parent.mkdir(parents=True,exist_ok=True); d.to_csv(a.out,index=False)
if len(d)>=2:
    rho=spearmanr(d.raw_fisher_z,d.clean_fisher_z).statistic; inv=pairwise_ranking_inversion(dict(zip(d.model,d.raw_fisher_z)),dict(zip(d.model,d.clean_fisher_z))); print(json.dumps({'spearman_model_ranking':rho,**inv},indent=2))
print(d.sort_values('raw_fisher_z',ascending=False).to_string(index=False) if len(d) else 'No evaluable runs')
