#!/usr/bin/env python3
"""Select high-leakage genes and one-to-one abundance/prevalence matched low-leakage controls."""
import argparse, pandas as pd, numpy as np
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument("--leakage",required=True); p.add_argument("--stats",required=True); p.add_argument("--out",required=True); p.add_argument("--n-high",type=int,default=32); p.add_argument("--n-low",type=int,default=32); p.add_argument("--min-r2",type=float,default=0.0); p.add_argument("--min-samples",type=int,default=1); a=p.parse_args()
l=pd.read_csv(a.leakage).dropna(subset=["gamma"]); s=pd.read_csv(a.stats); d=l.merge(s,on="gene",how="inner")
if "n_samples" in d: d=d[d.n_samples>=a.min_samples]
d=d[d.r2.fillna(-np.inf)>=a.min_r2].copy()
if len(d)<a.n_high+a.n_low: raise RuntimeError(f"Not enough eligible genes ({len(d)})")
high=d.nlargest(a.n_high,"gamma").copy(); high_set=set(high.gene.astype(str)); low_pool=d[~d.gene.astype(str).isin(high_set)].nsmallest(max(a.n_low*8,a.n_low),"gamma").copy()
features=[c for c in ["log_mean_count","prevalence","specificity","moran_i"] if c in d.columns]
if not {"log_mean_count","prevalence"}.issubset(features): raise RuntimeError("stats must contain log_mean_count and prevalence")
mu=d[features].mean(); sd=d[features].std().replace(0,1).fillna(1); used=set(); matches=[]
for _,h in high.iterrows():
    cand=low_pool[~low_pool.gene.astype(str).isin(used)].copy()
    if cand.empty: break
    diff=(cand[features]-h[features])/sd; cand["match_distance"]=np.sqrt((diff**2).sum(1)); m=cand.nsmallest(1,"match_distance").iloc[0]
    used.add(str(m.gene)); matches.append(m)
low=pd.DataFrame(matches)
if len(low)<min(a.n_low,a.n_high): raise RuntimeError(f"Could match only {len(low)} low-leakage controls")
high=high.head(len(low)); panel=pd.concat([high.assign(group="high"),low.assign(group="low")],ignore_index=True)
panel.to_csv(Path(a.out).with_suffix(".csv"),index=False); Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text("\n".join(panel.gene.astype(str))+"\n")
print(panel[["gene","group","gamma",*features]+(["match_distance"] if "match_distance" in panel else [])].to_string(index=False))
