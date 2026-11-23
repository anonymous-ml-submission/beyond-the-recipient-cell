#!/usr/bin/env python3
"""Test context–leakage coupling using Fisher-z donor/gene effects and adjusted inference."""
import argparse, json, pandas as pd, numpy as np
from scipy.stats import spearmanr, pearsonr
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--small',required=True); p.add_argument('--large',required=True); p.add_argument('--leakage',required=True); p.add_argument('--stats',default=None); p.add_argument('--out',required=True); p.add_argument('--bootstrap',type=int,default=2000); p.add_argument('--seed',type=int,default=2026); a=p.parse_args()
s=pd.read_csv(a.small); l=pd.read_csv(a.large); g=pd.read_csv(a.leakage)
keys=['gene']+(['donor_id'] if 'donor_id' in s.columns and 'donor_id' in l.columns else [])
s=s[keys+['pcc']].rename(columns={'pcc':'pcc_small'}); l=l[keys+['pcc']].rename(columns={'pcc':'pcc_large'})
d=s.merge(l,on=keys).merge(g,on='gene'); eps=1e-6
d['pcc_small_clip']=d.pcc_small.clip(-1+eps,1-eps); d['pcc_large_clip']=d.pcc_large.clip(-1+eps,1-eps)
d['z_small']=np.arctanh(d.pcc_small_clip); d['z_large']=np.arctanh(d.pcc_large_clip); d['context_gain_z']=d.z_large-d.z_small; d['context_gain_pcc']=d.pcc_large-d.pcc_small
if a.stats: d=d.merge(pd.read_csv(a.stats),on='gene',how='left')
Path(a.out).parent.mkdir(parents=True,exist_ok=True); d.to_csv(a.out,index=False)
cov=[c for c in ['gamma','log_mean_count','prevalence','specificity','moran_i'] if c in d.columns]
gene=d.groupby('gene',as_index=False).agg(context_gain_z=('context_gain_z','mean'),context_gain_pcc=('context_gain_pcc','mean'),gamma=('gamma','first'),r2=('r2','first'),**({c:(c,'first') for c in cov if c not in {'gamma'}}))
gene.to_csv(Path(a.out).with_suffix('.by_gene.csv'),index=False)
summary={}
for x in ['gamma','r2']:
    z=gene[['context_gain_z',x]].dropna()
    if len(z)>=3:
        summary[f'{x}_pearson_z']=float(pearsonr(z.context_gain_z,z[x]).statistic); summary[f'{x}_spearman_z']=float(spearmanr(z.context_gain_z,z[x]).statistic); summary[f'{x}_n_genes']=len(z)
# Primary adjusted gene-level model: genes are units after donor aggregation.
z=gene[['context_gain_z',*cov]].dropna() if cov else pd.DataFrame()
if 'gamma' in cov and len(z)>=max(10,len(cov)+3):
    X=sm.add_constant(z[cov].astype(float)); fit=sm.OLS(z.context_gain_z.astype(float),X).fit(cov_type='HC3')
    summary['gene_HC3_beta_gamma']=float(fit.params['gamma']); summary['gene_HC3_se_gamma']=float(fit.bse['gamma']); summary['gene_HC3_p_gamma']=float(fit.pvalues['gamma'])
    print(fit.summary().as_text())
# Donor-aware repeated-measures check: donor fixed effects, SE clustered by gene.
if 'donor_id' in d.columns and 'gamma' in cov:
    zz=d[['context_gain_z','gene','donor_id',*cov]].dropna().copy()
    if zz.donor_id.nunique()>=3 and zz.gene.nunique()>=10:
        formula='context_gain_z ~ '+' + '.join(cov)+' + C(donor_id)'
        f2=smf.ols(formula,data=zz).fit(cov_type='cluster',cov_kwds={'groups':zz['gene']})
        summary['donorFE_clusterGene_beta_gamma']=float(f2.params['gamma']); summary['donorFE_clusterGene_se_gamma']=float(f2.bse['gamma']); summary['donorFE_clusterGene_p_gamma']=float(f2.pvalues['gamma'])
# Hierarchical bootstrap: resample donors, then genes; refit adjusted gene-level coefficient.
if a.bootstrap>0 and 'donor_id' in d.columns and 'gamma' in cov:
    zz=d[['context_gain_z','gene','donor_id',*cov]].dropna().copy(); donors=np.array(sorted(zz.donor_id.astype(str).unique())); rng=np.random.default_rng(a.seed); bs=[]
    if len(donors)>=3:
        for _ in range(a.bootstrap):
            draw=rng.choice(donors,size=len(donors),replace=True); parts=[]
            for slot,don in enumerate(draw):
                q=zz[zz.donor_id.astype(str)==don].copy(); q['boot_slot']=slot; parts.append(q)
            b=pd.concat(parts,ignore_index=True); bg=b.groupby('gene',as_index=False).agg(context_gain_z=('context_gain_z','mean'),**({c:(c,'first') for c in cov}))
            if len(bg)<max(8,len(cov)+2): continue
            bg=bg.iloc[rng.integers(0,len(bg),size=len(bg))].reset_index(drop=True)
            try:
                X=sm.add_constant(bg[cov].astype(float)); bf=sm.OLS(bg.context_gain_z.astype(float),X).fit(); bs.append(float(bf.params['gamma']))
            except Exception: pass
        if len(bs)>=100:
            lo,hi=np.quantile(bs,[0.025,0.975]); summary['hier_boot_beta_gamma_median']=float(np.median(bs)); summary['hier_boot_beta_gamma_ci95']=[float(lo),float(hi)]; summary['hier_boot_n']=len(bs)
Path(a.out).with_suffix('.stats.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
