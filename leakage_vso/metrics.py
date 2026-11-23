from __future__ import annotations
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr


def _safe_corr(a,b,kind="pearson"):
    a=np.asarray(a,float); b=np.asarray(b,float); m=np.isfinite(a)&np.isfinite(b)
    if m.sum()<3 or np.std(a[m])<1e-12 or np.std(b[m])<1e-12: return np.nan
    return float((pearsonr if kind=="pearson" else spearmanr)(a[m],b[m]).statistic)


def gene_metrics(y_true,y_pred,genes):
    rows=[]
    for j,g in enumerate(genes):
        yt=y_true[:,j]; yp=y_pred[:,j]
        rows.append({"gene":g,"pcc":_safe_corr(yt,yp),"spearman":_safe_corr(yt,yp,"spearman"),
                     "mae":float(np.mean(np.abs(yt-yp))),"rmse":float(np.sqrt(np.mean((yt-yp)**2)))})
    return pd.DataFrame(rows)


def donor_gene_metrics(y_true,y_pred,genes,donor_ids):
    donor_ids=np.asarray(donor_ids).astype(str); rows=[]
    for d in sorted(np.unique(donor_ids)):
        ix=np.where(donor_ids==d)[0]
        if len(ix)<3: continue
        gm=gene_metrics(y_true[ix],y_pred[ix],genes); gm.insert(0,"donor_id",d); gm["n_cells"]=len(ix); rows.append(gm)
    return pd.concat(rows,ignore_index=True) if rows else pd.DataFrame(columns=["donor_id","gene","pcc","spearman","mae","rmse","n_cells"])


def pairwise_ranking_inversion(scores_raw: dict[str,float],scores_clean: dict[str,float]):
    keys=sorted(set(scores_raw)&set(scores_clean)); inv=tot=0
    for i,a in enumerate(keys):
        for b in keys[i+1:]:
            dr=scores_raw[a]-scores_raw[b]; dc=scores_clean[a]-scores_clean[b]
            if dr==0 or dc==0: continue
            tot+=1; inv += (dr*dc<0)
    return {"n_pairs":tot,"n_inversions":inv,"inversion_fraction":float(inv/tot) if tot else np.nan}
