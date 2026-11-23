from __future__ import annotations
from dataclasses import dataclass
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.spatial import cKDTree

@dataclass
class LeakageFit:
    gene:str; gamma:float; beta:float; r2:float; n:int


def build_radius_graph(coords:np.ndarray,radius_um:float):
    coords=np.asarray(coords,float); tree=cKDTree(coords); rows=[]; cols=[]; dist=[]
    for i,idxs in enumerate(tree.query_ball_point(coords,r=radius_um)):
        for j in idxs:
            if j==i: continue
            rows.append(i); cols.append(j); dist.append(float(np.linalg.norm(coords[i]-coords[j])))
    return np.asarray(rows,np.int64),np.asarray(cols,np.int64),np.asarray(dist,float),len(coords)


def graph_neighbor_signal(graph,values,beta:float):
    rows,cols,dist,n=graph; values=np.asarray(values,float); out=np.zeros(n,float)
    if len(rows): np.add.at(out,rows,np.exp(-beta*dist)*values[cols])
    return out


def kernel_neighbor_signal(coords,values,beta,radius_um,exclude_self=True):
    # Backward-compatible wrapper; graph implementation avoids repeated tree construction in multi-beta fits.
    return graph_neighbor_signal(build_radius_graph(coords,radius_um),values,beta)


def fit_gene_leakage(coords,raw,clean,betas=(0.02,0.05,0.1,0.2,0.4),radius_um=40.0,graph=None):
    y=np.maximum(np.asarray(raw,float)-np.asarray(clean,float),0.0)
    if np.var(y)<1e-12: return LeakageFit("",0.0,np.nan,np.nan,len(y))
    graph=graph if graph is not None else build_radius_graph(coords,radius_um); best=None; denom_y=((y-y.mean())**2).sum()
    for beta in betas:
        s=graph_neighbor_signal(graph,np.asarray(clean,float),beta)
        gamma=max(0.0,float(np.dot(s,y)/(np.dot(s,s)+1e-12))); pred=gamma*s
        r2=1.0-float(((y-pred)**2).sum()/denom_y)
        if best is None or r2>best.r2: best=LeakageFit("",gamma,float(beta),r2,len(y))
    return best


def estimate_leakage_table(coords,raw_X,clean_X,genes,betas=(0.02,0.05,0.1,0.2,0.4),radius_um=40.0,min_clean_sum=20.0):
    graph=build_radius_graph(coords,radius_um); rows=[]
    for gi,g in enumerate(genes):
        rc=raw_X[:,gi].toarray().ravel() if sp.issparse(raw_X) else np.asarray(raw_X[:,gi]).ravel()
        cc=clean_X[:,gi].toarray().ravel() if sp.issparse(clean_X) else np.asarray(clean_X[:,gi]).ravel()
        if cc.sum()<min_clean_sum: continue
        f=fit_gene_leakage(coords,rc,cc,betas,radius_um,graph=graph); f.gene=g; rows.append(f.__dict__)
    return pd.DataFrame(rows).sort_values(["r2","gamma"],ascending=False) if rows else pd.DataFrame(columns=["gene","gamma","beta","r2","n"])


def synthetic_ranking_reversal(gamma=1.0,p_source=0.5,n=10000,seed=0):
    rng=np.random.default_rng(seed); s=rng.binomial(1,p_source,size=n).astype(float); endo=np.zeros(n); obs=gamma*s
    f_bio=np.zeros(n); f_leak=gamma*s; mse=lambda a,b:float(np.mean((a-b)**2))
    return {"raw_bio":mse(obs,f_bio),"raw_leak":mse(obs,f_leak),"endo_bio":mse(endo,f_bio),"endo_leak":mse(endo,f_leak)}
