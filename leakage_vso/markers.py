from __future__ import annotations
import numpy as np, pandas as pd, scipy.sparse as sp


def select_exclusive_markers(clean_adata, cell_types: pd.Series, min_source_cells=50, max_recipient_mean=0.05, top_k=10):
    X=clean_adata.X
    rows=[]
    types=sorted(pd.Series(cell_types).dropna().astype(str).unique())
    for ct in types:
        src=np.asarray(pd.Series(cell_types).astype(str)==ct)
        if src.sum()<min_source_cells: continue
        for gi,g in enumerate(clean_adata.var_names):
            col=X[:,gi].toarray().ravel() if sp.issparse(X) else np.asarray(X[:,gi]).ravel()
            sm=float(col[src].mean()); om=float(col[~src].mean()) if (~src).any() else 0.
            if sm<=0: continue
            spec=sm/(om+1e-3)
            if om <= max_recipient_mean:
                rows.append({"gene":g,"source_cell_type":ct,"source_mean":sm,"other_mean":om,"specificity":spec})
    df=pd.DataFrame(rows)
    if df.empty: return df
    return df.sort_values(["source_cell_type","specificity"],ascending=[True,False]).groupby("source_cell_type").head(top_k).reset_index(drop=True)
