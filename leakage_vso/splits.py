from __future__ import annotations
import pandas as pd
from .utils import stable_int_hash


def assign_donor_splits(cells: pd.DataFrame, donor_col="donor_id", seed=0, train=0.7,val=0.15):
    if donor_col not in cells:
        raise ValueError(f"Missing {donor_col}; CVPR protocol requires donor-level split")
    donors=sorted(cells[donor_col].astype(str).unique())
    if len(donors)<3:
        raise ValueError(f"Need >=3 donors for train/val/test; found {len(donors)}. Use explicit demo_split for single-donor demos only.")
    score={d: stable_int_hash(seed,d)/float(2**31-1) for d in donors}
    label={d:("train" if score[d]<train else "val" if score[d]<train+val else "test") for d in donors}
    out=cells.copy(); out["split"]=out[donor_col].astype(str).map(label)
    if set(out.split.unique()) != {"train","val","test"}:
        # deterministic rank fallback for small donor counts
        order=sorted(donors,key=lambda d:score[d]); n=len(order)
        ntr=max(1,int(round(n*train))); nval=max(1,int(round(n*val))); ntr=min(n-2,ntr); nval=min(n-ntr-1,nval)
        label={d:("train" if i<ntr else "val" if i<ntr+nval else "test") for i,d in enumerate(order)}
        out["split"]=out[donor_col].astype(str).map(label)
    return out
