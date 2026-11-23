from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from .models import ContextRegressor, EndoVSO
from .metrics import gene_metrics, donor_gene_metrics
from .utils import device_from_string
from .train_engine import predict


def load_checkpoint(path,device):
    ck=torch.load(path,map_location=device,weights_only=False); cfg=ck["cfg"]; genes=ck["genes"]
    common=dict(n_genes=len(genes),backbone=cfg["model"].get("backbone","resnet18"),pretrained=False,freeze_backbone=False)
    if cfg["model"].get("method","context")=="endovso": m=EndoVSO(**common,shared_encoder=cfg["model"].get("shared_encoder",True)); method="endovso"
    else: m=ContextRegressor(**common); method="context"
    m.load_state_dict(ck["model"]); m.to(device); return m,cfg,genes,method


def _write_metrics(outdir,name,y,p,genes,donor_ids):
    yt=np.log1p(y); pt=np.log1p(p)
    gene_metrics(yt,pt,genes).to_csv(outdir/f"metrics_{name}.csv",index=False)
    donor_gene_metrics(yt,pt,genes,donor_ids).to_csv(outdir/f"metrics_{name}_by_donor.csv",index=False)


def evaluate_checkpoint(checkpoint,dataset,outdir,batch_size=64,workers=4,device="auto"):
    dev=device_from_string(device); m,cfg,genes,method=load_checkpoint(checkpoint,dev)
    dl=DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=workers,pin_memory=True)
    raw,preds,clean,meta=predict(m,dl,dev,method); outdir=Path(outdir); outdir.mkdir(parents=True,exist_ok=True)
    donors=np.array([m["donor_id"] for m in meta],dtype=str)
    if method=="endovso":
        _write_metrics(outdir,"raw",raw,preds["obs"],genes,donors)
        if clean is not None: _write_metrics(outdir,"clean",clean,preds["endo"],genes,donors)
        if clean is not None:
            residual=np.maximum(raw-clean,0); _write_metrics(outdir,"leak",residual,preds["leak"],genes,donors)
        save={"raw":raw,"pred_endo":preds["endo"],"pred_leak":preds["leak"],"pred_obs":preds["obs"]}
    else:
        p=preds["context"]; _write_metrics(outdir,"raw",raw,p,genes,donors)
        if clean is not None: _write_metrics(outdir,"clean",clean,p,genes,donors)
        save={"raw":raw,"pred":p}
    save.update({"clean":clean if clean is not None else np.array([]),"genes":np.array(genes),
                 "sample_id":np.array([m["sample_id"] for m in meta]),"cell_id":np.array([m["cell_id"] for m in meta]),"donor_id":donors})
    np.savez_compressed(outdir/"predictions.npz",**save)
