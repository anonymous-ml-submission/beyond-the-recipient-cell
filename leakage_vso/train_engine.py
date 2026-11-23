from __future__ import annotations
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
from .models import ContextRegressor, EndoVSO
from .losses import poisson_loss, log1p_huber_loss, endovso_loss
from .utils import seed_everything, device_from_string, atomic_json

@torch.no_grad()
def predict(model,loader,device,method):
    model.eval(); raw=[]; clean=[]; meta=[]; pred_context=[]; pred_endo=[]; pred_leak=[]; pred_obs=[]
    for b in loader:
        if method=="endovso":
            out=model(b["outer"].to(device),b["ring"].to(device))
            pred_endo.append(out["endo"].cpu().numpy()); pred_leak.append(out["leak"].cpu().numpy()); pred_obs.append(out["obs"].cpu().numpy())
        else:
            pred_context.append(model(b["outer"].to(device)).cpu().numpy())
        raw.append(b["raw"].numpy())
        if "clean" in b: clean.append(b["clean"].numpy())
        meta.extend({"sample_id":s,"cell_id":c,"donor_id":d} for s,c,d in zip(b["sample_id"],b["cell_id"],b["donor_id"]))
    preds={"context":np.concatenate(pred_context) if pred_context else None,
           "endo":np.concatenate(pred_endo) if pred_endo else None,
           "leak":np.concatenate(pred_leak) if pred_leak else None,
           "obs":np.concatenate(pred_obs) if pred_obs else None}
    return np.concatenate(raw),preds,(np.concatenate(clean) if clean else None),meta


def train_model(cfg,train_ds,val_ds,genes,outdir):
    seed_everything(int(cfg.get("seed",0))); device=device_from_string(cfg.get("device","auto"))
    method=cfg["model"].get("method","context")
    common=dict(n_genes=len(genes),backbone=cfg["model"].get("backbone","resnet18"),pretrained=cfg["model"].get("pretrained",True),freeze_backbone=cfg["model"].get("freeze_backbone",False))
    model=EndoVSO(**common,shared_encoder=cfg["model"].get("shared_encoder",True)) if method=="endovso" else ContextRegressor(**common)
    model.to(device)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=float(cfg["training"].get("lr",1e-4)),weight_decay=float(cfg["training"].get("weight_decay",1e-4)))
    bs=int(cfg["training"].get("batch_size",32)); nw=int(cfg["training"].get("workers",4))
    train_loader=DataLoader(train_ds,batch_size=bs,shuffle=True,num_workers=nw,pin_memory=True)
    val_loader=DataLoader(val_ds,batch_size=bs,shuffle=False,num_workers=nw,pin_memory=True)
    outdir=Path(outdir); outdir.mkdir(parents=True,exist_ok=True); best=float("inf"); history=[]
    for epoch in range(int(cfg["training"].get("epochs",20))):
        model.train(); losses=[]
        for b in train_loader:
            opt.zero_grad(set_to_none=True)
            if method=="endovso":
                if "clean" not in b: raise ValueError("EndoVSO requires corrected/clean targets")
                out=model(b["outer"].to(device),b["ring"].to(device))
                loss,_=endovso_loss(out,b["raw"].to(device),b["clean"].to(device),
                    lambda_endo=float(cfg["loss"].get("lambda_endo",1)),lambda_obs=float(cfg["loss"].get("lambda_obs",1)),lambda_leak=float(cfg["loss"].get("lambda_leak",0.25)))
            else:
                key=cfg["data"].get("target","raw")
                if key=="clean" and "clean" not in b: raise ValueError("clean target requested but unavailable")
                pred=model(b["outer"].to(device)); loss=(log1p_huber_loss(pred,b[key].to(device)) if key=="clean" else poisson_loss(pred,b[key].to(device)))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step(); losses.append(float(loss.detach()))
        yraw,preds,yclean,_=predict(model,val_loader,device,method)
        if method=="endovso":
            # Select on endogenous validation target because that is the scientific output.
            if yclean is None: raise RuntimeError("EndoVSO validation requires clean target")
            p=preds["endo"]; target=yclean
        else:
            p=preds["context"]; target=yclean if cfg["data"].get("target")=="clean" and yclean is not None else yraw
        val_loss=float(np.mean((np.log1p(p)-np.log1p(target))**2))
        row={"epoch":epoch,"train_loss":float(np.mean(losses)),"val_log_mse":val_loss}; history.append(row)
        if val_loss<best:
            best=val_loss; torch.save({"model":model.state_dict(),"cfg":cfg,"genes":genes},outdir/"best.pt")
        atomic_json(outdir/"history.json",history)
    return outdir/"best.pt"
