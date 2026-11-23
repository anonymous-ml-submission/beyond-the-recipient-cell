#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from leakage_vso.config import load_yaml
from leakage_vso.dataset import CellImageDataset
from leakage_vso.train_engine import train_model
p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--out",required=True); a=p.parse_args(); cfg=load_yaml(a.config)
root=cfg["data"]["prepared_root"]
train_sample_ids=cfg["data"].get("train_sample_ids")
val_sample_ids=cfg["data"].get("val_sample_ids")
if not train_sample_ids:
    raise RuntimeError(
        "config data.train_sample_ids is required; "
        "refusing ambiguous train routing"
    )
if not val_sample_ids:
    raise RuntimeError(
        "config data.val_sample_ids is required; "
        "refusing ambiguous validation routing"
    )
if set(map(str,train_sample_ids)) & set(map(str,val_sample_ids)):
    raise RuntimeError(
        "train_sample_ids and val_sample_ids overlap"
    )
if isinstance(sample_ids,str): sample_ids=[sample_ids]
genes=cfg["data"].get("genes")
if isinstance(genes,str): genes=[x.strip() for x in Path(genes).read_text().splitlines() if x.strip()]
if not genes: raise ValueError("data.genes must be a list or text file")
kwargs=dict(prepared_root=root,genes=genes,target=cfg["data"].get("target","raw"),context_um=cfg["data"].get("context_um",64),center_um=cfg["data"].get("center_um",24),model_mpp=cfg["data"].get("model_mpp",0.5),canvas_um=cfg["data"].get("canvas_um",cfg["data"].get("context_um",64)),ring_shuffle=cfg["data"].get("ring_shuffle",False))
tr=CellImageDataset(
    **kwargs,
    sample_ids=train_sample_ids,
    split="train",
    augment=True,
)
va=CellImageDataset(
    **kwargs,
    sample_ids=val_sample_ids,
    split="val",
    augment=False,
)
print(f"train={len(tr)} val={len(va)} genes={len(genes)}")
train_model(cfg,tr,va,genes,a.out)
