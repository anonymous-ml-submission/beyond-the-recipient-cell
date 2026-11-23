#!/usr/bin/env python3
import argparse
from leakage_vso.evaluate import load_checkpoint,evaluate_checkpoint
from leakage_vso.dataset import CellImageDataset
from leakage_vso.utils import device_from_string
p=argparse.ArgumentParser(); p.add_argument("--checkpoint",required=True); p.add_argument("--root",required=True); p.add_argument("--samples",nargs="+",required=True); p.add_argument("--out",required=True); p.add_argument("--split",default="test"); p.add_argument("--device",default="auto"); a=p.parse_args()
_,cfg,genes,_=load_checkpoint(a.checkpoint,device_from_string(a.device))
ds=CellImageDataset(a.root,a.samples,genes,target="both" if cfg["model"].get("method")=="endovso" else cfg["data"].get("target","raw"),context_um=cfg["data"].get("context_um",64),center_um=cfg["data"].get("center_um",24),model_mpp=cfg["data"].get("model_mpp",0.5),canvas_um=cfg["data"].get("canvas_um",cfg["data"].get("context_um",64)),split=a.split,augment=False,ring_shuffle=cfg["data"].get("ring_shuffle",False))
evaluate_checkpoint(a.checkpoint,ds,a.out,device=a.device)
