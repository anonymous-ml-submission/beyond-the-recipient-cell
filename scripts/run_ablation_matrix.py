#!/usr/bin/env python3
"""Generate audit-first ablations with a FIXED physical/network canvas.

Primary 24/64/128 µm runs use identical canvas_um and model_mpp, hence identical tensor
shape. Only the central amount of visible tissue changes.
"""
import argparse, copy, subprocess, sys
from pathlib import Path
import yaml
p=argparse.ArgumentParser(); p.add_argument('--base',required=True); p.add_argument('--out',required=True); p.add_argument('--execute',action='store_true'); a=p.parse_args()
base=yaml.safe_load(Path(a.base).read_text()); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
if 'model_mpp' not in base.get('data',{}): raise RuntimeError('base config must set data.model_mpp')
canvas=float(base.get('data',{}).get('canvas_um',128))
if canvas<128: raise RuntimeError('Primary ablation requires data.canvas_um >= 128 um')
abl=[]
for width in [24,64,128]:
    c=copy.deepcopy(base); c['model']['method']='context'; c['data']['target']='raw'; c['data']['context_um']=width; c['data']['canvas_um']=canvas; c['data']['center_um']=min(float(c['data'].get('center_um',24)),width); abl.append((f'A_raw_ctx{width}',c))
for target in ['raw','clean']:
    c=copy.deepcopy(base); c['model']['method']='context'; c['data']['target']=target; c['data']['context_um']=64; c['data']['canvas_um']=canvas; abl.append((f'B_{target}_ctx64',c))
for lo,ll in [(1,0.25),(1,0),(0,0.25)]:
    c=copy.deepcopy(base); c['model']['method']='endovso'; c['data']['target']='both'; c['data']['context_um']=64; c['data']['canvas_um']=canvas; c.setdefault('loss',{}); c['loss']['lambda_obs']=lo; c['loss']['lambda_leak']=ll; abl.append((f'C_anchored_obs{lo}_leak{ll}',c))
for shared in [True,False]:
    c=copy.deepcopy(base); c['model']['method']='endovso'; c['data']['target']='both'; c['model']['shared_encoder']=shared; c['data']['context_um']=64; c['data']['canvas_um']=canvas; abl.append((f'D_anchored_shared{int(shared)}',c))
for shuffled in [False,True]:
    c=copy.deepcopy(base); c['model']['method']='endovso'; c['data']['target']='both'; c['data']['ring_shuffle']=shuffled; c['data']['context_um']=64; c['data']['canvas_um']=canvas; abl.append((f'E_ring_shuffle{int(shuffled)}',c))
for name,cfg in abl:
    rd=out/name; rd.mkdir(exist_ok=True); cp=rd/'config.yaml'; cp.write_text(yaml.safe_dump(cfg,sort_keys=False))
    if a.execute: subprocess.run([sys.executable,'scripts/train.py','--config',str(cp),'--out',str(rd/'train')],check=True)
print(f'wrote {len(abl)} ablations to {out}; fixed canvas_um={canvas}')
