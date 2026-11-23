#!/usr/bin/env python3
"""Registration QA montage with cell-boundary overlays, not centroid dots alone."""
import argparse, math
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from leakage_vso.prepared import PreparedSample
from leakage_vso.wsi import TiffWSI
from leakage_vso.geometry import sort_polygon_vertices
p=argparse.ArgumentParser(); p.add_argument('--root',required=True); p.add_argument('--sample',required=True); p.add_argument('--out',required=True); p.add_argument('--polygons',default=None); p.add_argument('--n',type=int,default=36); p.add_argument('--context-um',type=float,default=80); p.add_argument('--seed',type=int,default=0); a=p.parse_args()
s=PreparedSample.load(a.root,a.sample); w=TiffWSI(s.image_path); pp=Path(a.polygons) if a.polygons else Path(a.root)/'samples'/a.sample/'cell_polygons.parquet'
if not pp.exists(): raise FileNotFoundError(f'Polygon overlay is required for registration QA: {pp}')
poly=pd.read_parquet(pp); poly['cell_id']=poly.cell_id.astype(str); rng=np.random.default_rng(a.seed); idx=rng.choice(len(s.cells),min(a.n,len(s.cells)),replace=False)
nc=int(math.ceil(math.sqrt(len(idx)))); nr=int(math.ceil(len(idx)/nc)); fig,axs=plt.subplots(nr,nc,figsize=(2.6*nc,2.6*nr)); axs=np.asarray(axs).reshape(-1); size=max(16,int(round(a.context_um/s.pixel_size_um)))
for ax,ii in zip(axs,idx):
    r=s.cells.iloc[ii]; cx=float(r.x_px); cy=float(r.y_px); im=w.crop_px(cx,cy,size); ax.imshow(im); x0=cx-size/2; y0=cy-size/2
    q=sort_polygon_vertices(poly[poly.cell_id==str(r.cell_id)])
    if len(q)>=3:
        xy=q[['x_px','y_px']].to_numpy(float); ax.plot(xy[:,0]-x0,xy[:,1]-y0,linewidth=1.2)
    ax.scatter([im.width/2],[im.height/2],s=10); ax.set_title(str(r.cell_id)[:14],fontsize=7); ax.axis('off')
for ax in axs[len(idx):]: ax.axis('off')
fig.tight_layout(); Path(a.out).parent.mkdir(parents=True,exist_ok=True); fig.savefig(a.out,dpi=200); print(a.out)
