#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad
from leakage_vso.io import read_mtx_bundle, intersect_genes, write_h5ad_safe
from leakage_vso.wsi import TiffWSI

p=argparse.ArgumentParser()
p.add_argument("--split-dir",required=True,help="Output of R/run_split_xenium.R")
p.add_argument("--he",required=True)
p.add_argument("--out",required=True)
p.add_argument("--sample-id",default="TENX_BREAST_IDC")
p.add_argument("--donor-id",default="TENX_BREAST_IDC_D1")
p.add_argument("--pixel-size-um",type=float,default=None,help="Override OME metadata only when verified")
p.add_argument("--origin-x-um",type=float,default=0.0)
p.add_argument("--origin-y-um",type=float,default=0.0)
p.add_argument("--x-offset-px",type=float,default=0.0)
p.add_argument("--y-offset-px",type=float,default=0.0)
p.add_argument("--demo-split",action="store_true",help="Spatial hash split for one-donor debugging only; NOT a CVPR result")
a=p.parse_args()
sp=Path(a.split_dir); out=Path(a.out)/"samples"/a.sample_id; out.mkdir(parents=True,exist_ok=True)
raw=read_mtx_bundle(sp,"raw_counts"); clean=read_mtx_bundle(sp,"clean_counts")
raw,clean=intersect_genes(raw,clean)
common_cells=raw.obs_names.intersection(clean.obs_names)
raw=raw[common_cells].copy(); clean=clean[common_cells].copy()
coords=pd.read_csv(sp/"coords.csv",dtype={"cell_id":str}).set_index("cell_id").loc[common_cells]
meta=pd.read_csv(sp/"cell_meta.csv",dtype={"cell_id":str}).set_index("cell_id").reindex(common_cells)
w=TiffWSI(a.he)
px=a.pixel_size_um or w.pixel_size_um_x
if px is None:
    raise RuntimeError("H&E OME-TIFF has no usable physical pixel size. Supply --pixel-size-um after verifying registration.")
if w.pixel_size_um_y and abs(w.pixel_size_um_y-px)/px>0.02 and a.pixel_size_um is None:
    raise RuntimeError("Anisotropic OME pixel size; supply verified transform explicitly")
x_um=coords.x_centroid.astype(float).to_numpy(); y_um=coords.y_centroid.astype(float).to_numpy()
x_px=(x_um-a.origin_x_um)/px+a.x_offset_px; y_px=(y_um-a.origin_y_um)/px+a.y_offset_px
cells=pd.DataFrame({"cell_id":common_cells.astype(str),"sample_id":a.sample_id,"donor_id":a.donor_id,"x_um":x_um,"y_um":y_um,"x_px":x_px,"y_px":y_px})
for c in ["first_type","second_type","spot_class","purification_status"]:
    if c in meta.columns: cells[c]=meta[c].astype(str).to_numpy()
if a.demo_split:
    # Debug only: deterministic coarse spatial blocks; README forbids reporting this as patient generalization.
    block=((np.floor(x_um/500).astype(int)*73856093)^(np.floor(y_um/500).astype(int)*19349663))%10
    cells["split"]=np.where(block<7,"train",np.where(block<9,"val","test"))
write_h5ad_safe(raw,out/"raw.h5ad"); write_h5ad_safe(clean,out/"clean.h5ad")
cells.to_parquet(out/"cells.parquet",index=False)
(out/"image_path.txt").write_text(str(Path(a.he).resolve())+"\n")
(out/"pixel_size_um.txt").write_text(str(px)+"\n")
(out/"provenance.json").write_text(json.dumps({"sample_id":a.sample_id,"he":str(Path(a.he).resolve()),"pixel_size_um":px,"note":"Prepared from exact raw+SPLIT corrected Xenium targets"},indent=2))
print(f"prepared {a.sample_id}: {len(cells)} cells, {raw.n_vars} genes, px={px} um/px")
