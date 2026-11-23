#!/usr/bin/env python3
"""GHIST patient-safe input adapter: prepare one slide and smoke one real tile.

Pinned upstream GHIST Framework is imported ONLY in smoke mode. No training.
No modifications to upstream repository. Original multihead GHIST train.py is
NOT compatible with this donor-aware dataset without a separate training driver.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DONOR_SECTIONS = {
    'P1': ('TENX191','TENX192','TENX193'),
    'P2': ('TENX195','TENX196'),
    'P3': ('TENX197','TENX198','TENX199'),
    'P4': ('TENX200','TENX201','TENX202'),
}
ROLE = {'P1': 'test', 'P2': 'train', 'P3': 'train', 'P4': 'val'}
CLASS = ('HISTO_Connective', 'HISTO_Inflammatory',
         'HISTO_Neoplastic', 'HISTO_Non-Neoplastic_Epithelial')
COMMIT = '917456be305fc82e92293ea272812e79675e821c'

def sha(path: Path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def canon(sample):
    matches = [d for d, samples in DONOR_SECTIONS.items() if sample in samples]
    if len(matches) != 1:
        raise ValueError(f'STOP: sample outside the locked 4-patient cohort: {sample}')
    return matches[0]

def validate_repo(repo):
    import subprocess
    assert (repo/'model/model.py').is_file(), 'GHIST Framework source missing'
    actual=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'], text=True).strip()
    assert actual == COMMIT, f'STOP: GHIST commit changed {actual}'
    return actual

def aligned_ann(root, sample):
    """Load only exact H&E-to-Xenium nuclei, never transcript-derived cell classes."""
    from shapely import wkb
    sdir = root/'data/prepared_primary2025/samples'/sample
    annp = root/'runs/independent_histo'/f'{sample}_cell_annotations.parquet'
    nucp = root/'runs/independent_histo'/f'{sample}_nuclei.parquet'
    cp = sdir/'cells.parquet'
    for p in (annp,nucp,cp,sdir/'raw.h5ad',sdir/'pixel_size_um.txt',sdir/'image_path.txt'):
        assert p.is_file(), f'STOP: missing input {p}'
    ann = pd.read_parquet(annp, columns=['cell_id','cell_type','annotation_status','hovernet_nucleus_id'])
    ann['cell_id'] = ann.cell_id.astype(str)
    assert ann.cell_id.is_unique, f'{sample}: duplicate annotation cell IDs'
    ann = ann.loc[ann.annotation_status.eq('unique_nucleus_inside_cell') & ann.cell_type.isin(CLASS),
                  ['cell_id','cell_type','hovernet_nucleus_id']].copy()
    assert ann.hovernet_nucleus_id.notna().all(), f'{sample}: missing nucleus mapping'
    ann['hovernet_nucleus_id'] = ann.hovernet_nucleus_id.astype(str)
    assert ann.hovernet_nucleus_id.is_unique, f'{sample}: nucleus reused across cells'
    cells = pd.read_parquet(cp,columns=['cell_id','x_px','y_px'])
    cells.cell_id = cells.cell_id.astype(str)
    assert cells.cell_id.is_unique, f'{sample}: duplicate prepared cell IDs'
    ann = ann.merge(cells,on='cell_id',how='left',validate='one_to_one')
    assert ann[['x_px','y_px']].notna().all().all(), f'{sample}: incomplete cell positions'
    nuc = pd.read_parquet(nucp,columns=['nucleus_id','x_px','y_px','geometry_wkb'])
    nuc.nucleus_id = nuc.nucleus_id.astype(str)
    assert nuc.nucleus_id.is_unique, f'{sample}: duplicate HoVer nuclei IDs'
    # Prepared cell xy and mapped nucleus xy have the same coordinate frame, but
    # they are NOT expected to be numerically equal: centroids refer to different objects.
    nuc = nuc.rename(columns={'x_px':'nucleus_x_px','y_px':'nucleus_y_px'})
    ann = ann.merge(nuc,left_on='hovernet_nucleus_id',right_on='nucleus_id',
                    how='left',validate='one_to_one')
    assert ann.geometry_wkb.notna().all(), f'{sample}: missing nucleus geometry'
    mpp=float((sdir/'pixel_size_um.txt').read_text().strip())
    assert 0.15 < mpp < 0.6, f'{sample}: suspect mpp {mpp}'
    image=Path((sdir/'image_path.txt').read_text().strip())
    assert image.is_file(), f'{sample}: WSI missing: {image}'
    native=round(128.0/mpp)
    assert native > 256 and native < 1000, f'{sample}: invalid native tile geometry'
    # Every nucleus is assigned to one nonoverlapping 128 um physical tile.
    tx=np.floor(ann.nucleus_x_px.to_numpy(dtype=float)/native).astype(np.int64)
    ty=np.floor(ann.nucleus_y_px.to_numpy(dtype=float)/native).astype(np.int64)
    ann['tx']=tx; ann['ty']=ty
    good=[]
    # Include only complete polygons, preserving exact nucleus-to-cell identity.
    for row in ann.itertuples(index=False):
        poly=wkb.loads(row.geometry_wkb)
        x0=row.tx*native; y0=row.ty*native
        b=poly.bounds
        good.append(poly.geom_type=='Polygon' and poly.is_valid and poly.area>0 and
                    b[0]>=x0 and b[1]>=y0 and b[2]<x0+native and b[3]<y0+native)
    ann['fully_contained']=good
    n_pre=len(ann)
    ann=ann.loc[ann.fully_contained].drop(columns=['fully_contained']).copy()
    assert len(ann)>0, f'{sample}: no fully-contained nucleus polygons'
    ann['cell_type_id']=ann.cell_type.map({v:i for i,v in enumerate(CLASS)}).astype(np.int8)
    ann=ann.sort_values(['ty','tx','nucleus_id','cell_id']).reset_index(drop=True)
    return ann,sdir,annp,nucp,cp,image,mpp,native,n_pre

def prepare(root, repo, sample, out):
    import anndata as ad
    import scipy.sparse as sp
    donor=canon(sample)
    actual=validate_repo(repo)
    assert not out.exists(), f'STOP: output already exists {out}'
    ann,sdir,annp,nucp,cp,image,mpp,native,n_pre=aligned_ann(root,sample)
    genes_path=root/'runs/frozen_primary2025/gene_axis_280.txt'
    assert genes_path.is_file(), f'STOP: missing frozen 280 genes {genes_path}'
    genes=[x.strip() for x in genes_path.read_text().splitlines() if x.strip()]
    assert len(genes)==280 and len(set(genes))==280, 'STOP: invalid 280-gene axis'
    h=ad.read_h5ad(sdir/'raw.h5ad')
    obs=pd.Index(h.obs_names.astype(str)); var=pd.Index(h.var_names.astype(str))
    assert obs.is_unique and var.is_unique, f'{sample}: duplicate H5AD labels'
    assert set(genes).issubset(var), f'{sample}: gene axis missing from H5AD'
    ridx=obs.get_indexer(ann.cell_id)
    assert (ridx>=0).all(), f'{sample}: H5AD missing eligible cell IDs'
    gidx=var.get_indexer(genes)
    expr=h.X[ridx,:][:,gidx]
    if sp.issparse(expr): expr=expr.toarray()
    expr=np.asarray(expr,dtype=np.float32)
    assert expr.shape==(len(ann),280), f'{sample}: bad expression geometry'
    assert np.isfinite(expr).all() and (expr>=0).all(), f'{sample}: invalid raw expression'
    assert np.allclose(expr, np.round(expr),atol=1e-4), f'{sample}: raw targets are not counts'
    assert len(ann)>0 and ann.cell_id.is_unique
    # Defensive no-clobber: stage and atomically rename, never mutate frozen inputs.
    staging=out.with_name(out.name+'.staging-'+str(os.getpid()))
    assert not staging.exists() and not out.exists()
    staging.mkdir(parents=True,exist_ok=False)
    try:
        ann[['cell_id','cell_type','cell_type_id','nucleus_id','geometry_wkb','tx','ty']].to_parquet(
            staging/'cells.parquet',index=False)
        np.save(staging/'raw_expression.npy',expr,allow_pickle=False)
        (staging/'genes.txt').write_text('\n'.join(genes)+'\n')
        manifest={'adapter_version':'1','repo_commit':actual,'sample':sample,'donor':donor,
                  'role':ROLE[donor],'context_um':128,'model_image_px':256,
                  'native_tile_px':native,'mpp_um_per_px':mpp,'image_path':str(image),
                  'eligible_cells_before_complete_polygon_gate':n_pre,
                  'eligible_cells':len(ann),'eligible_tiles':int(ann.groupby(['ty','tx']).ngroups),
                  'exact_mapped_biological_classes':list(CLASS),'genes_sha256':sha(genes_path),
                  'expression_target':'raw_counts',
                  'input_sha256':{str(p.relative_to(root)):sha(p) for p in (annp,nucp,cp,sdir/'raw.h5ad',sdir/'pixel_size_um.txt',sdir/'image_path.txt')},
                  'analysis_label':'GHIST_PATIENT_SAFE_ADAPTER_ONLY_NOT_OFFICIAL_TRAINING'}
        (staging/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        os.rename(staging,out)
    except Exception:
        # Preserve intermediate state if failure needs investigation, never overwrite.
        raise
    print(json.dumps({k:manifest[k] for k in ('sample','donor','role','eligible_cells','eligible_tiles','native_tile_px')},indent=2))
    print('GHIST_ADAPTER_PREPARED; NO TRAINING; NO GPU')

class TileDataset:
    """No test data enters train-mode datasets or normalization.

    On-demand patch rasterization from frozen nucleus WKB at 128 um physical FOV.
    All patch-local numeric labels 1..N map to the same ordered RNA rows.
    """
    def __init__(self, out):
        from shapely import wkb
        self.out=out
        self.manifest=json.loads((out/'manifest.json').read_text())
        self.cells=pd.read_parquet(out/'cells.parquet')
        self.expr=np.load(out/'raw_expression.npy',mmap_mode='r',allow_pickle=False)
        assert len(self.cells)==len(self.expr) and self.expr.shape[1]==280
        self.tiles=[(k,np.asarray(idx,dtype=np.int64)) for k,idx in self.cells.groupby(['ty','tx'],sort=True).indices.items()]
        self.native=int(self.manifest['native_tile_px'])
        self.image_path=Path(self.manifest['image_path'])
        assert self.image_path.is_file()
        self._wsi=None
    def __len__(self): return len(self.tiles)
    def __getitem__(self,idx):
        from PIL import Image,ImageDraw
        from shapely import wkb
        from leakage_vso.wsi import TiffWSI
        (ty,tx),sel=self.tiles[idx]
        if self._wsi is None: self._wsi=TiffWSI(self.image_path)
        n=self.native; x0=tx*n; y0=ty*n
        crop=self._wsi.crop_px(x0+n/2,y0+n/2,n).convert('RGB')
        hist=np.asarray(crop.resize((256,256),Image.Resampling.BILINEAR),dtype=np.float32)
        mask=Image.new('I',(256,256),0)
        draw=ImageDraw.Draw(mask)
        type_mask=Image.new('I',(256,256),0)
        types_draw=ImageDraw.Draw(type_mask)
        rows=self.cells.iloc[sel]
        for i, r in enumerate(rows.itertuples(index=False),start=1):
            poly=wkb.loads(r.geometry_wkb)
            coords=[((x-x0)*256/n,(y-y0)*256/n) for x,y in poly.exterior.coords]
            draw.polygon(coords,fill=int(i))
            types_draw.polygon(coords,fill=int(r.cell_type_id)+1)
            for ring in poly.interiors:
                holes=[((x-x0)*256/n,(y-y0)*256/n) for x,y in ring.coords]
                draw.polygon(holes,fill=0)
                types_draw.polygon(holes,fill=0)
        mask=np.asarray(mask,dtype=np.int64)
        type_mask=np.asarray(type_mask,dtype=np.int64)
        visible=np.unique(mask);visible=visible[visible!=0]
        # If downsampling drops nuclei, do not silently misalign labels.
        assert np.array_equal(visible,np.arange(1,len(sel)+1)), (
            f'STOP: rasterization dropped nucleus in sample {self.manifest["sample"]}, tile {(ty,tx)}; '
            'adjust patch geometry before continuing')
        classes=rows.cell_type_id.to_numpy(dtype=np.int64)
        for i,t in enumerate(classes,start=1):
            assert np.all(type_mask[mask==i]==t+1)
        return {'he':np.transpose(hist,(2,0,1)), 'mask':mask, 'type_mask':type_mask,
                'expr':np.asarray(self.expr[sel],dtype=np.float32), 'class':classes,
                'cell_id':rows.cell_id.astype(str).tolist(),'tile':(int(ty),int(tx))}

def smoke(root,repo,out,device):
    import torch
    import torch.nn.functional as F
    validate_repo(repo)
    ds=TileDataset(out)
    # Prefer a modest-size tile that minimizes unnecessary GPU/CPU work.
    cand=[i for i,(k,indices) in enumerate(ds.tiles) if 2 <= len(indices) <= 24]
    assert cand, 'STOP: no usable two-to-24-cell tile'
    sel=min(cand,key=lambda i:len(ds.tiles[i][1]))
    item=ds[sel]
    assert item['expr'].shape[1]==280 and np.isfinite(item['he']).all()
    # Actual upstream model import. Do not install dependencies automatically.
    sys.path.insert(0,str(repo))
    from model.model import Framework
    if device=='cuda':
        assert torch.cuda.is_available(), 'STOP: no CUDA device on lmin'
        dev=torch.device('cuda:0')
    else:dev=torch.device('cpu')
    model=Framework(n_classes=len(CLASS),n_genes=280,emb_dim=64,device=dev,
                    n_ref=None,use_avgexp=False,use_celltype=True,use_neighb=False).to(dev)
    model.eval()
    he=torch.from_numpy(item['he'][None]).float().to(dev)
    # Only for smoke: image scale/normalization is not locked for model training.
    he=(he-128.0)/64.0
    mask=torch.from_numpy(item['mask'][None]).long().to(dev)
    ncell=torch.tensor([[len(item['cell_id'])]],dtype=torch.long,device=dev)
    ct=torch.from_numpy(item['class'][None]).long().to(dev)
    expr=torch.from_numpy(np.log1p(item['expr'][None])).float().to(dev)
    with torch.inference_mode():
        outs=model(he,mask,ncell,None,ct,expr)
    outmap=outs[1]; out_expr=outs[3]; refs=outs[10]
    assert outmap.shape==(1,len(CLASS)+1,256,256), outmap.shape
    assert out_expr.shape==expr.shape[1:],(out_expr.shape,expr.shape)
    assert torch.isfinite(out_expr).all() and torch.isfinite(outmap).all()
    assert torch.equal(outs[2],ct[0]) and torch.equal(refs,expr[0]),'target alignment failed'
    print('GHIST_UPSTREAM_FRAMEWORK_FORWARD_PASS')
    print('SAMPLE',ds.manifest['sample'],'ROLE',ds.manifest['role'],
          'TILE',item['tile'],'CELLS',len(item['cell_id']),
          'EXPR_OUTPUT',tuple(out_expr.shape),'PIXEL_MAP',tuple(outmap.shape))
    print('STOP: This checks input contract and forward pass ONLY. No patient-safe trainer, loss policy, or trained result yet.')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--repo',type=Path,default=Path('external_methods/GHIST'))
    p.add_argument('--out-root',type=Path,default=Path('runs/ghist_patient_adapter_v1'))
    p.add_argument('--sample',required=True,choices=sum((list(v) for v in DONOR_SECTIONS.values()),[]))
    p.add_argument('--mode',required=True,choices=['prepare','smoke'])
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    a=p.parse_args()
    root=a.root.resolve();repo=(root/a.repo).resolve();out=(root/a.out_root/a.sample).resolve()
    assert out != root and root in out.parents, 'STOP: unsafe output root'
    if a.mode=='prepare':
        out.parent.mkdir(parents=True,exist_ok=True)
        prepare(root,repo,a.sample,out)
    else:
        assert out.is_dir(),f'STOP: prepare the slide first: {out}'
        smoke(root,repo,out,a.device)
if __name__=='__main__':main()
