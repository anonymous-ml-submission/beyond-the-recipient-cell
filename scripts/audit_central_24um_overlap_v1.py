#!/usr/bin/env python3
"""Read-only geometric central-24µm overlap audit of frozen intervention edits.

Audits original frozen source/control polygons for P2/P3/P4 and optional frozen
same-H&E-class rank1 decoys. Uses native H&E pixel coordinates and per-slide mpp.
Does not modify any pair file, exclusion rule, prediction, or model output.
This is a geometry check, NOT an exact rasterized intervention-mask check.
"""
from pathlib import Path
from collections import defaultdict
import argparse, json
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, box
from shapely.validation import make_valid, explain_validity
from collections import Counter
REPAIRED = {}

PATIENT={'TENX195':'P2','TENX196':'P2','TENX197':'P3','TENX198':'P3','TENX199':'P3',
         'TENX200':'P4','TENX201':'P4','TENX202':'P4'}
KEY=['gene','recipient_cell_id','source_cell_id','control_cell_id']

def overlap_area_um2(vertices,rec,halfpx,mpp,section='',cell_id=''):
    geom=Polygon(vertices)
    if not geom.is_valid:
        reason=explain_validity(geom)
        fixed=make_valid(geom)
        if fixed.is_empty or not fixed.is_valid or fixed.area<=0:
            raise RuntimeError(
                ('UNRECOVERABLE_POLYGON',section,cell_id,reason))
        REPAIRED[(section,cell_id)]=reason
        geom=fixed
    if geom.is_empty or geom.area<=0:
        raise RuntimeError(
            ('ZERO_AREA_POLYGON',section,cell_id))
    central=box(float(rec[0])-halfpx,float(rec[1])-halfpx,
                float(rec[0])+halfpx,float(rec[1])+halfpx)
    return float(geom.intersection(central).area)*mpp*mpp

def sample_audit(root,sid,threshold=1e-8):
    pat=PATIENT[sid]
    sd=root/'data/prepared_primary2025/samples'/sid
    original=root/'runs/independent_histo/donor_source_control_rotation_original_v1'/pat/f'{sid}_ctx128_intervention.csv'
    assert original.is_file(),original
    old=pd.read_csv(original,usecols=KEY, dtype={c:str for c in KEY}).drop_duplicates().reset_index(drop=True)
    assert not old.duplicated(KEY).any() and len(old)>0
    match=root/'runs/independent_histo/same_morph_decoys_v2'/sid/'decoy_rank1_pairs.csv'
    new=pd.read_csv(match,dtype={c:str for c in KEY+['old_control_cell_id']}) if match.is_file() else pd.DataFrame()
    if len(new):
        assert not new.duplicated(KEY+['old_control_cell_id']).any()
        links=old.merge(new[KEY[:3]+['old_control_cell_id','control_cell_id']],
            left_on=KEY,right_on=KEY[:3]+['old_control_cell_id'],
            how='inner',validate='one_to_one',suffixes=('_old','_decoy'))
        assert len(links)==len(new),('DECOY_NOT_SUBSET_OF_FROZEN',sid,len(links),len(new))
    else: links=pd.DataFrame()
    needed=set(old['recipient_cell_id'])|set(old['source_cell_id'])|set(old['control_cell_id'])
    if len(new):needed|=set(new['control_cell_id'])
    cells=pd.read_parquet(sd/'cells.parquet',columns=['cell_id','x_px','y_px'])
    cells.cell_id=cells.cell_id.astype(str)
    cells=cells.loc[cells.cell_id.isin(needed)]
    assert cells.cell_id.is_unique
    cent={str(c): (float(x),float(y)) for c,x,y in cells.itertuples(index=False,name=None)}
    assert set(old.recipient_cell_id).issubset(cent),('MISSING_RECIPIENT_CENTERS',sid)
    verts=pd.read_parquet(sd/'cell_polygons.parquet',columns=['cell_id','x_px','y_px','vertex_order'])
    verts.cell_id=verts.cell_id.astype(str)
    verts=verts.loc[verts.cell_id.isin(needed)].sort_values(['cell_id','vertex_order'])
    polys={c:g[['x_px','y_px']].to_numpy(dtype=float)
           for c,g in verts.groupby('cell_id',sort=False)}
    assert needed.issubset(polys),('MISSING_EDIT_POLYGONS',sid,len(needed-set(polys)))
    mpp=float((sd/'pixel_size_um.txt').read_text().strip())
    assert .15<mpp<.6
    half=12.0/mpp
    cache={}
    def area(rec,edited):
        k=(rec,edited)
        if k not in cache:
            cache[k]=overlap_area_um2(polys[edited],cent[rec],half,mpp,sid,edited)
        return cache[k]
    records=[]
    def summarize(d, cohort, ctrl_col):
        rows=[]
        for row in d.itertuples(index=False):
            rec=str(row.recipient_cell_id);src=str(row.source_cell_id)
            ctrl=str(getattr(row,ctrl_col))
            av=area(rec,src);bv=area(rec,ctrl)
            rows.append((av,bv))
        z=np.asarray(rows,dtype=float)
        assert z.shape==(len(d),2)
        records.append({'section':sid,'patient':pat,'cohort':cohort,'n_frozen_gene_pairs':len(d),
            'source_touch_pairs':int((z[:,0]>threshold).sum()),
            'control_touch_pairs':int((z[:,1]>threshold).sum()),
            'either_touch_pairs':int(((z[:,0]>threshold)|(z[:,1]>threshold)).sum()),
            'source_touch_fraction':float(np.mean(z[:,0]>threshold)),
            'control_touch_fraction':float(np.mean(z[:,1]>threshold)),
            'either_touch_fraction':float(np.mean((z[:,0]>threshold)|(z[:,1]>threshold))),
            'max_source_overlap_um2':float(z[:,0].max()),
            'max_control_overlap_um2':float(z[:,1].max()),'histo_mpp':mpp})
    summarize(old,'original_full','control_cell_id')
    if len(links):
        # Original subset and same-class control are evaluated on the exact same frozen source/recipient pairs.
        summarize(links,'original_rank1_subset','control_cell_id_old')
        summarize(links,'sameclass_rank1_subset','control_cell_id_decoy')
    return records

def self_test():
    mpp=0.25
    rec=(0,0);half=12/mpp
    far=np.array([[65,65],[75,65],[65,75]])
    inside=np.array([[3,3],[9,3],[3,9]])
    assert overlap_area_um2(far,rec,half,mpp)==0
    assert overlap_area_um2(inside,rec,half,mpp)>0
    bowtie=np.array([[1,1],[9,9],[1,9],[9,1]])
    assert overlap_area_um2(bowtie,rec,half,mpp,'TEST','BOWTIE')>0
    REPAIRED.clear()
    print('CENTRAL_OVERLAP_SYNTHETIC_GEOMETRY_PASS')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--project',type=Path,default=Path('.'))
    p.add_argument('--out',type=Path,default=None,
                   help='optional new CSV path; refuses existing destination')
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:self_test();return
    root=a.project.resolve()
    if a.out is not None: assert not a.out.exists(),('REFUSE_OVERWRITE',a.out)
    allrows=[]
    for sid in PATIENT:
        rows=sample_audit(root,sid)
        allrows.extend(rows)
        print('CENTRAL_EDIT_GEOMETRY',sid,json.dumps(rows),flush=True)
    print('AUDIT_ONLY_REPAIRED_POLYGON_COUNT',len(REPAIRED))
    print('REPAIRED_BY_SECTION',
          dict(Counter(s for s,c in REPAIRED)))
    for (sid,cid),reason in sorted(REPAIRED.items())[:20]:
        print('INVALID_ORIGINAL_GEOMETRY',sid,cid,reason)
    frame=pd.DataFrame(allrows)
    if a.out is not None:
        a.out.parent.mkdir(parents=True,exist_ok=True)
        frame.to_csv(a.out,index=False)
        print('AUDIT_SAVED',a.out)
    print('CENTRAL_24UM_GEOMETRIC_OVERLAP_AUDIT_PASS')
    print('CAUTION: geometric central-square overlap, not pixel-exact edited mask and not a retroactive exclusion rule.')
if __name__=='__main__':main()
