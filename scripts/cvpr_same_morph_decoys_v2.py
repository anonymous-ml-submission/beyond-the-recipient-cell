#!/usr/bin/env python3
"""Outcome-blind H&E-matched source-decoy candidates for completed P2/P3/P4.

Reads ONLY identity columns from the already-completed original intervention CSV,
then H&E cell geometry/class/nucleus geometry. It never loads original predictions,
marker outcomes, gene counts, or image-model outputs to choose decoys.
It produces one frozen pair CSV per rank (1,2,3), directly compatible with the
already verified scripts/source_cell_intervention.py. Do not edit original results.
"""
from __future__ import annotations
import argparse, hashlib, json
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

PANEL_C=('MMP2','SFRP4','CCDC80','CXCL12','FBLN1')
PANEL_I=('TRAC','CD3E','IL7R','CD8A','CCL5')
ALL_GENES=set(PANEL_C+PANEL_I)
PATS={'TENX195':'P2','TENX196':'P2','TENX197':'P3','TENX198':'P3','TENX199':'P3',
      'TENX200':'P4','TENX201':'P4','TENX202':'P4'}
CLASSES={'HISTO_Connective','HISTO_Inflammatory','HISTO_Neoplastic','HISTO_Non-Neoplastic_Epithelial'}
KEY=['gene','recipient_cell_id','source_cell_id','control_cell_id']

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()


def original_pair_identities(path):
    """Select columns at read time. Never inspect old outcomes while matching."""
    columns=set(KEY+['intervention'])
    x=pd.read_csv(path,usecols=lambda c:c in columns,dtype={q:str for q in KEY})
    assert set(x.columns)==columns,('ORIGINAL_CSV_SCHEMA',set(x.columns))
    assert not x[KEY].isna().any().any(),'NULL_FROZEN_PAIR_IDENTITIES'
    assert set(x.gene).issubset(ALL_GENES) and set(x.intervention)=={'blur','inpaint'}
    assert not x.duplicated(KEY+['intervention']).any(),'DUPLICATE_FROZEN_INTERVENTION'
    groups=x.groupby(KEY,sort=False).intervention.agg(lambda s:tuple(sorted(s)))
    assert groups.map(lambda t:t==('blur','inpaint')).all(),'UNPAIRED_ORIGINAL_MODES'
    x=x[KEY].drop_duplicates().sort_values(KEY,kind='mergesort').reset_index(drop=True)
    assert len(x)*2==int(groups.size)*2
    return x


def load_geometry(root,sid,confidence):
    sd=root/'data/prepared_primary2025/samples'/sid
    cp=sd/'cells.parquet'
    ap=root/'runs/independent_histo'/f'{sid}_cell_annotations.parquet'
    pp=sd/'cell_polygons.parquet'
    npth=root/'runs/independent_histo'/f'{sid}_nuclei.parquet'
    for f in (cp,ap,pp,npth): assert f.is_file(),f'MISSING {f}'
    mpp=float((sd/'pixel_size_um.txt').read_text().strip())
    assert 0.15<mpp<0.6
    c=pd.read_parquet(cp,columns=['cell_id','x_px','y_px','polygon_area_um2'])
    a=pd.read_parquet(ap,columns=['cell_id','cell_type','annotation_status'])
    for d in (c,a):
        d.cell_id=d.cell_id.astype(str)
        assert d.cell_id.is_unique,(sid,'DUPLICATE_ANNOTATION_OR_CELL_ID')
    assert len(c)==len(a) and set(c.cell_id)==set(a.cell_id)
    c=c.merge(a,on='cell_id',validate='one_to_one')
    c=c.loc[c.annotation_status.eq('unique_nucleus_inside_cell') & c.cell_type.isin(CLASSES)].copy()
    coords=c[['x_px','y_px','polygon_area_um2']].to_numpy(float)
    c=c.loc[np.isfinite(coords).all(axis=1)&(c.polygon_area_um2.to_numpy()>0)].copy()
    verts=pd.read_parquet(pp,columns=['cell_id','x_px','y_px'])
    verts.cell_id=verts.cell_id.astype(str)
    bounds=(verts.groupby('cell_id',sort=False).agg(xmin=('x_px','min'),xmax=('x_px','max'),
         ymin=('y_px','min'),ymax=('y_px','max'),nverts=('x_px','size')))
    c=c.join(bounds,on='cell_id',how='left')
    c=c.loc[c.nverts.ge(3) & np.isfinite(c[['xmin','xmax','ymin','ymax']]).all(axis=1)]
    assert len(c)>100
    n=pd.read_parquet(npth,columns=['x_px','y_px','histo_type','histo_prob'])
    n=n.loc[n.histo_type.isin(['Connective','Inflammatory','Neoplastic','Non-Neoplastic Epithelial']) &
            pd.to_numeric(n.histo_prob,errors='coerce').ge(confidence)].copy()
    nxy=n[['x_px','y_px']].to_numpy(float)*mpp
    assert len(nxy)>100 and np.isfinite(nxy).all()
    return c.set_index('cell_id'),mpp,cKDTree(nxy),(cp,ap,pp,npth)


def visible(r,recipient,half_px):
    return (r.xmin>=recipient.x_px-half_px and r.xmax<recipient.x_px+half_px and
            r.ymin>=recipient.y_px-half_px and r.ymax<recipient.y_px+half_px)


def run(args):
    root=args.project.resolve();sid=args.sample
    assert sid in PATS
    op=args.original.resolve();out=args.out.resolve()
    assert op.is_file(),f'MISSING_COMPLETED_ORIGINAL_RESULT {op}'
    assert not out.exists(),f'REFUSE_OVERWRITE {out}'
    assert args.n_decoys>=1 and args.n_decoys<=5
    assert all(t>0 for t in (args.max_distance_um,args.distance_caliper_um,args.log_area_caliper,args.log_d5_caliper))
    old=original_pair_identities(op)
    cells,mpp,ntree,paths=load_geometry(root,sid,args.prob_min)
    halfpx=64/mpp
    trees={};group={}
    for cls in sorted(CLASSES):
        z=cells.loc[cells.cell_type.eq(cls)]
        if len(z):
            group[cls]=z
            trees[cls]=cKDTree(z[['x_px','y_px']].to_numpy(float)*mpp)
    fail=Counter();a_by_rank={i:[] for i in range(1,args.n_decoys+1)}
    accepted_diagnostics=[]
    src_d5_cache={}
    decoy_d5_cache={}
    d5_count=6
    def d5(pix):
        xy=(float(pix[0])*mpp,float(pix[1])*mpp)
        ds=ntree.query(xy,k=d5_count)[0]
        value=float(ds[-1])
        assert np.isfinite(value) and value>0
        return value
    for q in old.itertuples(index=False):
        rid=str(q.recipient_cell_id);src=str(q.source_cell_id);oldc=str(q.control_cell_id)
        if rid not in cells.index or src not in cells.index:
            fail['source_or_recipient_not_exact_H&E']+=1;continue
        rec=cells.loc[rid];source=cells.loc[src]
        if len({rid,src,oldc})!=3 or source.cell_type==rec.cell_type:
            fail['invalid_source_recipient_identity_or_class']+=1;continue
        expected_src,expected_rec=(('HISTO_Connective','HISTO_Inflammatory') if q.gene in PANEL_C
                                    else ('HISTO_Inflammatory','HISTO_Connective'))
        assert source.cell_type==expected_src and rec.cell_type==expected_rec, (
          'FROZEN_GENE_SOURCE_RECIPIENT_MORPHOLOGY_CONTRACT',sid,q.gene,src,rid)
        if not visible(source,rec,halfpx):
            fail['source_polygon_outside_context']+=1;continue
        rxy=np.array([rec.x_px,rec.y_px])*mpp
        sxy=np.array([source.x_px,source.y_px])*mpp
        ds=float(np.linalg.norm(sxy-rxy))
        if not 0<ds<=args.max_distance_um:
            fail['source_out_of_distance_gate']+=1;continue
        cls=source.cell_type
        query_r=min(args.max_distance_um,ds+args.distance_caliper_um)
        cand_idx=trees[cls].query_ball_point(rxy,query_r)
        if not cand_idx:
            fail['no_same_morphology_candidates']+=1;continue
        z=group[cls].iloc[cand_idx]
        z=z.loc[~z.index.isin([rid,src,oldc])].copy()
        if z.empty:
            fail['no_distinct_candidates']+=1;continue
        xy=z[['x_px','y_px']].to_numpy(float)*mpp
        dd=np.linalg.norm(xy-rxy,axis=1)
        areaerr=np.abs(np.log(z.polygon_area_um2.to_numpy(float)/float(source.polygon_area_um2)))
        vis=(z.xmin.to_numpy()>=rec.x_px-halfpx)&(z.xmax.to_numpy()<rec.x_px+halfpx)&\
            (z.ymin.to_numpy()>=rec.y_px-halfpx)&(z.ymax.to_numpy()<rec.y_px+halfpx)
        gate=(np.abs(dd-ds)<=args.distance_caliper_um)&(areaerr<=args.log_area_caliper)&vis&(dd>0)
        if not gate.any():
            fail['no_distance_area_visibility_common_support']+=1;continue
        z=z.iloc[np.flatnonzero(gate)];dd=dd[gate];areaerr=areaerr[gate]
        skey=(src,)
        if skey not in src_d5_cache:src_d5_cache[skey]=d5([source.x_px,source.y_px])
        sd5=src_d5_cache[skey]
        cand_d5=[]
        for cid,xyi in zip(z.index.astype(str),z[['x_px','y_px']].to_numpy(float)):
            if cid not in decoy_d5_cache: decoy_d5_cache[cid]=d5(xyi)
            cand_d5.append(decoy_d5_cache[cid])
        cand_d5=np.asarray(cand_d5,dtype=float)
        errd5=np.abs(np.log(cand_d5/sd5))
        ok=errd5<=args.log_d5_caliper
        if not ok.any():
            fail['no_source_density_common_support']+=1;continue
        z=z.iloc[np.flatnonzero(ok)];dd=dd[ok];areaerr=areaerr[ok];errd5=errd5[ok]
        cost=np.abs(dd-ds)/args.distance_caliper_um+areaerr/args.log_area_caliper+errd5/args.log_d5_caliper
        order=np.lexsort((z.index.to_numpy(str),cost))
        accepted_diagnostics.append({'sample':sid,'gene':q.gene,'recipient_cell_id':rid,'source_cell_id':src,
            'old_control_cell_id':oldc,'number_eligible_decoys':len(order)})
        for rank,i in enumerate(order[:args.n_decoys],1):
            cr=z.iloc[int(i)];cid=str(z.index[int(i)])
            assert cid not in (rid,src,oldc) and cr.cell_type==source.cell_type
            pair={'gene':str(q.gene),'recipient_cell_id':rid,'source_cell_id':src,'control_cell_id':cid,
                  'old_control_cell_id':oldc,'source_cell_type':source.cell_type,
                  'recipient_cell_type':rec.cell_type,'control_cell_type':cr.cell_type,
                  'source_distance_um':ds,'control_distance_um':float(dd[int(i)]),
                  'distance_error_um':float(abs(dd[int(i)]-ds)),
                  'log_area_ratio_abs':float(areaerr[int(i)]),'log_source_5nn_ratio_abs':float(errd5[int(i)]),
                  'source_5nn_um':sd5,'matched_decoy_rank':rank,'available_decoys':len(order),
                  'sample':sid,'donor_id':PATS[sid]}
            a_by_rank[rank].append(pair)
    assert sum(len(v) for v in a_by_rank.values())>0,'NO_COMMON_SUPPORT; NO_OUTPUTS_WRITTEN'
    # A full 3-decoy cohort is a separate sensitivity, never silently substituted
    # for the single best-matched decoy primary analysis.
    overview={'sample':sid,'patient':PATS[sid],'original_unique_frozen_pairs':len(old),
      'accepted_per_rank':{str(k):len(v) for k,v in a_by_rank.items()},
      'coverage_per_rank':{str(k):len(v)/len(old) for k,v in a_by_rank.items()},
      'failures':dict(fail),'prespecified_calipers':{
        'max_source_distance_um':args.max_distance_um,'distance_error_um':args.distance_caliper_um,
        'source_control_log_area_ratio':args.log_area_caliper,
        'source_control_log_5nn_distance_ratio':args.log_d5_caliper,
        'H&E_nucleus_confidence_min':args.prob_min,'context_um':128},
      'method':'outcome blind H&E morph/distance/area/source-density matching; donor frozen; no transcript or prediction accessed',
      'original_completed_intervention_csv_sha256':sha(op),'input_hehe_geometry_sha256':{str(f):sha(f) for f in paths}}
    out.mkdir(parents=True,exist_ok=False)
    for rank,rows in a_by_rank.items():
        if rows:
            file=out/f'decoy_rank{rank}_pairs.csv'
            pd.DataFrame(rows).to_csv(file,index=False)
    pd.DataFrame(accepted_diagnostics).to_csv(out/'candidate_support.csv',index=False)
    (out/'coverage.json').write_text(json.dumps(overview,indent=2)+'\n')
    print('DECOY_MATCH_COMPLETE',json.dumps({'sample':sid,'original':len(old),
          'accepted_per_rank':overview['accepted_per_rank'],'coverage_per_rank':overview['coverage_per_rank'],
          'failures':overview['failures']},sort_keys=True),flush=True)
    print('NO_IMAGE_MODEL_RESULTS_OR_GENE_TARGETS_READ_DURING_MATCHING')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--project',type=Path,default=Path('.'))
    p.add_argument('--sample',required=True,choices=sorted(PATS))
    p.add_argument('--original',required=True,type=Path,help='already-completed SOURCE/control intervention CSV for this section')
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--n-decoys',type=int,default=3)
    p.add_argument('--prob-min',type=float,default=0.6)
    p.add_argument('--max-distance-um',type=float,default=40)
    p.add_argument('--distance-caliper-um',type=float,default=5)
    p.add_argument('--log-area-caliper',type=float,default=0.7)
    p.add_argument('--log-d5-caliper',type=float,default=0.7)
    args=p.parse_args()
    run(args)

if __name__=='__main__':main()
