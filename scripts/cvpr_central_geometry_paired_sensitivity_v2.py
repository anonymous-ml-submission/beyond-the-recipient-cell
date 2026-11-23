#!/usr/bin/env python3
"""Post-hoc, read-only input sensitivity: compare same frozen source/old/same-class
interventions on rank-1 pairs whose THREE edited cell polygons avoid the
recipient's central 24-um square. DOES NOT assert pixel-exact mask exclusion.

Original/intervention CSVs are never modified. Requires all eight completed
rank-1 CSVs and the original verified patient-level report.
"""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from shapely.validation import make_valid, explain_validity

PATS = {
    'P2': ('TENX195','TENX196'),
    'P3': ('TENX197','TENX198','TENX199'),
    'P4': ('TENX200','TENX201','TENX202'),
}
PANEL = {
    'MMP2':'Connective_to_Inflammatory','SFRP4':'Connective_to_Inflammatory',
    'CCDC80':'Connective_to_Inflammatory','CXCL12':'Connective_to_Inflammatory',
    'FBLN1':'Connective_to_Inflammatory',
    'TRAC':'Inflammatory_to_Connective','CD3E':'Inflammatory_to_Connective',
    'IL7R':'Inflammatory_to_Connective','CD8A':'Inflammatory_to_Connective',
    'CCL5':'Inflammatory_to_Connective',
}
K = ['gene','recipient_cell_id','source_cell_id','intervention']
OLD_KEY = K+['control_cell_id']
NEW_KEY = K+['old_control_cell_id']


def need(ok,why):
    if not ok:raise RuntimeError('STOP: '+str(why))


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''):
            h.update(b)
    return h.hexdigest()


def polygon_from_vertices(vertices, sid, cellid, repairs):
    arr=np.asarray(vertices,dtype='float64')
    need(arr.ndim==2 and arr.shape[1]==2 and len(arr)>=3 and np.isfinite(arr).all(),
         (sid,cellid,'INVALID_COORDINATES'))
    g=Polygon(arr)
    if not g.is_valid:
        reason=explain_validity(g)
        fixed=make_valid(g)
        parts=[]
        def collect(x):
            if x.geom_type in ('Polygon','MultiPolygon'):
                if x.geom_type=='Polygon':parts.append(x)
                else:
                    for z in x.geoms:collect(z)
            elif hasattr(x,'geoms'):
                for z in x.geoms:collect(z)
        collect(fixed)
        need(bool(parts),(sid,cellid,'UNREPAIRABLE_POLYGON',reason))
        g=unary_union(parts)
        need(g.is_valid and g.area>0,(sid,cellid,'INVALID_AFTER_MAKE_VALID',reason))
        repairs.append({'sample':sid,'cell_id':cellid,'reason':reason,
                        'original_area_px2':float(Polygon(arr).area),
                        'repaired_area_px2':float(g.area)})
    need(g.area>0 and g.is_valid,(sid,cellid,'ZERO_AREA_POLYGON'))
    return g


def load_areas(root,sid,pairs,repairs):
    sd=root/'data/prepared_primary2025/samples'/sid
    needed=set(pairs.recipient_cell_id) | set(pairs.source_cell_id) | set(pairs.control_cell_id_old) | set(pairs.control_cell_id_same)
    cell=pd.read_parquet(sd/'cells.parquet',columns=['cell_id','x_px','y_px'])
    cell.cell_id=cell.cell_id.astype(str)
    cell=cell[cell.cell_id.isin(needed)]
    need(cell.cell_id.is_unique,(sid,'DUPLICATE_CELLS'))
    centers={str(k):(float(x),float(y)) for k,x,y in cell.itertuples(index=False,name=None)}
    need(set(pairs.recipient_cell_id).issubset(centers),(sid,'MISSING_RECIPIENT'))
    vert=pd.read_parquet(sd/'cell_polygons.parquet',columns=['cell_id','x_px','y_px','vertex_order'])
    vert.cell_id=vert.cell_id.astype(str)
    vert=vert[vert.cell_id.isin(needed)].sort_values(['cell_id','vertex_order'])
    polys={str(k):polygon_from_vertices(gr[['x_px','y_px']].to_numpy(),sid,str(k),repairs)
           for k,gr in vert.groupby('cell_id',sort=False)}
    need(needed.issubset(polys),(sid,'MISSING_POLYGONS',len(needed-set(polys))))
    mpp=float((sd/'pixel_size_um.txt').read_text().strip())
    need(.15<mpp<.6,(sid,'MISSING_MPP'))
    half=12.0/mpp
    cache={}
    def area(rec,edited):
        key=(rec,edited)
        if key not in cache:
            cx,cy=centers[rec]
            sq=box(cx-half,cy-half,cx+half,cy+half)
            cache[key]=float(polys[edited].intersection(sq).area)*mpp*mpp
        return cache[key]
    return [
        (area(r.recipient_cell_id,r.source_cell_id),
         area(r.recipient_cell_id,r.control_cell_id_old),
         area(r.recipient_cell_id,r.control_cell_id_same))
        for r in pairs.itertuples(index=False)
    ]


def effects(df,subset):
    result=[]
    for col,method in [
        ('old_effect','ORIGINAL_SUBSET_rank1'),
        ('same_effect','SAME_CLASS_rank1'),
        ('difference','SAME_MINUS_OLD_rank1')
    ]:
        g=df.groupby(['patient','sample','panel','intervention','gene'],as_index=False)[col].mean()
        # 5 genes in each panel per section. No silent missing groups.
        panel_ct=g.groupby(['patient','sample','panel','intervention']).gene.nunique()
        need(len(panel_ct)==8*2*2 and panel_ct.eq(5).all(),
             ('UNBALANCED_GENE_PANELS',subset,col,panel_ct.to_dict()))
        sec=g.groupby(['patient','sample','panel','intervention'],as_index=False)[col].mean()
        pat=sec.groupby(['patient','panel','intervention'],as_index=False)[col].mean()
        need(len(pat)==3*2*2,('MISSING_PATIENT_PANEL_MODE',subset,col))
        pat=pat.rename(columns={col:'effect'})
        pat['subset']=subset;pat['comparison']=method
        result.append(pat)
    return pd.concat(result,ignore_index=True)


def ingest(root,sid,patient,repairs):
    base=root/'runs/independent_histo'
    o=base/'donor_source_control_rotation_original_v1'/patient/f'{sid}_ctx128_intervention.csv'
    n=base/'same_morph_image_results_v2/rank1'/f'{sid}_same_morph_intervention.csv'
    marker=n.with_suffix('.json')
    matched=base/'same_morph_decoys_v2'/sid/'decoy_rank1_pairs.csv'
    for f in (o,n,marker,matched):need(f.is_file(),('MISSING',str(f)))
    ck=json.loads(marker.read_text())
    need((ck.get('sample'),ck.get('patient'),ck.get('rank'))==(sid,patient,1),(sid,'MANIFEST_IDENTITY'))
    for key,p in [('new_predictions_sha256',n),('new_pairs_sha256',matched),('completed_original_sha256',o)]:
        need(ck.get(key)==sha(p),(sid,'SHA256_MISMATCH',key))
    old=pd.read_csv(o,dtype={k:str for k in OLD_KEY})
    new=pd.read_csv(n,dtype={k:str for k in NEW_KEY+['control_cell_id']})
    matches=pd.read_csv(matched,dtype={k:str for k in NEW_KEY+['control_cell_id']})
    need(not old.duplicated(OLD_KEY).any() and not new.duplicated(NEW_KEY).any(),(sid,'DUPLICATE_PAIR'))
    need(len(new)==2*len(matches) and len(new)==ck['n_rows'],(sid,'PAIR_COUNT_MISMATCH'))
    need((new.groupby(['gene','recipient_cell_id','source_cell_id','old_control_cell_id']).intervention.nunique()==2).all(),
         (sid,'MISSING_BLUR_OR_INPAINT'))
    cols=OLD_KEY+['delta_source','delta_source_minus_control','pred_original']
    joined=new.merge(old[cols],left_on=NEW_KEY,right_on=OLD_KEY,how='left',validate='one_to_one',
                     suffixes=('_same','_old'),indicator=True)
    need(len(joined)==len(new) and (joined['_merge']=='both').all(),(sid,'IDENTITY_JOIN_MISMATCH'))
    for c in ('delta_source_minus_control_same','delta_source_minus_control_old'):
        need(np.isfinite(joined[c].to_numpy(float)).all(),(sid,'NONFINITE',c))
    srerr=np.max(np.abs(joined.delta_source_same.to_numpy(float)-joined.delta_source_old.to_numpy(float)))
    prerr=np.max(np.abs(joined.pred_original_same.to_numpy(float)-joined.pred_original_old.to_numpy(float)))
    need(srerr<=1e-4 and prerr<=1e-4,(sid,'REPLAY_MISMATCH',srerr,prerr))
    joined['old_effect']=joined.delta_source_minus_control_old.astype(float)
    joined['same_effect']=joined.delta_source_minus_control_same.astype(float)
    joined['difference']=joined.same_effect-joined.old_effect
    joined['patient']=patient
    joined['sample']=sid
    joined['panel']=joined.gene.map(PANEL)
    need(joined.panel.notna().all(),(sid,'UNEXPECTED_PANEL_GENE'))
    # Old 'control_cell_id' became 'control_cell_id_old'; new one is '_same'.
    for c in ('control_cell_id_old','control_cell_id_same'):
        need(c in joined.columns,(sid,'JOIN_RENAME_ERROR',c,list(joined.columns)))
    ar=np.asarray(load_areas(root,sid,joined,repairs),dtype=float)
    need(ar.shape==(len(joined),3) and np.isfinite(ar).all(),(sid,'BAD_OVERLAP_AREAS'))
    joined['source_overlap_um2']=ar[:,0]
    joined['old_control_overlap_um2']=ar[:,1]
    joined['same_control_overlap_um2']=ar[:,2]
    # Pair-based eligibility (not outcomes); positive intersection area at 1e-8 µm².
    joined['strict_center_clear']=(ar<=1e-8).all(axis=1)
    joined['source_center_clear']=ar[:,0]<=1e-8
    # Every geometric pair must have identical eligibility for blur and inpaint.
    grp=joined.groupby(['gene','recipient_cell_id','source_cell_id','old_control_cell_id'])
    need(grp.strict_center_clear.nunique().eq(1).all(),(sid,'MODES_GEOMETRY_DISAGREE'))
    keys=['patient','sample','panel','gene','recipient_cell_id','source_cell_id',
          'old_control_cell_id','control_cell_id_same','intervention']
    carry=keys+['old_effect','same_effect','difference','source_overlap_um2',
                'old_control_overlap_um2','same_control_overlap_um2','strict_center_clear']
    return joined[carry].copy(),{'section':sid,'patient':patient,
          'matched_pairs':len(joined)//2,'strict_clear_pairs':int(joined.strict_center_clear.sum())//2,
          'strict_clear_pct':100*float(joined.strict_center_clear.mean()),
          'source_touch_pairs':int((~joined.source_center_clear).sum())//2,
          'old_control_touch_pairs':int((joined.old_control_overlap_um2>1e-8).sum())//2,
          'same_control_touch_pairs':int((joined.same_control_overlap_um2>1e-8).sum())//2,
          'replay_source_error':float(srerr),'replay_prediction_error':float(prerr)}


def run(root,out):
    need(not out.exists(),('REFUSE_OVERWRITE',out))
    need((root/'runs/independent_histo/patient_evidence_sameclass_rank1_v2/patient_panel_effects.csv').is_file(),
         'MISSING_PUBLISHED_PATIENT_REPORT')
    repairs=[]; frames=[];coverage=[]; input_shas={}
    for patient,sids in PATS.items():
        for sid in sids:
            sub,c=ingest(root,sid,patient,repairs)
            frames.append(sub);coverage.append(c)
            print('READ_ONLY_GEOMETRY_PAIRED',sid,c,flush=True)
    df=pd.concat(frames,ignore_index=True)
    full=effects(df,'ALL_MATCHED')
    clear=df.loc[df.strict_center_clear].copy()
    need(len(clear)>0,'NO_GEOMETRICALLY_CLEAR_PAIRS')
    filtered=effects(clear,'STRICT_NO_CENTRAL_POLYGON_OVERLAP')
    # Verify the recomputed unfiltered means exactly reproduce the prior published analysis.
    previous=pd.read_csv(root/'runs/independent_histo/patient_evidence_sameclass_rank1_v2/patient_panel_effects.csv')
    baseline=previous[previous.comparison.isin(full.comparison.unique())]
    k=['patient','panel','intervention','comparison']
    audit=full.merge(baseline,on=k,validate='one_to_one',indicator=True,suffixes=('_new','_previous'))
    need(len(audit)==36 and (audit._merge=='both').all(),('PREVIOUS_REPORT_JOIN',len(audit)))
    err=float(np.max(np.abs(audit.effect_new.to_numpy()-audit.effect_previous.to_numpy())))
    need(err<1e-9,('PREVIOUS_REPORT_NOT_REPRODUCED',err))
    combined=pd.concat([full,filtered],ignore_index=True)
    ag=combined.groupby(['subset','comparison'],as_index=False).effect.mean().rename(columns={'effect':'equal_patient_panel_mode_mean'})
    old=ag[(ag.comparison=='ORIGINAL_SUBSET_rank1')].set_index('subset').equal_patient_panel_mode_mean
    new=ag[(ag.comparison=='SAME_CLASS_rank1')].set_index('subset').equal_patient_panel_mode_mean
    attenuation=(1-new/old).rename('attenuation_ratio').reset_index()
    print('PRIOR_PATIENT_REPORT_REPLAY_PASS',err)
    print('GEOMETRY_CLEAR_COVERAGE',int(len(clear)//2),'/',int(len(df)//2),'=',float(len(clear)/len(df)))
    print('EQUAL_PATIENT_SUMMARY\n',ag.to_string(index=False))
    print('ATTENUATION\n',attenuation.to_string(index=False))
    print('PATIENT_PAIRWISE_CONTRAST\n',combined[combined.comparison=='SAME_MINUS_OLD_rank1'].groupby(['subset','patient']).effect.mean().to_string())
    out.mkdir(parents=True,exist_ok=False)
    df.to_csv(out/'paired_geometry_per_intervention.csv',index=False)
    pd.DataFrame(coverage).to_csv(out/'section_coverage.csv',index=False)
    combined.to_csv(out/'patient_panel_effects_stratified.csv',index=False)
    ag.to_csv(out/'equal_patient_summary.csv',index=False)
    attenuation.to_csv(out/'attenuation.csv',index=False)
    pd.DataFrame(repairs,columns=['sample','cell_id','reason','original_area_px2','repaired_area_px2']).to_csv(out/'geometry_repairs.csv',index=False)
    provenance={'status':'POSTHOC_GEOMETRIC_SENSITIVITY_ONLY',
                'no_inference_or_training':True,'central_square_um':24,
                'reference_region':'recipient_centered_square',
                'strict_rule':'source, original control, same-class control ALL area intersection <= 1e-8 um2',
                'pixel_exact_mask_verified':False,
                'repair_count':len(repairs),
                'baseline_patient_report_max_error':err,
                'script_sha256':sha(Path(__file__)),
                'n_matched_pair_rows':len(df)//2,'n_clear_pair_rows':len(clear)//2}
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('POSTHOC_GEOMETRY_CONTROLLED_PATIENT_SENSITIVITY_PASS',out)
    print('CAUTION: post-hoc selection, limited matched support; polygons NOT pixel-exact edited masks')


def self_test():
    repairs=[]
    bad=[[0,0],[5,5],[0,5],[5,0]]
    g=polygon_from_vertices(bad,'S','bad',repairs)
    need(g.area>0 and len(repairs)==1,'SELF_TEST_INVALID_POLYGON_REPAIR')
    good=polygon_from_vertices([[0,0],[1,0],[1,1],[0,1]],'S','good',repairs)
    need(round(good.intersection(box(0,0,.5,.5)).area,6)==.25,'SELF_TEST_INTERSECTION')
    # Two edit methods, 5 genes, one section each across all patients.
    rows=[]
    for pat in PATS:
        for sid in PATS[pat]:
            for gene,panel in PANEL.items():
                for mode in ['blur','inpaint']:
                    rows.append({'patient':pat,'sample':sid,'panel':panel,'gene':gene,
                                 'intervention':mode,'old_effect':.02,'same_effect':.005,'difference':-.015})
    z=effects(pd.DataFrame(rows),'FAKE')
    need(len(z)==36 and np.allclose(z[z.comparison=='SAME_MINUS_OLD_rank1'].effect,-.015),
         'SELF_TEST_WEIGHTING')
    print('GEOMETRY_MAKE_VALID_AND_SECTION_PATIENT_AGGREGATION_SYNTHETIC_PASS')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--project',type=Path,default=Path('.'))
    p.add_argument('--out',type=Path,
                   default=Path('runs/independent_histo/central_geometry_paired_sensitivity_v2'))
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:self_test()
    else:run(a.project.resolve(),a.out if a.out.is_absolute() else (a.project.resolve()/a.out))
