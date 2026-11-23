#!/usr/bin/env python3
"""Read-only, independent CVPR_Bio2 final rank-1 evidence verification.

Verifies all 8 source/decoy GPU files, frozen inputs, replay provenance,
recomputes nested patient means from raw CSV rows, and compares the already
existing patient_panel_effects.csv. Never trains, rewrites, or creates results.

Run from the project root with project modules on PYTHONPATH.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

PATS = {
    'P2':['TENX195','TENX196'],
    'P3':['TENX197','TENX198','TENX199'],
    'P4':['TENX200','TENX201','TENX202'],
}
PANEL_C = ['MMP2','SFRP4','CCDC80','CXCL12','FBLN1']
PANEL_I = ['TRAC','CD3E','IL7R','CD8A','CCL5']
GENES = PANEL_C+PANEL_I
KEY = ['gene','recipient_cell_id','source_cell_id','intervention']
ORIG_KEY = KEY+['control_cell_id']
NEW_KEY = KEY+['old_control_cell_id']

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as fh:
        for chunk in iter(lambda:fh.read(8<<20),b''): h.update(chunk)
    return h.hexdigest()

def require(condition,message):
    if not condition: raise RuntimeError('STOP: '+str(message))

def patient_means(data):
    a=data.groupby(['comparison','patient','sample','panel','intervention','gene'],as_index=False).agg(gene_mean=('delta_source_minus_control','mean'))
    b=a.groupby(['comparison','patient','sample','panel','intervention'],as_index=False).agg(section_mean=('gene_mean','mean'),n_genes=('gene','nunique'))
    require(b.n_genes.eq(5).all(),'panel with missing gene')
    c=b.groupby(['comparison','patient','panel','intervention'],as_index=False).agg(effect=('section_mean','mean'),n_sections=('sample','nunique'))
    return c

def run(root):
    base=root/'runs/independent_histo'
    original_root=base/'donor_source_control_rotation_original_v1'
    match_root=base/'same_morph_decoys_v2'
    new_root=base/'same_morph_image_results_v2'/'rank1'
    report=base/'patient_evidence_sameclass_rank1_v2'/'patient_panel_effects.csv'
    source_script=root/'scripts/source_cell_intervention.py'
    require(source_script.is_file(),f'missing intervention code {source_script}')
    actual_source_sha=digest(source_script)
    rows=[]; summary=[]; checkpoint_shas={}; count=0
    for patient,sections in PATS.items():
        for sid in sections:
            o=original_root/patient/f'{sid}_ctx128_intervention.csv'
            n=new_root/f'{sid}_same_morph_intervention.csv'
            manifest=n.with_suffix('.json')
            matches=match_root/sid/'decoy_rank1_pairs.csv'
            coverage=match_root/sid/'coverage.json'
            for f in [o,n,manifest,matches,coverage]:require(f.is_file(),f'missing {f}')
            m=json.loads(manifest.read_text());cov=json.loads(coverage.read_text())
            require((m.get('sample'),m.get('patient'),m.get('rank'))==(sid,patient,1),f'identity {sid}')
            for key,p in [('new_predictions_sha256',n),('new_pairs_sha256',matches),('completed_original_sha256',o)]:
                require(m.get(key)==digest(p),f'{sid} SHA256 {key}')
            require(cov['original_completed_intervention_csv_sha256']==digest(o),f'{sid} coverage original sha')
            require(m['source_intervention_sha256']==actual_source_sha,f'{sid} intervention code changed')
            ck=Path(m['checkpoint_path'])
            if not ck.is_absolute():ck=root/ck
            require(ck.is_file() and digest(ck)==m['checkpoint_sha256'],f'{sid} checkpoint changed')
            checkpoint_shas.setdefault(patient,set()).add(m['checkpoint_sha256'])
            require(m['max_source_delta_repetition_error']<=1e-4,f'{sid} source replay')
            require(m['max_prediction_repetition_error']<=1e-4,f'{sid} original pred replay')
            old=pd.read_csv(o,dtype={c:str for c in ORIG_KEY})
            new=pd.read_csv(n,dtype={c:str for c in NEW_KEY+['control_cell_id']})
            selected=pd.read_csv(matches,dtype={c:str for c in KEY+['old_control_cell_id','control_cell_id']})
            require(len(selected)==m['n_pairs']==cov['accepted_per_rank']['1'],f'{sid} pair count')
            require(len(new)==m['n_rows']==2*len(selected),f'{sid} new mode rows')
            require(not old.duplicated(ORIG_KEY).any() and not new.duplicated(NEW_KEY).any(),f'{sid} duplicate key')
            require(set(new.intervention)=={'blur','inpaint'},f'{sid} intervention coverage')
            require((new.groupby(['gene','recipient_cell_id','source_cell_id','old_control_cell_id']).intervention.nunique()==2).all(),f'{sid} unpaired blur/inpaint')
            if 'pred_original' in old and 'pred_original' in new:
                check=new.merge(old[ORIG_KEY+['delta_source','pred_original','delta_source_minus_control']],
                    left_on=NEW_KEY,right_on=ORIG_KEY,validate='one_to_one',how='left',suffixes=('_new','_old'))
            else:
                raise RuntimeError(f'STOP: {sid} prediction replay columns missing')
            require(len(check)==len(new) and check['delta_source_old'].notna().all(),f'{sid} frozen identity join')
            err_src=float(np.max(np.abs(check.delta_source_new-check.delta_source_old)))
            err_pred=float(np.max(np.abs(check.pred_original_new-check.pred_original_old)))
            require(err_src<=1e-4 and err_pred<=1e-4,f'{sid} source replay recomputation')
            for frame,col in [(old,'delta_source_minus_control'),(new,'delta_source_minus_control')]:
                require(np.isfinite(frame[col]).all(),f'{sid} nonfinite effect')
                require(np.allclose(frame.delta_source-frame.delta_control,frame[col],atol=1e-5,rtol=1e-5),f'{sid} delta arithmetic')
            old['sample']=sid;old['patient']=patient
            new['sample']=sid;new['patient']=patient
            # Frozen original rows on exactly the same rank1 identity set.
            old_subset=old.merge(new[NEW_KEY],left_on=ORIG_KEY,right_on=NEW_KEY,validate='one_to_one',how='inner')
            require(len(old_subset)==len(new),f'{sid} old subset not exact same identities')
            for label,frame in [('ORIGINAL_FULL',old),('ORIGINAL_SUBSET_rank1',old_subset),('SAME_CLASS_rank1',new)]:
                d=frame.copy();d['comparison']=label;rows.append(d)
            paired=check.copy()
            paired['delta_source_minus_control']=check['delta_source_minus_control_new']-check['delta_source_minus_control_old']
            paired['comparison']='SAME_MINUS_OLD_rank1';paired['sample']=sid;paired['patient']=patient
            rows.append(paired)
            count+=len(selected)
            summary.append({'section':sid,'patient':patient,'n_original_pairs':len(old)//2,'matched_pairs':len(selected),
                            'pct_matched':100*len(selected)/(len(old)/2),'source_replay_abs_error':err_src,
                            'original_prediction_replay_abs_error':err_pred,'checkpoint_sha256':m['checkpoint_sha256']})
            print('SECTION_REPLAY_AND_HASH_PASS',sid,'matched',len(selected),'replay_err',err_src,err_pred,flush=True)
    for pat,h in checkpoint_shas.items():require(len(h)==1,f'{pat} multiple donor checkpoint hashes')
    require(report.is_file(),f'patient summary missing {report}')
    allrows=pd.concat(rows,ignore_index=True)
    allrows['panel']=allrows.gene.map({**{g:'Connective_to_Inflammatory' for g in PANEL_C},
                                     **{g:'Inflammatory_to_Connective' for g in PANEL_I}})
    require(allrows.panel.notna().all(),'unrecognized gene marker')
    reconstructed=patient_means(allrows)
    existing=pd.read_csv(report)
    keys=['comparison','patient','panel','intervention']
    aligned=reconstructed.merge(existing,on=keys,how='outer',validate='one_to_one',indicator=True,suffixes=('_recomputed','_reported'))
    require((aligned._merge=='both').all() and len(aligned)==48,'missing/extra patient panel rows')
    require((aligned.n_sections_recomputed==aligned.n_sections_reported).all(),'n_sections differs')
    err=float(np.max(np.abs(aligned.effect_recomputed-aligned.effect_reported)))
    require(err<1e-9,f'patient effects changed; max error {err}')
    wide=reconstructed.pivot(index=['patient','panel','intervention'],columns='comparison',values='effect')
    arithmetic=(wide['SAME_CLASS_rank1']-wide['ORIGINAL_SUBSET_rank1']-wide['SAME_MINUS_OLD_rank1']).abs().max()
    require(arithmetic<1e-9,f'patient contrast arithmetic {arithmetic}')
    print('\nRECONSTRUCTED_PATIENT_EFFECT_MAX_ERROR',err,'PAIRED_CONTRAST_MAX_ERROR',arithmetic)
    print('RANK1_MATCHED_TOTAL',count,'ORIGINAL_TOTAL',sum(x['n_original_pairs'] for x in summary),'COVERAGE',count/sum(x['n_original_pairs'] for x in summary))
    means=reconstructed.groupby('comparison').effect.mean()
    print('EQUAL_PATIENT_PANEL_MODE_MEANS',means.to_dict())
    print('ATTENUATION_1_MINUS_SAME_OVER_ORIGINAL_SUBSET',1-means['SAME_CLASS_rank1']/means['ORIGINAL_SUBSET_rank1'])
    print('PATIENT_MEANS:\n',reconstructed.pivot_table(index='patient',columns='comparison',values='effect',aggfunc='mean').to_string())
    print('SECTION_COVERAGE:\n',pd.DataFrame(summary)[['section','n_original_pairs','matched_pairs','pct_matched']].to_string(index=False))
    a=base/'ghist_resnet18_exact_p1_v1'/'common_finite_genes.txt'
    b=base/'ghist_morph_exact_v2_test'/'primary_common_genes.txt'
    if a.is_file() and b.is_file():
        aa=a.read_text().splitlines();bb=b.read_text().splitlines()
        require(len(aa)==len(set(aa))==225 and len(bb)==len(set(bb))==225,'invalid finite gene count/duplicates')
        require(set(aa)==set(bb),'five-model common gene SET mismatch')
        print('FIVE_MODEL_GENE_SET_EQUAL_PASS',len(aa),'order_identical',aa==bb,
              'SHA_RESNET',digest(a),'SHA_MORPH',digest(b))
    else:
        print('FIVE_MODEL_GENE_SET_NOT_VERIFIED_MISSING_LIST_FILES',a.is_file(),b.is_file())
    print('ALL_EIGHT_RANK1_AND_PAIRED_PATIENT_EVIDENCE_VERIFIED_READ_ONLY')

def self_test():
    df=pd.DataFrame({'comparison':['X']*4,'patient':['P2']*4,'sample':['S']*4,
      'panel':['Connective_to_Inflammatory']*4,'intervention':['blur']*4,
      'gene':['MMP2','SFRP4','CCDC80','CXCL12'],
      'delta_source_minus_control':[1.,2.,3.,4.]})
    try:patient_means(df)
    except RuntimeError as e:require('missing gene' in str(e),'expected missing gene caught')
    else:raise RuntimeError('self-test failed: missing-gene accepted')
    print('SYNTHETIC_MISSING_GENE_FAIL_CLOSED_PASS')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--self-test',action='store_true');args=p.parse_args()
    if args.self_test:self_test()
    else:run(args.root.resolve())
