#!/usr/bin/env python3
"""Run geometry-matched decoy image perturbation with *existing* frozen P2/P3/P4 CNNs.

No training. Per-section checkpoint selection must come from explicit --checkpoints
JSON, or --discover-checkpoints with exactly one SHA-distinct verified candidate.
Runs original source_cell_intervention.py unchanged, validates source-delta
numerical equivalence to the ORIGINAL completed run on the same frozen pairs,
then atomically promotes outputs. Stops on mismatches or existing results.
"""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd

PATS={'TENX195':'P2','TENX196':'P2','TENX197':'P3','TENX198':'P3','TENX199':'P3',
      'TENX200':'P4','TENX201':'P4','TENX202':'P4'}
KEY=['gene','recipient_cell_id','source_cell_id','intervention']

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()

def ensure_checkpoint(path,patient,root,genes):
    import torch
    from collections.abc import Mapping
    f=Path(path).resolve()
    assert f.is_file(),f'CHECKPOINT_NOT_FOUND {f}'
    ck=torch.load(f,map_location='cpu',weights_only=False)
    assert isinstance(ck,Mapping) and 'cfg' in ck and 'genes' in ck and 'model' in ck, 'INVALID_CHECKPOINT_STRUCTURE'
    assert list(ck['genes'])==genes,('GENE_AXIS_MISMATCH',f)
    cfg=ck['cfg']; assert isinstance(cfg,Mapping)
    data=cfg.get('data',{});model=cfg.get('model',{})
    assert str(data.get('target'))=='raw',('NOT_RAW_TRAINED',f,data.get('target'))
    assert float(data.get('context_um',-1))==128,('NOT_128_CONTEXT',f)
    assert model.get('method','context')=='context',('NOT_CONTEXT_REGRESSOR',f)
    # Any patient split metadata that exists must explicitly exclude the patient.
    forbidden={'P2':{'TENX195','TENX196'},'P3':{'TENX197','TENX198','TENX199'},
               'P4':{'TENX200','TENX201','TENX202'}}[patient]
    for field in ('train_samples','samples_train','train_sections'):
        if field in data:
            assert not set(data[field])&forbidden,('TEST_PATIENT_LEAKAGE',patient,f,field)
    del ck
    return f,sha(f)

def discover(root,patient,genes):
    roots=[root/'runs/primary2025_donor_rotation_v1',root/'runs/verified_donor_rotation_v1',
           root/'runs/independent_histo',root/'runs/primary2025_context_v1']
    candidates=[]
    for base in roots:
        if not base.is_dir():continue
        for f in base.rglob('best.pt'):
            parts='/'.join(f.parts).lower()
            if patient.lower() not in parts and ('test_'+patient).lower() not in parts:continue
            if '128' not in parts or 'raw' not in parts:continue
            try:
                cf,h=ensure_checkpoint(f,patient,root,genes)
            except (AssertionError,KeyError,ValueError,RuntimeError,EOFError):continue
            candidates.append((cf,h))
    hashes={h for _,h in candidates}
    if len(hashes)!=1:
        raise RuntimeError('STOP: checkpoint discovery ambiguous/unavailable for '+patient+': '+
           repr([(str(p),h) for p,h in candidates])+'; supply explicit --checkpoints JSON')
    return sorted(candidates,key=lambda x:str(x[0]))[0][0]


def run(args):
    root=args.project.resolve();sid=args.sample;patient=PATS[sid]
    ckmap={}
    genes=[g.strip() for g in (root/'runs/frozen_primary2025/gene_axis_280.txt').read_text().splitlines() if g.strip()]
    assert len(genes)==280
    if args.checkpoints:
        ckmap=json.loads(args.checkpoints.read_text())
        assert set(ckmap)=={'P2','P3','P4'},'CHECKPOINT_JSON_MUST_HAVE_P2_P3_P4'
        checkfile=Path(ckmap[patient]);checkfile=checkfile if checkfile.is_absolute() else root/checkfile
    else:
        checkfile=discover(root,patient,genes)
    checkpoint,cksha=ensure_checkpoint(checkfile,patient,root,genes)
    src=root/'scripts/source_cell_intervention.py'
    assert src.is_file()
    mdir=args.match_root.resolve()/sid
    pfile=mdir/f'decoy_rank{args.rank}_pairs.csv'
    orig=args.original_root.resolve()/patient/f'{sid}_ctx128_intervention.csv'
    assert pfile.is_file(),f'MISSING_MATCHED_DECoys {pfile}'
    assert orig.is_file(),f'MISSING_ORIGINAL {orig}'
    cov=json.loads((mdir/'coverage.json').read_text())
    assert cov['sample']==sid and cov['patient']==patient
    assert cov['original_completed_intervention_csv_sha256']==sha(orig),'ORIGINAL_RESULT_CHANGED_AFTER_MATCHING'
    pair=pd.read_csv(pfile,dtype={c:str for c in ('gene','recipient_cell_id','source_cell_id','control_cell_id','old_control_cell_id')})
    assert len(pair)>0 and not pair.duplicated(['gene','recipient_cell_id','source_cell_id','old_control_cell_id']).any()
    assert pair['matched_decoy_rank'].eq(args.rank).all()
    assert pair.control_cell_type.eq(pair.source_cell_type).all()
    assert pair.distance_error_um.le(cov['prespecified_calipers']['distance_error_um']+1e-6).all()
    out=args.out_root.resolve()/f'rank{args.rank}'/f'{sid}_same_morph_intervention.csv'
    assert not out.exists(),f'REFUSE_OVERWRITE {out}'
    out.parent.mkdir(parents=True,exist_ok=True)
    tmp=out.with_name(out.name+'.partial.'+str(os.getpid()))
    assert not tmp.exists(),f'REFUSE_EXISTING_TEMP {tmp}'
    sdir=root/'data/prepared_primary2025/samples'/sid
    command=[sys.executable,'-u',str(src),'--checkpoint',str(checkpoint),
      '--root',str(root/'data/prepared_primary2025'),'--sample',sid,
      '--pairs',str(pfile),'--polygons',str(sdir/'cell_polygons.parquet'),
      '--out',str(tmp),'--device',args.device,'--modes','blur','inpaint']
    print('EXISTING_CHECKPOINT',patient,str(checkpoint),'SHA256',cksha,flush=True)
    print('ORIGINAL_INTERVENTION_CODE_SHA256',sha(src),flush=True)
    print('PAIRS_SHA256',sha(pfile),'NUMBER',len(pair),flush=True)
    subprocess.run(command,check=True,cwd=root)
    new=pd.read_csv(tmp,dtype={c:str for c in KEY+['old_control_cell_id','control_cell_id']})
    assert set(new.intervention)=={'blur','inpaint'} and len(new)==len(pair)*2,'MISSING_OR_DUPLICATE_INTERVENTIONS'
    assert not new.duplicated(KEY+['old_control_cell_id']).any()
    old=pd.read_csv(orig,usecols=KEY+['control_cell_id','pred_original','delta_source'],
                    dtype={c:str for c in KEY+['control_cell_id']})
    assert not old.duplicated(KEY+['control_cell_id']).any(), 'DUPLICATE_ORIGINAL_ROWS'
    merged=new.merge(old,left_on=KEY+['old_control_cell_id'],right_on=KEY+['control_cell_id'],
                     how='left',validate='one_to_one',suffixes=('_new','_orig'))
    assert len(merged)==len(new) and merged.delta_source_orig.notna().all(),'INCOMPLETE_ORIGINAL_PAIR_ALIGNMENT'
    err_source=float(np.max(np.abs(merged.delta_source_new.to_numpy(float)-merged.delta_source_orig.to_numpy(float))))
    err_original=float(np.max(np.abs(merged.pred_original_new.to_numpy(float)-merged.pred_original_orig.to_numpy(float))))
    # Same deterministic H&E crop, original checkpoint and source edit, so these
    # *should* agree. A mismatch is a protocol/weights difference, not tolerated.
    assert err_source<=args.max_repeat_error and err_original<=args.max_repeat_error,(
        'SOURCE_REPRODUCIBILITY_FAILED',sid,err_source,err_original,cksha)
    assert np.isfinite(new[['delta_source','delta_control','delta_source_minus_control']].to_numpy(float)).all()
    os.rename(tmp,out)
    info={'sample':sid,'patient':patient,'rank':args.rank,'checkpoint_sha256':cksha,
          'checkpoint_path':str(checkpoint),'source_intervention_sha256':sha(src),
          'new_pairs_sha256':sha(pfile),'completed_original_sha256':sha(orig),
          'n_pairs':len(pair),'n_rows':len(new),'max_source_delta_repetition_error':err_source,
          'max_prediction_repetition_error':err_original,'new_predictions_sha256':sha(out)}
    (out.with_suffix('.json')).write_text(json.dumps(info,indent=2)+'\n')
    print('SAME_MORPH_REAL_IMAGE_INTERVENTION_PASS',json.dumps(info),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--project',type=Path,default=Path('.'))
    p.add_argument('--sample',choices=sorted(PATS),required=True)
    p.add_argument('--rank',type=int,default=1)
    p.add_argument('--checkpoints',type=Path,help='JSON mapping P2,P3,P4 to original frozen raw-ctx128 checkpoint paths; omit only if auto-discovery unambiguous')
    p.add_argument('--match-root',type=Path,default=Path('runs/independent_histo/same_morph_decoys_v2'))
    p.add_argument('--original-root',type=Path,default=Path('runs/independent_histo/donor_source_control_rotation_original_v1'))
    p.add_argument('--out-root',type=Path,default=Path('runs/independent_histo/same_morph_image_results_v2'))
    p.add_argument('--device',default='cuda')
    p.add_argument('--max-repeat-error',type=float,default=1e-4)
    a=p.parse_args()
    assert a.rank in (1,2,3) and 0<a.max_repeat_error<=1e-3
    run(a)
if __name__=='__main__':main()


