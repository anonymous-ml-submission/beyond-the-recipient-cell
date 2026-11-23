#!/usr/bin/env python3
"""CPU matched-cell morphology/image sensitivity; original P2 freeze unchanged."""
from pathlib import Path
import hashlib, json, sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'cvpr_fair_baselines_v1'
ORIG = BASE / 'runs/P2_freeze_v1'
DER = BASE / 'runs/P2_morph_matched_v1'
DEST = DER / 'matched_ridge_v1'
TMP = DER / '.matched_ridge_v1.partial'
sys.path.insert(0, str(BASE))
import cvpr_fair_baselines_v1 as b

frozen, manifest = b.read_freeze(ORIG)
audit = json.loads((DER / 'manifest.json').read_text())
assert audit['original_manifest_sha256'] == b.sha(ORIG / 'manifest.json')
assert audit['counts_by_role'] == {'train':3998,'val':869,'test':729}
assert manifest['morph_template'] is None and len(manifest['genes']) == 280
assert not DEST.exists() and not TMP.exists(), 'STOP: existing derived results or partial run'
ALPHAS = (0.1,1.,10.,100.,1000.,10000.,100000.,1000000.)
EXPECTED = {'train':3998,'val':869,'test':729}
MODELS = ('M1','M2','M3','imagenet-r18','phikon-v2')

def get_morph(ctx, role):
    xx,keys=[],[]
    for sid,rec in manifest['samples'].items():
        if rec['role']!=role: continue
        p=DER/'features/morph'/f'ctx{ctx}'/f'{sid}.npz'
        assert p.is_file(), p
        sel=b.read_selection(frozen,manifest,sid)
        with np.load(p, allow_pickle=False) as z:
            ids=z['cell_id'].astype(str).tolist()
            ridx=z['row_index'].tolist()
            feat=z['feature'].astype(np.float64)
            assert z['gene'].tolist()==manifest['genes']
        picked=sel[sel.cell_id.isin(ids)]
        assert len(ids)==len(set(ids))==len(picked)
        assert ids==picked.cell_id.tolist() and ridx==picked.row_index.tolist()
        assert feat.shape==(len(ids),23) and np.isfinite(feat).all()
        xx.append(feat)
        keys.extend((sid,cid) for cid in ids)
    X=np.concatenate(xx)
    assert len(keys)==len(set(keys))==EXPECTED[role]
    return keys,X

def align_image(ctx, role, name, keys):
    source=b.load_feature_set(frozen,manifest,name,ctx,role,name)
    source_keys=list(zip(source['sample_id'].tolist(),source['cell_id'].tolist()))
    lookup={key:i for i,key in enumerate(source_keys)}
    assert len(lookup)==len(source_keys)
    assert set(keys).issubset(lookup), f'{name} ctx{ctx} {role}: missing morphology-mapped cell'
    pos=np.array([lookup[key] for key in keys],dtype=np.int64)
    return {k:source[k][pos] for k in ('feature','raw','clean')}

TMP.mkdir()
summary_rows=[]
checks=[]
for ctx in (24,128):
    samples={}
    for role in ('train','val','test'):
        keys, morph=get_morph(ctx,role)
        r18=align_image(ctx,role,'imagenet-r18',keys)
        phikon=align_image(ctx,role,'phikon-v2',keys)
        assert np.array_equal(r18['raw'],phikon['raw']), f'raw mismatch ctx{ctx} {role}'
        assert np.array_equal(r18['clean'],phikon['clean']), f'clean mismatch ctx{ctx} {role}'
        samples[role]={'keys':keys,'morph':morph,'imagenet-r18':r18['feature'],
                       'phikon-v2':phikon['feature'],'raw':r18['raw'],'clean':r18['clean']}
        print('MATCHED_ALIGN_PASS',ctx,role,len(keys),flush=True)
    truth=samples['test']
    if ctx==24:
        reference_keys=truth['keys'];reference_raw=truth['raw'];reference_clean=truth['clean']
    else:
        assert truth['keys']==reference_keys
        assert np.array_equal(truth['raw'],reference_raw)
        assert np.array_equal(truth['clean'],reference_clean)
    for name in MODELS:
        def feat(role):
            d=samples[role]
            if name=='M1': return d['morph'][:,:4]
            if name=='M2': return d['morph'][:,:16]
            if name=='M3': return d['morph']
            return d[name]
        (xt,xv,xe),mean,std=b.standardize(*(feat(role) for role in ('train','val','test')))
        for target in ('raw','clean'):
            yt=np.log1p(samples['train'][target].astype(np.float64))
            yv=np.log1p(samples['val'][target].astype(np.float64))
            trials=[];best=None
            for alpha in ALPHAS:
                coef,bias=b.ridge_solver(xt,yt,alpha)
                mse=float(np.mean((xv@coef+bias-yv)**2))
                trials.append((alpha,mse))
                if best is None or mse<best[0]: best=(mse,alpha,coef,bias)
            loss,alpha,coef,bias=best
            pred=np.expm1(np.clip(xe@coef+bias,0,20)).astype(np.float32)
            assert np.isfinite(pred).all() and (pred>=0).all()
            sids=np.array([x[0] for x in truth['keys']],dtype='U')
            ids=np.array([x[1] for x in truth['keys']],dtype='U')
            by_gene,group=b.compute_scores(pred,truth['raw'],truth['clean'],sids,
                       manifest['genes'],b.donor_mapping(set(sids)))
            d=TMP / f'{name}_ctx{ctx}_{target}'
            d.mkdir()
            np.savez_compressed(d/'test_predictions.npz',prediction=pred,raw=truth['raw'],
                clean=truth['clean'],sample_id=sids,cell_id=ids,gene=np.asarray(manifest['genes'],dtype='U'))
            by_gene.to_csv(d/'metrics_by_gene_section.csv',index=False)
            group.to_csv(d/'metrics_summary.csv',index=False)
            pd.DataFrame(trials,columns=['alpha','val_log_mse']).to_csv(d/'validation_alphas.csv',index=False)
            (d/'provenance.json').write_text(json.dumps({
                'analysis':'POSTHOC_EXACT_HNE_MATCHED_SUBSET_SENSITIVITY',
                'method':name,'context_um':ctx,'trained_target':target,
                'n_train':3998,'n_val':869,'n_test':729,
                'alpha':alpha,'val_log_mse':loss,'alphas':ALPHAS,
                'original_manifest_sha256':b.sha(ORIG/'manifest.json'),
                'derived_manifest_sha256':b.sha(DER/'manifest.json'),
                'source_sha256':b.sha(BASE/'cvpr_fair_baselines_v1.py'),
                'prediction_sha256':b.sha(d/'test_predictions.npz'),
                'note':'Image heads refitted on the same HoVer-Net exact-mapped cells; neighborhood uses mapped nuclei only. One test patient.'
            },indent=2))
            for _,r in group.loc[group.group.eq('all')].iterrows():
                summary_rows.append({'model':name,'ctx_um':ctx,'trained_target':target,
                                     'evaluated_target':r['target'],'alpha':alpha,
                                     'mean_fisher_z':r['mean_fisher_z'],'n_cells':r['n_cells'],
                                     'n_genes_finite':r['n_genes_finite']})
            print('FIT_PASS',name,ctx,target,'alpha',alpha,'val_log_mse',round(loss,6),flush=True)

pd.DataFrame(summary_rows).to_csv(TMP/'matched_summary.csv',index=False)
(TMP/'analysis_manifest.json').write_text(json.dumps({
    'original_manifest_sha256':b.sha(ORIG/'manifest.json'),
    'morph_manifest_sha256':b.sha(DER/'manifest.json'),
    'alpha_grid':ALPHAS,'train_cells':3998,'val_cells':869,'test_cells':729,
    'n_heads':20,'n_reported_rows':len(summary_rows),
    'limitation':'Posthoc exact-mapped subset, restricted neighborhood of mapped nuclei, one test patient.'
},indent=2))
assert len(summary_rows)==40
TMP.rename(DEST)
print('MATCHED_MORPH_RIDGE_ALL_20_PASS',DEST,flush=True)
print(pd.read_csv(DEST/'matched_summary.csv').to_string(index=False))

