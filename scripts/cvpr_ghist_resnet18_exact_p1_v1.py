#!/usr/bin/env python3
"""CVPR_Bio2: fair *evaluation-cohort* adapted GHIST vs original raw-CTX128 ResNet-18.

Read-only inputs; NO training or inference, NO use of P4 ResNet test results.
Primary result: GHIST minus ResNet, same P1 cell IDs, same raw log1p truth,
identical frozen genes finite for BOTH models in ALL THREE P1 sections,
per-section Pearson r / Fisher z and log-MSE, equal-section patient aggregate.

Native original-run scores are retained SEPARATELY with incomparable denominators.
Huge original ResNet NPZ pred/raw are extracted to temporary NPY files for mmap;
all temporaries are removed at the end. Python/Numpy/Pandas/Torch required.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd

P1 = ('TENX191', 'TENX192', 'TENX193')
P1_COUNTS = {'TENX191':209467,'TENX192':375529,'TENX193':254480}
TRAIN = ['TENX195','TENX196','TENX197','TENX198','TENX199']
VAL = ['TENX200','TENX201','TENX202']
R_CLIP = 0.999999
VAR_MIN = 1e-10
CHUNK = 8192


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def need(condition, message):
    if not condition:
        raise RuntimeError('STOP: ' + str(message))


def axis(values, name):
    x = [str(v) for v in values]
    need(len(x)==280 and len(set(x))==280, name+' must contain 280 unique genes')
    return np.asarray(x, dtype='U')


def permutation(values, frozen, name):
    v = axis(values, name)
    ii = pd.Index(v).get_indexer(frozen)
    need((ii>=0).all(), name+' differs from frozen gene set')
    need(len(set(ii))==len(ii), name+' contains duplicate mapped gene')
    return ii, bool(np.array_equal(v, frozen))


def key_index(sample, cell):
    sample = np.asarray(sample).astype('U')
    cell = np.asarray(cell).astype('U')
    need(len(sample)==len(cell), 'sample/cell length mismatch')
    mi = pd.MultiIndex.from_arrays([sample,cell], names=['sample','cell_id'])
    need(mi.is_unique, 'duplicate (sample_id,cell_id) in ResNet predictions')
    return mi


def load_original_resnet_metadata(path):
    need(path.is_file(),f'missing ResNet prediction archive: {path}')
    with np.load(path,allow_pickle=False) as d:
        for k in ('genes','sample_id','cell_id','pred','raw'):
            need(k in d.files, 'ResNet predictions.npz missing '+k)
        genes=d['genes'].astype('U')
        samples=d['sample_id'].astype('U')
        ids=d['cell_id'].astype('U')
        donors=d['donor_id'].astype('U') if 'donor_id' in d.files else None
    return genes,samples,ids,donors


def extract_npz_arrays(path,keys,tmp):
    """Stream only large raw/pred arrays from existing NPZ; no 2-GB RAM spike."""
    arr={}
    with zipfile.ZipFile(path,'r') as zf:
        entries=set(zf.namelist())
        for key in keys:
            member=key+'.npy'
            need(member in entries, f'NPZ member {member} missing')
            dest=tmp/(key+'.npy')
            with zf.open(member,'r') as src,dest.open('wb') as dst:
                shutil.copyfileobj(src,dst,length=8<<20)
            arr[key]=np.load(dest,mmap_mode='r',allow_pickle=False)
    return arr


class Moments:
    def __init__(self, n_genes=280):
        self.n = 0
        self.sp = np.zeros(n_genes,dtype=np.float64)
        self.sy = np.zeros(n_genes,dtype=np.float64)
        self.spp = np.zeros(n_genes,dtype=np.float64)
        self.syy = np.zeros(n_genes,dtype=np.float64)
        self.spy = np.zeros(n_genes,dtype=np.float64)
        self.sse = np.zeros(n_genes,dtype=np.float64)

    def update(self, pred, truth):
        p=np.asarray(pred,dtype=np.float64)
        y=np.asarray(truth,dtype=np.float64)
        need(p.ndim==2 and p.shape==y.shape and p.shape[1]==len(self.sp),
             'scorer shape mismatch')
        need(np.isfinite(p).all() and np.isfinite(y).all(), 'nonfinite matched prediction or target')
        self.n += len(p)
        self.sp += p.sum(axis=0);self.sy += y.sum(axis=0)
        self.spp += np.einsum('ij,ij->j',p,p)
        self.syy += np.einsum('ij,ij->j',y,y)
        self.spy += np.einsum('ij,ij->j',p,y)
        d=p-y
        self.sse += np.einsum('ij,ij->j',d,d)

    def metrics(self):
        need(self.n>=3, 'fewer than 3 matched cells')
        n=self.n
        vp=np.maximum(0.0,self.spp-self.sp**2/n)
        vy=np.maximum(0.0,self.syy-self.sy**2/n)
        cov=self.spy-self.sp*self.sy/n
        valid=(vp>VAR_MIN)&(vy>VAR_MIN)
        r=np.full(len(vp),np.nan,dtype=np.float64)
        r[valid]=np.clip(cov[valid]/np.sqrt(vp[valid]*vy[valid]),-R_CLIP,R_CLIP)
        z=np.arctanh(r)
        return r,z,self.sse/n,valid


def test_ghist_config(run):
    confp=run/'run.json'
    ck=run/'best.pt'
    need(confp.is_file() and ck.is_file(), 'GHIST best.pt/run.json missing')
    cfg=json.loads(confp.read_text())
    need(cfg.get('train_samples')==TRAIN, 'GHIST trained on unexpected sections')
    need(cfg.get('val_samples')==VAL, 'GHIST validation samples mismatch')
    need(cfg.get('test_samples')==list(P1), 'GHIST test split mismatch')
    need(cfg.get('target')=='log1p(raw_counts)', 'GHIST expression target is not log1p(raw)')
    need(cfg.get('uses_neighborhood') is False, 'GHIST neighborhood policy mismatch')
    need(cfg.get('uses_avgexp') is False, 'GHIST average expression policy mismatch')
    return cfg, sha256(ck)


def test_resnet_checkpoint(ck,frozen):
    need(ck.is_file(),f'original ResNet checkpoint missing: {ck}')
    import torch
    obj=torch.load(ck,map_location='cpu',weights_only=False)
    need(isinstance(obj,dict) and 'cfg' in obj and 'genes' in obj,
         'original checkpoint lacks cfg/genes')
    cfg=obj['cfg']; rgenes=np.asarray(obj['genes']).astype('U')
    need(np.array_equal(rgenes,frozen), 'original checkpoint gene order != frozen axis')
    data=cfg.get('data',{}); model=cfg.get('model',{})
    need(str(data.get('target','raw'))=='raw','ResNet is not raw-target trained')
    need(float(data.get('context_um',-1))==128,'ResNet is not CTX128')
    need(str(model.get('method','context'))=='context','not original ContextRegressor')
    need(str(model.get('backbone','resnet18')).lower()=='resnet18','not ResNet18 backbone')
    # Preserve original config fields rather than invent a training procedure.
    cfg_compact={'model':model,'data':data,'training':cfg.get('training',{}),
                 'split_keys_present':[k for k in cfg if ('split' in k.lower() or 'sample' in k.lower())]}
    del obj
    return cfg_compact,sha256(ck)


def native_reference_metrics(ghist, resnet_eval, original_counts, genes):
    """Read ORIGINAL reported scores, not recomputed matched scores."""
    orig=[]
    rfile=resnet_eval/'metrics_raw.csv'
    need(rfile.is_file(), 'original ResNet eval/metrics_raw.csv missing')
    r=pd.read_csv(rfile)
    need({'gene','pcc','rmse'}.issubset(r.columns), 'unexpected ResNet original metric schema')
    need(len(r)==280 and r.gene.is_unique and set(r.gene)==set(genes), 'bad ResNet gene metrics')
    r=r.set_index('gene').reindex(genes)
    for gene,entry in r.iterrows():
        orig.append({'model':'Original_ResNet18_CTX128','scope':'P1_pooled_original_run',
                     'sample':'P1_pooled','gene':gene,'pearson_r':entry.pcc,
                     'fisher_z':np.arctanh(np.clip(entry.pcc,-R_CLIP,R_CLIP)) if np.isfinite(entry.pcc) else np.nan,
                     'log_mse':float(entry.rmse)**2 if np.isfinite(entry.rmse) else np.nan,
                     'n_cells':sum(original_counts.values())})
    for sid in P1:
        base=ghist/'test_full'/sid
        metric=base/'gene_section_metrics.csv'
        status=base/'complete.json'
        need(metric.is_file() and status.is_file(), f'missing GHIST original-run stats {sid}')
        mm=pd.read_csv(metric)
        state=json.loads(status.read_text())
        need({'gene','pearson_r','fisher_z'}.issubset(mm.columns), 'unexpected GHIST original gene stats')
        need(len(mm)==280 and mm.gene.is_unique and set(mm.gene)==set(genes), 'bad GHIST original gene metrics '+sid)
        need(state['sample']==sid and state['n_cells']>0,'GHIST original section status mismatch')
        mm=mm.set_index('gene').reindex(genes)
        for gene,entry in mm.iterrows():
            orig.append({'model':'GHIST_adapted','scope':'P1_section_original_run',
                         'sample':sid,'gene':gene,'pearson_r':entry.pearson_r,
                         'fisher_z':entry.fisher_z,'log_mse':np.nan,
                         'n_cells':int(state['n_cells'])})
    return pd.DataFrame(orig)


def summarize_native(orig,ghist):
    rows=[]
    for (model,scope,sid),d in orig.groupby(['model','scope','sample'],sort=False):
        s={'model':model,'scope':scope,'sample':sid,'n_cells':int(d.n_cells.iloc[0]),
           'valid_pearson_genes':int(np.isfinite(d.pearson_r).sum()),
           'valid_fisher_genes':int(np.isfinite(d.fisher_z).sum()),
           'mean_gene_r_on_own_valid':float(d.pearson_r.mean()),
           'mean_gene_fisher_z_on_own_valid':float(d.fisher_z.mean()),
           'mean_gene_log_mse':float(d.log_mse.mean()) if d.log_mse.notna().any() else np.nan}
        if model=='GHIST_adapted':
            info=json.loads((ghist/'test_full'/sid/'complete.json').read_text())
            s['all_280_genes_log_mse_from_original_eval']=float(info['log_mse'])
        else:
            s['all_280_genes_log_mse_from_original_eval']=float(d.log_mse.mean())
        rows.append(s)
    return pd.DataFrame(rows)


def calc_comparison(args, expected=P1_COUNTS):
    project=args.project.resolve()
    ghist=args.ghist_run.resolve()
    res_eval=args.resnet_eval.resolve()
    out=args.out.resolve()
    need(project.is_dir() and ghist.is_dir() and res_eval.is_dir(), 'project / GHIST / ResNet path missing')
    need(not out.exists(),f'output exists; refusing overwrite: {out}')
    gene_file=project/'runs/frozen_primary2025/gene_axis_280.txt'
    need(gene_file.is_file(),f'frozen gene axis missing: {gene_file}')
    frozen=axis([s.strip() for s in gene_file.read_text().splitlines() if s.strip()],'frozen')
    gh_cfg,gh_ck_sha=test_ghist_config(ghist)
    resnet_ck=args.resnet_checkpoint.resolve()
    r_cfg,r_ck_sha=test_resnet_checkpoint(resnet_ck,frozen)

    rp=res_eval/'predictions.npz'
    rgenes,rsid,rcid,rdonor=load_original_resnet_metadata(rp)
    rperm,r_axis_equal=permutation(rgenes,frozen,'ResNet predictions')
    index=key_index(rsid,rcid)
    cnt={s:int((rsid==s).sum()) for s in P1}
    need(set(np.unique(rsid))==set(P1),'original ResNet evaluation is NOT P1 only')
    if expected is not None:
        need(cnt==expected, f'wrong original P1 ResNet evaluation size: {cnt}; expected {expected}')
    if rdonor is not None:
        need(len(np.unique(rdonor))==1,'original P1 evaluation contains multiple donor IDs')
    print('ORIGINAL_RESNET_P1_ONLY_PASS',cnt,'genes',len(frozen),flush=True)

    native=native_reference_metrics(ghist,res_eval,cnt,frozen)
    nsummary=summarize_native(native,ghist)
    scratch=args.scratch.resolve() if args.scratch else None
    if scratch is not None:
        need(scratch.is_dir(),'scratch directory must exist (will NOT create it)')

    gene_rows=[];coverage=[];global_g=Moments();global_r=Moments()
    checks=[]
    with tempfile.TemporaryDirectory(prefix='cvpr_p1_exact_r18_',dir=str(scratch) if scratch else None) as tmp:
        print('STREAM_EXTRACT_RESNET_RAW_AND_PRED_TO_TEMPORARY_MMAP',flush=True)
        arrays=extract_npz_arrays(rp,('raw','pred'),Path(tmp))
        raw,rawpred=arrays['raw'],arrays['pred']
        need(raw.shape==rawpred.shape==(len(rsid),280), 'ResNet prediction/raw array shape mismatch')
        need(np.issubdtype(raw.dtype,np.number) and np.issubdtype(rawpred.dtype,np.number),
             'ResNet unexpected prediction dtype')
        # Internal consistency: re-score the original unfiltered 839,476 P1 cells,
        # then verify its saved metrics_raw.csv came from these exact predictions.
        # This is CPU analysis of SAVED arrays, not repeat model inference.
        full_resnet=Moments()
        for start in range(0,len(rsid),CHUNK):
            sel=slice(start,min(start+CHUNK,len(rsid)))
            yy=np.log1p(np.asarray(raw[sel,:][:,rperm],dtype=np.float64))
            pp=np.log1p(np.asarray(rawpred[sel,:][:,rperm],dtype=np.float64))
            need(np.isfinite(yy).all() and np.isfinite(pp).all(),
                 'original full ResNet arrays contain nonfinite log1p values')
            full_resnet.update(pp,yy)
        all_r,_,all_mse,all_valid=full_resnet.metrics()
        ref=native.loc[(native.model=='Original_ResNet18_CTX128')]
        ref=ref.set_index('gene').reindex(frozen)
        observed_r=ref.pearson_r.to_numpy(dtype=np.float64)
        observed_mse=ref.log_mse.to_numpy(dtype=np.float64)
        need(np.array_equal(np.isfinite(observed_r),all_valid),
             'ResNet original finite-correlation gene set differs from predictions')
        if all_valid.any():
            rerr=float(np.max(np.abs(all_r[all_valid]-observed_r[all_valid])))
            need(rerr<5e-4, f'original ResNet saved PCC does not reproduce: max error {rerr}')
        merr=float(np.max(np.abs(all_mse-observed_mse)))
        need(merr<5e-4, f'original ResNet saved log-MSE does not reproduce: max error {merr}')
        print('RESNET_ORIGINAL_P1_FULL_RUN_METRIC_REPLAY_PASS',
              'cells',full_resnet.n,'max_MSE_error',merr,flush=True)
        for sid in P1:
            gp=ghist/'test_full'/sid/'predictions.npz'
            meta_path=gp.with_name('complete.json')
            need(gp.is_file() and meta_path.is_file(),f'GHIST P1 section missing {sid}')
            meta=json.loads(meta_path.read_text())
            need(meta['sample']==sid and meta['checkpoint_sha256']==gh_ck_sha,
                 f'GHIST section {sid}: checkpoint mismatch')
            actual_sha=sha256(gp)
            need(actual_sha==meta['predictions_sha256'],f'GHIST {sid} predictions changed')
            with np.load(gp,allow_pickle=False) as d:
                for k in ('genes','cell_id','sample_id','pred_log1p','raw_log1p'):
                    need(k in d.files,f'GHIST {sid} missing {k}')
                ggenes=d['genes'].astype('U')
                gpidx,g_equal=permutation(ggenes,frozen,f'GHIST {sid}')
                gs=str(d['sample_id'].item())
                gids=d['cell_id'].astype('U')
                pred_g=d['pred_log1p']
                truth_g=d['raw_log1p']
            need(gs==sid,'GHIST mismatched section label')
            need(len(gids)==len(set(gids)),'GHIST duplicate nucleus-mapped cell IDs '+sid)
            need(pred_g.shape==truth_g.shape==(len(gids),280),f'GHIST output dimensions wrong {sid}')
            need(len(gids)==int(meta['n_cells']),f'GHIST cell-count status mismatch {sid}')
            # GHIST original per-section metrics must also replay from the
            # saved complete-section prediction files (not just match a CSV name).
            gh_native=Moments()
            for st in range(0,len(gids),CHUNK):
                ix=slice(st,min(st+CHUNK,len(gids)))
                gh_native.update(np.asarray(pred_g[ix,:][:,gpidx]),
                                 np.asarray(truth_g[ix,:][:,gpidx]))
            native_r,_,native_mse,native_valid=gh_native.metrics()
            reference=native.loc[(native.model=='GHIST_adapted') & (native['sample']==sid)]
            reference=reference.set_index('gene').reindex(frozen)
            expected_r=reference.pearson_r.to_numpy(dtype=np.float64)
            need(np.array_equal(np.isfinite(expected_r),native_valid),
                 f'GHIST {sid} original finite-gene set differs from saved predictions')
            if native_valid.any():
                gh_err=float(np.max(np.abs(native_r[native_valid]-expected_r[native_valid])))
                need(gh_err<5e-4,f'GHIST {sid} published-run Pearson replay mismatch {gh_err}')
            reported_mse=float(meta['log_mse'])
            actual_mse=float(native_mse.mean())
            need(abs(reported_mse-actual_mse)<5e-4,
                 f'GHIST {sid} complete.json log-MSE differs from prediction replay')
            print('GHIST_ORIGINAL_SECTION_METRIC_REPLAY_PASS',sid,
                  'valid_genes',int(native_valid.sum()),flush=True)
            idx=index.get_indexer(pd.MultiIndex.from_arrays([
                np.repeat(sid,len(gids)),gids]))
            rows=np.flatnonzero(idx>=0).astype(np.int64)
            ires=idx[rows]
            need(len(rows)>=100,f'no credible common-cell cohort in {sid}')
            need(len(np.unique(ires))==len(ires),f'duplicate matched ResNet cells {sid}')
            gacc=Moments();racc=Moments()
            max_raw_error=0.0
            for start in range(0,len(rows),CHUNK):
                sel=rows[start:start+CHUNK]
                res_idx=ires[start:start+CHUNK]
                # ResNet saved *counts*, GHIST saved log1p *counts*.
                yy=np.log1p(np.asarray(raw[res_idx,:][:,rperm],dtype=np.float64))
                rp_log=np.log1p(np.asarray(rawpred[res_idx,:][:,rperm],dtype=np.float64))
                gp_log=np.asarray(pred_g[sel,:][:,gpidx],dtype=np.float64)
                gy=np.asarray(truth_g[sel,:][:,gpidx],dtype=np.float64)
                need(np.isfinite(yy).all() and np.isfinite(rp_log).all() and np.isfinite(gp_log).all(),
                     f'nonfinite model output/truth {sid}')
                need((raw[res_idx,:]>=0).all(),f'negative raw counts {sid}')
                err=float(np.max(np.abs(yy-gy)))
                max_raw_error=max(max_raw_error,err)
                need(err<=1e-4,f'MISMATCHED_RAW_TRUTH {sid}: max |log1p(R18 raw)-GHIST raw|={err}')
                gacc.update(gp_log,yy)
                racc.update(rp_log,yy)
                global_g.update(gp_log,yy)
                global_r.update(rp_log,yy)
            need(gacc.n==racc.n==len(rows),'mismatched section accumulator lengths')
            gr,gz,gm,gv=gacc.metrics();rr,rz,rm,rv=racc.metrics()
            for j,g in enumerate(frozen):
                gene_rows.append({'sample':sid,'gene':g,'n_shared_cells':len(rows),
                                  'GHIST_r':gr[j],'ResNet_r':rr[j],
                                  'GHIST_z':gz[j],'ResNet_z':rz[j],
                                  'GHIST_log_mse':gm[j],'ResNet_log_mse':rm[j],
                                  'both_valid_fisher':bool(gv[j] and rv[j])})
            coverage.append({'sample':sid,'GHIST_full_cells':len(gids),
                             'ResNet_original_section_cells':cnt[sid],
                             'shared_cells':len(rows),
                             'pct_GHIST_matched':100*len(rows)/len(gids),
                             'pct_ResNet_matched':100*len(rows)/cnt[sid],
                             'GHIST_pred_frozen_axis_ordered':g_equal,
                             'max_abs_log1p_raw_truth_difference':max_raw_error})
            checks.append({'sample':sid,'ghist_prediction_sha256':actual_sha,
                           'max_abs_log1p_truth_error':max_raw_error})
            print(f'MATCHED_P1_SECTION {sid} GHIST={len(gids)} ResNet={cnt[sid]} '
                  f'SHARED={len(rows)} TRUTH_MAX_ERR={max_raw_error:.3g}',flush=True)
            del pred_g,truth_g
    detail=pd.DataFrame(gene_rows)
    valid=detail.groupby('gene')['both_valid_fisher'].agg(['sum','size'])
    accepted=valid.index[(valid['sum']==3)&(valid['size']==3)].tolist()
    accepted=[g for g in frozen if g in set(accepted)]
    need(len(accepted)>0,'NO_COMMON_VALID_GENES_ACROSS_BOTH_MODELS_AND_3_SECTIONS')
    excluded=[str(g) for g in frozen if g not in set(accepted)]
    x=detail.loc[detail.gene.isin(accepted)].copy()
    need(len(x)==len(accepted)*3,'fixed gene universe error')
    x['delta_z_GHIST_minus_ResNet']=x.GHIST_z-x.ResNet_z
    x['delta_log_mse_GHIST_minus_ResNet']=x.GHIST_log_mse-x.ResNet_log_mse
    bysection=x.groupby('sample',sort=False).agg(
        shared_cells=('n_shared_cells','first'),common_finite_genes=('gene','nunique'),
        GHIST_mean_pearson=('GHIST_r','mean'),ResNet_mean_pearson=('ResNet_r','mean'),
        GHIST_mean_fisher_z=('GHIST_z','mean'),ResNet_mean_fisher_z=('ResNet_z','mean'),
        GHIST_mean_log_mse=('GHIST_log_mse','mean'),ResNet_mean_log_mse=('ResNet_log_mse','mean'),
        GHIST_minus_ResNet_delta_z=('delta_z_GHIST_minus_ResNet','mean'),
        GHIST_minus_ResNet_delta_log_mse=('delta_log_mse_GHIST_minus_ResNet','mean'),
    ).reset_index()
    bysection['GHIST_equivalent_r']=np.tanh(bysection.GHIST_mean_fisher_z)
    bysection['ResNet_equivalent_r']=np.tanh(bysection.ResNet_mean_fisher_z)
    bysection['GHIST_lower_log_mse']=bysection.GHIST_mean_log_mse < bysection.ResNet_mean_log_mse

    gpr,gpz,gpm,gvalid=global_g.metrics()
    rpr,rpz,rpm,rvalid=global_r.metrics()
    pooled=pd.DataFrame({'gene':frozen,'GHIST_r':gpr,'ResNet_r':rpr,
                         'GHIST_z':gpz,'ResNet_z':rpz,
                         'GHIST_log_mse':gpm,'ResNet_log_mse':rpm,
                         'both_pooled_valid':gvalid&rvalid})
    need(pooled.loc[pooled.gene.isin(accepted),'both_pooled_valid'].all(),
         'primary gene lost finite correlation when pooling sections')
    pooled=pooled.loc[pooled.gene.isin(accepted)].copy()

    # Scientific warning: exactly one test patient; 3 sections are NOT independent patients.
    row={'patient':'P1','n_sections':3,'n_shared_cells':int(bysection.shared_cells.sum()),
         'n_common_finite_genes':len(accepted),
         'equal_section_mean_GHIST_z':float(bysection.GHIST_mean_fisher_z.mean()),
         'equal_section_mean_ResNet_z':float(bysection.ResNet_mean_fisher_z.mean()),
         'equal_section_delta_z':float(bysection.GHIST_minus_ResNet_delta_z.mean()),
         'equal_section_GHIST_log_mse':float(bysection.GHIST_mean_log_mse.mean()),
         'equal_section_ResNet_log_mse':float(bysection.ResNet_mean_log_mse.mean()),
         'equal_section_delta_log_mse':float(bysection.GHIST_minus_ResNet_delta_log_mse.mean()),
         'pooled_cell_GHIST_z':float(pooled.GHIST_z.mean()),
         'pooled_cell_ResNet_z':float(pooled.ResNet_z.mean()),
         'pooled_cell_delta_z':float((pooled.GHIST_z-pooled.ResNet_z).mean()),
         'pooled_cell_GHIST_log_mse':float(pooled.GHIST_log_mse.mean()),
         'pooled_cell_ResNet_log_mse':float(pooled.ResNet_log_mse.mean()),
         'pooled_cell_delta_log_mse':float((pooled.GHIST_log_mse-pooled.ResNet_log_mse).mean())}
    orig_summary=nsummary.copy()
    coverage=pd.DataFrame(coverage)
    print('\nCOMMON_FINITE_GENES',len(accepted),'/280','EXCLUDED',len(excluded),flush=True)
    print('\n===== PRIMARY MATCHED GHIST MINUS RESNET18 (P1 ONLY) =====')
    print(bysection.to_string(index=False,float_format=lambda y:f'{y:.6f}'),flush=True)
    print('\n===== EQUAL-SECTION P1 (ONE PATIENT) =====')
    for k,v in row.items():print(k,v)
    print('\n===== ORIGINAL RUN METRICS: DIFFERENT UNMATCHED DENOMINATORS =====')
    print(orig_summary.to_string(index=False,float_format=lambda y:f'{y:.6f}'),flush=True)
    out.mkdir(parents=True,exist_ok=False)
    detail.to_csv(out/'all_280_gene_by_section_metrics.csv',index=False)
    x.to_csv(out/'primary_fixed_common_gene_metrics.csv',index=False)
    bysection.to_csv(out/'primary_matched_section_scores.csv',index=False)
    pd.DataFrame([row]).to_csv(out/'primary_patient_summary.csv',index=False)
    pooled.to_csv(out/'secondary_pooled_cell_gene_metrics.csv',index=False)
    coverage.to_csv(out/'matched_cell_coverage.csv',index=False)
    native.to_csv(out/'original_unmatched_run_gene_metrics.csv',index=False)
    orig_summary.to_csv(out/'original_unmatched_run_summary.csv',index=False)
    (out/'common_finite_genes.txt').write_text('\n'.join(accepted)+'\n')
    pd.DataFrame({'excluded_from_primary_gene_universe':excluded}).to_csv(out/'excluded_genes.csv',index=False)
    provenance={
        'analysis':'P1 GHIST_adapted vs original raw-trained ResNet18 CTX128',
        'scope':'P1 independent held-out test patient, three correlated tissue sections',
        'not_p4_comparison':True,
        'primary':'same (sample_id,cell_id), log1p(raw count) truth, fixed valid gene set across BOTH models and ALL P1 sections; equal-section Fisher z',
        'finite_gene_rules':{'n_cells_min':3,'variance_sumsquares_gt':VAR_MIN,'pearson_clip':R_CLIP},
        'frozen_gene_axis_sha256':sha256(gene_file),
        'frozen_gene_count':len(frozen),'common_finite_genes':accepted,
        'GHIST_checkpoint_sha256':gh_ck_sha,'GHIST_run_json_sha256':sha256(ghist/'run.json'),
        'ResNet18_original_checkpoint_sha256':r_ck_sha,'ResNet18_predictions_sha256':sha256(rp),
        'ResNet18_original_metrics_raw_sha256':sha256(res_eval/'metrics_raw.csv'),
        'ResNet18_predictions_genes_in_exact_frozen_order':r_axis_equal,
        'GHIST_prediction_hash_checks':checks,
        'model_training_differences':{
            'GHIST_adapted':{
                'train_sections':TRAIN,'validation_sections':VAL,'test_sections':list(P1),
                'inputs':'128um image tile plus external HoVer-Net nucleus instance masks; cell type auxiliary supervision',
                'target':'log1p(raw Xenium)',
                'training_objectives':gh_cfg.get('losses'),
                'neighbor_head_enabled':gh_cfg.get('uses_neighborhood'),
                'reference_expression_enabled':gh_cfg.get('uses_avgexp'),
                'model_selection':gh_cfg.get('model_selection'),
                'run_config':gh_cfg},
            'ResNet18_original':{
                'inputs':'128um cell-centered H&E image, no externally supplied nucleus mask to CNN',
                'target':'raw Xenium counts',
                'training_objective':'Poisson on raw counts (original ContextRegressor training code)',
                'validation':'original run validation configuration; do not infer from predictions.npz',
                'checkpoint_config':r_cfg,
            }},
        'interpretation':'fair matched test cohort, NOT identical architecture/training regime; one independent P1 test patient',
    }
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2,default=str)+'\n')
    print('\nGHIST_RESNET18_EXACT_P1_COMPARISON_PASS',out,flush=True)
    return row,bysection,orig_summary


def self_test():
    # Fast failure-path and metric math self-check; no project files required.
    import math
    rng=np.random.default_rng(43)
    n=500
    y=np.log1p(rng.poisson(3,(n,280)).astype(np.float32))
    a=y+rng.normal(0,0.13,y.shape)
    b=y+rng.normal(0,0.24,y.shape)
    ma=Moments();mb=Moments()
    for start in range(0,n,33):
        ma.update(a[start:start+33],y[start:start+33]);mb.update(b[start:start+33],y[start:start+33])
    ar,az,ae,av=ma.metrics();br,bz,be,bv=mb.metrics()
    reference=np.array([np.corrcoef(a[:,j],y[:,j])[0,1] for j in range(280)])
    need(np.allclose(ar,reference,atol=1e-9),'Pearson math mismatch')
    need(np.all(av&bv) and np.mean(az-bz)>0 and np.mean(ae-be)<0,
         'Paired score math mismatch')
    const=Moments()
    const.update(np.ones((20,280)),np.ones((20,280)))
    need(not const.metrics()[3].any(),'zero-variance gene not excluded')
    # An exact permutation is accepted only by reordering back to frozen gene order.
    genes=np.asarray([f'g{i:03d}' for i in range(280)],dtype='U')
    perm=rng.permutation(280)
    ix,eq=permutation(genes[perm],genes,'permuted')
    need(not eq and np.array_equal(genes[perm][ix],genes),'gene reorder check failed')
    try:permutation(np.asarray(['bad']+list(genes[1:])),genes,'bad')
    except RuntimeError:pass
    else:raise RuntimeError('self-test failed to stop wrong gene universe')
    # Read the known NPZ contract from a small compressed fixture and mmap arrays.
    with tempfile.TemporaryDirectory() as td:
        path=Path(td)/'a.npz'
        np.savez_compressed(path,genes=genes,sample_id=np.array(['TENX191']*10),
                            cell_id=np.array([f'cell{i}' for i in range(10)]),
                            pred=np.ones((10,280),np.float32),raw=np.ones((10,280),np.float32))
        with tempfile.TemporaryDirectory() as scratch:
            mm=extract_npz_arrays(path,('raw','pred'),Path(scratch))
            need(mm['raw'].shape==(10,280) and np.isfinite(mm['pred']).all(),
                 'NPZ streamed extraction failed')
    print('GHIST_RESNET18_SYNTHETIC_MATH_AND_NPZ_CONTRACT_PASS')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--project',type=Path,default=Path('.'))
    ap.add_argument('--ghist-run',type=Path,default=Path('runs/ghist_donor_adapted_v1/seed2026'))
    ap.add_argument('--resnet-eval',type=Path,default=Path('runs/primary2025_context_v1/A_raw_ctx128/eval'))
    ap.add_argument('--resnet-checkpoint',type=Path,required=True,help='Path to the original ResNet18 checkpoint')
    ap.add_argument('--out',type=Path,default=Path('runs/independent_histo/ghist_resnet18_exact_p1_v1'))
    ap.add_argument('--scratch',type=Path,default=None,help='existing directory for temporary ~2GB ResNet .npy extraction')
    ap.add_argument('--self-test',action='store_true',help='synthetic math/NPZ tests only; no project data')
    args=ap.parse_args()
    if args.self_test:self_test();return
    calc_comparison(args)

if __name__=='__main__':
    main()
