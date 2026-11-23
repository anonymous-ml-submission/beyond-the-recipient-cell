#!/usr/bin/env python3
"""CPU-only H&E morphology/neighborhood baseline for frozen Primary2025 P1 split.

Modes:
  preflight : read metadata and schemas only; write nothing
  extract   : for ONE section, cache H&E-only features at BOTH 24/128 um
  fit       : once all 11 caches exist, fit/score M1/M2/M3 on raw or clean labels

Reference geometry: prepared cell x_px/y_px and independently exported HoVer-Net
nucleus x_px/y_px, converted to microns using each section's pixel_size_um.
Neighbor pools use ALL high-confidence H&E-detected biological nuclei, not
only exact Xenium-mapped nuclei. Recipients are restricted to exact mapping.
"""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

TRAIN=['TENX195','TENX196','TENX197','TENX198','TENX199']
VAL=['TENX200','TENX201','TENX202']
TEST=['TENX191','TENX192','TENX193']
ALL=TRAIN+VAL+TEST
FROZEN_SHA={
    'runs/frozen_primary2025/gene_axis_280.txt':'14947279372641dafffe7ae755b9da7ec3b361786c4274ec554e226fdde98713',
    'runs/frozen_primary2025/model_split_routing_v1.json':'34c7218f1b358e51a3a0bacc8384df8c3b103161207048f6bf18c7b07343e7c8',
    'runs/frozen_primary2025/deleakage_all11_clean_targets_v1.json':'445d9d57f22f7e21609928a605eee8618e9457779c3e61bc0e0e64cf96bc57ce',
}
CLASSES=['Connective','Inflammatory','Neoplastic','Non-Neoplastic Epithelial']
FNAME=['bias']+['recipient_'+x for x in CLASSES]+['near_frac_'+x for x in CLASSES]+['log_nuc_density']+['nearest_'+x for x in CLASSES]
assert len(FNAME)==14

def check_freeze(root):
    for rel,exp in FROZEN_SHA.items():
        p=expected(root/rel)
        h=hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
        if h.hexdigest()!=exp: raise RuntimeError(('FROZEN_SHA_MISMATCH',str(p),h.hexdigest()))
    print('FROZEN_INPUT_HASHES_PASS',flush=True)

def expected(path):
    p=Path(path)
    if not p.is_file(): raise RuntimeError(f'MISSING: {p}')
    return p

def paths(root,sid):
    return (root/'data/prepared_primary2025/samples'/sid,
            root/'runs/independent_histo'/f'{sid}_cell_annotations.parquet',
            root/'runs/independent_histo'/f'{sid}_nuclei.parquet')

def get_meta(root,sid,include_nuclei=False):
    sdir,an,nuc=paths(root,sid)
    cp=expected(sdir/'cells.parquet'); an=expected(an); nuc=expected(nuc)
    mpp=float(expected(sdir/'pixel_size_um.txt').read_text().strip())
    if not 0<mpp<10: raise RuntimeError((sid,'INVALID_MPP',mpp))
    cells=pd.read_parquet(cp,columns=['cell_id','x_px','y_px'])
    ann=pd.read_parquet(an,columns=['cell_id','cell_type','annotation_status','hovernet_nucleus_id'])
    for d in (cells,ann):
        d['cell_id']=d.cell_id.astype(str)
        if not d.cell_id.is_unique: raise RuntimeError((sid,'DUPLICATE_CELL_ID'))
    if len(cells)!=len(ann) or not set(cells.cell_id)==set(ann.cell_id):
        raise RuntimeError((sid,'CELL_ANNOTATION_UNIVERSE_MISMATCH'))
    c=cells.merge(ann,on='cell_id',how='left',validate='one_to_one',sort=False)
    if c.annotation_status.isna().any(): raise RuntimeError((sid,'ANNOTATIONS_INCOMPLETE'))
    valid=c.annotation_status.eq('unique_nucleus_inside_cell') & c.cell_type.isin(['HISTO_'+x.replace(' ','_') for x in CLASSES])
    c=c.loc[valid].copy()
    c['class_idx']=c.cell_type.map({'HISTO_'+x.replace(' ','_'):i for i,x in enumerate(CLASSES)}).astype(int)
    if len(c)==0 or c.hovernet_nucleus_id.isna().any(): raise RuntimeError((sid,'NO_VALID_RECIPIENTS'))
    nuclei=None
    if include_nuclei:
        nuclei=pd.read_parquet(nuc,columns=['nucleus_id','x_px','y_px','histo_type','histo_prob'])
        nuclei['nucleus_id']=nuclei.nucleus_id.astype(str)
        if not nuclei.nucleus_id.is_unique: raise RuntimeError((sid,'DUPLICATE_NUCLEI'))
    return c,nuclei,mpp,len(cells)

def preflight(root):
    check_freeze(root)
    genes=[g.strip() for g in expected(root/'runs/frozen_primary2025/gene_axis_280.txt').read_text().splitlines() if g.strip()]
    if len(genes)!=280 or len(set(genes))!=280: raise RuntimeError('FROZEN_GENE_AXIS_INVALID')
    for sid in ALL:
        sdir,_,_=paths(root,sid)
        expected(sdir/'raw.h5ad'); expected(sdir/'clean.h5ad')
        c,n,mpp,total=get_meta(root,sid,include_nuclei=True)
        n=n.loc[n.histo_type.isin(CLASSES) & pd.to_numeric(n.histo_prob,errors='coerce').ge(0.6)]
        if len(n)<100: raise RuntimeError((sid,'TOO_FEW_HISTO_NUCLEI'))
        missing=set(c.hovernet_nucleus_id.astype(str))-set(n.nucleus_id) # confidence filter may exclude some mapped nuclei
        print(f'{sid}: mapped_biological={len(c)} / {total}; high_conf_nuclei={len(n)}; mapped_nuclei_not_high_conf={len(missing)}; mpp={mpp}',flush=True)
    print('PREFLIGHT PASS: metadata only; neither training nor patient test used')

def extract(root,out,sid,prob_min):
    check_freeze(root)
    c,n,mpp,total=get_meta(root,sid,include_nuclei=True)
    n=n.loc[n.histo_type.isin(CLASSES) & pd.to_numeric(n.histo_prob,errors='coerce').ge(prob_min)].copy()
    if len(n)<100: raise RuntimeError((sid,'TOO_FEW_NUCLEI'))
    n['class_idx']=n.histo_type.map({x:i for i,x in enumerate(CLASSES)}).astype(int)
    nxy=n[['x_px','y_px']].to_numpy(float)*mpp
    cxy=c[['x_px','y_px']].to_numpy(float)*mpp
    if not np.isfinite(nxy).all() or not np.isfinite(cxy).all(): raise RuntimeError('NONFINITE_COORDS')
    ids=n.nucleus_id.to_numpy(str)
    nclass=n.class_idx.to_numpy(int)
    cid=c.hovernet_nucleus_id.astype(str).to_numpy(str)
    classes=c.class_idx.to_numpy(int)
    tree=cKDTree(nxy)
    # Check that the exact nucleus is geometrically near its recipient.
    loc=pd.Index(ids).get_indexer(cid)
    if (loc<0).sum():
        # mapped nuclei could be below confidence cutoff: only check those included
        print('MAPPED_NUCLEI_BELOW_CONFIDENCE_FILTER',int((loc<0).sum()),flush=True)
    keep=loc>=0
    if keep.any():
        offset=np.linalg.norm(nxy[loc[keep]]-cxy[keep],axis=1)
        if np.nanquantile(offset,0.99)>25:
            raise RuntimeError((sid,'MAPPED_NUCLEUS_COORDINATE_MISMATCH_Q99_UM',np.quantile(offset,0.99)))
    # Require exact raw index matching; no assumed row order.
    import anndata as ad
    raw=ad.read_h5ad(expected(paths(root,sid)[0]/'raw.h5ad'),backed='r')
    try:
        obs=pd.Index(raw.obs_names.astype(str))
        idx=obs.get_indexer(c.cell_id.astype(str))
        if (idx<0).any() or len(obs)!=total: raise RuntimeError((sid,'RAW_CELL_ID_MISMATCH'))
    finally:
        raw.file.close()
    order=np.argsort(idx)
    idx=idx[order]; c=c.iloc[order]; cxy=cxy[order]; cid=cid[order]; classes=classes[order]
    # Precompute one-hot recipients identically for both contexts.
    rx=np.zeros((len(c),5),dtype=np.float32)
    rx[:,0]=1
    rx[np.arange(len(c)),classes+1]=1
    all_features={}
    for ctx in (24,128):
        X=np.zeros((len(c),14),dtype=np.float32); X[:,:5]=rx
        half=ctx/2.0; r=half*math.sqrt(2)
        for lo in range(0,len(c),512):
            if lo % 25600 == 0: print('EXTRACT_PROGRESS',sid,ctx,lo,'/',len(c),flush=True)
            hi=min(len(c),lo+512)
            neighbors=tree.query_ball_point(cxy[lo:hi],r)
            for ii,cand in enumerate(neighbors):
                k=lo+ii
                if not cand: continue
                cp=cxy[k]
                dxy=np.abs(nxy[cand]-cp)
                inside=np.maximum(dxy[:,0],dxy[:,1])<=half
                if not inside.any(): continue
                selected=np.asarray(cand,dtype=np.int64)[inside]
                # Do not count the recipient's own mapped nucleus.
                selected=selected[ids[selected]!=cid[k]]
                if selected.size==0: continue
                counts=np.bincount(nclass[selected],minlength=4)
                X[k,5:9]=counts/counts.sum()
                X[k,9]=np.log1p(len(selected)*10000.0/(ctx*ctx))/10.0
                ds=np.linalg.norm(nxy[selected]-cp,axis=1)
                for j in range(4):
                    matched=nclass[selected]==j
                    X[k,10+j]=(min(float(ds[matched].min()),r)/r if matched.any() else 1.0)
        # All recipient IDs must be the same in both contexts.
        all_features[ctx]=X
        print(sid,ctx,'cells',len(c),'nonzero_neighbor_fraction',float((X[:,5:9].sum(1)>0).mean()),flush=True)
    # No overwrite; prevent partial output at known paths.
    out.mkdir(parents=True,exist_ok=True)
    for ctx,X in all_features.items():
        dst=out/f'{sid}_ctx{ctx}.npz'
        if dst.exists(): raise RuntimeError(f'REFUSE_OVERWRITE: {dst}')
    for ctx,X in all_features.items():
        dst=out/f'{sid}_ctx{ctx}.npz'
        np.savez_compressed(dst,x=X,raw_row=idx.astype(np.int64),cell_id=c.cell_id.to_numpy(str),features=np.array(FNAME),sample=np.array(sid),prob_min=np.array(prob_min))
        print('SAVED',dst,flush=True)

def y_chunks(root,sid,indices,genes,chunk=8192):
    import anndata as ad
    sdir,_,_=paths(root,sid)
    raw=ad.read_h5ad(expected(sdir/'raw.h5ad'),backed='r')
    clean=ad.read_h5ad(expected(sdir/'clean.h5ad'),backed='r')
    try:
        if list(raw.obs_names.astype(str))!=list(clean.obs_names.astype(str)): raise RuntimeError((sid,'RAW_CLEAN_CELL_ORDER_MISMATCH'))
        if list(raw.var_names.astype(str))!=list(clean.var_names.astype(str)): raise RuntimeError((sid,'RAW_CLEAN_GENE_ORDER_MISMATCH'))
        ri=pd.Index(raw.var_names.astype(str)).get_indexer(genes)
        if (ri<0).any(): raise RuntimeError((sid,'MISSING_GENES'))
        if not np.all(np.diff(indices)>=0): raise RuntimeError('RAW_ROW_INDICES_NOT_SORTED')
        for lo in range(0,len(indices),chunk):
            rows=indices[lo:lo+chunk]
            data=[]
            for a in (raw,clean):
                tmp=a.X[rows,:][:,ri]
                if hasattr(tmp,'toarray'): tmp=tmp.toarray()
                arr=np.asarray(tmp,dtype=np.float64)
                if arr.shape!=(len(rows),280) or not np.isfinite(arr).all() or (arr<0).any():
                    raise RuntimeError((sid,'INVALID_EXPRESSION',arr.shape))
                data.append(np.log1p(arr))
            yield lo,lo+len(rows),data[0],data[1]
    finally:
        raw.file.close()
        clean.file.close()

def loadcache(out,sid,ctx):
    z=np.load(expected(out/f'{sid}_ctx{ctx}.npz'),allow_pickle=False)
    if z['sample'].item()!=sid or list(z['features'])!=FNAME: raise RuntimeError((sid,'FEATURE_SCHEMA_MISMATCH'))
    X=z['x']; rows=z['raw_row']
    if X.shape!=(len(rows),14) or not np.isfinite(X).all() or len(rows)==0: raise RuntimeError((sid,'FEATURE_SHAPE_ERROR'))
    if len(np.unique(rows))!=len(rows): raise RuntimeError((sid,'DUPLICATE_ROWS'))
    return X,rows

def design(X,model):
    return X[:,:{'M1':5,'M2':9,'M3':14}[model]].astype(np.float64,copy=False)

def train(root,out,ctx,model,target,genes,alpha):
    p={'M1':5,'M2':9,'M3':14}[model]
    xx=np.zeros((p,p),np.float64); xy=np.zeros((p,280),np.float64); count=0
    for sid in TRAIN:
        X,idx=loadcache(out,sid,ctx); X=design(X,model)
        for lo,hi,raw,clean in y_chunks(root,sid,idx,genes):
            Y=raw if target=='raw' else clean; V=X[lo:hi]
            xx+=V.T@V; xy+=V.T@Y; count+=len(Y)
        print('TRAIN_ACCUMULATED',sid,count,flush=True)
    assert count>0
    xx/=count; xy/=count
    reg=np.eye(p)*alpha; reg[0,0]=0
    return np.linalg.solve(xx+reg,xy)

def metrics_for_section(root,out,sid,ctx,model,coef,genes):
    X,idx=loadcache(out,sid,ctx); X=design(X,model)
    stats={t:np.zeros((5,280),np.float64) for t in ('raw','clean')}
    N=0
    for lo,hi,raw,clean in y_chunks(root,sid,idx,genes):
        pred=X[lo:hi]@coef
        for t,y in [('raw',raw),('clean',clean)]:
            st=stats[t]
            st[0]+=y.sum(axis=0); st[1]+=(y*y).sum(axis=0)
            st[2]+=pred.sum(axis=0); st[3]+=(pred*pred).sum(axis=0)
            st[4]+=(y*pred).sum(axis=0)
        N+=len(pred)
    rows=[]
    for t,s in stats.items():
        cov=s[4]-s[0]*s[2]/N
        vy=s[1]-s[0]**2/N; vp=s[3]-s[2]**2/N
        valid=(vy>1e-10)&(vp>1e-10)
        r=np.full(280,np.nan)
        r[valid]=np.clip(cov[valid]/np.sqrt(vy[valid]*vp[valid]),-0.999999,0.999999)
        z=np.arctanh(r)
        for j,g in enumerate(genes):
            rows.append(dict(sample=sid,patient={'TENX191':'P1','TENX192':'P1','TENX193':'P1','TENX195':'P2','TENX196':'P2','TENX197':'P3','TENX198':'P3','TENX199':'P3','TENX200':'P4','TENX201':'P4','TENX202':'P4'}[sid],context_um=ctx,model=model,training_target='__TRAIN_TARGET__',evaluation=t,gene=g,n_cells=N,pearson_r=float(r[j]),fisher_z=float(z[j])))
    return rows

def fit(root,out,result_dir,alpha):
    """One expression read per section per stage, NOT per model/target/context."""
    check_freeze(root)
    if result_dir.exists(): raise RuntimeError(f'REFUSE_OVERWRITE {result_dir}')
    genes=[g.strip() for g in expected(root/'runs/frozen_primary2025/gene_axis_280.txt').read_text().splitlines() if g.strip()]
    if len(genes)!=280 or len(set(genes))!=280: raise RuntimeError('INVALID_FROZEN_GENES')
    for sid in ALL:
        z24=np.load(expected(out/f'{sid}_ctx24.npz'),allow_pickle=False)
        z128=np.load(expected(out/f'{sid}_ctx128.npz'),allow_pickle=False)
        if not np.array_equal(z24['raw_row'],z128['raw_row']): raise RuntimeError((sid,'CTX_CELL_UNIVERSE_DIFFERS'))
        if not np.array_equal(z24['cell_id'],z128['cell_id']): raise RuntimeError((sid,'CTX_CELL_ID_DIFFERENCE'))
        if not (float(z24['prob_min'])==float(z128['prob_min'])==0.6):
            raise RuntimeError((sid,'PROB_THRESHOLD_MISMATCH'))
    keys=[(ctx,model) for ctx in (24,128) for model in ('M1','M2','M3')]
    xx={k:np.zeros(({'M1':5,'M2':9,'M3':14}[k[1]],)*2,np.float64) for k in keys}
    xy={k:{t:np.zeros(({'M1':5,'M2':9,'M3':14}[k[1]],280),np.float64) for t in ('raw','clean')} for k in keys}
    n_train=0
    for sid in TRAIN:
        data={ctx:loadcache(out,sid,ctx) for ctx in (24,128)}
        rowidx=data[24][1]
        for lo,hi,raw,clean in y_chunks(root,sid,rowidx,genes):
            n_train+=len(raw)
            for ctx,model in keys:
                V=design(data[ctx][0][lo:hi],model)
                xx[ctx,model]+=V.T@V
                xy[ctx,model]['raw']+=V.T@raw
                xy[ctx,model]['clean']+=V.T@clean
        print('TRAIN_ACCUMULATED_ONCE',sid,'cumulative_cells',n_train,flush=True)
    if n_train<=0: raise RuntimeError('EMPTY_TRAIN')
    coefs={}
    for ctx,model in keys:
        p=xx[ctx,model].shape[0]
        reg=np.eye(p)*alpha; reg[0,0]=0
        A=xx[ctx,model]/n_train+reg
        for t in ('raw','clean'):
            coefs[ctx,t,model]=np.linalg.solve(A,xy[ctx,model][t]/n_train)
    for t in ('raw','clean'):
        if not np.allclose(coefs[24,t,'M1'],coefs[128,t,'M1'],rtol=0,atol=1e-10):
            raise RuntimeError(('M1_CONTEXT_INVARIANT_FAILED',t))
    results=[]
    # Test labels are read only AFTER all coefficients are fixed.
    for sid in VAL+TEST:
        data={ctx:loadcache(out,sid,ctx) for ctx in (24,128)}
        rowidx=data[24][1]; N=0
        accum={(k,ev):np.zeros((5,280),np.float64)
               for k in coefs for ev in ('raw','clean')}
        for lo,hi,raw,clean in y_chunks(root,sid,rowidx,genes):
            N+=len(raw)
            for ctx,target,model in coefs:
                pred=design(data[ctx][0][lo:hi],model)@coefs[ctx,target,model]
                for ev,Y in (('raw',raw),('clean',clean)):
                    st=accum[(ctx,target,model),ev]
                    st[0]+=Y.sum(axis=0); st[1]+=(Y*Y).sum(axis=0)
                    st[2]+=pred.sum(axis=0); st[3]+=(pred*pred).sum(axis=0)
                    st[4]+=(Y*pred).sum(axis=0)
        for (ctx,target,model),ev in accum:
            st=accum[(ctx,target,model),ev]
            cov=st[4]-st[0]*st[2]/N
            vy=st[1]-st[0]**2/N; vp=st[3]-st[2]**2/N
            valid=(vy>1e-10)&(vp>1e-10)
            r=np.full(280,np.nan)
            r[valid]=np.clip(cov[valid]/np.sqrt(vy[valid]*vp[valid]),-0.999999,0.999999)
            z=np.arctanh(r)
            patient={'TENX191':'P1','TENX192':'P1','TENX193':'P1','TENX195':'P2','TENX196':'P2','TENX197':'P3','TENX198':'P3','TENX199':'P3','TENX200':'P4','TENX201':'P4','TENX202':'P4'}[sid]
            for j,g in enumerate(genes):
                results.append(dict(sample=sid,patient=patient,context_um=ctx,model=model,
                                    training_target=target,evaluation=ev,gene=g,n_cells=N,
                                    pearson_r=float(r[j]),fisher_z=float(z[j])))
        print('EVALUATED_ONCE',sid,'cells',N,flush=True)
    result_dir.mkdir(parents=True,exist_ok=False)
    detail=pd.DataFrame(results)
    detail.to_csv(result_dir/'gene_section_metrics.csv',index=False)
    patient=detail.groupby(['patient','context_um','model','training_target','evaluation','gene'],as_index=False).agg(mean_section_fisher_z=('fisher_z','mean'),n_sections=('sample','nunique'),n_cells_sum=('n_cells','sum'))
    patient.to_csv(result_dir/'patient_gene_metrics.csv',index=False)
    patient.groupby(['patient','context_um','model','training_target','evaluation'],as_index=False).agg(mean_gene_fisher_z=('mean_section_fisher_z','mean'),median_gene_fisher_z=('mean_section_fisher_z','median'),valid_genes=('mean_section_fisher_z','count')).to_csv(result_dir/'patient_summary.csv',index=False)
    np.savez_compressed(result_dir/'coefs.npz',**{f'ctx{ctx}_{t}_{m}':c for (ctx,t,m),c in coefs.items()})
    (result_dir/'protocol.json').write_text(json.dumps({'split':{'train':TRAIN,'val':VAL,'test':TEST},'targets':['raw','clean'],'contexts_um':[24,128],
                'models':{'M1':FNAME[:5],'M2':FNAME[:9],'M3':FNAME},'alpha_normalized_XtX':alpha,
                'high_conf_prob_min':0.6,'patient_sampling_unit':'donor',
                'important':'restricted exact H&E mapped recipient subset; score H&E-image models on identical cells before direct performance comparisons',
                'read_optimization':'one H5AD pass per section for fitting / per section for evaluation'},indent=2)+'\n')
    print('CPU_MORPHOLOGY_BASELINE_COMPLETE',result_dir)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',required=True,choices=['preflight','extract','fit'])
    ap.add_argument('--root',type=Path,default=Path('.'))
    ap.add_argument('--features',type=Path,default=Path('runs/independent_histo/morph_neigh_v1/features'))
    ap.add_argument('--result',type=Path,default=Path('runs/independent_histo/morph_neigh_v1/results'))
    ap.add_argument('--sample',choices=ALL)
    ap.add_argument('--prob-min',type=float,default=0.6)
    ap.add_argument('--alpha',type=float,default=0.001)
    a=ap.parse_args()
    if not 0<=a.prob_min<=1 or not a.alpha>0: raise RuntimeError('INVALID_NUMERIC_PARAMETER')
    if a.mode=='extract' and a.prob_min!=0.6: raise RuntimeError('PROB_THRESHOLD_FROZEN_TO_0.6_FOR_THIS_V1')
    if a.mode=='preflight': preflight(a.root)
    elif a.mode=='extract':
        if not a.sample: raise RuntimeError('--sample required for extract')
        extract(a.root,a.features,a.sample,a.prob_min)
    else: fit(a.root,a.features,a.result,a.alpha)

if __name__=='__main__': main()

