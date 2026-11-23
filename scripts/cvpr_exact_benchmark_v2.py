#!/usr/bin/env python3
"""Exact held-out GHIST / frozen morphology model comparison.

Never retrains. Fixes gene order, common-gene denominators, common-cell universe,
and verifies RNA targets against the original prepared H5AD (optional but default).
The primary statistic uses the same genes for every model and every section.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

SPLIT = {"val": ("TENX200", "TENX201", "TENX202"),
         "test": ("TENX191", "TENX192", "TENX193")}
MODELS = ("GHIST_adapted", "M1", "M2", "M3")
DIM = {"M1": 5, "M2": 9, "M3": 14}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):
            h.update(b)
    return h.hexdigest()


def correlation(p, y):
    """Gene-wise Pearson on matched cells; no impossible zero-variance scores."""
    assert p.shape == y.shape and p.ndim == 2
    p=np.asarray(p,dtype=np.float64)
    y=np.asarray(y,dtype=np.float64)
    assert np.isfinite(p).all() and np.isfinite(y).all()
    pc=p-p.mean(axis=0); yc=y-y.mean(axis=0)
    pv=(pc*pc).sum(axis=0); yv=(yc*yc).sum(axis=0)
    ok=(pv>1e-10)&(yv>1e-10)
    r=np.full(p.shape[1],np.nan)
    r[ok]=np.clip((pc*yc).sum(axis=0)[ok]/np.sqrt(pv[ok]*yv[ok]),-0.999999,0.999999)
    z=np.arctanh(r)
    mae=np.abs(p-y).mean(axis=0)
    rmse=np.sqrt(((p-y)**2).mean(axis=0))
    return r,z,mae,rmse


def check_targets(root,sid,ids,truth,genes,idx, ncheck=128):
    """Independent GHIST raw-target validation using raw.h5ad and morph row indices."""
    import anndata as ad
    f=root/'data/prepared_primary2025/samples'/sid/'raw.h5ad'
    assert f.is_file(),f'MISSING_H5AD {f}'
    rng=np.random.default_rng(271828)
    pick=np.sort(rng.choice(len(ids),size=min(ncheck,len(ids)),replace=False))
    d=ad.read_h5ad(f,backed='r')
    try:
        names=pd.Index(d.obs_names.astype(str))
        gidx=pd.Index(d.var_names.astype(str)).get_indexer(genes)
        assert (gidx>=0).all(),'RNA_GENE_AXIS_NOT_PRESENT'
        assert np.array_equal(names.to_numpy()[idx[pick]],ids[pick]),'RNA_RAW_ROW_CELL_ID_MISMATCH'
        # Backed H5AD readers may require monotonically increasing row indices.
        ordered=np.argsort(idx[pick])
        rows=idx[pick][ordered]
        raw=d.X[rows,:][:,gidx]
        if hasattr(raw,'toarray'):raw=raw.toarray()
        raw=np.asarray(raw,dtype=np.float32)[np.argsort(ordered)]
        raw=np.log1p(raw)
        err=float(np.max(np.abs(raw-truth[pick])))
        assert err<1e-4,('RAW_TARGET_DISAGREEMENT',sid,err)
        return {'verified_cells':len(pick),'max_log1p_error':err,'h5ad_sha256':None}
    finally:
        d.file.close()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--project',type=Path,default=Path('.'))
    p.add_argument('--ghist-run',type=Path,required=True)
    p.add_argument('--morph-root',type=Path,default=Path('runs/independent_histo/morph_neigh_v1'))
    p.add_argument('--role',choices=SPLIT,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--skip-h5ad-target-check',action='store_true',help='Not recommended; audit records omitted verification')
    a=p.parse_args()
    root=a.project.resolve();gh=a.ghist_run.resolve();mo=a.morph_root.resolve();out=a.out.resolve()
    assert not out.exists(),f'REFUSE_OVERWRITE {out}'
    genes_file=root/'runs/frozen_primary2025/gene_axis_280.txt'
    genes=np.asarray([s.strip() for s in genes_file.read_text().splitlines() if s.strip()],dtype=str)
    assert genes.shape==(280,) and len(set(genes))==280,'FROZEN_GENE_AXIS_INVALID'
    cf=mo/'results/coefs.npz'
    assert cf.is_file(), f'MISSING_MORPH_COEFS {cf}'
    with np.load(cf,allow_pickle=False) as c:
        coefs={m:np.asarray(c[f'ctx128_raw_{m}'],dtype=np.float64) for m in DIM}
    assert all(coefs[m].shape==(DIM[m],280) for m in DIM),'BAD_COEFFICIENT_SHAPES'
    assert all(np.isfinite(c).all() for c in coefs.values())
    ck=gh/'best.pt';assert ck.is_file(),f'MISSING_CKPT {ck}'
    ckhash=sha(ck)
    genehash=sha(genes_file)
    records=[];cover=[];checks=[];inputhash={}
    for sid in SPLIT[a.role]:
        gp=gh/f'{a.role}_full'/sid/'predictions.npz'
        mp=mo/'features'/f'{sid}_ctx128.npz'
        complete=gp.with_name('complete.json')
        for f in (gp,mp,complete):
            assert f.is_file(),f'MISSING {f}'
        meta=json.loads(complete.read_text())
        assert meta['sample']==sid and meta['checkpoint_sha256']==ckhash,('GHIST_CHECKPOINT_HASH_MISMATCH',sid)
        assert meta['predictions_sha256']==sha(gp),('GHIST_PREDICTION_HASH_MISMATCH',sid)
        inputhash[str(gp)]=meta['predictions_sha256'];inputhash[str(mp)]=sha(mp)
        with np.load(gp,allow_pickle=False) as x:
            ggenes=x['genes'].astype(str)
            assert np.array_equal(ggenes,genes),('GHIST_GENE_ORDER_MISMATCH',sid)
            gids=x['cell_id'].astype(str)
            pred=np.asarray(x['pred_log1p'],dtype=np.float32)
            truth=np.asarray(x['raw_log1p'],dtype=np.float32)
            assert x['sample_id'].item()==sid,('GHIST_SAMPLE_MISMATCH',sid)
        with np.load(mp,allow_pickle=False) as z:
            mids=z['cell_id'].astype(str)
            X=z['x'].astype(np.float32)
            ridx=z['raw_row'].astype(np.int64)
            assert z['sample'].item()==sid
            assert list(z['features'])==['bias','recipient_Connective','recipient_Inflammatory','recipient_Neoplastic','recipient_Non-Neoplastic Epithelial','near_frac_Connective','near_frac_Inflammatory','near_frac_Neoplastic','near_frac_Non-Neoplastic Epithelial','log_nuc_density','nearest_Connective','nearest_Inflammatory','nearest_Neoplastic','nearest_Non-Neoplastic Epithelial']
        assert X.shape==(len(mids),14) and len(ridx)==len(mids)
        assert pred.shape==truth.shape==(len(gids),280)
        assert len(set(gids))==len(gids) and len(set(mids))==len(mids)
        assert np.isfinite(pred).all() and np.isfinite(truth).all() and np.isfinite(X).all()
        loc=pd.Index(mids).get_indexer(gids)
        present=loc>=0
        n=int(present.sum())
        assert n>=100,('NO_COMMON_CELLS',sid,n)
        mX=X[loc[present]].astype(np.float64)
        yy=truth[present]
        ids=gids[present]
        if not a.skip_h5ad_target_check:
            checks.append({'sample':sid,**check_targets(root,sid,ids,yy,genes,ridx[loc[present]])})
        cover.append({'sample':sid,'ghist_n':len(gids),'morph_n':len(mids),'matched_n':n,
                      'match_frac_ghist':n/len(gids),'match_frac_morph':n/len(mids),
                      'excluded_ghist_n':len(gids)-n})
        for model in MODELS:
            pp=pred[present] if model=='GHIST_adapted' else mX[:,:DIM[model]]@coefs[model]
            r,z,mae,rmse=correlation(pp,yy)
            for j,g in enumerate(genes):
                records.append({'sample':sid,'role':a.role,'model':model,'gene':g,'n_cells':n,
                                'r':r[j],'z':z[j],'mae_log1p':mae[j],'rmse_log1p':rmse[j]})
        print('MATCHED_SECTION',sid,'COMMON_CELLS',n,'GHIST',len(gids),'MORPH',len(mids),flush=True)
    detail=pd.DataFrame(records)
    table=detail.pivot(index=['sample','gene'],columns='model',values='z').reindex(columns=MODELS)
    # PRIMARY: identical 280-gene *intersection* across every model AND every section.
    mask=np.isfinite(table).all(axis=1).unstack('sample')
    valid=mask.all(axis=1)
    cg=valid.index[valid].tolist()
    assert len(cg)>0,'NO_COMMON_VALID_GENE_UNIVERSE'
    sub=detail.loc[detail.gene.isin(cg)].copy()
    sums=(sub.groupby(['sample','model'],as_index=False)
          .agg(mean_z=('z','mean'),mean_mae_log1p=('mae_log1p','mean'),n_genes=('gene','nunique')))
    assert sums.n_genes.eq(len(cg)).all()
    pt=sums.pivot(index='sample',columns='model',values='mean_z')[list(MODELS)]
    delta=pd.DataFrame({f'GHIST_minus_{m}':pt['GHIST_adapted']-pt[m] for m in DIM})
    print('COMMON_GENES_ALL_MODELS_ALL_SECTIONS',len(cg),'/',len(genes))
    print('SECTION_MEAN_FISHER_Z_ON_FIXED_GENE_UNIVERSE')
    print(pt.to_string(float_format=lambda x:f'{x:.6f}'))
    print('EQUAL_SECTION_MEAN_Z')
    print(pt.mean().to_string(float_format=lambda x:f'{x:.6f}'))
    print('GHIST_MINUS_MORPH_EQUAL_SECTION_DELTA_Z')
    print(delta.mean().to_string(float_format=lambda x:f'{x:.6f}'))
    out.mkdir(parents=True,exist_ok=False)
    detail.to_csv(out/'all_280_gene_section_metrics.csv',index=False)
    sub.to_csv(out/'primary_common_gene_metrics.csv',index=False)
    sums.to_csv(out/'section_summary.csv',index=False)
    pt.to_csv(out/'section_mean_z.csv')
    delta.to_csv(out/'section_paired_delta_z.csv')
    pd.DataFrame(cover).to_csv(out/'cell_coverage.csv',index=False)
    (out/'primary_common_genes.txt').write_text('\n'.join(cg)+'\n')
    (out/'provenance.json').write_text(json.dumps({
        'role':a.role,'patients':'P1 only' if a.role=='test' else 'P4 only',
        'model_train_patients':['P2','P3'],'selection_patient':'P4',
        'ghist_checkpoint_sha256':ckhash,'frozen_gene_axis_sha256':genehash,
        'morph_coefs_sha256':sha(cf),'input_sha256':inputhash,
        'sample_target_spotchecks':checks,'sample_target_spotcheck_skipped':a.skip_h5ad_target_check,
        'common_valid_genes':cg,'common_valid_gene_count':len(cg),
        'estimand':'equal section mean within one patient, per-gene Fisher z on matched cell IDs, fixed common gene set',
        'warning':'matching evaluation does not match training cohorts or supervision; GHIST adapted with known input nucleus masks',
    },indent=2)+'\n')
    print('EXACT_COMMON_COHORT_BENCHMARK_PASS',out)

if __name__=='__main__':main()
