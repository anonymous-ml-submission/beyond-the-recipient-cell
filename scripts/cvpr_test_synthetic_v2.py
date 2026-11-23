#!/usr/bin/env python3
"""Small self-contained tests; no biological source data, pyarrow, GPU or training."""
from pathlib import Path
import tempfile, subprocess, sys, hashlib, json
import numpy as np, pandas as pd
import importlib.util

HERE=Path(__file__).resolve().parent

def call(script,*args):
    cmd=[sys.executable,str(HERE/script),*map(str,args)]
    p=subprocess.run(cmd,capture_output=True,text=True)
    if p.returncode:
        print('STDOUT',p.stdout[-3000:]);print('STDERR',p.stderr[-4000:])
        raise RuntimeError(f'{script}: exit {p.returncode}')
    return p.stdout

def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()

def matched_test(base):
    root=base/'project';genes=[f'G{i:03d}' for i in range(280)]
    gfile=root/'runs/frozen_primary2025/gene_axis_280.txt'
    gfile.parent.mkdir(parents=True)
    gfile.write_text('\n'.join(genes)+'\n')
    morph=root/'runs/independent_histo/morph_neigh_v1'
    (morph/'results').mkdir(parents=True)
    (morph/'features').mkdir(parents=True)
    r=np.random.default_rng(12)
    coefs={f'ctx128_raw_{m}':r.normal(size=(k,280))*0.03 for m,k in [('M1',5),('M2',9),('M3',14)]}
    np.savez_compressed(morph/'results/coefs.npz',**coefs)
    gh=root/'runs/ghist_donor_adapted_v1/seed2026';gh.mkdir(parents=True)
    (gh/'best.pt').write_text('synthetic checkpoint')
    features=['bias']+['recipient_'+x for x in ['Connective','Inflammatory','Neoplastic','Non-Neoplastic Epithelial']]+['near_frac_'+x for x in ['Connective','Inflammatory','Neoplastic','Non-Neoplastic Epithelial']]+['log_nuc_density']+['nearest_'+x for x in ['Connective','Inflammatory','Neoplastic','Non-Neoplastic Epithelial']]
    for sid in ['TENX191','TENX192','TENX193']:
        f=gh/'test_full'/sid;f.mkdir(parents=True)
        n=144
        ids=np.array([f'{sid}_c{i:04d}' for i in range(n)])
        X=r.normal(size=(n,14)).astype(np.float32);X[:,0]=1
        truth=np.maximum(0,r.normal(loc=1.,scale=.5,size=(n,280))).astype('f4')
        pred=(truth*.7+r.normal(size=(n,280))*.3).astype('f4')
        # Same always-zero target gene in all sections should be excluded everywhere.
        truth[:,0]=0
        np.savez_compressed(f/'predictions.npz',genes=np.array(genes),cell_id=ids,
                            sample_id=np.array(sid),pred_log1p=pred,raw_log1p=truth)
        (f/'complete.json').write_text(json.dumps({'sample':sid,'checkpoint_sha256':sha(gh/'best.pt'),
                    'predictions_sha256':sha(f/'predictions.npz')}))
        # Permute order to verify exact ID-based joining rather than row numbers.
        order=r.permutation(n)
        np.savez_compressed(morph/'features'/f'{sid}_ctx128.npz',cell_id=ids[order],
            x=X[order],raw_row=order,features=np.array(features),sample=np.array(sid))
    out=root/'match_test'
    output=call('cvpr_exact_benchmark_v2.py','--project',root,'--ghist-run',gh,
      '--morph-root',morph,'--role','test','--out',out,'--skip-h5ad-target-check')
    assert 'COMMON_GENES_ALL_MODELS_ALL_SECTIONS 279 / 280' in output
    assert (out/'section_paired_delta_z.csv').is_file()
    assert len((out/'primary_common_genes.txt').read_text().splitlines())==279
    # Alter GHIST gene order and ensure hard stop, not silent reordering.
    f=gh/'test_full'/'TENX191'/'predictions.npz'
    with np.load(f) as z: content={k:z[k] for k in z.files}
    content['genes'][0],content['genes'][1]=content['genes'][1],content['genes'][0]
    np.savez_compressed(f,**content)
    # Update prediction checksum to bypass that check and isolate gene-order guard.
    d=f.with_name('complete.json');meta=json.loads(d.read_text());meta['predictions_sha256']=sha(f);d.write_text(json.dumps(meta))
    p=subprocess.run([sys.executable,str(HERE/'cvpr_exact_benchmark_v2.py'), '--project',str(root),
      '--ghist-run',str(gh),'--morph-root',str(morph),'--role','test','--out',str(root/'must_fail'),
      '--skip-h5ad-target-check'],capture_output=True,text=True)
    assert p.returncode and 'GHIST_GENE_ORDER_MISMATCH' in (p.stdout+p.stderr)
    return 'MATCHED_BENCHMARK_AND_WRONG_GENE_ORDER_GUARDS_PASS'

def patient_test(base):
    old=base/'original';new=base/'new';match=base/'match'
    panel1=('MMP2','SFRP4','CCDC80','CXCL12','FBLN1')
    panel2=('TRAC','CD3E','IL7R','CD8A','CCL5')
    pids={'P2':['TENX195','TENX196'],'P3':['TENX197','TENX198','TENX199'],
          'P4':['TENX200','TENX201','TENX202']}
    for pat,sids in pids.items():
        for sid in sids:
            (old/pat).mkdir(parents=True,exist_ok=True)
            (new/'rank1').mkdir(parents=True,exist_ok=True)
            (match/sid).mkdir(parents=True,exist_ok=True)
            rows=[];rnew=[];pairs=[]
            for gene in panel1+panel2:
                for k in range(2):
                    a={'gene':gene,'recipient_cell_id':f'{sid}_recipient{k}',
                       'source_cell_id':f'{sid}_src{k}','control_cell_id':f'{sid}_old{k}'}
                    b={**a,'old_control_cell_id':a['control_cell_id'],
                       'control_cell_id':f'{sid}_decoy{k}', 'matched_decoy_rank':1,
                       'source_cell_type':'HISTO_Connective','control_cell_type':'HISTO_Connective'}
                    pairs.append(b)
                    for mode in ('blur','inpaint'):
                        v=0.12 if gene in panel1 else 0.08
                        rows.append({**a,'intervention':mode,'pred_original':1.5,
                                     'delta_source':v,'delta_control':v-.03,
                                     'delta_source_minus_control':.03})
                        rnew.append({**b,'intervention':mode,'pred_original':1.5,
                                    'delta_source':v,'delta_control':v-.01,
                                    'delta_source_minus_control':.01})
            op=old/pat/f'{sid}_ctx128_intervention.csv';pd.DataFrame(rows).to_csv(op,index=False)
            f=match/sid/'decoy_rank1_pairs.csv';pd.DataFrame(pairs).to_csv(f,index=False)
            fp=new/'rank1'/f'{sid}_same_morph_intervention.csv';pd.DataFrame(rnew).to_csv(fp,index=False)
            (fp.with_suffix('.json')).write_text(json.dumps({'sample':sid,'patient':pat,'rank':1,
              'new_predictions_sha256':sha(fp),'new_pairs_sha256':sha(f),'completed_original_sha256':sha(op)}))
            (match/sid/'coverage.json').write_text(json.dumps({'sample':sid,'patient':pat,
              'original_completed_intervention_csv_sha256':sha(op),
              'coverage_per_rank':{'1':1.},'accepted_per_rank':{'1':len(pairs)},
              'original_unique_frozen_pairs':len(pairs)}))
    result=call('cvpr_patient_evidence_v2.py','--original-root',old,'--decoy-result-root',new,
      '--match-root',match,'--ranks','1','--out',base/'evidence')
    st=pd.read_csv(base/'evidence/patient_uncertainty.csv')
    assert set(st.comparison)=={'ORIGINAL_FULL','ORIGINAL_SUBSET_rank1','SAME_CLASS_rank1','SAME_MINUS_OLD_rank1'}
    assert (st.patients==3).all()
    assert np.allclose(st.loc[st.comparison=='SAME_CLASS_rank1','mean_equal_patient'],.01)
    assert np.allclose(st.loc[st.comparison=='ORIGINAL_FULL','mean_equal_patient'],.03)
    assert np.allclose(st.loc[st.comparison=='SAME_MINUS_OLD_rank1','mean_equal_patient'],-.02)
    assert (base/'evidence/patient_effects.pdf').is_file()
    return 'PAIRED_PATIENT_UNCERTAINTY_AND_FIGURES_PASS'


def decoy_test(base):
    spec=importlib.util.spec_from_file_location('decoy',HERE/'cvpr_same_morph_decoys_v2.py')
    dec=importlib.util.module_from_spec(spec);spec.loader.exec_module(dec)
    # This tests the mathematical gate using an in-memory H&E geometry fixture.
    import pandas as pd
    n=150
    rng=np.random.default_rng(37)
    cells=pd.DataFrame({'cell_id':[f'c{i}' for i in range(n)],
        'x_px':np.concatenate(([50,90,91,89,95],rng.uniform(0,3000,n-5))),
        'y_px':np.concatenate(([50,50,51,49,52],rng.uniform(0,3000,n-5))),
        'polygon_area_um2':np.repeat(45.0,n)})
    ann=pd.DataFrame({'cell_id':cells.cell_id,'cell_type':['HISTO_Inflammatory']+
         ['HISTO_Connective']*(n-1),'annotation_status':['unique_nucleus_inside_cell']*n})
    ncs=pd.DataFrame({'x_px':cells.x_px,'y_px':cells.y_px,
       'histo_type':['Inflammatory']+['Connective']*(n-1),'histo_prob':[.9]*n})
    verts=[]
    for r in cells.itertuples():
        for x,y in ((r.x_px-2,r.y_px-2),(r.x_px+2,r.y_px-2),(r.x_px+2,r.y_px+2),(r.x_px-2,r.y_px+2)):
            verts.append({'cell_id':r.cell_id,'x_px':x,'y_px':y})
    vertdf=pd.DataFrame(verts)
    project=base/'project';sid='TENX195'
    sd=project/'data/prepared_primary2025/samples'/sid
    hist=project/'runs/independent_histo'
    sd.mkdir(parents=True,exist_ok=True);hist.mkdir(parents=True,exist_ok=True)
    (sd/'pixel_size_um.txt').write_text('0.274\n')
    for f in (sd/'cells.parquet',hist/f'{sid}_cell_annotations.parquet',sd/'cell_polygons.parquet',hist/f'{sid}_nuclei.parquet'):
        f.touch()
    orig=base/'original.csv'
    origrows=[]
    for g in ('MMP2','SFRP4'):
        for intervention in ('blur','inpaint'):
            origrows.append({'gene':g,'recipient_cell_id':'c0','source_cell_id':'c1','control_cell_id':'c4',
                             'intervention':intervention})
    pd.DataFrame(origrows).to_csv(orig,index=False)
    original=pd.read_parquet
    def fake_read_parquet(path,*args,**kwargs):
        v=str(path)
        if v.endswith('cell_annotations.parquet'):return ann.copy()
        if v.endswith('cell_polygons.parquet'):return vertdf.copy()
        if v.endswith('_nuclei.parquet'):return ncs.copy()
        if v.endswith('cells.parquet'):return cells.copy()
        raise RuntimeError(v)
    pd.read_parquet=fake_read_parquet
    try:
        from argparse import Namespace
        opts=Namespace(project=project,sample=sid,original=orig,out=base/'out',
             n_decoys=2,prob_min=.6,max_distance_um=40.,distance_caliper_um=5.,
             log_area_caliper=.7,log_d5_caliper=.7)
        dec.run(opts)
    finally:
        pd.read_parquet=original
    cov=json.loads((base/'out/coverage.json').read_text())
    assert cov['original_unique_frozen_pairs']==2
    assert cov['accepted_per_rank']['1']==2
    pair=pd.read_csv(base/'out/decoy_rank1_pairs.csv')
    assert (pair.source_cell_type==pair.control_cell_type).all()
    assert not pair.control_cell_id.isin(['c0','c1','c4']).any()
    return 'OUTCOME_BLIND_GEOMETRY_DECOY_MATCH_PASS'


def gpu_runner_contract_test(base):
    import torch
    root=base/'project';sid='TENX195';pat='P2'
    genes=[f'G{i:03d}' for i in range(280)]
    gf=root/'runs/frozen_primary2025/gene_axis_280.txt'
    gf.parent.mkdir(parents=True,exist_ok=True)
    gf.write_text('\n'.join(genes)+'\n')
    ck=root/'runs/primary2025_donor_rotation_v1/test_P2/raw_ctx128/best.pt'
    ck.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'cfg':{'data':{'target':'raw','context_um':128,'train_samples':['TENX197']},
                       'model':{'method':'context'}},'genes':genes,'model':{}},ck)
    scripts=root/'scripts';scripts.mkdir(parents=True)
    scripts.joinpath('source_cell_intervention.py').write_text("""
import argparse,pandas as pd
p=argparse.ArgumentParser()
for name in ('checkpoint','root','sample','pairs','polygons','out','device'):
 p.add_argument('--'+name,required=True)
p.add_argument('--modes',nargs='+',default=['blur','inpaint'])
a=p.parse_args()
rows=[]
for r in pd.read_csv(a.pairs).to_dict('records'):
 for mode in a.modes:
  rows.append({**r,'intervention':mode,'pred_original':1.5,'delta_source':0.2,
       'delta_control':0.1,'delta_source_minus_control':0.1})
pd.DataFrame(rows).to_csv(a.out,index=False)
""")
    (root/'data/prepared_primary2025/samples'/sid).mkdir(parents=True)
    (root/'data/prepared_primary2025/samples'/sid/'cell_polygons.parquet').touch()
    orig=root/'runs/independent_histo/donor_source_control_rotation_original_v1'/pat/f'{sid}_ctx128_intervention.csv'
    orig.parent.mkdir(parents=True)
    original=[]
    for mode in ('blur','inpaint'):
       original.append({'gene':'MMP2','recipient_cell_id':'r0','source_cell_id':'s0',
                        'control_cell_id':'o0','intervention':mode,'pred_original':1.5,
                        'delta_source':0.2,'delta_control':0.02,'delta_source_minus_control':.18})
    pd.DataFrame(original).to_csv(orig,index=False)
    mdir=root/'runs/independent_histo/same_morph_decoys_v2'/sid
    mdir.mkdir(parents=True)
    pfile=mdir/'decoy_rank1_pairs.csv'
    pd.DataFrame([{'gene':'MMP2','recipient_cell_id':'r0','source_cell_id':'s0',
       'control_cell_id':'d0','old_control_cell_id':'o0','matched_decoy_rank':1,
       'source_cell_type':'HISTO_Connective','control_cell_type':'HISTO_Connective',
       'distance_error_um':2}]).to_csv(pfile,index=False)
    (mdir/'coverage.json').write_text(json.dumps({'sample':sid,'patient':pat,
     'prespecified_calipers':{'distance_error_um':5},
     'original_completed_intervention_csv_sha256':sha(orig)}))
    command=['--project',str(root),'--sample',sid,'--rank','1',
              '--match-root',str(mdir.parent),
              '--original-root',str(orig.parent.parent),
              '--out-root',str(root/'new')]
    output=call('cvpr_decoy_gpu_runner_v2.py',*command)
    assert 'SAME_MORPH_REAL_IMAGE_INTERVENTION_PASS' in output
    out=root/'new/rank1/TENX195_same_morph_intervention.csv'
    assert out.is_file() and out.with_suffix('.json').is_file()
    return 'DECoy_GPU_RUNNER_FROZEN_CHECKPOINT_AND_REPLAY_GUARD_PASS'


def main():
    with tempfile.TemporaryDirectory() as t:
        root=Path(t)
        print(matched_test(root/'matched'),flush=True)
        print(patient_test(root/'patient'),flush=True)
        print(decoy_test(root/'decoy'),flush=True)
        print(gpu_runner_contract_test(root/'runner'),flush=True)
    print('ALL_SYNTHETIC_CONTRACT_TESTS_PASS')

if __name__=='__main__':main()
