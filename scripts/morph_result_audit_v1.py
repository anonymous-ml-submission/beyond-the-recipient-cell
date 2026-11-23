#!/usr/bin/env python3
"""Read-only audit/report of completed M1/M2/M3 morphology results, P4+P1.
Never trains, recalculates features, or modifies frozen artifacts.
"""
from __future__ import annotations
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd

EXPECTED_SPLIT = {
  'train': ['TENX195','TENX196','TENX197','TENX198','TENX199'],
  'val': ['TENX200','TENX201','TENX202'],
  'test': ['TENX191','TENX192','TENX193'],
}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--results',type=Path,default=Path('runs/independent_histo/morph_neigh_v1/results'))
    args=ap.parse_args()
    root=args.results
    for f in ('patient_summary.csv','patient_gene_metrics.csv',
              'gene_section_metrics.csv','protocol.json','coefs.npz'):
        assert (root/f).is_file(),f'STOP: missing {root/f}'
    protocol=json.loads((root/'protocol.json').read_text())
    assert protocol['split']==EXPECTED_SPLIT, 'STOP: incorrect donor split'
    assert protocol['contexts_um']==[24,128]
    assert protocol['alpha_normalized_XtX']==0.001
    d=pd.read_csv(root/'patient_summary.csv')
    required=['patient','context_um','model','training_target','evaluation',
              'mean_gene_fisher_z','valid_genes']
    assert set(required).issubset(d.columns)
    assert len(d)==48,f'STOP: expected 48 patient model summaries; got {len(d)}'
    assert set(d.patient)=={'P1','P4'}
    assert set(d.context_um)=={24,128}
    assert set(d.model)=={'M1','M2','M3'}
    assert set(d.training_target)=={'raw','clean'}
    assert set(d.evaluation)=={'raw','clean'}
    assert not d.duplicated(required[:5]).any(),'STOP: duplicate model groups'
    assert d.valid_genes.between(0,280).all()
    with np.load(root/'coefs.npz',allow_pickle=False) as cf:
        expected=[f'ctx{ctx}_{target}_{model}' for ctx in (24,128)
                  for target in ('raw','clean') for model in ('M1','M2','M3')]
        assert set(expected)==set(cf.files),'STOP: wrong coefficient set'
        assert all(np.isfinite(cf[k]).all() for k in expected),'STOP: nonfinite coefficients'
        assert all(np.array_equal(cf[f'ctx24_{t}_M1'],cf[f'ctx128_{t}_M1'])
                   or np.allclose(cf[f'ctx24_{t}_M1'],cf[f'ctx128_{t}_M1'],rtol=0,atol=1e-10)
                   for t in ('raw','clean')),'STOP: M1 context invariance failed'
    print('MORPH_RESULT_CONTRACT_PASS')
    print('SCOPE: P4 VALIDATION and P1 HELD-OUT TEST, NOT four independent test donors')
    print('METRIC: mean per-section per-gene Fisher-z averaged across genes; tanh(mean-z) is equivalent r')
    print('TRAINING: fixed alpha=0.001, not validation selected')
    bad=d.loc[d.valid_genes<280,required]
    if len(bad):
        print('NOTE: Some groups have fewer than 280 valid genes:')
        print(bad.to_string(index=False))
    for patient,role in [('P4','VALIDATION'),('P1','HELD-OUT TEST')]:
      for training_target in ('raw','clean'):
        for evaluation in ('raw','clean'):
          q=d.loc[(d.patient==patient)&(d.training_target==training_target)&(d.evaluation==evaluation)]
          p=q.pivot(index='model',columns='context_um',values='mean_gene_fisher_z').reindex(index=['M1','M2','M3'],columns=[24,128])
          assert np.isfinite(p.to_numpy()).all(), 'STOP: nonfinite summary'
          assert np.isclose(p.loc['M1',24],p.loc['M1',128],atol=1e-8,rtol=0), 'STOP: M1 context mismatch'
          print(f'\n===== {patient} {role} / TRAIN={training_target} / EVAL={evaluation} =====')
          for model in ('M1','M2','M3'):
              z24=float(p.loc[model,24]);z128=float(p.loc[model,128])
              print(f'{model}: z24={z24:.6f} z128={z128:.6f} equiv_r24={np.tanh(z24):.6f} equiv_r128={np.tanh(z128):.6f}')
          print('GAINS (FISHER-Z):')
          print('M2_minus_M1_ctx24=',round(p.loc['M2',24]-p.loc['M1',24],6),
                'M3_minus_M2_ctx24=',round(p.loc['M3',24]-p.loc['M2',24],6))
          print('M2_minus_M1_ctx128=',round(p.loc['M2',128]-p.loc['M1',128],6),
                'M3_minus_M2_ctx128=',round(p.loc['M3',128]-p.loc['M2',128],6))
          print('M3_ctx128_minus_ctx24=',round(p.loc['M3',128]-p.loc['M3',24],6))
    print('\nMORPH_REPORT_COMPLETE; READ_ONLY')

if __name__=='__main__': main()

