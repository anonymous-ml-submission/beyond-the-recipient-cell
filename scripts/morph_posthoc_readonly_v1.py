#!/usr/bin/env python3
"""Read-only per-section and prespecified-marker audit of completed M1/M2/M3.

No retraining, no feature extraction, no results mutation, no gene inference.
All comparisons are paired on existing section-gene metrics. Patient count,
not cell count, determines the independent biological replication count.
"""
from __future__ import annotations
import argparse,hashlib
from pathlib import Path
import pandas as pd
import numpy as np
P1=('TENX191','TENX192','TENX193')
P4=('TENX200','TENX201','TENX202')
MARKERS=('MMP2','SFRP4','CCDC80','CXCL12','FBLN1')
ORTHO=('TRAC','CD3E','IL7R','CD8A','CCL5')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--results',type=Path,default=Path('runs/independent_histo/morph_neigh_v1/results'))
    a=p.parse_args()
    src=a.results/'gene_section_metrics.csv'
    assert src.is_file(),f'STOP: missing {src}'
    data=pd.read_csv(src)
    cols={'sample','patient','gene','context_um','model','training_target','evaluation','fisher_z','n_cells'}
    assert cols <= set(data.columns), f'STOP: missing columns {cols-set(data.columns)}'
    assert data['sample'].isin(P1+P4).all() and len(data)==6*280*2*3*2*2
    assert not data.duplicated(['sample','gene','context_um','model','training_target','evaluation']).any()
    print('INPUT_SHA256',hashlib.sha256(src.read_bytes()).hexdigest())
    print('READ_ONLY: no output files and no training')
    for patient,sids in [('P4',P4),('P1',P1)]:
      for train_target,eval_target in [('raw','raw'),('clean','clean')]:
        q=data[(data['sample'].isin(sids))&(data.training_target==train_target)&(data.evaluation==eval_target)]
        # Within each section-gene, compare identical targets and cell sets.
        u=q.pivot(index=['sample','gene'],columns=['context_um','model'],values='fisher_z')
        assert u.shape==(840,6)
        d=pd.DataFrame(index=u.index)
        d['M2_minus_M1_ctx128']=u[128,'M2']-u[128,'M1']
        d['M3_minus_M2_ctx128']=u[128,'M3']-u[128,'M2']
        d['M2_ctx128_minus_ctx24']=u[128,'M2']-u[24,'M2']
        d['M3_ctx128_minus_ctx24']=u[128,'M3']-u[24,'M3']
        print(f'\n===== {patient} train={train_target} evaluate={eval_target} =====')
        for sid,sec in d.groupby(level='sample'):
          print('SECTION',sid,'n_valid_gene_pairs',sec.notna().all(axis=1).sum(),
            'M2-M1_128',round(sec.M2_minus_M1_ctx128.mean(),6),
            'M3-M2_128',round(sec.M3_minus_M2_ctx128.mean(),6),
            'M2_128-24',round(sec.M2_ctx128_minus_ctx24.mean(),6),
            'fraction_genes_M2_improves',round(float((sec.M2_minus_M1_ctx128>0).mean()),4))
        for panel,gg in [('all_280',None),('CONNECTIVE_5',MARKERS),('INFLAMMATORY_5',ORTHO)]:
          dd=d if gg is None else d[d.index.get_level_values('gene').isin(gg)]
          if gg is not None and len(set(dd.index.get_level_values('gene')))!=len(gg):
             print('PANEL_NOT_COMPLETE',panel,'available',sorted(set(dd.index.get_level_values('gene'))))
          print('PANEL',panel,'n_gene_section_pairs',len(dd),
            'M2-M1_128',round(dd.M2_minus_M1_ctx128.mean(),6),
            'M3-M2_128',round(dd.M3_minus_M2_ctx128.mean(),6),
            'M2_128-24',round(dd.M2_ctx128_minus_ctx24.mean(),6))
    print('\nMORPH_POSTHOC_READONLY_PASS')
    print('CAUTION: P4 and P1 each represent ONE patient; gene/cell counts are not independent patient replicates.')
    print('CAUTION: 128um M2/M3 model selection cannot be changed based on P1 test performance.')
if __name__=='__main__':main()

