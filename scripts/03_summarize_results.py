#!/usr/bin/env python3
"""Read-only complete-marker, complete-section, equal-patient outcome summaries."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

SECTIONS={'P2':('TENX195','TENX196'),'P3':('TENX197','TENX198','TENX199'),'P4':('TENX200','TENX201','TENX202')}
CONNECTIVE={'MMP2','SFRP4','CCDC80','CXCL12','FBLN1'}
INFLAMMATORY={'TRAC','CD3E','IL7R','CD8A','CCL5'}
GENES=CONNECTIVE|INFLAMMATORY

def check_result(path, pair_path):
    assert path.is_file() and path.stat().st_size>0, f'MISSING_RESULT: {path}'
    p=pd.read_csv(pair_path)
    d=pd.read_csv(path)
    assert set(p.gene)==GENES and len(p)>0, f'PAIR_GENE_INCOMPLETE: {pair_path}'
    assert len(d)==len(p)*2, f'RESULT_ROWS {path}: {len(d)} vs {len(p)*2}'
    assert set(d.gene)==GENES and set(d.intervention)=={'blur','inpaint'},path
    expected=p.groupby('gene').size().sort_index()
    for mode in ('blur','inpaint'):
        actual=d[d.intervention==mode].groupby('gene').size().sort_index()
        assert actual.equals(expected),f'{path}: counts mismatch for {mode}'
    cols=['pred_original','pred_source_masked','pred_control_masked','delta_source','delta_control','delta_source_minus_control']
    assert set(cols).issubset(d.columns),f'{path}: missing prediction columns'
    assert np.isfinite(d[cols].to_numpy(dtype=float)).all(),f'{path}: nonfinite predictions'
    assert np.allclose((d.delta_source-d.delta_control).to_numpy(),d.delta_source_minus_control.to_numpy(),atol=1e-6,rtol=1e-5),path
    return d

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',required=True)
    parser.add_argument('--pairs',required=True)
    args=parser.parse_args()
    results=Path(args.results)
    pairroot=Path(args.pairs)
    rows=[]
    for p,sids in SECTIONS.items():
        for sid in sids:
            result=results/p/f'{sid}_ctx128_intervention.csv'
            pair=pairroot/f'{sid}_pairs.csv'
            d=check_result(result,pair)
            d['patient']=p
            d['section']=sid
            d['panel']=np.where(d.gene.isin(CONNECTIVE),'Connective_to_Inflammatory','Inflammatory_to_Connective')
            rows.append(d)
            print(f'COMPLETE {p} {sid} pairs={len(d)//2} blur={len(d)//2} inpaint={len(d)//2}',flush=True)
    D=pd.concat(rows,ignore_index=True)
    # Analysis unit is patient. Genes and sections are averaged within patients.
    gene_section=(D.groupby(['patient','section','panel','intervention','gene'],as_index=False)
                  .agg(effect=('delta_source_minus_control','mean'),n_pairs=('gene','size'),
                       source_delta=('delta_source','mean'),control_delta=('delta_control','mean')))
    section_panel=gene_section.groupby(['patient','section','panel','intervention'],as_index=False).agg(effect=('effect','mean'),genes=('gene','nunique'))
    patient_panel=section_panel.groupby(['patient','panel','intervention'],as_index=False).agg(effect=('effect','mean'),sections=('section','nunique'))
    cohort=patient_panel.groupby(['panel','intervention'],as_index=False).agg(equal_patient_mean=('effect','mean'),n_patients=('patient','nunique'),positive_patients=('effect',lambda v:int((v>0).sum())))
    assert (cohort.n_patients==3).all()
    results.mkdir(parents=True,exist_ok=True)
    for name,data in [('gene_section_effects.csv',gene_section),('section_panel_effects.csv',section_panel),('patient_panel_effects.csv',patient_panel),('cohort_equal_patient_effects.csv',cohort)]:
        path=results/name
        assert not path.exists(),f'STOP: no overwrite {path}'
        data.to_csv(path,index=False)
    print('=== PATIENT EFFECTS (includes negative values) ===')
    print(patient_panel.to_string(index=False))
    print('=== EQUAL PATIENT SUMMARY (n=3) ===')
    print(cohort.to_string(index=False))
    print('DONOR_INTERVENTION_ALL_EIGHT_RESULTS_AUDITED')
if __name__=='__main__': main()
