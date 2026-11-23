#!/usr/bin/env python3
"""Attach independently generated cell annotations to a prepared sample with strict ID checks.

This script is deliberately separate from molecular target preparation. Halo/source-intervention
claims require independently defined source/recipient/control cell classes; the repository must not
silently infer them from the same gene whose leakage is being tested.
"""
import argparse
from pathlib import Path
import pandas as pd

p=argparse.ArgumentParser()
p.add_argument('--root',required=True)
p.add_argument('--sample',required=True)
p.add_argument('--annotations',required=True,help='CSV/TSV/Parquet with cell_id and annotation column')
p.add_argument('--annotation-col',required=True)
p.add_argument('--out-col',default='cell_type')
p.add_argument('--allow-missing',action='store_true',help='Engineering only; paper mode requires complete annotation coverage')
a=p.parse_args()

d=Path(a.root)/'samples'/a.sample
cells=pd.read_parquet(d/'cells.parquet'); cells['cell_id']=cells.cell_id.astype(str)
q=Path(a.annotations)
if q.suffix.lower()=='.parquet': ann=pd.read_parquet(q)
elif q.suffix.lower() in {'.csv','.tsv'}: ann=pd.read_csv(q,sep='\t' if q.suffix.lower()=='.tsv' else ',')
else: raise ValueError(f'Unsupported annotation file: {q}')
if 'cell_id' not in ann or a.annotation_col not in ann:
    raise RuntimeError(f'annotations require cell_id and {a.annotation_col}')
ann=ann[['cell_id',a.annotation_col]].copy(); ann['cell_id']=ann.cell_id.astype(str)
if ann.cell_id.duplicated().any():
    dup=ann.loc[ann.cell_id.duplicated(),'cell_id'].head().tolist(); raise RuntimeError(f'duplicate annotation cell IDs: {dup}')
extra=set(ann.cell_id)-set(cells.cell_id)
if extra: raise RuntimeError(f'{len(extra)} annotation cell IDs are absent from prepared sample; wrong sample/ID system?')
merged=cells.merge(ann,on='cell_id',how='left',validate='one_to_one')
miss=merged[a.annotation_col].isna() | (merged[a.annotation_col].astype(str).str.strip()=='')
if miss.any() and not a.allow_missing:
    raise RuntimeError(f'{int(miss.sum())}/{len(merged)} prepared cells lack independent annotation; refusing paper mode')
merged[a.out_col]=merged[a.annotation_col].astype('string')
if a.annotation_col!=a.out_col: merged=merged.drop(columns=[a.annotation_col])
merged.to_parquet(d/'cells.parquet',index=False)
print(f'attached {a.out_col} to {len(merged)-int(miss.sum())}/{len(merged)} cells in {a.sample}')
