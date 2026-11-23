#!/usr/bin/env python3
"""Download a real multi-donor HEST Xenium subset after HF access is accepted.

We intentionally query HEST metadata at run time rather than hard-code slide IDs.
"""
import argparse, pandas as pd
from huggingface_hub import hf_hub_download, snapshot_download

p=argparse.ArgumentParser()
p.add_argument("--out",default="data/hest")
p.add_argument("--organ",default=None)
p.add_argument("--oncotree",default=None)
p.add_argument("--max-samples",type=int,default=12)
p.add_argument("--metadata",default="HEST_v1_3_0.csv")
a=p.parse_args()
repo="MahmoodLab/hest"
meta_path=hf_hub_download(repo,a.metadata,repo_type="dataset")
m=pd.read_csv(meta_path)
# HEST metadata schemas have evolved; detect technology column conservatively.
tech_col=next((c for c in ["technology","tech","st_technology"] if c in m.columns),None)
if tech_col is None:
    raise RuntimeError(f"Could not locate technology column in {list(m.columns)}")
sel=m[m[tech_col].astype(str).str.contains("xenium",case=False,na=False)].copy()
if a.organ is not None and "organ" in sel: sel=sel[sel.organ.astype(str).str.lower()==a.organ.lower()]
if a.oncotree is not None and "oncotree_code" in sel: sel=sel[sel.oncotree_code.astype(str)==a.oncotree]
# Prefer unique patients/donors where metadata exposes them.
donor_col=next((c for c in ["patient","patient_id","donor_id","donor"] if c in sel.columns),None)
if donor_col:
    sel=sel.sort_values("id").drop_duplicates(donor_col,keep="first")
sel=sel.head(a.max_samples)
if len(sel)==0: raise RuntimeError("Query selected zero Xenium samples")
print(sel[[c for c in ["id",tech_col,"organ","oncotree_code",donor_col] if c]].to_string(index=False))
patterns=[]
for sid in sel.id.astype(str):
    # Official HEST naming is id-based across wsis/st/transcripts/xenium_seg/metadata.
    patterns += [f"wsis/{sid}*",f"st/{sid}*",f"transcripts/{sid}*",f"xenium_seg/{sid}*",f"metadata/{sid}*"]
patterns += [a.metadata]
snapshot_download(repo_id=repo,repo_type="dataset",local_dir=a.out,allow_patterns=patterns)
sel.to_csv(f"{a.out}/selected_xenium_samples.csv",index=False)
