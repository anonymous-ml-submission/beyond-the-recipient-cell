#!/usr/bin/env python3
"""Non-destructive GHIST-vs-Primary2025 compatibility gate. Does not train or install.

Official GHIST v1 trains/validates on regions of a single H&E slide;
it is NOT donor-split out of the box. Audit that before adapting/training.
"""
from pathlib import Path
import argparse, hashlib, json, subprocess, sys

SPLITS = {
    "train": ["TENX195", "TENX196", "TENX197", "TENX198", "TENX199"],
    "val": ["TENX200", "TENX201", "TENX202"],
    "test": ["TENX191", "TENX192", "TENX193"],
}

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', type=Path, default=Path('.'))
    ap.add_argument('--repo', type=Path, required=True, help='Existing official GHIST clone')
    ap.add_argument('--out', type=Path, default=None, help='Optional report path; must not exist')
    a=ap.parse_args()
    project=a.project.resolve(); repo=a.repo.resolve()
    issues=[]; report={'intent':'READ_ONLY_GHIST_COMPATIBILITY_AUDIT','repo':str(repo), 'project':str(project), 'frozen_split':SPLITS}
    if not (repo/'train.py').is_file() or not (repo/'dataio/dataset_input.py').is_file():
        issues.append('GHIST source files missing. Clone official SydneyBioX/GHIST first.')
    else:
        script=(repo/'train.py').read_text()
        ds=(repo/'dataio/dataset_input.py').read_text()
        report['train_sha256']=sha(repo/'train.py')
        report['dataset_sha256']=sha(repo/'dataio/dataset_input.py')
        required={'--config_file':script,'--fold_id':script,'--gpu_id':script,
                  'fp_expr':script,'DataProcessing':script,'divisions_fold':ds,
                  'fp_nuc_seg':ds,'fp_nuc_sizes':ds,'fp_hist':ds,'fp_cell_type':ds}
        for name, body in required.items():
            if name not in body: issues.append(f'Official-interface assumption changed: {name}')
        # This is a project-specific requirement: native region-fold single-slide != patient split.
        issues.append('BLOCK: official GHIST v1 loader uses a single fp_hist/fp_nuc_seg and regions_val within-slide; a tested multi-slide, donor-held-out loader is REQUIRED before training')
        if 'classes.index("B")' in script or "classes.index('B')" in script:
            issues.append('BLOCK: optional GHIST neighborhood head expects B/Myeloid/Malignant; frozen H&E coarse classes do not supply those distinctions')
        try:
            p=subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'],check=True,capture_output=True,text=True)
            report['commit']=p.stdout.strip()
        except (OSError, subprocess.CalledProcessError): issues.append('Cannot determine GHIST git commit')
        try:
            demo=json.loads((repo/'configs/config_demo.json').read_text())
            report['official_demo_components']=demo['comps']
            report['official_demo_data_fields']=list(demo['data_sources_train_val'].keys())
            report['official_regions_val']=demo['regions_val']['divisions']
        except Exception as e: issues.append(f'GHIST demo config unavailable/incompatible: {e}')
    frozen=project/'runs/frozen_primary2025/gene_axis_280.txt'
    if not frozen.is_file(): issues.append(f'Missing frozen gene list: {frozen}')
    else:
        genes=[s.strip() for s in frozen.read_text().splitlines() if s.strip()]
        if len(genes)!=280 or len(set(genes))!=280: issues.append('Frozen gene list is not 280 unique genes')
        else: report['gene_axis_sha256']=sha(frozen)
    for split, sids in SPLITS.items():
        for sid in sids:
            p=project/'data/prepared_primary2025/samples'/sid
            for name in ('raw.h5ad','cells.parquet','image_path.txt','pixel_size_um.txt'):
                if not (p/name).is_file(): issues.append(f'Missing {sid}/{name}')
            an=project/'runs/independent_histo'/f'{sid}_cell_annotations.parquet'
            nuc=project/'runs/independent_histo'/f'{sid}_nuclei.parquet'
            if not an.is_file(): issues.append(f'Missing H&E matched annotations: {an}')
            if not nuc.is_file(): issues.append(f'Missing H&E nuclei: {nuc}')
    # Intentional: don't pass the official loader as a donor-valid benchmark.
    report['status']='BLOCKED_FOR_TRAINING' if issues else 'SOURCE_PREFLIGHT_PASS_NOT_TRAIN_READY'
    report['issues']=issues
    print('GHIST COMMIT:',report.get('commit','UNKNOWN'))
    print('STATUS:',report['status'])
    for i in issues: print(' -',i)
    if a.out:
        if a.out.exists(): raise SystemExit(f'REFUSE OVERWRITE {a.out}')
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(report,indent=2)+'\n')
        print('REPORT:',a.out)
    if issues: sys.exit(2)

if __name__=='__main__': main()
