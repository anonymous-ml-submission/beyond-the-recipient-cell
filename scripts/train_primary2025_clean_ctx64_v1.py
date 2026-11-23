#!/usr/bin/env python3
"""Guarded corrected-target ContextRegressor baseline using frozen RAW outer path.

Requires execution from project repository root and a distinct configuration
whose only difference from A_raw_ctx64/config.yaml is data.target: clean.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from leakage_vso.dataset import _dense_rows
from leakage_vso.train_engine import train_model

RAW_ENTRY = Path('scripts/train_primary2025_context_v1.py')
RAW_CONFIG = Path('runs/primary2025_context_v1/A_raw_ctx64/config.yaml')

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

spec = importlib.util.spec_from_file_location('frozen_raw_context_training', RAW_ENTRY)
assert spec is not None and spec.loader is not None
raw_stage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(raw_stage)

class CleanContextDataset(raw_stage.RawContextDataset):
    """Identical image/augmentation path; adds aligned corrected targets."""
    def __getitem__(self, idx):
        out = super().__getitem__(idx)
        si, li = self.rows[idx]
        sample, gidx = self.samples[si]
        if sample.clean is None:
            raise RuntimeError(f'Missing clean target: {sample.sample_id}')
        vals = _dense_rows(sample.clean.X, [li], gidx)[0]
        clean = torch.from_numpy(vals)
        if clean.shape != out['raw'].shape:
            raise RuntimeError(f'Clean/raw shape mismatch: {sample.sample_id}')
        if not torch.isfinite(clean).all() or (clean < 0).any():
            raise RuntimeError(f'Invalid corrected target: {sample.sample_id}')
        out['clean'] = clean
        return out

def dataset(cfg, genes, ids, split, augment):
    return CleanContextDataset(
        prepared_root=cfg['data']['prepared_root'],
        sample_ids=ids,
        genes=genes,
        target='clean',
        context_um=cfg['data']['context_um'],
        center_um=cfg['data']['center_um'],
        model_mpp=cfg['data']['model_mpp'],
        canvas_um=cfg['data']['canvas_um'],
        split=split,
        augment=augment,
        ring_shuffle=False,
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out')
    ap.add_argument('--preflight-only', action='store_true')
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    original = yaml.safe_load(RAW_CONFIG.read_text())
    edited = yaml.safe_load(RAW_CONFIG.read_text())
    edited['data']['target'] = 'clean'
    if cfg != edited or original['data']['target'] != 'raw':
        raise RuntimeError('Only data.target=clean may differ from frozen RAW CTX64 config')
    assert cfg['data']['context_um'] == 64
    assert cfg['data']['canvas_um'] == 128
    assert cfg['data']['model_mpp'] == 0.5
    assert cfg['model']['method'] == 'context'
    assert cfg['model']['backbone'] == 'resnet18'
    assert cfg['training']['epochs'] == 30
    assert cfg['training']['workers'] == 6
    assert cfg['training']['batch_size'] == 48

    genes = raw_stage.genes_from_cfg(cfg)
    tr = dataset(cfg, genes, raw_stage.TRAIN_IDS, 'train', not args.preflight_only)
    va = dataset(cfg, genes, raw_stage.VAL_IDS, 'val', False)
    if len(tr) != raw_stage.EXPECTED_TRAIN_N or len(va) != raw_stage.EXPECTED_VAL_N:
        raise RuntimeError(f'Split counts changed: train={len(tr)} val={len(va)}')

    if args.preflight_only:
        for name, ds in [('train', tr), ('val', va)]:
            indices = np.linspace(0, len(ds)-1, num=96, dtype=np.int64).tolist()
            dl = DataLoader(Subset(ds, indices), batch_size=16,
                            shuffle=False, num_workers=6)
            n = 0
            changed = 0
            for b in dl:
                assert b['outer'].shape[1:] == (3, 256, 256)
                assert b['raw'].shape[1:] == b['clean'].shape[1:] == (280,)
                assert torch.isfinite(b['clean']).all()
                assert torch.all(b['clean'] >= 0)
                changed += torch.count_nonzero(b['raw'] != b['clean']).item()
                n += b['outer'].shape[0]
            assert n == len(indices)
            if changed == 0:
                raise RuntimeError(f'{name}: checked labels all unchanged: validate correction')
            print(f'{name.upper()} CLEAN 6-WORKER PREFLIGHT: PASS; sampled={n}, changed_values={changed}', flush=True)
        return

    if args.out is None:
        raise RuntimeError('--out required')
    if Path(args.out).exists():
        raise RuntimeError(f'Existing training directory: {args.out}')
    print('CLEAN CTX64 TRAINING', flush=True)
    print('train',len(tr),'val',len(va),'genes',len(genes),'test NONE', flush=True)
    train_model(cfg, tr, va, genes, args.out)
    best = Path(args.out)/'best.pt'
    hist = Path(args.out)/'history.json'
    if not best.is_file() or not hist.is_file():
        raise RuntimeError('Missing training artifacts')
    print('TRAINING COMPLETE', str(best), 'sha256', sha(best), flush=True)

if __name__ == '__main__':
    main()
