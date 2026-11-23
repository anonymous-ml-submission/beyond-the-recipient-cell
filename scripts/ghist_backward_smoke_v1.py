#!/usr/bin/env python3
"""One REAL training-gradient smoke test for pinned GHIST using the existing
patient-safe TileDataset. Does not save model/checkpoints or modify any data.

This is NOT a trained model, and is NOT a frozen image-normalization policy.
The loss terms are the non-neighborhood terms of pinned GHIST train.py.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path('.'))
    ap.add_argument('--sample', choices=['TENX195','TENX196','TENX197','TENX198','TENX199'], default='TENX195')
    ap.add_argument('--device', choices=['cuda','cpu'], default='cuda')
    args = ap.parse_args()
    root = args.root.resolve()
    repo = root/'external_methods/GHIST'
    out = root/'runs/ghist_patient_adapter_v1'/args.sample
    assert out.is_dir(), 'STOP: prepare training slide before gradient test'
    sys.path.insert(0, str(root/'scripts'))
    from ghist_patient_adapter_v1 import TileDataset, validate_repo, CLASS
    validate_repo(repo)
    ds = TileDataset(out)
    assert ds.manifest['role'] == 'train', 'STOP: gradient step must use TRAIN patient only'
    assert ds.manifest['expression_target'] == 'raw_counts'
    assert ds.manifest['repo_commit'] == '917456be305fc82e92293ea272812e79675e821c'
    assert ds.expr.shape[1] == 280

    # Deterministic modest-sized tile. Do not use any test/val annotation or target.
    choices = [i for i, (_, inds) in enumerate(ds.tiles) if 3 <= len(inds) <= 24]
    assert choices, 'STOP: no valid training tile with 3..24 cells'
    choice = min(choices, key=lambda i: len(ds.tiles[i][1]))
    tile = ds[choice]
    import torch
    import torch.nn.functional as F
    sys.path.insert(0,str(repo))
    from model.model import Framework
    torch.manual_seed(2026)
    dev = torch.device('cuda:0' if args.device == 'cuda' else 'cpu')
    if dev.type == 'cuda':
        assert torch.cuda.is_available(), 'STOP: CUDA missing'
    model = Framework(n_classes=len(CLASS),n_genes=280,emb_dim=64,
                      device=dev,n_ref=None,use_avgexp=False,
                      use_celltype=True,use_neighb=False).to(dev)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=0)

    he = torch.as_tensor(tile['he'][None].copy(),dtype=torch.float32,device=dev)
    # Retain the *same provisional smoke-only transform* as the prior forward test.
    # DO NOT use this as a publication training normalization without a frozen policy.
    he = (he - 128.0)/64.0
    masks = torch.as_tensor(tile['mask'][None].copy(),dtype=torch.long,device=dev)
    seg_classes = torch.as_tensor(tile['type_mask'][None].copy(),dtype=torch.long,device=dev)
    class_targets = torch.as_tensor(tile['class'][None].copy(),dtype=torch.long,device=dev)
    expr = torch.as_tensor(np.log1p(tile['expr'][None]).copy(),dtype=torch.float32,device=dev)
    n_cells = torch.tensor([[len(tile['cell_id'])]],dtype=torch.long,device=dev)
    assert torch.all((seg_classes>=0)&(seg_classes<=len(CLASS)))

    optimizer.zero_grad(set_to_none=True)
    outputs = model(he,masks,n_cells,None,class_targets,expr)
    assert len(outputs) == 14, f'STOP: unexpected upstream outputs {len(outputs)}'
    (out_cell_type,out_map,batch_ct_pc,out_expr,
     _,_,out_cell_type_expr,fv_cell_type_expr,
     out_cell_type_gt_expr,fv_cell_type_gt_expr,batch_expr_pc,
     _,_,_) = outputs
    assert out_map.shape == (1,len(CLASS)+1,256,256)
    assert out_expr.shape == (len(tile['cell_id']),280)
    assert torch.equal(batch_ct_pc,class_targets[0]), 'STOP: class alignment changed'
    assert torch.equal(batch_expr_pc,expr[0]), 'STOP: expression alignment changed'

    # Exact non-neighborhood GHIST multi-head losses from pinned train.py.
    losses = {
      'segmentation_CE': F.cross_entropy(out_map,seg_classes),
      'histology_celltype_CE': F.cross_entropy(out_cell_type,batch_ct_pc),
      'expr_celltype_CE': F.cross_entropy(out_cell_type_expr,batch_ct_pc),
      'expr_MSE': F.mse_loss(out_expr,batch_expr_pc),
      'celltype_embedding_cosine': 100 * F.cosine_embedding_loss(
          fv_cell_type_expr,fv_cell_type_gt_expr,
          torch.ones(len(tile['cell_id']),device=dev)),
      'celltype_logits_MSE': F.mse_loss(out_cell_type_expr,out_cell_type_gt_expr),
    }
    for k, v in losses.items():
        assert torch.isfinite(v).item(),f'STOP: nonfinite loss {k}: {v.item()}'
        print('LOSS',k,round(v.item(),6),flush=True)
    total=sum(losses.values())
    total.backward()
    grad_count=0
    for name,param in model.named_parameters():
        if param.grad is None: continue
        assert torch.isfinite(param.grad).all().item(),f'STOP: nonfinite gradient {name}'
        grad_count+=1
    assert grad_count>0, 'STOP: zero parameters received gradients'
    optimizer.step()
    for name,param in model.named_parameters():
        assert torch.isfinite(param).all().item(), f'STOP: nonfinite updated weight {name}'
    print('GHIST_REAL_TRAIN_STEP_GRADIENT_PASS')
    print('SAMPLE',args.sample,'N_CELLS',len(tile['cell_id']),
          'TILE',tile['tile'],'TOTAL_LOSS',round(total.item(),6),
          'PARAMETER_TENSORS_WITH_GRAD',grad_count)
    print('NO_CHECKPOINT_NO_SAVED_WEIGHTS_NO_TRAINING_RUN')
    print('TRAINING_NOT_AUTHORIZED: freeze image normalization, optimizer policy, donor-safe validation, and final loss selection first')

if __name__=='__main__':
    main()
