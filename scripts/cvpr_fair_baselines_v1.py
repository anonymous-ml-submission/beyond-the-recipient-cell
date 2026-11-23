#!/usr/bin/env python3
"""CVPR_Bio2 independent pilot: frozen-cell, source-safe expression baselines.

No original data/config/code modification. Official GHIST/sCellST training is NOT
implemented or misrepresented here. Strict official-prediction ingestion is offered.

Usage: python cvpr_fair_baselines_v1.py -h
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BIO_CLASSES = ("Neoplastic", "Non-Neoplastic Epithelial", "Inflammatory", "Connective")
CONTEXTS = (24, 128)
EXPECTED_GENE_SHA = '14947279372641dafffe7ae755b9da7ec3b361786c4274ec554e226fdde98713'
EXPECTED_CELLS = {
    'TENX191':209467,'TENX192':375529,'TENX193':254480,
    'TENX195':380739,'TENX196':358519,'TENX197':308531,
    'TENX198':373150,'TENX199':150145,
    'TENX200':192898,'TENX201':335709,'TENX202':201955,
}
ROLE_KEYS = ("train_sample_ids", "val_sample_ids", "test_sample_ids")
DEFAULT_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)


def fail(message):
    raise RuntimeError("STOP: " + str(message))


def require(condition, message):
    if not condition:
        fail(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''):
            h.update(b)
    return h.hexdigest()


def stable_int(*values):
    d = hashlib.sha256('|'.join(map(str, values)).encode()).digest()
    return int.from_bytes(d[:8], 'little')


def save_json_exclusive(path, data):
    with Path(path).open('x') as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write('\n')


def path_from_cfg(value, project):
    p = Path(value).expanduser()
    return p if p.is_absolute() else project / p


def config_contract(project, config, prepared_override=None, genes_override=None):
    import yaml
    config = Path(config).resolve()
    require(config.is_file(), f'configuration not found: {config}')
    cfg = yaml.safe_load(config.read_text())
    d = cfg['data']
    require(float(d['canvas_um']) == 128 and float(d['model_mpp']) == .5,
            'requires original 128-um/0.5-um-per-pixel fixed-canvas protocol')
    require(int(cfg['seed']) == 2026, 'unrecognized original seed')
    require(str(cfg['model']['method']) == 'context', 'expected context model config')
    roles = {k: list(map(str, d[k])) for k in ROLE_KEYS}
    require(all(len(v) > 0 and len(v) == len(set(v)) for v in roles.values()), 'empty/duplicated sample ids')
    require(len(set(sum(roles.values(), []))) == sum(map(len, roles.values())), 'patient/sample split overlap')
    expected = {'TENX191','TENX192','TENX193','TENX195','TENX196',
                'TENX197','TENX198','TENX199','TENX200','TENX201','TENX202'}
    require(set(sum(roles.values(), [])) == expected, 'sample universe must be the frozen 11 Primary2025 sections')
    donor_map=donor_mapping(expected)
    by_role={k:set(donor_map[s] for s in ids) for k,ids in roles.items()}
    require(len(set().union(*by_role.values())) == sum(len(v) for v in by_role.values()),
            'DONOR LEAKAGE: same patient appears in multiple train/val/test roles')
    gene_path = path_from_cfg(genes_override or d['genes'], project)
    require(gene_path.is_file(), f'gene axis missing: {gene_path}')
    genes = [x.strip() for x in gene_path.read_text().splitlines() if x.strip()]
    require(len(genes) == 280 and len(set(genes)) == 280, 'require exact 280 unique gene axis')
    require(sha(gene_path)==EXPECTED_GENE_SHA, 'gene-axis SHA differs from locked Primary2025')
    prepared = path_from_cfg(prepared_override or d['prepared_root'], project)
    require(prepared.is_dir(), f'prepared_root not found: {prepared}')
    return cfg, roles, genes, prepared, gene_path


def annotation_path(template, sid, project):
    require('{sid}' in template, '--morph-template must contain literal {sid}')
    return path_from_cfg(template.replace('{sid}', sid), project)


def load_ann(path):
    require(path.is_file(), f'HoVerNet annotation file missing: {path}')
    if path.suffix.lower() in ('.parquet', '.pq'):
        a = pd.read_parquet(path)
    else:
        a = pd.read_csv(path, low_memory=False)
    require({'cell_id', 'hovernet_type', 'annotation_status'} <= set(a.columns),
            f'{path}: needs cell_id,hovernet_type,annotation_status from independent HoVer-Net')
    a = a[a.annotation_status.eq('unique_nucleus_inside_cell') &
          a.hovernet_type.isin(BIO_CLASSES)].copy()
    a['cell_id'] = a['cell_id'].astype(str)
    require(a.cell_id.is_unique, f'{path}: duplicate exact mapped cell_ids')
    require(len(a), f'{path}: no exact mapped biological cells')
    return a


def read_freeze(frozen):
    frozen = Path(frozen).resolve()
    p = frozen / 'manifest.json'
    require(p.is_file(), f'manifest missing: {p}')
    data = json.loads(p.read_text())
    require(data.get('version') == 'cvpr-fair-v1', f'unknown manifest version: {p}')
    for sid, rec in data['samples'].items():
        sel = frozen / rec['selection_file']
        require(sel.is_file() and sha(sel) == rec['selection_sha256'], f'selection altered: {sel}')
    return frozen, data


def read_selection(frozen, data, sid):
    rec = data['samples'][sid]
    df = pd.read_csv(frozen / rec['selection_file'], dtype={'cell_id': str})
    require({'row_index', 'cell_id'} <= set(df.columns), f'{sid}: selection schema')
    require(len(df) == rec['selected'], f'{sid}: selection length mismatch')
    require(df.cell_id.is_unique and df.row_index.is_unique, f'{sid}: duplicate selection')
    return df


def cmd_preflight(a):
    project = Path(a.project).resolve()
    cfg, roles, genes, prepared, gene_path = config_contract(project, a.config, getattr(a,'prepared_root',None), getattr(a,'gene_axis',None))
    print('PROJECT', project)
    print('CONFIG', Path(a.config).resolve(), sha(a.config))
    print('PREPARED', prepared)
    print('GENE_AXIS', len(genes), sha(gene_path))
    print('CANVAS', cfg['data']['canvas_um'], 'MPP', cfg['data']['model_mpp'])
    for role, ids in roles.items():
        print(role, ','.join(ids))
        for sid in ids:
            s = prepared / 'samples' / sid
            for name in ('cells.parquet','raw.h5ad','clean.h5ad','pixel_size_um.txt','image_path.txt'):
                require((s / name).is_file(), f'{sid}: missing {name} under {s}')
            img = Path((s / 'image_path.txt').read_text().strip())
            require(img.is_file(), f'{sid}: H&E unavailable at {img}')
            mpp = float((s / 'pixel_size_um.txt').read_text().strip())
            require(mpp > 0, f'{sid}: invalid pixel size')
            if a.morph_template:
                p = annotation_path(a.morph_template, sid, project)
                require(p.is_file(), f'{sid}: morphology input unavailable {p}')
            print('  FILES_PASS', sid, 'mpp', mpp, 'H&E', img)
    print('PREFLIGHT_PASS (metadata/files only; not a full assay alignment audit)')


def cmd_freeze(a):
    project = Path(a.project).resolve()
    cfg, roles, genes, prepared, gene_path = config_contract(project, a.config, getattr(a,'prepared_root',None), getattr(a,'gene_axis',None))
    root = Path(a.out).resolve()
    require(not root.exists(), f'refuse overwrite: {root}')
    require(any(int(x) > 0 for x in (a.train_cap, a.val_cap, a.test_cap)) or a.accept_full,
            'all caps zero -> full cohort: pass --accept-full explicitly')
    sizes = {'train':int(a.train_cap), 'val':int(a.val_cap), 'test':int(a.test_cap)}
    require(all(n >= 0 for n in sizes.values()), 'caps must be nonnegative')
    root.mkdir(parents=True)
    (root / 'selections').mkdir()
    recs = {}
    for role, key in (('train',ROLE_KEYS[0]),('val',ROLE_KEYS[1]),('test',ROLE_KEYS[2])):
        for sid in roles[key]:
            cells = pd.read_parquet(prepared / 'samples' / sid / 'cells.parquet', columns=['cell_id'])
            cells['cell_id'] = cells.cell_id.astype(str)
            require(len(cells)==EXPECTED_CELLS[sid],
                    f'{sid}: prepared cell count changed: {len(cells)} vs frozen {EXPECTED_CELLS[sid]}')
            require(cells.cell_id.is_unique, f'{sid}: duplicate primary cell_id')
            all_idx = np.arange(len(cells), dtype=np.int64)
            anno_hash = None
            if a.morph_template:
                p = annotation_path(a.morph_template, sid, project)
                ann = load_ann(p)
                keep = cells.cell_id.isin(set(ann.cell_id))
                all_idx = np.flatnonzero(keep.to_numpy())
                require(len(all_idx) >= 100, f'{sid}: insufficient exact mapped cells')
                anno_hash = sha(p)
            cap = sizes[role]
            if cap > 0 and cap < len(all_idx):
                rng = np.random.default_rng(stable_int(a.seed, sid, role))
                chosen = np.sort(rng.choice(all_idx, size=cap, replace=False))
            else:
                chosen = all_idx
            require(len(chosen) >= 10, f'{sid}: too few selected cells')
            sel = pd.DataFrame({'row_index':chosen, 'cell_id':cells.cell_id.iloc[chosen].to_numpy()})
            relative = f'selections/{sid}.csv'
            sel.to_csv(root / relative, index=False)
            recs[sid] = {'role':role,'selection_file':relative,
                         'selection_sha256':sha(root / relative), 'selected':len(chosen),
                         'eligible':len(all_idx),'total_cells':len(cells),
                         'morph_annotation_sha256':anno_hash}
            print('FROZEN_SELECTION', role, sid, len(chosen), '/', len(cells), 'eligible',len(all_idx),flush=True)
    cap_flag = any(sizes[k] != 0 and any(recs[s]['selected'] < recs[s]['eligible'] for s in roles[ROLE_KEYS[i]])
                   for i,k in enumerate(('train','val','test')))
    manifest = {'version':'cvpr-fair-v1','seed':a.seed,'config':str(Path(a.config).resolve()),
                'config_sha256':sha(a.config),'project':str(project),'prepared_root':str(prepared),
                'prepared_root_override':getattr(a,'prepared_root',None),'gene_axis_override':getattr(a,'gene_axis',None),
                'gene_axis':str(gene_path),'gene_axis_sha256':sha(gene_path),
                'center_um':float(cfg['data']['center_um']),
                'genes':genes,'roles':roles,'caps_per_slide':sizes,
                'morph_template':a.morph_template,
                'restricted_to_exact_hne_morphology':bool(a.morph_template),
                'sampled_pilot':bool(cap_flag), 'samples':recs}
    save_json_exclusive(root / 'manifest.json', manifest)
    print('FROZEN',root,'sampled_pilot=',bool(cap_flag))
    print('NOTE: no test expression used for cell selection')


def make_loader(project, prepared, sid, genes, selection, context, batch, center_um):
    import torch
    from torch.utils.data import DataLoader, Subset
    from torchvision.transforms import functional as TF
    from leakage_vso.dataset import CellImageDataset, _dense_rows, _resize_physical, _zero_outside_center

    class StrictDataset(CellImageDataset):
        def __getitem__(self, idx):
            si, li = self.rows[idx]
            s, gidx = self.samples[si]
            row = s.cells.iloc[li]
            wsi = self._get_wsi(s)
            px = max(8, int(round(self.canvas_um / s.pixel_size_um)))
            outer = wsi.crop_px(float(row.x_px), float(row.y_px), px)
            outer = _resize_physical(outer, self.canvas_um, self.model_mpp)
            outer_t = _zero_outside_center(self.to_tensor(outer), self.context_um, self.canvas_um)
            require(tuple(outer_t.shape)==(3,256,256),f'{sid}: invalid fixed-canvas tensor shape')
            raw = _dense_rows(s.raw.X, [li], gidx)[0]
            require(s.clean is not None, f'{sid}: missing correction')
            clean = _dense_rows(s.clean.X, [li], gidx)[0]
            return {'x':outer_t,'raw':torch.as_tensor(raw,dtype=torch.float32),
                    'clean':torch.as_tensor(clean,dtype=torch.float32),
                    'cell_id':str(row.cell_id),'row_index':int(li)}

    ds = StrictDataset(prepared_root=str(prepared),sample_ids=[sid],genes=genes,
                       target='both',context_um=context,center_um=center_um,
                       model_mpp=.5,canvas_um=128.0,split=None,augment=False,ring_shuffle=False)
    indices = selection.row_index.astype(int).tolist()
    require(len(ds.rows) >= max(indices) + 1, f'{sid}: dataset has fewer rows than expected')
    for j in [0,len(indices)//2,-1]:
        idx = indices[j]
        require(tuple(ds.rows[idx]) == (0,idx), f'{sid}: dataset row order differs from cells.parquet')
        require(str(ds.samples[0][0].cells.iloc[idx].cell_id) == str(selection.cell_id.iloc[j]),
                f'{sid}: row/cell_id alignment mismatch')
    loader = DataLoader(Subset(ds,indices), batch_size=batch, shuffle=False,
                        num_workers=0, pin_memory=False)
    return ds,loader


def create_encoder(name, device):
    import torch
    from torch import nn
    if name == 'imagenet-r18':
        from torchvision.models import resnet18, ResNet18_Weights
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Identity()
        model.to(device).eval()
        return model, None, 512, 'imagenet-v1, ResNet18 frozen linear probe'
    if name == 'phikon-v2':
        from transformers import AutoImageProcessor, AutoModel
        ident = 'owkin/phikon-v2'
        processor = AutoImageProcessor.from_pretrained(ident, use_fast=False)
        model = AutoModel.from_pretrained(ident).to(device).eval()
        return model, processor, 1024, ident
    if name == 'uni2-h':
        import timm
        from torchvision.transforms import functional as TF
        params = dict(img_size=224,patch_size=14,depth=24,num_heads=24,
                      init_values=1e-5,embed_dim=1536,mlp_ratio=2.66667*2,
                      num_classes=0,no_embed_class=True,
                      mlp_layer=timm.layers.SwiGLUPacked,act_layer=torch.nn.SiLU,
                      reg_tokens=8,dynamic_img_size=True)
        model = timm.create_model('hf-hub:MahmoodLab/UNI2-h',pretrained=True,**params).to(device).eval()
        return model,None,1536,'MahmoodLab/UNI2-h (gated, 681M params)'
    fail('unknown encoder')


def encode_batch(x, encoder, processor, name, device, amp):
    import torch
    import torch.nn.functional as F
    if name == 'imagenet-r18':
        inp = x.to(device)
        with torch.inference_mode(), torch.autocast(device_type='cuda',enabled=amp):
            z = encoder(inp)
    elif name == 'uni2-h':
        inp = F.interpolate(x.to(device),size=(224,224),mode='bilinear',align_corners=False)
        with torch.inference_mode(), torch.autocast(device_type='cuda',enabled=amp):
            z = encoder(inp)
    else:
        # Dataset uses ImageNet-normalized tensor with *zero in masked region*.
        # Rebuild RGB and let the official Phikon processor normalize it once.
        from PIL import Image
        mean = torch.tensor([.485,.456,.406],dtype=torch.float32)[None,:,None,None]
        std = torch.tensor([.229,.224,.225],dtype=torch.float32)[None,:,None,None]
        rgb = ((x.float()*std+mean).clamp(0,1)*255).round().byte().permute(0,2,3,1).numpy()
        images = [Image.fromarray(img,'RGB') for img in rgb]
        inp = processor(images=images,return_tensors='pt')['pixel_values'].to(device)
        with torch.inference_mode(), torch.autocast(device_type='cuda',enabled=amp):
            outputs = encoder(pixel_values=inp)
            z = outputs.last_hidden_state[:,0,:]
    require(z.ndim == 2 and z.shape[0] == len(x), 'invalid encoder output shape')
    require(torch.isfinite(z).all().item(), 'nonfinite encoder embedding')
    return z.float().cpu().numpy()


def cmd_extract(a):
    import torch
    frozen,data = read_freeze(a.frozen)
    require(a.sid in data['samples'], f'unknown section {a.sid}')
    require(a.context in CONTEXTS, 'context must be 24 or 128')
    require(1 <= a.batch <= 64, '--batch out of safety bounds')
    sid = a.sid
    sel = read_selection(frozen,data,sid)
    out = frozen / 'features' / a.encoder / f'ctx{a.context}' / f'{sid}.npz'
    require(not out.exists(), f'refuse overwrite: {out}')
    require(0 == len(list(out.parent.glob(f'.{sid}.*.partial'))), 'stale partial file exists')
    project = Path(data['project'])
    # Allow mounting the frozen manifest on a different machine, but ensure same genes/config.
    prepared = Path(data['prepared_root'])
    require(prepared.is_dir(),f'original prepared_root unavailable: {prepared}')
    require(Path(data['gene_axis']).is_file() and sha(data['gene_axis']) == data['gene_axis_sha256'],
            'frozen gene axis moved or modified')
    require(Path(data['config']).is_file() and sha(data['config']) == data['config_sha256'],
            'frozen config changed')
    torch.manual_seed(2026)
    np.random.seed(2026)
    device = 'cuda' if (a.device == 'auto' and torch.cuda.is_available()) else ('cpu' if a.device == 'auto' else a.device)
    require(device == 'cpu' or torch.cuda.is_available(), 'GPU requested but CUDA unavailable')
    model,processor,dim,backbone = create_encoder(a.encoder,device)
    ds,loader = make_loader(project,prepared,sid,data['genes'],sel,a.context,a.batch, data['center_um'])
    emb=[]; raws=[]; cleans=[]; seen=[]; indices=[]
    start=time.time()
    try:
        for k,b in enumerate(loader):
            z = encode_batch(b['x'],model,processor,a.encoder,device,device=='cuda' and a.amp)
            emb.append(z)
            raws.append(b['raw'].numpy())
            cleans.append(b['clean'].numpy())
            seen.extend(map(str,b['cell_id']))
            indices.extend(b['row_index'].tolist())
            if k % max(1,a.log_every) == 0:
                print('EXTRACT',sid,'ctx',a.context,'encoder',a.encoder,'rows',len(seen),'/',len(sel),flush=True)
    finally:
        for w in ds._wsi.values():
            with contextlib.suppress(Exception):w.close()
    require(len(seen) == len(sel),'extract count mismatch')
    require(seen == sel.cell_id.tolist() and indices == sel.row_index.tolist(),'extraction misaligned with frozen selection')
    F=np.concatenate(emb).astype(np.float32)
    raw=np.concatenate(raws).astype(np.float32)
    clean=np.concatenate(cleans).astype(np.float32)
    require(F.shape==(len(sel),dim) and raw.shape==clean.shape==(len(sel),280),'unexpected extracted shape')
    require(np.isfinite(F).all() and np.isfinite(raw).all() and np.isfinite(clean).all(),'nonfinite features/labels')
    require((raw>=-1e-5).all() and (clean>=-1e-5).all(),'negative count target encountered')
    require((clean<=raw+1e-3).all(),'corrected exceeds raw unexpectedly: audit correction')
    out.parent.mkdir(parents=True,exist_ok=True)
    # All source paths in manifest; inputs unchanged. Do not store checkpoints here.
    np.savez_compressed(out, feature=F,raw=raw,clean=clean,
                        cell_id=np.array(seen,dtype='U'),row_index=np.array(indices,dtype=np.int64),
                        gene=np.array(data['genes'],dtype='U'),
                        sample_id=np.array([sid]*len(seen),dtype='U'))
    save_json_exclusive(out.with_suffix('.json'),
        {'method':'frozen_encoder_linear_probe','encoder':a.encoder,'context_um':a.context,
         'section':sid,'n':len(sel),'input_manifest_sha256':sha(frozen/'manifest.json'),
         'feature_sha256':sha(out),'encoder_name':backbone,
         'seconds':round(time.time()-start,3),'amp':bool(a.amp),'device':device})
    print('EXTRACT_PASS',out,'rows',len(sel),'seconds',round(time.time()-start,1))


def build_morph_features(cells, ann, sel, half):
    from scipy.spatial import cKDTree
    c=cells[['cell_id','x_px','y_px']].copy()
    c.cell_id=c.cell_id.astype(str)
    require(c.cell_id.is_unique,'morph: duplicate cell ids')
    d=ann[['cell_id','hovernet_type']].merge(c,on='cell_id',how='inner',validate='one_to_one')
    require(len(d)==len(ann),'some morph annotations lack primary cell coordinates')
    require(d.hovernet_type.isin(BIO_CLASSES).all(),'invalid morphology type')
    xy=d[['x_px','y_px']].to_numpy(dtype=np.float64)
    require(np.isfinite(xy).all(),'morph xy nonfinite')
    lookup=pd.Series(np.arange(len(d),dtype=np.int64),index=d.cell_id)
    require(sel.cell_id.isin(lookup.index).all(),'morph selection lacks exact matched labels')
    ids=lookup.loc[sel.cell_id].to_numpy(dtype=np.int64)
    labels=pd.Categorical(d.hovernet_type,categories=BIO_CLASSES).codes
    tree=cKDTree(xy)
    f=[]
    for i in ids:
        center=xy[i]
        candidates=tree.query_ball_point(center,half*math.sqrt(2)+1e-8)
        cand=np.asarray(candidates,dtype=np.int64)
        cand=cand[cand!=i]
        dd=xy[cand]-center
        in_box=(np.abs(dd[:,0]) <= half+1e-8)&(np.abs(dd[:,1]) <= half+1e-8)
        cand=cand[in_box]
        dist=np.linalg.norm(dd[in_box],axis=1)
        order=np.argsort(dist,kind='mergesort')
        cand=cand[order];dist=dist[order]
        kinds=labels[cand]
        features=list(np.eye(len(BIO_CLASSES),dtype=np.float32)[labels[i]])
        for k in (5,10,20):
            cur=kinds[:k]
            features.extend((np.bincount(cur,minlength=len(BIO_CLASSES))/float(k)).tolist())
        for typ in range(len(BIO_CLASSES)):
            m=dist[kinds==typ]
            features.append(np.log1p(float(m[0]) if len(m) else half*math.sqrt(2)))
        features.extend([np.log1p(len(cand)),
                         np.log1p(float(dist[:5].mean()) if len(dist) else half*math.sqrt(2)),
                         np.log1p(float(dist[:20].mean()) if len(dist) else half*math.sqrt(2))])
        f.append(features)
    return np.array(f,dtype=np.float32)


def cmd_morph(a):
    frozen,data=read_freeze(a.frozen)
    require(data['morph_template'], 'freeze must specify --morph-template for exact mapped population')
    require(a.sid in data['samples'],'sid not in manifest')
    out=frozen/'features'/'morph'/f'ctx{a.context}'/f'{a.sid}.npz'
    require(not out.exists(), f'refuse overwrite: {out}')
    project=Path(data['project'])
    sdir=Path(data['prepared_root'])/'samples'/a.sid
    sel=read_selection(frozen,data,a.sid)
    p=annotation_path(data['morph_template'],a.sid,project)
    require(sha(p)==data['samples'][a.sid]['morph_annotation_sha256'],'morph file differs from freeze')
    ann=load_ann(p)
    cells=pd.read_parquet(sdir/'cells.parquet',columns=['cell_id','x_px','y_px'])
    mpp=float((sdir/'pixel_size_um.txt').read_text().strip())
    require(0<mpp<5,'invalid mpp')
    # Convert native WSI pixels to micrometers for physically correct FOV.
    cells=cells.copy();cells.x_px=cells.x_px.astype(float)*mpp;cells.y_px=cells.y_px.astype(float)*mpp
    F=build_morph_features(cells,ann,sel,a.context/2)
    require(np.isfinite(F).all(),'morph feature nonfinite')
    out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,feature=F,cell_id=sel.cell_id.to_numpy(dtype='U'),
                        row_index=sel.row_index.to_numpy(dtype=np.int64),
                        sample_id=np.array([a.sid]*len(sel),dtype='U'),gene=np.array(data['genes'],dtype='U'))
    save_json_exclusive(out.with_suffix('.json'),
        {'method':'morph-only-exact-hne','context_um':a.context,'section':a.sid,
         'n':len(sel),'input_manifest_sha256':sha(frozen/'manifest.json'),
         'feature_sha256':sha(out),'morph_annotation_sha256':sha(p),
         'method_note':'H&E HoVer-Net class and neighbors within frozen physical canvas only; no assay-derived features'})
    print('MORPH_PASS',out,'shape',F.shape)


def load_feature_set(frozen,data,method,context,role,labels_from):
    all_F=[];all_y=[];all_ids=[];all_sids=[];all_raw=[];all_clean=[]
    for sid,rec in data['samples'].items():
        if rec['role']!=role:continue
        p=frozen/'features'/method/f'ctx{context}'/f'{sid}.npz'
        require(p.is_file(),f'missing extracted {method} features: {p}')
        side=p.with_suffix('.json')
        require(side.is_file(),f'missing provenance: {side}')
        meta=json.loads(side.read_text())
        require(meta['feature_sha256']==sha(p),'feature changed since extraction')
        require(meta['input_manifest_sha256']==sha(frozen/'manifest.json'),'manifest not matching features')
        sel=read_selection(frozen,data,sid)
        with np.load(p,allow_pickle=False) as b:
            ids=b['cell_id'].astype(str);rows=b['row_index']
            require(ids.tolist()==sel.cell_id.tolist() and rows.tolist()==sel.row_index.tolist(),f'{sid}: feature IDs misaligned')
            require(b['gene'].tolist()==data['genes'],f'{sid}: gene axis mismatch')
            F=b['feature'].astype(np.float64)
            require(len(F)==len(ids) and np.isfinite(F).all(),f'{sid}: feature invalid')
            all_F.append(F);all_ids.extend(ids.tolist());all_sids.extend([sid]*len(ids))
            if method == 'morph':
                lp=frozen/'features'/labels_from/f'ctx{context}'/f'{sid}.npz'
                require(lp.is_file(),f'morph needs assay labels extracted by --labels-from encoder: {lp}')
                with np.load(lp,allow_pickle=False) as b2:
                    require(ids.tolist()==b2['cell_id'].astype(str).tolist(),f'{sid}: morph/labels different cells')
                    raw=b2['raw'].astype(np.float32);clean=b2['clean'].astype(np.float32)
            else:
                raw=b['raw'].astype(np.float32);clean=b['clean'].astype(np.float32)
            require(raw.shape==clean.shape==(len(ids),280),f'{sid}: gene shape mismatch')
            require(np.isfinite(raw).all() and np.isfinite(clean).all(),'nonfinite targets')
            all_raw.append(raw);all_clean.append(clean)
    require(len(all_F)>0,f'no {role} features')
    F=np.concatenate(all_F);raw=np.concatenate(all_raw);clean=np.concatenate(all_clean)
    return dict(feature=F, raw=raw,clean=clean,
                sample_id=np.array(all_sids,dtype='U'),cell_id=np.array(all_ids,dtype='U'))


def standardize(Xtrain,Xval,Xtest):
    mean=Xtrain.mean(axis=0)
    std=Xtrain.std(axis=0)
    std=np.where(std<1e-6,1.0,std)
    return [(x-mean)/std for x in (Xtrain,Xval,Xtest)],mean,std


def ridge_solver(xt,yt,alpha):
    # Exact multi-output ridge with unpenalized intercept, Cholesky on features.
    from scipy.linalg import cho_factor, cho_solve
    require(alpha>0,'ridge alpha must be positive')
    mu_x=xt.mean(axis=0);mu_y=yt.mean(axis=0)
    xc=xt-mu_x;yc=yt-mu_y
    G=xc.T@xc
    G.flat[::len(G)+1]+=alpha
    coef=cho_solve(cho_factor(G,check_finite=False),xc.T@yc,check_finite=False)
    bias=mu_y-mu_x@coef
    return coef,bias


def pcc_columns(y,p):
    require(y.shape==p.shape and y.ndim==2,'PCC shape mismatch')
    y=y.astype(np.float64);p=p.astype(np.float64)
    cy=y-y.mean(axis=0);cp=p-p.mean(axis=0)
    yvar=(cy*cy).sum(axis=0);pvar=(cp*cp).sum(axis=0)
    denom=np.sqrt(yvar*pvar)
    r=np.divide((cy*cp).sum(axis=0),denom,out=np.full(y.shape[1],np.nan),where=denom>1e-14)
    # Explicit reporting convention: an unvarying prediction has zero
    # predictive correlation when the observed target varies. Do not silently
    # exclude a weak baseline's constant-output genes from its mean score.
    r[(yvar>1e-14)&(pvar<=1e-14)]=0.0
    # If the held-out target is constant, PCC is genuinely undefined.
    return np.clip(r,-1,1)


def compute_scores(pred, raw, clean, sids, genes, donor_map):
    out=[];summary=[]
    for group_col,ids in [('all',np.ones(len(sids),dtype=bool))]+[(s,sids==s) for s in sorted(set(sids))]:
        if ids.sum()<3:continue
        for tgt,target in [('raw',raw),('clean',clean)]:
            pc=pcc_columns(target[ids],pred[ids])
            for g,val in zip(genes,pc):
                out.append({'group':group_col,'donor_id':donor_map.get(group_col,'ALL'),
                            'target':tgt,'gene':g,'pcc':val,'n_cells':int(ids.sum())})
            finite=np.isfinite(pc)
            z=np.arctanh(np.clip(pc[finite],-1+1e-6,1-1e-6))
            summary.append({'group':group_col,'target':tgt,'n_cells':int(ids.sum()),
                            'n_genes_finite':int(finite.sum()),'mean_fisher_z':float(np.mean(z)) if len(z) else np.nan,
                            'equiv_pcc':float(np.tanh(np.mean(z))) if len(z) else np.nan})
    result=pd.DataFrame(out)
    # Separate grouping by independent patient, pool sections per patient.
    for donor in sorted(set(donor_map[s] for s in set(sids))):
        ids=np.isin(sids,[s for s in set(sids) if donor_map[s]==donor])
        if ids.sum()<3:continue
        for tgt,target in [('raw',raw),('clean',clean)]:
            pc=pcc_columns(target[ids],pred[ids]);f=np.isfinite(pc)
            z=np.arctanh(np.clip(pc[f],-1+1e-6,1-1e-6))
            summary.append({'group':f'donor:{donor}','target':tgt,'n_cells':int(ids.sum()),
                            'n_genes_finite':int(f.sum()),'mean_fisher_z':float(np.mean(z)) if len(z) else np.nan,
                            'equiv_pcc':float(np.tanh(np.mean(z))) if len(z) else np.nan})
    return result,pd.DataFrame(summary)


def donor_mapping(sids):
    # Primary2025 IDs established by frozen cohort provenance; not inferred from slide numbers.
    return {s:('Patient 1' if s in ('TENX191','TENX192','TENX193')
              else 'Patient 2' if s in ('TENX195','TENX196')
              else 'Patient 3' if s in ('TENX197','TENX198','TENX199')
              else 'Patient 4') for s in sids}


def save_predictions(frozen,data,out,pred,truth,meta):
    genes=data['genes'];raw=truth['raw'];clean=truth['clean'];sids=truth['sample_id'];ids=truth['cell_id']
    require(pred.shape==raw.shape==clean.shape and pred.shape[1]==280,'prediction shape invalid')
    require(np.isfinite(pred).all(),'prediction has nan/inf')
    require((pred>=0).all(),'prediction negative')
    np.savez_compressed(out/'test_predictions.npz',prediction=pred.astype(np.float32),
                        raw=raw,clean=clean,sample_id=sids,cell_id=ids,gene=np.array(genes,dtype='U'))
    donor_map=donor_mapping(set(sids))
    per_gene,summary=compute_scores(pred,raw,clean,sids,genes,donor_map)
    per_gene.to_csv(out/'metrics_by_gene_section.csv',index=False)
    summary.to_csv(out/'metrics_summary.csv',index=False)
    meta.update({'frozen_manifest_sha256':sha(frozen/'manifest.json'),
                 'sampled_pilot':data['sampled_pilot'],
                 'n_independent_test_patients':len(set(donor_map.values())),
                 'genes':280,'n_test_cells':len(ids),
                 'label':'PILOT_SUBSAMPLE' if data['sampled_pilot'] else 'ALL_ELIGIBLE_CELLS',
                 'test_prediction_sha256':sha(out/'test_predictions.npz')})
    save_json_exclusive(out/'provenance.json',meta)
    print(summary.to_string(index=False))
    print('TEST_PASS',out,'TEST_PATIENTS',meta['n_independent_test_patients'],'PILOT',data['sampled_pilot'])


def cmd_fit(a):
    frozen,data=read_freeze(a.frozen)
    require(a.context in CONTEXTS,'invalid context')
    require(a.labels_from != 'morph','labels_from must be image encoder')
    out=Path(a.out).resolve()
    require(not out.exists(),f'refuse overwrite: {out}')
    tr=load_feature_set(frozen,data,a.method,a.context,'train',a.labels_from)
    va=load_feature_set(frozen,data,a.method,a.context,'val',a.labels_from)
    te=load_feature_set(frozen,data,a.method,a.context,'test',a.labels_from)
    require(tr['feature'].shape[1]==va['feature'].shape[1]==te['feature'].shape[1],'feature dim mismatch')
    (xt,xv,xe),mean,std=standardize(tr['feature'],va['feature'],te['feature'])
    yt=np.log1p(tr[a.target].astype(np.float64))
    yv=np.log1p(va[a.target].astype(np.float64))
    best=None; history=[]
    for alpha in a.alphas:
        coef,bias=ridge_solver(xt,yt,alpha)
        val_pred=xv@coef+bias
        score=float(np.mean((val_pred-yv)**2))
        history.append({'alpha':alpha,'val_log_mse':score})
        print('VAL_ALPHA',alpha,'log_mse',score,flush=True)
        if best is None or score<best[0]:best=(score,alpha,coef,bias)
    require(best is not None,'no alpha selected')
    _,alpha,coef,bias=best
    pred=np.expm1(np.clip(xe@coef+bias,0,20)).astype(np.float32)
    require(np.isfinite(pred).all(),'numerical overflow')
    out.mkdir(parents=True)
    pd.DataFrame(history).to_csv(out/'validation_alphas.csv',index=False)
    np.savez_compressed(out/'frozen_head.npz',coef=coef.astype(np.float32),bias=bias.astype(np.float32),
                        mean=mean.astype(np.float32),std=std.astype(np.float32))
    save_predictions(frozen,data,out,pred,te,
        {'method':a.method,'trained_target':a.target,'context_um':a.context,
         'head':'frozen-encoder multioutput ridge (train standardized, train-only fit)',
         'validation_alpha':alpha,'val_log_mse':best[0],
         'train_sample_ids':data['roles']['train_sample_ids'],
         'val_sample_ids':data['roles']['val_sample_ids'],
         'test_sample_ids':data['roles']['test_sample_ids'],
         'n_train_cells':len(tr['raw']),'n_val_cells':len(va['raw']),
         'note':'not fine-tuned; cannot claim reproduction of GHIST or sCellST'})


def cmd_official(a):
    frozen,data=read_freeze(a.frozen)
    require(a.method_name.strip() and a.method_name not in ('morph','imagenet-r18','phikon-v2','uni2-h'),
            'official method name must identify external code and be distinct from local probe')
    source=Path(a.predictions).resolve()
    prov=Path(a.training_provenance).resolve()
    require(source.is_file() and prov.is_file(),'official prediction CSV and provenance JSON must exist')
    external=json.loads(prov.read_text())
    for k in ('official_repository_url','official_commit','train_sample_ids',
              'val_sample_ids','test_sample_ids','checkpoint_sha256','method_name'):
        require(k in external,f'official provenance missing {k}')
    require(external['method_name']==a.method_name,'method name inconsistent')
    for k in ROLE_KEYS:
        require(set(external[k]) == set(data['roles'][k]),f'official model uses wrong {k}')
    require(len(external['checkpoint_sha256']) == 64 and all(c in '0123456789abcdef' for c in external['checkpoint_sha256'].lower()),
            'official checkpoint SHA256 must be 64 hexadecimal characters')
    require(len(external['official_commit']) == 40 and all(c in '0123456789abcdef' for c in external['official_commit'].lower()),
            'official Git commit must be 40 hexadecimal characters')
    # Pull original test truth only from an already-extracted independent encoder:
    # external method must not read assay expression from test for inference.
    te=load_feature_set(frozen,data,a.labels_from,a.context,'test',a.labels_from)
    df=pd.read_csv(source,dtype={'cell_id':str,'sample_id':str},low_memory=False)
    genes=data['genes']
    require(set(df.columns)==set(['sample_id','cell_id']+genes),
            'official predictions must be wide format with exactly sample_id,cell_id and the 280 original genes')
    require(not df.duplicated(['sample_id','cell_id']).any(),'official duplicate test cell ids')
    truth_ids=list(zip(te['sample_id'].tolist(),te['cell_id'].tolist()))
    require(set(zip(df.sample_id,df.cell_id))==set(truth_ids),
            'official predictions test cell universe differs from frozen selection')
    key=pd.MultiIndex.from_frame(df[['sample_id','cell_id']])
    pos=key.get_indexer(pd.MultiIndex.from_tuples(truth_ids))
    require((pos>=0).all(),'official predictions mapping failed')
    pred=df.iloc[pos][genes].apply(pd.to_numeric,errors='raise').to_numpy(dtype=np.float32)
    require(np.isfinite(pred).all(),'official predictions nonfinite')
    require((pred>=0).all(),'official prediction negative; provide raw-count-scale inverse transformed predictions')
    out=Path(a.out).resolve();require(not out.exists(),f'refuse overwrite {out}')
    out.mkdir(parents=True)
    save_predictions(frozen,data,out,pred,te,
        {'method':a.method_name,'trained_target':external.get('trained_target','unverified'),
         'context_um':a.context,'type':'OFFICIAL_EXTERNAL_PREDICTIONS',
         'official_predictions_sha256':sha(source),
         'official_training_provenance_sha256':sha(prov),
         'external_provenance':external,
         'note':'training claimed by submitted provenance; auditor cannot independently prove no test leakage'})


def cmd_compare(a):
    require(len(a.runs)>=2,'compare >=2 runs')
    out=Path(a.out).resolve();require(not out.exists(),f'refuse overwrite {out}')
    inputs=[];rows=[]
    for r in map(Path,a.runs):
        p=r/'provenance.json';npz=r/'test_predictions.npz'
        require(p.is_file() and npz.is_file(),f'missing run artifacts: {r}')
        meta=json.loads(p.read_text())
        require(meta['test_prediction_sha256']==sha(npz),f'test predictions changed: {r}')
        with np.load(npz,allow_pickle=False) as d:
            dat={k:d[k] for k in ('prediction','raw','clean','sample_id','cell_id','gene')}
        require(dat['prediction'].shape==(len(dat['sample_id']),280),f'invalid {r}')
        inputs.append((str(r),meta,dat))
    base=inputs[0][2];g0=inputs[0][1]['frozen_manifest_sha256']
    for name,meta,other in inputs[1:]:
        require(meta['frozen_manifest_sha256']==g0,'different frozen manifests cannot be fairly ranked')
        for k in ('gene','sample_id','cell_id','raw','clean'):
            require(np.array_equal(base[k],other[k]),f'{name}: mismatch in {k} / test rows; cannot compare')
    donor_map=donor_mapping(set(base['sample_id']))
    donors=sorted(set(donor_map.values()))
    # Same predeclared scored-gene denominator for every model and target.
    # Undefined labels are shared; constant model predictions score zero.
    for tgt in ('raw','clean'):
        target=base[tgt]
        ref=np.isfinite(pcc_columns(target,inputs[0][2]['prediction']))
        for name,_,dat in inputs[1:]:
            valid=np.isfinite(pcc_columns(target,dat['prediction']))
            require(np.array_equal(ref,valid),
                    f'{name}: different valid-scored genes for {tgt}; refuse misleading ranking')
    for name,meta,dat in inputs:
        model=f"{Path(name).name}:{meta['method']}:{meta.get('trained_target','?')}:{meta.get('context_um','?')}"
        _,summary=compute_scores(dat['prediction'],dat['raw'],dat['clean'],dat['sample_id'],list(dat['gene']),donor_map)
        for _,rec in summary.iterrows():
            rows.append({'model':model,'run':name,**rec.to_dict(),'sampled_pilot':meta['sampled_pilot']})
    out.mkdir(parents=True)
    tab=pd.DataFrame(rows);tab.to_csv(out/'matched_scores.csv',index=False)
    # A single held-out patient is descriptive only.
    aggregate=tab[tab.group=='all'][['model','target','mean_fisher_z','equiv_pcc','n_cells','n_genes_finite']]
    aggregate.to_csv(out/'model_ranking.csv',index=False)
    save_json_exclusive(out/'comparison_provenance.json',
        {'runs':a.runs,'test_patients':donors,'n_independent_patients':len(donors),
         'manifest_sha256':g0,'sampled_pilot':inputs[0][1]['sampled_pilot'],
         'patient_inference':'not available' if len(donors)<3 else 'use independent donor unit; 4 folds still small',
         'note':'different heads/training objectives; observational model comparison, not leakage causality'})
    print(aggregate.to_string(index=False))
    print('COMPARE_PASS',out,'patients=',len(donors))




def cmd_aggregate_folds(a):
    """Donor-level aggregation of independently frozen rotation folds.

    Each input run must have exactly one held-out patient and a distinct donor
    per method/context/target. No cell/section-level bootstrap is allowed.
    """
    require(len(a.runs)>=2,'aggregate-folds requires multiple run outputs')
    out=Path(a.out).resolve();require(not out.exists(),f'refuse overwrite: {out}')
    rows=[];gene_ref=None
    seen=set()
    for r in map(Path,a.runs):
        pp=r/'provenance.json';npz=r/'test_predictions.npz'
        require(pp.is_file() and npz.is_file(),f'run missing: {r}')
        meta=json.loads(pp.read_text())
        require(meta['test_prediction_sha256']==sha(npz),f'prediction hash mismatch: {r}')
        with np.load(npz,allow_pickle=False) as q:
            pred=q['prediction'];raw=q['raw'];clean=q['clean']
            sid=q['sample_id'].astype(str);genes=q['gene'].astype(str)
        if gene_ref is None: gene_ref=genes
        require(np.array_equal(gene_ref,genes),'different gene axes across folds')
        donors={donor_mapping([s])[s] for s in set(sid.tolist())}
        require(len(donors)==1,f'{r}: not one independent held-out patient')
        donor=next(iter(donors))
        key=(str(meta['method']),int(meta['context_um']),str(meta.get('trained_target','?')),donor)
        require(key not in seen,f'duplicate patient/model run: {key}')
        seen.add(key)
        for target,yy in [('raw',raw),('clean',clean)]:
            pcc=pcc_columns(yy,pred)
            valid=np.isfinite(pcc)
            z=np.arctanh(np.clip(pcc[valid],-1+1e-6,1-1e-6))
            require(len(z)>0,f'{r}: no finite PCC')
            rows.append({'patient':donor,'method':key[0],
                         'context_um':key[1],'trained_target':key[2],
                         'evaluation_target':target,'n_genes_finite':int(valid.sum()),
                         'mean_fisher_z':float(z.mean()),'n_test_cells':len(sid),
                         'sampled_pilot':meta['sampled_pilot'],'run':str(r)})
    dat=pd.DataFrame(rows)
    # No cross-fold comparison is allowed unless all folds use equal
    # target-specific gene denominators, or all 280 genes are available.
    groups=[]
    keys=['method','context_um','trained_target','evaluation_target']
    for k,g in dat.groupby(keys,sort=True):
        require(not g.patient.duplicated().any(),f'within-method duplicate patients: {k}')
        groups.append((k,g.set_index('patient')))
    import itertools
    pair_rows=[]
    rng=np.random.default_rng(20261030)
    for (ka,da),(kb,db) in itertools.combinations(groups,2):
        if ka[-1] != kb[-1]: continue
        common=sorted(set(da.index)&set(db.index))
        if not common:continue
        x=da.loc[common];y=db.loc[common]
        require(np.array_equal(x.n_genes_finite.to_numpy(),y.n_genes_finite.to_numpy()),
                f'different scored-gene denominator for {ka} vs {kb}')
        d=x.mean_fisher_z.to_numpy()-y.mean_fisher_z.to_numpy()
        n=len(d)
        low=high=float('nan')
        if n>=3:
            boot=np.mean(rng.choice(d,size=(5000,n),replace=True),axis=1)
            low,high=np.quantile(boot,[.025,.975]).tolist()
        pair_rows.append({'model_a':str(ka),'model_b':str(kb),
                          'target':ka[-1],'n_independent_patients':n,
                          'mean_paired_delta_fisher_z':float(d.mean()),
                          'median_paired_delta_fisher_z':float(np.median(d)),
                          'positive_patients':int((d>0).sum()),
                          'donor_bootstrap_ci_low':low,'donor_bootstrap_ci_high':high,
                          'patients':','.join(common)})
    out.mkdir(parents=True)
    dat.to_csv(out/'per_patient_scores.csv',index=False)
    pd.DataFrame(pair_rows).to_csv(out/'donor_paired_method_comparisons.csv',index=False)
    save_json_exclusive(out/'provenance.json',
        {'runs':a.runs,'patient_is_independent_unit':True,'bootstrap_seed':20261030,
         'min_patients_for_bootstrap':3,'genes':gene_ref.tolist(),
         'note':'All comparisons are descriptive when based on few donors; independent folds can use different train patients.'})
    print(dat[['patient','method','context_um','trained_target','evaluation_target',
               'mean_fisher_z','n_genes_finite']].to_string(index=False))
    print('FOLD_AGGREGATION_PASS',out,'unique patients=',dat.patient.nunique(),
          'paired comparisons=',len(pair_rows))


def cmd_bridge(a):
    """Export a role-safe, cell-indexed bridge for OFFICIAL external pipelines.

    This does NOT convert Xenium to GHIST/sCellST native format; each native
    method must be trained using its own official repository and dependencies.
    """
    frozen,data=read_freeze(a.frozen)
    out=Path(a.out).resolve();require(not out.exists(),f'refuse overwrite: {out}')
    out.mkdir(parents=True)
    rows=[]
    for sid,record in data['samples'].items():
        sdir=Path(data['prepared_root'])/'samples'/sid
        selection=read_selection(frozen,data,sid)
        cells=pd.read_parquet(sdir/'cells.parquet',columns=['cell_id','x_px','y_px'])
        cells.cell_id=cells.cell_id.astype(str)
        coord=cells.iloc[selection.row_index.to_numpy(dtype=np.int64)]
        require(coord.cell_id.tolist()==selection.cell_id.tolist(),f'{sid}: coordinate ID mismatch')
        imagepath=Path((sdir/'image_path.txt').read_text().strip())
        require(imagepath.is_file(),f'{sid}: image missing')
        mpp=float((sdir/'pixel_size_um.txt').read_text().strip())
        meta=pd.DataFrame({'sample_id':sid,'cell_id':selection.cell_id,
                           'role':record['role'],'x_px':coord.x_px.to_numpy(),
                           'y_px':coord.y_px.to_numpy(),'x_um':coord.x_px.to_numpy()*mpp,
                           'y_um':coord.y_px.to_numpy()*mpp,'pixel_size_um':mpp,
                           'wsi_path':str(imagepath)})
        rows.append(meta)
    meta=pd.concat(rows,ignore_index=True)
    require(not meta.duplicated(['sample_id','cell_id']).any(),'bridge duplicate cell ids')
    for role in ('train','val','test'):
        q=meta[meta.role==role].copy()
        q.to_csv(out/f'{role}_cells.csv',index=False)
        if role=='test':continue
        features=load_feature_set(frozen,data,a.labels_from,a.context,role,a.labels_from)
        keys=pd.MultiIndex.from_frame(q[['sample_id','cell_id']])
        expected=pd.MultiIndex.from_arrays([features['sample_id'],features['cell_id']])
        ix=expected.get_indexer(keys)
        require((ix>=0).all(),f'bridge missing {role} labels')
        for target in ('raw','clean'):
            labels=features[target][ix]
            df=pd.concat([q[['sample_id','cell_id']].reset_index(drop=True),
                          pd.DataFrame(labels,columns=data['genes'])],axis=1)
            df.to_csv(out/f'{role}_{target}_labels.csv.gz',index=False,compression='gzip')
    (out/'gene_axis_280.txt').write_text('\n'.join(data['genes'])+'\n')
    save_json_exclusive(out/'bridge_provenance.json',
        {'purpose':'OFFICIAL_METHOD_ROLE_SAFE_BRIDGE_NOT_NATIVE_FORMAT',
         'frozen_manifest_sha256':sha(frozen/'manifest.json'),
         'test_expression_exported':False,'sampled_pilot':data['sampled_pilot'],
         'warning':'Official GHIST/sCellST need their original training pipeline, additional data, and method-specific preprocessing. No paper method reimplementation is claimed.',
         'train_sample_ids':data['roles']['train_sample_ids'],
         'val_sample_ids':data['roles']['val_sample_ids'],
         'test_sample_ids':data['roles']['test_sample_ids']})
    print('BRIDGE_PASS',out,'NO_TEST_EXPRESSION_EXPORTED')


def cmd_intervention(a):
    # Works for exact exported *model predictions* for same previously frozen polygon edits.
    p=Path(a.input);require(p.is_file(),f'intervention table not found: {p}')
    df=pd.read_csv(p)
    cols={'model','patient','sample_id','recipient_id','source_id','control_id',
          'operator','gene','pred_original','pred_source','pred_control'}
    require(cols <= set(df.columns),f'intervention CSV missing: {sorted(cols-set(df.columns))}')
    key=['model','sample_id','recipient_id','source_id','control_id','operator','gene']
    require(not df.duplicated(key).any(),'duplicated intervention key')
    require(set(df.operator)<=set(('blur','inpaint')),'operator must be blur or inpaint')
    for c in ('pred_original','pred_source','pred_control'):
        df[c]=pd.to_numeric(df[c],errors='raise')
        require(np.isfinite(df[c]).all(),f'nonfinite {c}')
    # Strict same frozen triples across all methods. Do not rank incomparable pairs.
    pair_key=['patient','sample_id','recipient_id','source_id','control_id','operator','gene']
    methods=sorted(df.model.unique().tolist())
    if len(methods)>1:
        ref=set(map(tuple,df.loc[df.model==methods[0],pair_key].itertuples(index=False,name=None)))
        for method in methods[1:]:
            oth=set(map(tuple,df.loc[df.model==method,pair_key].itertuples(index=False,name=None)))
            require(ref==oth, f'intervention pairs not identical across {methods[0]} vs {method}')
    df['delta_source']=df.pred_original-df.pred_source
    df['delta_control']=df.pred_original-df.pred_control
    df['source_minus_control']=df.delta_source-df.delta_control
    df['normalized_source_minus_control']=df.source_minus_control/(1.0+df.pred_original.abs())
    # No gene/section treated as independent patient: sample-level and patient-level direction.
    summ=df.groupby(['model','patient','operator','gene'],as_index=False).agg(
        n_pairs=('source_minus_control','size'),mean_effect=('source_minus_control','mean'),
        median_effect=('source_minus_control','median'),
        mean_scale_normalized_effect=('normalized_source_minus_control','mean'))
    out=Path(a.out);require(not out.exists(),f'refuse overwrite {out}')
    out.mkdir(parents=True)
    summ.to_csv(out/'patient_gene_operator_effects.csv',index=False)
    print(summ.to_string(index=False,max_rows=50))
    print('INTERVENTION_SUMMARY_PASS; compatible only if source/control edits reuse identical frozen polygons')


def cmd_check_run(a):
    frozen,data=read_freeze(a.frozen)
    print('FROZEN_MANIFEST_PASS',sha(frozen/'manifest.json'))
    print('SAMPLE_PILOT',data['sampled_pilot'])
    print('SAMPLE_COUNT',sum(v['selected'] for v in data['samples'].values()))
    print('EXACT_HNE_MORPH',data['restricted_to_exact_hne_morphology'])
    for sid,s in data['samples'].items():
        print(s['role'],sid,s['selected'],'/',s['eligible'])
    print('CHECK_PASS')


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    sub=parser.add_subparsers(dest='cmd',required=True)
    def add_proj(p):
        p.add_argument('--project',default='.')
        p.add_argument('--config',required=True)
        p.add_argument('--prepared-root',default=None,help='explicit data path override; original YAML is never edited')
        p.add_argument('--gene-axis',default=None,help='explicit frozen 280-gene text path override; SHA still enforced')
    p=sub.add_parser('preflight');add_proj(p);p.add_argument('--morph-template');p.set_defaults(func=cmd_preflight)
    p=sub.add_parser('freeze');add_proj(p);p.add_argument('--out',required=True)
    p.add_argument('--train-cap',type=int,default=1024);p.add_argument('--val-cap',type=int,default=512)
    p.add_argument('--test-cap',type=int,default=512);p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--morph-template');p.add_argument('--accept-full',action='store_true');p.set_defaults(func=cmd_freeze)
    p=sub.add_parser('check');p.add_argument('--frozen',required=True);p.set_defaults(func=cmd_check_run)
    p=sub.add_parser('extract');p.add_argument('--frozen',required=True);p.add_argument('--sid',required=True)
    p.add_argument('--context',type=int,choices=CONTEXTS,required=True)
    p.add_argument('--encoder',choices=('imagenet-r18','phikon-v2','uni2-h'),required=True)
    p.add_argument('--device',default='auto',choices=('auto','cuda','cpu'))
    p.add_argument('--batch',type=int,default=4);p.add_argument('--amp',action='store_true')
    p.add_argument('--log-every',type=int,default=200);p.set_defaults(func=cmd_extract)
    p=sub.add_parser('morph');p.add_argument('--frozen',required=True);p.add_argument('--sid',required=True)
    p.add_argument('--context',type=int,choices=CONTEXTS,required=True);p.set_defaults(func=cmd_morph)
    p=sub.add_parser('fit');p.add_argument('--frozen',required=True);p.add_argument('--method',required=True,
                                      choices=('imagenet-r18','phikon-v2','uni2-h','morph'))
    p.add_argument('--context',type=int,choices=CONTEXTS,required=True)
    p.add_argument('--target',choices=('raw','clean'),required=True)
    p.add_argument('--labels-from',default='imagenet-r18',choices=('imagenet-r18','phikon-v2','uni2-h'))
    p.add_argument('--alphas',type=float,nargs='+',default=DEFAULT_ALPHAS)
    p.add_argument('--out',required=True);p.set_defaults(func=cmd_fit)
    p=sub.add_parser('score-official');p.add_argument('--frozen',required=True)
    p.add_argument('--method-name',required=True);p.add_argument('--predictions',required=True)
    p.add_argument('--training-provenance',required=True);p.add_argument('--context',type=int,choices=CONTEXTS,required=True)
    p.add_argument('--labels-from',default='imagenet-r18',choices=('imagenet-r18','phikon-v2','uni2-h'))
    p.add_argument('--out',required=True);p.set_defaults(func=cmd_official)
    p=sub.add_parser('aggregate-folds');p.add_argument('--runs',nargs='+',required=True)
    p.add_argument('--out',required=True);p.set_defaults(func=cmd_aggregate_folds)
    p=sub.add_parser('export-bridge');p.add_argument('--frozen',required=True)
    p.add_argument('--context',type=int,choices=CONTEXTS,default=128)
    p.add_argument('--labels-from',default='imagenet-r18',choices=('imagenet-r18','phikon-v2','uni2-h'))
    p.add_argument('--out',required=True);p.set_defaults(func=cmd_bridge)
    p=sub.add_parser('compare');p.add_argument('--runs',nargs='+',required=True)
    p.add_argument('--out',required=True);p.set_defaults(func=cmd_compare)
    p=sub.add_parser('interventions');p.add_argument('--input',required=True)
    p.add_argument('--out',required=True);p.set_defaults(func=cmd_intervention)
    args=parser.parse_args()
    if args.cmd=='fit':
        require(all(x>0 for x in args.alphas),'all alphas must be positive')
    args.func(args)


if __name__ == '__main__':
    main()

