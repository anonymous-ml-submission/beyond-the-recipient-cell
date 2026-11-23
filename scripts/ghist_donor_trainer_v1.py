#!/usr/bin/env python3
"""Donor-held-out ADAPTED GHIST baseline for CVPR_Bio2; upstream Framework unchanged.

Train: P2/P3 ONLY; model selection: P4 ONLY; final test: P1 ONLY.
Raw 280-gene log1p targets; upstream GHIST six non-neighborhood losses.
No inferred B/Myeloid/Malignant labels. Fixed, non-data-fitted RGB normalization.
One WSI tile / forward; variable nucleus count, on-demand 256px nucleus raster.
This is a new adapted baseline, NOT upstream's exact published training pipeline.

Requires the existing validated ghist_patient_adapter_v1.py and its 11 outputs.
No install, no changes to source data, no GPU0/GPU1 auto-occupancy.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, random, sys, time
from pathlib import Path
from collections import defaultdict
import numpy as np

TRAIN=['TENX195','TENX196','TENX197','TENX198','TENX199']
VAL=['TENX200','TENX201','TENX202']
TEST=['TENX191','TENX192','TENX193']
PIN='917456be305fc82e92293ea272812e79675e821c'


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()


def requirements(root):
    import torch
    sys.path.insert(0,str(root/'scripts'))
    import ghist_patient_adapter_v1 as ad
    ad.validate_repo(root/'external_methods/GHIST')
    assert len(ad.CLASS)==4
    return torch,ad


def validate_manifests(root,ad, samples):
    import pandas as pd
    ref=(root/'runs/frozen_primary2025/gene_axis_280.txt')
    assert ref.is_file()
    genes=[z.strip() for z in ref.read_text().splitlines() if z.strip()]
    assert len(genes)==280 and len(set(genes))==280
    manifests={}
    for sid in samples:
        f=root/'runs/ghist_patient_adapter_v1'/sid
        assert f.is_dir(),f'STOP: unprepared {sid}'
        m=json.loads((f/'manifest.json').read_text())
        assert m['repo_commit']==PIN and m['sample']==sid
        assert m['donor']==ad.canon(sid) and m['role']==ad.ROLE[m['donor']]
        assert m['expression_target']=='raw_counts'
        assert m['context_um']==128 and m['model_image_px']==256
        assert m['exact_mapped_biological_classes']==list(ad.CLASS)
        assert m['genes_sha256']==sha(ref)
        assert [s.strip() for s in (f/'genes.txt').read_text().splitlines() if s.strip()]==genes
        arr=np.load(f/'raw_expression.npy',mmap_mode='r',allow_pickle=False)
        assert arr.ndim==2 and arr.shape==(m['eligible_cells'],280) and arr.dtype==np.float32
        assert (f/'cells.parquet').is_file()
        # Read Parquet footer only; no unnecessary H&E / RNA batch pass.
        import pyarrow.parquet as pq
        assert pq.read_metadata(f/'cells.parquet').num_rows==m['eligible_cells']
        assert Path(m['image_path']).is_file()
        manifests[sid]=m
    return genes,manifests


class StableTiles:
    """Read existing adapter tiles, retaining all nuclei visible after downsampling.

    Dropped subpixel nuclei are excluded from BOTH raster labels and RNA targets
    using a deterministic geometry-only mask. The number excluded is reported.
    """
    def __init__(self, adapter, out):
        self.base=adapter.TileDataset(out)
        self.manifest=self.base.manifest
        self.n_total=0
        self.n_dropped=0
        self.n_empty=0

    def __len__(self):return len(self.base)

    def __getitem__(self,idx):
        from PIL import Image,ImageDraw
        from shapely import wkb
        from leakage_vso.wsi import TiffWSI
        ds=self.base
        (ty,tx),sel=ds.tiles[idx]
        rows=ds.cells.iloc[sel]
        n=ds.native;x0=tx*n;y0=ty*n
        labimg=Image.new('I',(256,256),0)
        draw=ImageDraw.Draw(labimg)
        for j,r in enumerate(rows.itertuples(index=False),1):
            polygon=wkb.loads(r.geometry_wkb)
            assert polygon.geom_type=='Polygon' and polygon.is_valid
            pts=[((x-x0)*256/n,(y-y0)*256/n) for x,y in polygon.exterior.coords]
            draw.polygon(pts,fill=j)
            for ring in polygon.interiors:
                draw.polygon([((x-x0)*256/n,(y-y0)*256/n) for x,y in ring.coords],fill=0)
        rawmask=np.asarray(labimg,dtype=np.int32)
        labels=np.unique(rawmask)
        labels=labels[labels>0]
        assert all(1<=int(j)<=len(sel) for j in labels)
        self.n_total+=len(sel)
        self.n_dropped+=len(sel)-len(labels)
        if len(labels)==0:
            self.n_empty+=1
            return None
        chosen=(labels-1).astype(np.int64)
        lut=np.zeros(len(sel)+1,dtype=np.int32)
        lut[labels]=np.arange(1,len(labels)+1,dtype=np.int32)
        mask=lut[rawmask]
        ct=rows.cell_type_id.to_numpy(dtype=np.int64)[chosen]
        typelut=np.zeros(len(labels)+1,dtype=np.int64)
        typelut[1:]=ct+1
        type_mask=typelut[mask]
        assert mask.max()==len(labels) and (type_mask<=4).all()
        if ds._wsi is None:ds._wsi=TiffWSI(ds.image_path)
        img=ds._wsi.crop_px(x0+n/2,y0+n/2,n).convert('RGB')
        hist=np.asarray(img.resize((256,256),Image.Resampling.BILINEAR),dtype=np.float32)
        if hist.shape!=(256,256,3) or not np.isfinite(hist).all():
            raise RuntimeError('STOP: malformed WSI tile')
        expression=np.asarray(ds.expr[sel[chosen]],dtype=np.float32)
        ids=rows.cell_id.iloc[chosen].astype(str).tolist()
        assert expression.shape==(len(ids),280)
        return {'he':hist.transpose(2,0,1),'mask':mask,'type_mask':type_mask,
                'class':ct,'expr':expression,'cell_id':ids,'tile':(int(ty),int(tx)),
                'lost':len(sel)-len(labels),'original_cells':len(sel)}


def new_model(root,dev):
    import torch
    sys.path.insert(0,str(root/'external_methods/GHIST'))
    from model.model import Framework
    from ghist_patient_adapter_v1 import CLASS
    return Framework(n_classes=len(CLASS),n_genes=280,emb_dim=64,
                     device=dev,n_ref=None,use_avgexp=False,
                     use_celltype=True,use_neighb=False).to(dev)


def forward(model,tile,dev,supervised=True):
    import torch
    he=torch.from_numpy(np.ascontiguousarray(tile['he'][None])).to(dev,dtype=torch.float32)
    he=(he-128.0)/64.0 # FROZEN fixed transform from existing forward/backward smoke
    mask=torch.from_numpy(np.ascontiguousarray(tile['mask'][None])).to(dev,dtype=torch.long)
    pixel_type=torch.from_numpy(np.ascontiguousarray(tile['type_mask'][None])).to(dev,dtype=torch.long)
    types=torch.from_numpy(np.ascontiguousarray(tile['class'][None])).to(dev,dtype=torch.long)
    y=torch.from_numpy(np.ascontiguousarray(np.log1p(tile['expr'][None]))).to(dev,dtype=torch.float32)
    n=torch.tensor([[len(tile['cell_id'])]],dtype=torch.long,device=dev)
    # During validation/test, NEVER give the model the assay target or cell-type
    # ground truth. Supervised training supplies them for auxiliary losses only.
    outputs=(model(he,mask,n,None,types,y) if supervised else
             model(he,mask,n,None,None,None,do_st_mlp=False))
    assert len(outputs)==14 and outputs[1].shape==(1,5,256,256)
    assert outputs[3].shape==(len(tile['cell_id']),280)
    if supervised:
        assert torch.equal(outputs[2],types[0]) and torch.equal(outputs[10],y[0])
    return outputs,pixel_type,y[0]


def train_losses(outs,pixel_type,y):
    import torch
    import torch.nn.functional as F
    (ct_pred, seg_pred, ct_targets, expr_pred, _, _, expr_ct_pred,
     fv_expr_ct, gt_ct_logits, fv_gt_ct, y_for_loss, _, _, _)=outs
    assert torch.equal(y,y_for_loss)
    losses={
      'seg_ce':F.cross_entropy(seg_pred,pixel_type),
      'hist_ct_ce':F.cross_entropy(ct_pred,ct_targets),
      'expr_ct_ce':F.cross_entropy(expr_ct_pred,ct_targets),
      'expr_mse':F.mse_loss(expr_pred,y_for_loss),
      'ct_embed_cos':100*F.cosine_embedding_loss(fv_expr_ct,fv_gt_ct,
                             torch.ones(len(ct_targets),device=y.device)),
      'ct_logits_mse':F.mse_loss(expr_ct_pred,gt_ct_logits),
    }
    assert all(torch.isfinite(v).item() for v in losses.values()), 'STOP: nonfinite GHIST loss'
    return losses


def enumerate_tiles(datasets,seed,cap=0):
    rng=np.random.default_rng(seed)
    groups=[]
    for sid,d in datasets.items():
        ix=rng.permutation(len(d))
        if cap:ix=ix[:cap]
        groups.append([(sid,int(i)) for i in ix])
    # Round-robin across slides; no patient/slide is entirely delayed to epoch end.
    for j in range(max(map(len,groups))):
        for group in groups:
            if j<len(group):yield group[j]


def val_tiles(datasets,limit=256):
    rng=np.random.default_rng(314159) # SAME fixed validation subset for every seed
    for sid,d in datasets.items():
        choices=rng.choice(len(d),size=min(limit,len(d)),replace=False)
        for i in sorted(choices):yield sid,int(i)


def evaluate_tiles(model,datasets,dev,indices,return_data=False):
    import torch
    sums=defaultdict(lambda:np.zeros((5,280),dtype=np.float64))
    n=defaultdict(int)
    mses=defaultdict(float)
    drop=defaultdict(int)
    orig=defaultdict(int)
    rows=defaultdict(lambda: {'ids':[],'pred':[],'truth':[]})
    model.eval()
    with torch.inference_mode():
        for sid,i in indices:
            tile=datasets[sid][i]
            if tile is None:continue
            outs,_,y=forward(model,tile,dev,supervised=False)
            p=outs[3].detach().float().cpu().numpy().astype(np.float64)
            yy=y.detach().float().cpu().numpy().astype(np.float64)
            assert np.isfinite(p).all() and np.isfinite(yy).all()
            st=sums[sid]
            st[0]+=yy.sum(0);st[1]+=(yy**2).sum(0)
            st[2]+=p.sum(0);st[3]+=(p**2).sum(0)
            st[4]+=(yy*p).sum(0)
            n[sid]+=len(p)
            mses[sid]+=np.square(p-yy).sum()
            drop[sid]+=tile['lost'];orig[sid]+=tile['original_cells']
            if return_data:
                rows[sid]['ids'].extend(tile['cell_id'])
                rows[sid]['pred'].append(p.astype(np.float32))
                rows[sid]['truth'].append(yy.astype(np.float32))
    assert all(v>0 for v in n.values()), 'STOP: evaluation produced no cells'
    mean_mse=np.mean([mses[sid]/(n[sid]*280) for sid in n])
    return dict(stats=sums,n=dict(n),mse=float(mean_mse),dropped=dict(drop),original=dict(orig),preds=rows)


def metrics_from_eval(r,genes,split):
    import pandas as pd
    out=[]
    for sid,st in r['stats'].items():
        N=r['n'][sid]
        cov=st[4]-st[0]*st[2]/N
        vy=st[1]-st[0]**2/N;vp=st[3]-st[2]**2/N
        valid=(vy>1e-10)&(vp>1e-10)
        R=np.full(280,np.nan)
        R[valid]=np.clip(cov[valid]/np.sqrt(vy[valid]*vp[valid]),-.999999,.999999)
        for j,g in enumerate(genes):
            out.append({'sample':sid,'patient':{'val':'P4','test':'P1'}[split],
                        'gene':g,'pearson_r':R[j],'fisher_z':np.arctanh(R[j]),
                        'n_cells':N,'original_nuclei_in_evaluated_tiles':r['original'].get(sid,0),
                        'subpixel_nuclei_excluded':r['dropped'].get(sid,0)})
    d=pd.DataFrame(out)
    summary=d.groupby('sample',as_index=False).agg(
        n_cells=('n_cells','first'),mean_gene_fisher_z=('fisher_z','mean'),
        valid_genes=('fisher_z','count'),subpixel_nuclei_excluded=('subpixel_nuclei_excluded','first'))
    return d,summary


def atomic_torch_save(torch,obj,path):
    tmp=path.with_name(path.name+'.tmp-'+str(os.getpid()))
    assert not tmp.exists()
    try:
        torch.save(obj,tmp)
        os.replace(tmp,path)
    finally:
        if tmp.exists():tmp.unlink()


def atomic_json(obj,path):
    tmp=path.with_name(path.name+'.tmp-'+str(os.getpid()))
    tmp.write_text(json.dumps(obj,indent=2)+'\n')
    os.replace(tmp,path)


def make_datasets(root,ad,samples):
    return {sid:StableTiles(ad,root/'runs/ghist_patient_adapter_v1'/sid) for sid in samples}


def run(args):
    root=args.root.resolve()
    assert (root/'scripts/ghist_patient_adapter_v1.py').is_file()
    assert args.epochs>0 and args.lr>0 and args.val_tiles_per_section>0
    assert args.train_tiles_per_section>=0 and args.accum_steps>0
    torch,ad=requirements(root)
    role_samples={'train':TRAIN,'val':VAL,'test':TEST}
    if args.mode=='audit':
        genes,m=validate_manifests(root,ad,TRAIN+VAL+TEST)
        for sid in TRAIN+VAL+TEST:
            a=m[sid]
            print('ADAPTER_CONTRACT_PASS',sid,a['role'],a['eligible_cells'],a['eligible_tiles'],flush=True)
        print('GHIST_11_SLIDE_DONOR_CONTRACT_PASS; NO_TRAINING_NO_TEST_TARGETS_LOADED')
        return
    if args.mode in ('pilot','train'):
        genes,manifest=validate_manifests(root,ad,TRAIN+VAL)
        assert not set(manifest)&set(TEST)
        ds_tr=make_datasets(root,ad,TRAIN)
        ds_val=make_datasets(root,ad,VAL)
    else:
        assert args.mode=='evaluate'
        assert args.eval_role in ('val','test')
        genes,manifest=validate_manifests(root,ad,role_samples[args.eval_role])
        ds_ev=make_datasets(root,ad,role_samples[args.eval_role])
    if args.device=='cuda':
        assert torch.cuda.is_available(), 'STOP: requested CUDA device unavailable'
        dev=torch.device('cuda:0')
    else:dev=torch.device('cpu')
    if args.mode in ('pilot','train'):
        torch.manual_seed(args.seed);random.seed(args.seed);np.random.seed(args.seed)
        model=new_model(root,dev)
        opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
        if args.mode=='pilot':
            model.train()
            step=0
            for sid,idx in enumerate_tiles(ds_tr,args.seed,cap=2):
                tile=ds_tr[sid][idx]
                if tile is None:continue
                opt.zero_grad(set_to_none=True)
                outs,mask,y=forward(model,tile,dev)
                ls=train_losses(outs,mask,y)
                loss=sum(ls.values())
                loss.backward()
                assert any(p.grad is not None for p in model.parameters())
                assert all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)
                opt.step()
                step+=1
                print('PILOT_STEP',sid,'CELLS',len(tile['cell_id']),'LOSS',float(loss.detach().cpu()),'LOST',tile['lost'],flush=True)
                if step>=5:break
            assert step==5,'STOP: fewer than five actual training tile steps'
            val=evaluate_tiles(model,ds_val,dev,val_tiles(ds_val,limit=2))
            print('GHIST_PATIENT_SAFE_PILOT_PASS','VAL_LOG_MSE',val['mse'],'SECTIONS',val['n'])
            print('NO_CHECKPOINT_NO_TRAINING_RUN_NO_P1_ACCESSED')
            return
        cfg={'model':'upstream_Framework','repo_commit':PIN,'adapter':'ghist_patient_adapter_v1',
             'run_script_sha256':sha(Path(__file__)),'adapter_sha256':sha(root/'scripts/ghist_patient_adapter_v1.py'),
             'target':'log1p(raw_counts)','image_normalization':'fixed (RGB-128)/64',
             'losses':['seg_ce','hist_ct_ce','expr_ct_ce','expr_mse','ct_embed_cos','ct_logits_mse'],
             'uses_neighborhood':False,'uses_avgexp':False,'uses_celltype':True,
             'train_samples':TRAIN,'val_samples':VAL,'test_samples':TEST,
             'epochs':args.epochs,'seed':args.seed,'lr':args.lr,'weight_decay':args.weight_decay,
             'gradient_accumulation':args.accum_steps,'train_tiles_per_section':args.train_tiles_per_section,
             'val_tiles_per_section':args.val_tiles_per_section,
             'model_selection':'minimum equal-section mean log-MSE on fixed P4 subset; P1 never used'}
        rd=args.out.resolve()
        assert rd!=root and root in rd.parents
        if args.resume:
            assert rd.is_dir() and (rd/'run.json').is_file() and (rd/'last.pt').is_file()
            assert json.loads((rd/'run.json').read_text())==cfg,'STOP: resume config changed'
            checkpoint=torch.load(rd/'last.pt',map_location=dev,weights_only=False)
            assert checkpoint['cfg']==cfg
            model.load_state_dict(checkpoint['model']);opt.load_state_dict(checkpoint['optimizer'])
            start=checkpoint['epoch']+1
            best_val=checkpoint['best_val']
        else:
            assert not rd.exists(),f'STOP: existing GHIST run directory {rd}'
            rd.mkdir(parents=True,exist_ok=False)
            (rd/'run.json').write_text(json.dumps(cfg,indent=2)+'\n')
            start=1;best_val=float('inf')
        for epoch in range(start,args.epochs+1):
            model.train();opt.zero_grad(set_to_none=True)
            steps=0;total=0.0;seen=0;dropped=0
            for sid,idx in enumerate_tiles(ds_tr,args.seed+epoch,cap=args.train_tiles_per_section):
                tile=ds_tr[sid][idx]
                if tile is None:continue
                outs,mask,y=forward(model,tile,dev)
                losses=train_losses(outs,mask,y)
                loss=sum(losses.values())
                (loss/args.accum_steps).backward()
                steps+=1;total+=float(loss.detach().cpu());seen+=len(tile['cell_id']);dropped+=tile['lost']
                if steps%args.accum_steps==0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),5.0,error_if_nonfinite=True)
                    opt.step();opt.zero_grad(set_to_none=True)
                if steps%500==0:
                    print('TRAIN_PROGRESS',epoch,steps,'mean_loss',round(total/steps,5),'cells',seen,'dropped',dropped,flush=True)
            assert steps>0,'STOP: no gradient steps'
            if steps%args.accum_steps:
                # Correct scale of the final incomplete accumulation group.
                rem=steps%args.accum_steps
                for p in model.parameters():
                    if p.grad is not None:p.grad.mul_(args.accum_steps/rem)
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.0,error_if_nonfinite=True)
                opt.step();opt.zero_grad(set_to_none=True)
            val=evaluate_tiles(model,ds_val,dev,val_tiles(ds_val,args.val_tiles_per_section))
            assert set(val['n'])==set(VAL),'STOP: missing P4 validation section'
            score=val['mse']
            improved=score<best_val
            if improved:best_val=score
            state={'epoch':epoch,'cfg':cfg,'model':model.state_dict(),
                   'optimizer':opt.state_dict(),'best_val':best_val,'validation_score':score}
            atomic_torch_save(torch,state,rd/'last.pt')
            if improved:atomic_torch_save(torch,state,rd/'best.pt')
            line={'epoch':epoch,'train_tiles':steps,'train_cells_seen':seen,
                  'train_subpixel_nuclei_excluded':dropped,
                  'train_mean_objective':total/steps,'p4_fixed_subset_log_mse':score,
                  'p4_val_cells':val['n'],'best_checkpoint':improved}
            with (rd/'history.jsonl').open('a') as f:f.write(json.dumps(line)+'\n')
            print('EPOCH_COMPLETE',json.dumps(line),flush=True)
        print('GHIST_ADAPTED_DONOR_TRAINING_COMPLETE',rd/'best.pt','BEST_P4_LOG_MSE',best_val)
        return
    # Distinct complete evaluation: never test until all training epochs finish.
    rd=args.out.resolve()
    ck=rd/'best.pt'
    last=rd/'last.pt'
    assert ck.is_file() and last.is_file() and (rd/'run.json').is_file(), 'STOP: train/select on P4 first'
    config=json.loads((rd/'run.json').read_text())
    latest=torch.load(last,map_location='cpu',weights_only=False)
    assert latest['cfg']==config and latest['epoch']==config['epochs'], 'STOP: training incomplete; P1 not permitted'
    del latest
    checkpoint=torch.load(ck,map_location=dev,weights_only=False)
    assert checkpoint['cfg']==config and checkpoint['epoch']<=config['epochs']
    assert config['train_samples']==TRAIN and config['val_samples']==VAL and config['test_samples']==TEST
    assert config['adapter_sha256']==sha(root/'scripts/ghist_patient_adapter_v1.py'), 'STOP: adapter changed since training'
    ck_sha=sha(ck)
    model=new_model(root,dev)
    model.load_state_dict(checkpoint['model'])
    dest=rd/f'{args.eval_role}_full'
    if not args.resume:
        assert not dest.exists(),f'STOP: evaluation output exists {dest}; use --resume after interruption'
        dest.mkdir(parents=True,exist_ok=False)
    else:
        assert dest.is_dir(),'STOP: no interrupted evaluation directory'
    chunks=[]
    for sid in role_samples[args.eval_role]:
        sub=dest/sid
        marker=sub/'complete.json'
        if sub.exists():
            assert args.resume and marker.is_file(), f'STOP: incomplete evaluation section {sid}'
            item=json.loads(marker.read_text())
            assert item['checkpoint_sha256']==ck_sha and item['sample']==sid
            print('EVAL_ALREADY_COMPLETE',sid,flush=True)
            chunks.append(sub)
            continue
        sub.mkdir(parents=False,exist_ok=False)
        d=ds_ev[sid]
        import itertools
        # No other sample's target labels or predictions are loaded in this iteration.
        outcome=evaluate_tiles(model,{sid:d},dev,((sid,i) for i in range(len(d))),return_data=True)
        assert sid in outcome['n'] and outcome['n'][sid]>0
        detail,summary=metrics_from_eval(outcome,genes,args.eval_role)
        detail.to_csv(sub/'gene_section_metrics.csv',index=False)
        summary.to_csv(sub/'section_summary.csv',index=False)
        r=outcome['preds'][sid]
        pred=np.concatenate(r['pred'],axis=0)
        truth=np.concatenate(r['truth'],axis=0)
        ids=np.asarray(r['ids'],dtype='U')
        assert len(ids)==len(set(ids)) and len(ids)==len(pred)==len(truth)
        np.savez_compressed(sub/'predictions.npz',sample_id=np.array(sid),
                 cell_id=ids,genes=np.asarray(genes,dtype='U'),pred_log1p=pred,raw_log1p=truth)
        status={'checkpoint_sha256':ck_sha,'sample':sid,'n_cells':outcome['n'][sid],
                'dropped_subpixel_nuclei':outcome['dropped'].get(sid,0),
                'original_nuclei':outcome['original'].get(sid,0),
                'log_mse':outcome['mse'],'n_gene_metrics':len(detail),
                'predictions_sha256':sha(sub/'predictions.npz')}
        (sub/'complete.json').write_text(json.dumps(status,indent=2)+'\n')
        chunks.append(sub)
        print('GHIST_SECTION_EVAL_PASS',json.dumps(status),flush=True)
    import pandas as pd
    detail=pd.concat([pd.read_csv(q/'gene_section_metrics.csv') for q in chunks],ignore_index=True)
    summary=pd.concat([pd.read_csv(q/'section_summary.csv') for q in chunks],ignore_index=True)
    assert len(detail)==280*len(chunks) and set(detail['sample'])==set(role_samples[args.eval_role])
    detail.to_csv(dest/'gene_section_metrics.csv',index=False)
    summary.to_csv(dest/'section_summary.csv',index=False)
    (dest/'provenance.json').write_text(json.dumps({'checkpoint_sha256':ck_sha,
       'role':args.eval_role,'selected_epoch':checkpoint['epoch'],'samples':role_samples[args.eval_role],
       'model_definition':'GHIST adapted; no neighborhood/avgexp; fixed RGB normalization'},indent=2)+'\n')
    print('GHIST_'+args.eval_role.upper()+'_EVALUATION_COMPLETE',dest,flush=True)
    print(summary.to_string(index=False),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',required=True,choices=['audit','pilot','train','evaluate'])
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--out',type=Path,default=Path('runs/ghist_donor_adapted_v1/seed2026'))
    p.add_argument('--eval-role',choices=['val','test'],default='val')
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--epochs',type=int,default=8)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--weight-decay',type=float,default=1e-2)
    p.add_argument('--accum-steps',type=int,default=8)
    p.add_argument('--train-tiles-per-section',type=int,default=0,help='0=all tiles; nonzero for budget-constrained exploratory runs only')
    p.add_argument('--val-tiles-per-section',type=int,default=256)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    run(a)
if __name__=='__main__':main()
