#!/usr/bin/env python3
"""Source-cell counterfactual intervention using actual cell polygons.

Runs two perturbations by default: local Gaussian blur and biharmonic inpainting. A pair is
used only if both source and matched-control polygons are fully inside the VISIBLE context.
"""
import argparse, numpy as np, pandas as pd, torch
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
from torchvision.transforms import functional as TF
from torchvision import transforms as T
from leakage_vso.prepared import PreparedSample
from leakage_vso.wsi import TiffWSI
from leakage_vso.evaluate import load_checkpoint
from leakage_vso.utils import device_from_string
from leakage_vso.geometry import sort_polygon_vertices

p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--root',required=True); p.add_argument('--sample',required=True); p.add_argument('--pairs',required=True); p.add_argument('--polygons',default=None); p.add_argument('--out',required=True); p.add_argument('--device',default='auto'); p.add_argument('--blur-radius',type=float,default=8.0); p.add_argument('--modes',nargs='+',default=['blur','inpaint'],choices=['blur','inpaint']); a=p.parse_args()
dev=device_from_string(a.device); model,cfg,genes,method=load_checkpoint(a.checkpoint,dev); model.eval(); s=PreparedSample.load(a.root,a.sample); wsi=TiffWSI(s.image_path)
pairs=pd.read_csv(a.pairs); poly_path=Path(a.polygons) if a.polygons else Path(a.root)/'samples'/a.sample/'cell_polygons.parquet'
if not poly_path.exists(): raise FileNotFoundError(f'Cell polygons required: {poly_path}')
poly=pd.read_parquet(poly_path); cells=s.cells.set_index(s.cells.cell_id.astype(str)); poly['cell_id']=poly.cell_id.astype(str)
required={'cell_id','x_px','y_px'}
if not required.issubset(poly.columns): raise RuntimeError(f'polygon parquet needs {required}')
context_um=float(cfg['data'].get('context_um',64)); canvas_um=float(cfg['data'].get('canvas_um',context_um)); model_mpp=float(cfg['data'].get('model_mpp',0.5))
canvas_native=max(8,int(round(canvas_um/s.pixel_size_um))); model_px=max(8,int(round(canvas_um/model_mpp))); visible_native=float(context_um/s.pixel_size_um)
norm=T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])

def crop_for(recipient_id):
    r=cells.loc[str(recipient_id)]; return wsi.crop_px(float(r.x_px),float(r.y_px),canvas_native),float(r.x_px),float(r.y_px)

def polygon_local(cx,cy,cell_id):
    q=sort_polygon_vertices(poly[poly.cell_id==str(cell_id)])
    if len(q)<3: raise RuntimeError(f'No polygon with >=3 vertices for {cell_id}')
    x0=cx-canvas_native/2; y0=cy-canvas_native/2
    return np.asarray([(float(x-x0),float(y-y0)) for x,y in q[['x_px','y_px']].to_numpy()],dtype=float)

def fully_visible(pts):
    half=visible_native/2; c=canvas_native/2
    return bool(np.all((pts[:,0]>=c-half)&(pts[:,0]<c+half)&(pts[:,1]>=c-half)&(pts[:,1]<c+half)))

def polygon_mask(img,pts):
    mask=Image.new('L',img.size,0); ImageDraw.Draw(mask).polygon([tuple(x) for x in pts],fill=255); return mask

def perturb(img,pts,mode):
    mask=polygon_mask(img,pts)
    if mode=='blur':
        blurred=img.filter(ImageFilter.GaussianBlur(radius=a.blur_radius)); return Image.composite(blurred,img,mask)
    from skimage.restoration import inpaint
    arr=np.asarray(img).astype(np.float32)/255.0; m=np.asarray(mask)>0
    out=inpaint.inpaint_biharmonic(arr,m,channel_axis=-1); return Image.fromarray(np.clip(out*255,0,255).astype(np.uint8))

def tensor(img):
    t=norm(TF.to_tensor(TF.resize(img,[model_px,model_px],antialias=True)))
    if context_um<canvas_um:
        h,w=t.shape[-2:]; frac=context_um/canvas_um; vh=max(1,int(round(h*frac))); vw=max(1,int(round(w*frac))); y0=(h-vh)//2; x0=(w-vw)//2
        z=torch.zeros_like(t); z[:,y0:y0+vh,x0:x0+vw]=t[:,y0:y0+vh,x0:x0+vw]; t=z
    return t.unsqueeze(0).to(dev)

def predict(img):
    x=tensor(img)
    with torch.no_grad():
        if method=='endovso': y=model(x,x)['endo']
        else: y=model(x)
    return y.squeeze(0).cpu().numpy()

rows=[]; skipped_visibility=0
for _,r in pairs.iterrows():
    g=str(r.gene)
    if g not in genes: continue
    img,cx,cy=crop_for(r.recipient_cell_id); sp=polygon_local(cx,cy,r.source_cell_id); cp=polygon_local(cx,cy,r.control_cell_id)
    if not fully_visible(sp) or not fully_visible(cp): skipped_visibility+=1; continue
    y0=predict(img); j=genes.index(g)
    base={**r.to_dict(),'pred_original':float(y0[j])}
    for mode in a.modes:
        ys=predict(perturb(img,sp,mode)); yc=predict(perturb(img,cp,mode))
        rows.append({**base,'intervention':mode,'pred_source_masked':float(ys[j]),'pred_control_masked':float(yc[j]),
                     'delta_source':float(y0[j]-ys[j]),'delta_control':float(y0[j]-yc[j]),'delta_source_minus_control':float((y0[j]-ys[j])-(y0[j]-yc[j]))})
out=pd.DataFrame(rows); Path(a.out).parent.mkdir(parents=True,exist_ok=True); out.to_csv(a.out,index=False)
print(f'rows={len(out)} skipped_not_fully_visible={skipped_visibility}')
if len(out): print(out.groupby(['intervention','gene'])[['delta_source','delta_control','delta_source_minus_control']].mean().to_string())
