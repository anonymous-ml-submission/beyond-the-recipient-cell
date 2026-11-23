from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from .prepared import PreparedSample
from .wsi import TiffWSI


def _dense_rows(X, idx, gene_idx):
    sub = X[idx][:, gene_idx]
    if hasattr(sub, "toarray"):
        sub = sub.toarray()
    return np.asarray(sub, dtype=np.float32)


def _resize_physical(img, width_um: float, model_mpp: float):
    n = max(8, int(round(float(width_um) / float(model_mpp))))
    return TF.resize(img, [n, n], antialias=True)


def _zero_outside_center(t: torch.Tensor, visible_um: float, canvas_um: float) -> torch.Tensor:
    """Hide tissue outside a centered physical FOV while preserving tensor geometry.

    `t` is already ImageNet-normalized, so zero is the ImageNet mean and is a neutral fill.
    """
    if visible_um >= canvas_um - 1e-9:
        return t
    h,w=t.shape[-2:]
    frac=float(visible_um)/float(canvas_um)
    vh=max(1,min(h,int(round(h*frac)))); vw=max(1,min(w,int(round(w*frac))))
    y0=(h-vh)//2; x0=(w-vw)//2
    out=torch.zeros_like(t); out[:,y0:y0+vh,x0:x0+vw]=t[:,y0:y0+vh,x0:x0+vw]
    return out


class CellImageDataset(Dataset):
    """Cell-centred H&E dataset with strict paper-protocol safeguards.

    `context_um` is the VISIBLE physical crop width. `canvas_um` is the physical width
    represented by the network tensor. For the primary CVPR context ablation set one
    fixed `canvas_um` (e.g. 128) for every context value, so 24/64/128 µm runs have
    identical MPP and identical tensor geometry and differ only in visible tissue.
    """
    def __init__(self, prepared_root, sample_ids, genes, target="raw", context_um=64.0,
                 center_um=24.0, model_mpp=0.5, canvas_um=None, split="train", split_col="split",
                 augment=False, ring_view=False, ring_shuffle=False):
        self.context_um=float(context_um); self.center_um=float(center_um); self.model_mpp=float(model_mpp)
        self.canvas_um=float(canvas_um if canvas_um is not None else context_um)
        if self.context_um < self.center_um:
            raise ValueError(f"context_um ({context_um}) must be >= center_um ({center_um})")
        if self.canvas_um < self.context_um:
            raise ValueError(f"canvas_um ({self.canvas_um}) must be >= context_um ({self.context_um})")
        if self.model_mpp <= 0: raise ValueError("model_mpp must be >0")
        self.samples=[]; self.rows=[]; self.genes=list(genes); self.target=target
        self.ring_view=ring_view; self.ring_shuffle=bool(ring_shuffle); self._wsi={}
        for sid in sample_ids:
            s=PreparedSample.load(prepared_root,sid)
            missing=[g for g in genes if g not in s.raw.var_names]
            if missing: raise ValueError(f"Sample {sid} is missing {len(missing)} requested genes; first={missing[:5]}")
            gene_idx=[s.raw.var_names.get_loc(g) for g in genes]
            if target in {"clean","both"} and s.clean is None:
                raise ValueError(f"Sample {sid} has no clean.h5ad; cannot use target={target}")
            if split is not None:
                if split_col not in s.cells.columns:
                    raise RuntimeError(f"Sample {sid} lacks required split column '{split_col}'. Refusing to fall back to all cells.")
                mask=s.cells[split_col].astype(str).to_numpy()==str(split)
                if not np.any(mask): raise RuntimeError(f"Sample {sid} has zero rows for requested split='{split}'")
            else: mask=np.ones(len(s.cells),dtype=bool)
            sample_idx=len(self.samples)
            for li in np.where(mask)[0]: self.rows.append((sample_idx,int(li)))
            self.samples.append((s,gene_idx))
        if not self.rows: raise RuntimeError(f"Dataset contains zero rows for split={split}")
        self.augment=bool(augment)
        self.to_tensor=T.Compose([T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

    def __len__(self): return len(self.rows)

    def _get_wsi(self,s):
        key=str(s.image_path)
        if key not in self._wsi: self._wsi[key]=TiffWSI(s.image_path)
        return self._wsi[key]

    def _views(self, s, row):
        wsi=self._get_wsi(s); x=float(row.x_px); y=float(row.y_px)
        canvas_px=max(8,int(round(self.canvas_um/s.pixel_size_um)))
        center_px=max(8,int(round(self.center_um/s.pixel_size_um)))
        outer=wsi.crop_px(x,y,canvas_px); center=wsi.crop_px(x,y,center_px)
        return center, outer

    def __getitem__(self,idx):
        si,li=self.rows[idx]; s,gidx=self.samples[si]; row=s.cells.iloc[li]
        center,outer=self._views(s,row)
        ring_source=outer
        if self.ring_shuffle and len(self.rows)>1:
            alt_idx=(idx*104729+17)%len(self.rows)
            if alt_idx==idx: alt_idx=(alt_idx+1)%len(self.rows)
            asi,ali=self.rows[alt_idx]; asamp,_=self.samples[asi]; arow=asamp.cells.iloc[ali]
            _,ring_source=self._views(asamp,arow)
        ring_np=np.asarray(ring_source).copy(); h,w=ring_np.shape[:2]
        frac=min(0.95,self.center_um/self.canvas_um)
        rh=max(1,int(round(h*frac))); rw=max(1,int(round(w*frac)))
        y0=(h-rh)//2; x0=(w-rw)//2; ring_np[y0:y0+rh,x0:x0+rw]=np.array([124,116,104],dtype=np.uint8)
        from PIL import Image
        ring=Image.fromarray(ring_np)
        if self.augment:
            import random
            if random.random()<0.5:
                center=TF.hflip(center); outer=TF.hflip(outer); ring=TF.hflip(ring)
            if random.random()<0.5:
                center=TF.vflip(center); outer=TF.vflip(outer); ring=TF.vflip(ring)
            angle=random.uniform(0.0,360.0); fill=(124,116,104)
            center=TF.rotate(center,angle,fill=fill); outer=TF.rotate(outer,angle,fill=fill); ring=TF.rotate(ring,angle,fill=fill)
        center=_resize_physical(center,self.center_um,self.model_mpp)
        outer=_resize_physical(outer,self.canvas_um,self.model_mpp)
        ring=_resize_physical(ring,self.canvas_um,self.model_mpp)
        center_t=self.to_tensor(center)
        outer_t=_zero_outside_center(self.to_tensor(outer),self.context_um,self.canvas_um)
        ring_t=_zero_outside_center(self.to_tensor(ring),self.context_um,self.canvas_um)
        raw=_dense_rows(s.raw.X,[li],gidx)[0]
        item={"center":center_t,"outer":outer_t,"ring":ring_t,"raw":torch.from_numpy(raw),
              "cell_id":str(row.cell_id),"sample_id":s.sample_id,"donor_id":str(row.get("donor_id",s.sample_id)),
              "x_um":float(row.get("x_um",np.nan)),"y_um":float(row.get("y_um",np.nan))}
        if "cell_type" in row.index: item["cell_type"]=str(row.cell_type)
        if s.clean is not None:
            clean=_dense_rows(s.clean.X,[li],gidx)[0]; item["clean"]=torch.from_numpy(clean)
            item["residual"]=torch.from_numpy(np.maximum(raw-clean,0))
        return item
