from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import anndata as ad
import pandas as pd
import numpy as np
import scipy.sparse as sp

@dataclass
class PreparedSample:
    root: Path
    sample_id: str
    cells: pd.DataFrame
    raw: ad.AnnData
    clean: ad.AnnData | None
    image_path: Path
    pixel_size_um: float

    @classmethod
    def load(cls, root: str | Path, sample_id: str):
        d=Path(root)/"samples"/sample_id
        cells=pd.read_parquet(d/"cells.parquet")
        raw=ad.read_h5ad(d/"raw.h5ad")
        clean_path=d/"clean.h5ad"; clean=ad.read_h5ad(clean_path) if clean_path.exists() else None
        image_path=Path((d/"image_path.txt").read_text().strip())
        pixel_size_um=float((d/"pixel_size_um.txt").read_text().strip())
        if len(cells)!=raw.n_obs:
            raise ValueError(f"Cell table/raw row-count mismatch in {sample_id}: {len(cells)} vs {raw.n_obs}")
        for x,name in ((raw,"raw"),(clean,"clean")):
            if x is None: continue
            if not x.obs_names.equals(pd.Index(cells.cell_id.astype(str))):
                raise ValueError(f"Cell order mismatch in {name} matrix for prepared sample {sample_id}")
        if clean is not None and not raw.var_names.equals(clean.var_names):
            raise ValueError(f"Gene axis/order mismatch raw vs clean in prepared sample {sample_id}")
        for x,name in ((raw,"raw"),(clean,"clean")):
            if x is None: continue
            vals=x.X.data if sp.issparse(x.X) else np.asarray(x.X).ravel()
            if len(vals) and (not np.isfinite(vals).all() or float(vals.min())<0):
                raise ValueError(f"{name} expression must be finite and nonnegative in {sample_id}")
        if pixel_size_um<=0: raise ValueError(f"Invalid pixel_size_um={pixel_size_um} for {sample_id}")
        return cls(Path(root),sample_id,cells,raw,clean,image_path,pixel_size_um)

    @property
    def genes(self): return self.raw.var_names.to_list()
