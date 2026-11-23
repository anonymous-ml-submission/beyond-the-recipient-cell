from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
from .wsi import TiffWSI


def validate_sample(sample_dir: str|Path, min_in_bounds=0.98):
    d=Path(sample_dir); cells=pd.read_parquet(d/"cells.parquet"); image=Path((d/"image_path.txt").read_text().strip())
    w=TiffWSI(image)
    x=cells.x_px.to_numpy(float); y=cells.y_px.to_numpy(float)
    ok=(x>=0)&(x<w.width)&(y>=0)&(y<w.height)
    frac=float(ok.mean())
    if frac<min_in_bounds:
        raise RuntimeError(f"Only {frac:.3%} centroids land inside image. Refusing to train; registration/scale is wrong.")
    return {"n_cells":len(cells),"fraction_in_bounds":frac,"image_width":w.width,"image_height":w.height,"ome_pixel_size_um_x":w.pixel_size_um_x,"ome_pixel_size_um_y":w.pixel_size_um_y}
