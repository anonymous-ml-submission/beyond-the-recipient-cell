import numpy as np, tifffile
from leakage_vso.wsi import TiffWSI

def test_crop(tmp_path):
    a=np.zeros((100,120,3),dtype=np.uint8); a[...,0]=123
    p=tmp_path/"x.tif"; tifffile.imwrite(p,a)
    w=TiffWSI(p); im=np.asarray(w.crop_px(60,50,20)); assert im.shape==(20,20,3); assert im[...,0].mean()==123
