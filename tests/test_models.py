import torch
from leakage_vso.models import ContextRegressor, EndoVSO

def test_forward_shapes():
    x=torch.randn(2,3,64,64)
    m=ContextRegressor(5,pretrained=False); y=m(x); assert y.shape==(2,5); assert (y>=0).all()
    e=EndoVSO(5,pretrained=False); o=e(x,x); assert o["endo"].shape==(2,5); assert torch.allclose(o["obs"],o["endo"]+o["leak"])
