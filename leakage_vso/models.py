from __future__ import annotations
import torch.nn as nn
from torchvision import models


def build_backbone(name:str="resnet18",pretrained:bool=True):
    if name=="resnet18":
        m=models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None); d=m.fc.in_features; m.fc=nn.Identity(); return m,d
    if name=="resnet50":
        m=models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None); d=m.fc.in_features; m.fc=nn.Identity(); return m,d
    if name.startswith("timm:"):
        import timm; mn=name.split(":",1)[1]; m=timm.create_model(mn,pretrained=pretrained,num_classes=0); return m,m.num_features
    raise ValueError(f"Unknown backbone {name}")

class ContextRegressor(nn.Module):
    def __init__(self,n_genes:int,backbone="resnet18",pretrained=True,freeze_backbone=False):
        super().__init__(); self.encoder,d=build_backbone(backbone,pretrained)
        if freeze_backbone:
            for p in self.encoder.parameters(): p.requires_grad=False
        self.head=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,n_genes),nn.Softplus())
    def forward(self,x): return self.head(self.encoder(x))

class AnchoredTwoView(nn.Module):
    """Corrected-target anchored decomposition baseline.

    FULL H&E context -> endogenous expression (true biology may be context dependent).
    Ring-only H&E context -> nonnegative residual/contamination component.
    clean/corrected targets anchor the endogenous branch; raw-clean residual anchors leakage.
    This is NOT claimed to identify endogenous/leakage from raw labels alone and is NOT a
    substitute for a physically constrained molecular leakage operator.
    """
    def __init__(self,n_genes:int,backbone="resnet18",pretrained=True,freeze_backbone=False,shared_encoder=True):
        super().__init__(); self.endo_enc,d=build_backbone(backbone,pretrained)
        self.leak_enc=self.endo_enc if shared_encoder else build_backbone(backbone,pretrained)[0]
        if freeze_backbone:
            for p in self.endo_enc.parameters(): p.requires_grad=False
            if not shared_encoder:
                for p in self.leak_enc.parameters(): p.requires_grad=False
        self.endo_head=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,n_genes),nn.Softplus())
        self.leak_head=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,n_genes),nn.Softplus())
    def forward(self,full_context,ring):
        e=self.endo_head(self.endo_enc(full_context)); l=self.leak_head(self.leak_enc(ring)); return {"endo":e,"leak":l,"obs":e+l}

# Backward-compatible import name. Documentation calls this AnchoredTwoView, not final EndoVSO.
EndoVSO=AnchoredTwoView
