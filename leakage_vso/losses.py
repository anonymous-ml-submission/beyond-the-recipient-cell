from __future__ import annotations
import torch
import torch.nn.functional as F

def poisson_loss(mu,target,weight=None,eps=1e-6):
    loss=F.poisson_nll_loss(mu.clamp_min(eps),target,log_input=False,full=False,reduction="none")
    if weight is not None: loss=loss*weight
    return loss.mean()

def log1p_huber_loss(pred,target,weight=None,delta=1.0):
    loss=F.huber_loss(torch.log1p(pred.clamp_min(0)),torch.log1p(target.clamp_min(0)),delta=delta,reduction="none")
    if weight is not None: loss=loss*weight
    return loss.mean()

def endovso_loss(outputs,raw,clean,confidence=None,lambda_endo=1.0,lambda_obs=1.0,lambda_leak=0.25):
    # Corrected targets from SPLIT/DeLeakage/DeSpotX need not be integer counts. Use a
    # log-scale robust loss for the anchored endogenous component. Raw Xenium counts keep
    # a Poisson observation likelihood. Residual supervision is auxiliary, not ground truth.
    residual=(raw-clean).clamp_min(0)
    l_endo=log1p_huber_loss(outputs["endo"],clean,confidence)
    l_obs=poisson_loss(outputs["obs"],raw,None)
    l_leak=F.smooth_l1_loss(torch.log1p(outputs["leak"]),torch.log1p(residual))
    total=lambda_endo*l_endo+lambda_obs*l_obs+lambda_leak*l_leak
    return total,{"endo":float(l_endo.detach()),"obs":float(l_obs.detach()),"leak":float(l_leak.detach())}
