import numpy as np
from leakage_vso.leakage import synthetic_ranking_reversal, fit_gene_leakage, kernel_neighbor_signal

def test_ranking_reversal():
    r=synthetic_ranking_reversal(gamma=.7,n=1000)
    assert r["raw_leak"] < r["raw_bio"]
    assert r["endo_bio"] < r["endo_leak"]

def test_fit_gamma_recovers_signal():
    coords=np.c_[np.arange(60,dtype=float),np.zeros(60)]
    clean=np.zeros(60); clean[::10]=5
    beta=.2; gamma=.4
    s=kernel_neighbor_signal(coords,clean,beta,6)
    raw=clean+gamma*s
    f=fit_gene_leakage(coords,raw,clean,betas=[.1,.2,.3],radius_um=6)
    assert abs(f.beta-beta)<1e-9
    assert abs(f.gamma-gamma)<1e-6
    assert f.r2>.99
