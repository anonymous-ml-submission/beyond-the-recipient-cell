from leakage_vso.metrics import pairwise_ranking_inversion

def test_inversion():
    r=pairwise_ranking_inversion({"a":2,"b":1},{"a":1,"b":2})
    assert r["n_inversions"]==1 and r["inversion_fraction"]==1
