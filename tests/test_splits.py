import pandas as pd
from leakage_vso.splits import assign_donor_splits

def test_donor_split_no_leakage():
    df=pd.DataFrame({"donor_id":[f"d{i}" for i in range(10) for _ in range(3)]})
    o=assign_donor_splits(df,seed=3)
    assert set(o.split)=={"train","val","test"}
    assert o.groupby("donor_id").split.nunique().max()==1
