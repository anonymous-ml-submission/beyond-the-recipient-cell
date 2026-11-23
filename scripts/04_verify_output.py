#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
p=argparse.ArgumentParser();p.add_argument('--result',required=True);p.add_argument('--pairs',required=True);a=p.parse_args()
x=pd.read_csv(a.result); pairs=pd.read_csv(a.pairs)
assert len(x)==len(pairs)*2,(a.result,len(x),len(pairs))
assert set(x.intervention)=={'blur','inpaint'}
assert set(x.gene)==set(pairs.gene)
for mode in ('blur','inpaint'):
    assert x[x.intervention==mode].groupby('gene').size().sort_index().equals(pairs.groupby('gene').size().sort_index()),mode
cols=['pred_original','pred_source_masked','pred_control_masked','delta_source','delta_control','delta_source_minus_control']
assert np.isfinite(x[cols].to_numpy(dtype=float)).all()
assert np.allclose(x.delta_source-x.delta_control,x.delta_source_minus_control,atol=1e-6,rtol=1e-5)
print('VERIFIED_RESULT',a.result,len(x),flush=True)
