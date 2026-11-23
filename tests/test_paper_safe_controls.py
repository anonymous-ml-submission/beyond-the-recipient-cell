import pandas as pd, numpy as np, subprocess, sys
from pathlib import Path

def test_pair_builder_requires_explicit_control_type(tmp_path):
    cells=pd.DataFrame({'cell_id':['r','s','c'],'x_um':[0.,5.,-5.],'y_um':[0.,0.,0.], 'cell_type':['R','S','C']})
    cp=tmp_path/'cells.csv'; cells.to_csv(cp,index=False)
    markers=pd.DataFrame({'gene':['g'],'source_cell_type':['S'],'recipient_cell_type':['R']})
    mp=tmp_path/'m.csv'; markers.to_csv(mp,index=False)
    out=tmp_path/'o.csv'
    r=subprocess.run([sys.executable,'scripts/make_source_intervention_pairs.py','--cells',str(cp),'--markers',str(mp),'--out',str(out)],capture_output=True,text=True)
    assert r.returncode!=0 and 'control_cell_type' in (r.stdout+r.stderr)

def test_pair_builder_uses_declared_control_type(tmp_path):
    cells=pd.DataFrame({'cell_id':['r','s','bad','good'],'x_um':[0.,5.,-5.,0.],'y_um':[0.,0.,0.,5.], 'cell_type':['R','S','X','C']})
    cp=tmp_path/'cells.csv'; cells.to_csv(cp,index=False)
    markers=pd.DataFrame({'gene':['g'],'source_cell_type':['S'],'recipient_cell_type':['R'],'control_cell_type':['C']})
    mp=tmp_path/'m.csv'; markers.to_csv(mp,index=False)
    out=tmp_path/'o.csv'
    r=subprocess.run([sys.executable,'scripts/make_source_intervention_pairs.py','--cells',str(cp),'--markers',str(mp),'--out',str(out),'--distance-caliper-um','1'],capture_output=True,text=True,check=True)
    q=pd.read_csv(out); assert len(q)==1 and q.loc[0,'control_cell_id']=='good' and q.loc[0,'control_cell_type']=='C'
