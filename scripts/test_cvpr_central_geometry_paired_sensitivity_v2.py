"""Synthetic eight-section integration test: mocks Parquet reads, never needs GPU."""
import tempfile,json,importlib.util
from pathlib import Path
from unittest.mock import patch
import pandas as pd

spec=importlib.util.spec_from_file_location('geo',Path(__file__).with_name('cvpr_central_geometry_paired_sensitivity_v2.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.self_test()
with tempfile.TemporaryDirectory() as td:
    root=Path(td); mapped={}; previous=[]
    for pat,sids in m.PATS.items():
        for sid in sids:
            sd=root/'data/prepared_primary2025/samples'/sid
            sd.mkdir(parents=True)
            (sd/'pixel_size_um.txt').write_text('0.25')
            cells=[];vertices=[];old=[];new=[];matches=[]
            for i,(gene,panel) in enumerate(m.PANEL.items()):
                for j in (0,1):
                    x=1000+200*i+70*j
                    rec=f'rec{i}x{j}';src=f'src{i}x{j}';oc=f'old{i}x{j}';nc=f'new{i}x{j}'
                    touch=j==0
                    for name,cx in ((rec,x),(src,x+(30 if touch else 90)),(oc,x-(30 if touch else 90)),(nc,x+95)):
                        cells.append((name,cx,2500))
                        for v,(a,b) in enumerate([(cx-5,2495),(cx+5,2495),(cx+5,2505),(cx-5,2505)]):
                            vertices.append((name,a,b,v))
                    matches.append({'gene':gene,'recipient_cell_id':rec,'source_cell_id':src,
                                    'old_control_cell_id':oc,'control_cell_id':nc})
                    for mode in ('blur','inpaint'):
                        old_effect=.015 if touch else .008
                        new_effect=.010 if touch else .005
                        old.append({'gene':gene,'recipient_cell_id':rec,'source_cell_id':src,'intervention':mode,
                                    'control_cell_id':oc,'delta_source':.02,'delta_control':.02-old_effect,
                                    'delta_source_minus_control':old_effect,'pred_original':.1})
                        new.append({'gene':gene,'recipient_cell_id':rec,'source_cell_id':src,'intervention':mode,
                                    'old_control_cell_id':oc,'control_cell_id':nc,
                                    'delta_source':.02,'delta_control':.02-new_effect,
                                    'delta_source_minus_control':new_effect,'pred_original':.1})
            mapped[str(sd/'cells.parquet')]=pd.DataFrame(cells,columns=['cell_id','x_px','y_px'])
            mapped[str(sd/'cell_polygons.parquet')]=pd.DataFrame(vertices,columns=['cell_id','x_px','y_px','vertex_order'])
            base=root/'runs/independent_histo'
            op=base/'donor_source_control_rotation_original_v1'/pat/f'{sid}_ctx128_intervention.csv'
            np_=base/'same_morph_image_results_v2/rank1'/f'{sid}_same_morph_intervention.csv'
            mp=base/'same_morph_decoys_v2'/sid/'decoy_rank1_pairs.csv'
            for p in (op,np_,mp):p.parent.mkdir(parents=True,exist_ok=True)
            pd.DataFrame(old).to_csv(op,index=False)
            pd.DataFrame(new).to_csv(np_,index=False)
            pd.DataFrame(matches).to_csv(mp,index=False)
            np_.with_suffix('.json').write_text(json.dumps({
                'sample':sid,'patient':pat,'rank':1,'n_rows':len(new),
                'new_predictions_sha256':m.sha(np_),
                'new_pairs_sha256':m.sha(mp),
                'completed_original_sha256':m.sha(op)}))
        for panel in set(m.PANEL.values()):
            for mode in ('blur','inpaint'):
                for comp,effect in [('ORIGINAL_SUBSET_rank1',.0115),
                                    ('SAME_CLASS_rank1',.0075),
                                    ('SAME_MINUS_OLD_rank1',-.004)]:
                    previous.append({'comparison':comp,'patient':pat,'panel':panel,
                                     'intervention':mode,'effect':effect,'n_sections':len(sids)})
    prior=root/'runs/independent_histo/patient_evidence_sameclass_rank1_v2/patient_panel_effects.csv'
    prior.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(previous).to_csv(prior,index=False)

    def mock_parquet(path,columns=None,**kwargs):
        df=mapped[str(path)].copy()
        return df if columns is None else df[columns].copy()

    with patch.object(m.pd,'read_parquet',mock_parquet):
        out=root/'runs/independent_histo/central_geometry_paired_sensitivity_v2'
        m.run(root,out)
        cov=pd.read_csv(out/'section_coverage.csv')
        assert (cov.matched_pairs==20).all() and (cov.strict_clear_pairs==10).all()
        dat=pd.read_csv(out/'attenuation.csv').set_index('subset')
        assert abs(dat.loc['ALL_MATCHED','attenuation_ratio']-(1-.0075/.0115))<1e-10
        assert abs(dat.loc['STRICT_NO_CENTRAL_POLYGON_OVERLAP','attenuation_ratio']-(1-.005/.008))<1e-10
        try:
            m.run(root,out)
        except RuntimeError as exc:
            assert 'REFUSE_OVERWRITE' in str(exc)
        else:
            raise AssertionError('Expected no-overwrite guard')
print('EIGHT_SECTION_SYNTHETIC_END_TO_END_AND_NO_OVERWRITE_PASS')
