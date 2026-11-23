import numpy as np, pandas as pd, anndata as ad, tifffile, pytest, torch
from pathlib import Path
from leakage_vso.dataset import CellImageDataset
from leakage_vso.prepared import PreparedSample


def _make_sample(tmp_path, with_split=True, clean_axis_ok=True):
    root=tmp_path; d=root/'samples'/'S1'; d.mkdir(parents=True)
    cells=pd.DataFrame({'cell_id':['c1','c2'],'donor_id':['d1','d1'],'x_px':[20.,40.],'y_px':[20.,40.]})
    if with_split: cells['split']=['train','val']
    cells.to_parquet(d/'cells.parquet',index=False)
    raw=ad.AnnData(np.array([[1,2],[3,4]],dtype=np.float32)); raw.obs_names=['c1','c2']; raw.var_names=['g1','g2']; raw.write_h5ad(d/'raw.h5ad')
    clean=ad.AnnData(np.array([[1,1],[2,3]],dtype=np.float32)); clean.obs_names=['c1','c2']; clean.var_names=['g1','g2'] if clean_axis_ok else ['g2','g1']; clean.write_h5ad(d/'clean.h5ad')
    img=np.full((64,64,3),180,np.uint8); tifffile.imwrite(d/'img.tif',img)
    (d/'image_path.txt').write_text(str((d/'img.tif').resolve())+'\n'); (d/'pixel_size_um.txt').write_text('0.5\n')
    return root


def test_missing_split_is_fatal(tmp_path):
    root=_make_sample(tmp_path,with_split=False)
    with pytest.raises(RuntimeError,match='lacks required split'):
        CellImageDataset(root,['S1'],['g1'],target='raw',split='train')


def test_zero_requested_split_is_fatal(tmp_path):
    root=_make_sample(tmp_path,with_split=True)
    with pytest.raises(RuntimeError,match='zero rows'):
        CellImageDataset(root,['S1'],['g1'],target='raw',split='test')


def test_raw_clean_gene_axis_must_match(tmp_path):
    root=_make_sample(tmp_path,clean_axis_ok=False)
    with pytest.raises(ValueError,match='Gene axis/order mismatch'):
        PreparedSample.load(root,'S1')


def test_primary_context_ablation_has_fixed_tensor_geometry(tmp_path):
    root=_make_sample(tmp_path)
    a=CellImageDataset(root,['S1'],['g1'],target='raw',split='train',context_um=24,canvas_um=128,center_um=24,model_mpp=0.5)
    b=CellImageDataset(root,['S1'],['g1'],target='raw',split='train',context_um=128,canvas_um=128,center_um=24,model_mpp=0.5)
    assert a[0]['outer'].shape[-2:]==b[0]['outer'].shape[-2:]==(256,256)
    assert a[0]['center'].shape[-2:]==b[0]['center'].shape[-2:]==(48,48)
    # invisible tissue is zero after ImageNet normalization; larger context exposes it
    assert torch.allclose(a[0]['outer'][:,0,0],torch.zeros(3))
    assert not torch.allclose(b[0]['outer'][:,128,128],torch.zeros(3))


def test_canvas_must_cover_visible_context(tmp_path):
    root=_make_sample(tmp_path)
    with pytest.raises(ValueError,match='canvas_um'):
        CellImageDataset(root,['S1'],['g1'],target='raw',split='train',context_um=128,canvas_um=64,center_um=24,model_mpp=0.5)


def test_polygon_vertices_are_sorted_and_centroid_stable():
    from leakage_vso.geometry import sort_polygon_vertices, polygon_summary
    q=pd.DataFrame({'cell_id':['c']*4,'x_px':[1,0,0,1],'y_px':[1,0,1,0],'vertex_order':[2,0,3,1]})
    qq=sort_polygon_vertices(q)
    assert qq.vertex_order.tolist()==[0,1,2,3]
    s=polygon_summary(qq)
    assert np.allclose([s.iloc[0].x_px,s.iloc[0].y_px],[0.5,0.5])
    assert np.isclose(s.iloc[0].polygon_area_px2,1.0)
