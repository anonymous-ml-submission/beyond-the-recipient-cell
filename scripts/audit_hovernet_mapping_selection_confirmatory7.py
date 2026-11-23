from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
SAMPLES = ['TENX192', 'TENX193', 'TENX196', 'TENX198', 'TENX199', 'TENX201', 'TENX202']
PREP = Path('data/prepared_primary2025/samples')
ANN = Path('runs/independent_histo')

def smd(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    denom = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0)
    if denom == 0:
        return 0.0
    return (np.mean(a) - np.mean(b)) / denom
rows = []
for sid in SAMPLES:
    print('\n' + '=' * 90)
    print(sid)
    print('=' * 90)
    cells = pd.read_parquet(PREP / sid / 'cells.parquet')
    ann = pd.read_parquet(ANN / f'{sid}_cell_annotations.parquet')
    cells['cell_id'] = cells['cell_id'].astype(str)
    ann['cell_id'] = ann['cell_id'].astype(str)
    x = cells.merge(ann[['cell_id', 'annotation_status']], on='cell_id', how='left', validate='one_to_one')
    if x['annotation_status'].isna().any():
        raise RuntimeError(f'{sid}: missing annotation rows after merge')
    exact = x['annotation_status'].eq('unique_nucleus_inside_cell')
    if 'polygon_area_um2' not in x.columns:
        raise RuntimeError(f'{sid}: polygon_area_um2 missing')
    if not {'x_um', 'y_um'}.issubset(x.columns):
        raise RuntimeError(f'{sid}: x_um/y_um missing')
    xy = x[['x_um', 'y_um']].to_numpy(dtype=float)
    tree = cKDTree(xy)
    d, _ = tree.query(xy, k=6, workers=-1)
    x['d5_um'] = d[:, 5]
    area = pd.to_numeric(x['polygon_area_um2'], errors='coerce')
    d5 = pd.to_numeric(x['d5_um'], errors='coerce')
    log_area = np.log1p(area)
    log_d5 = np.log1p(d5)
    rec = {'sample': sid, 'n_cells': len(x), 'n_exact': int(exact.sum()), 'exact_fraction': float(exact.mean()), 'area_median_exact': float(area[exact].median()), 'area_median_nonexact': float(area[~exact].median()), 'area_log_smd': float(smd(log_area[exact], log_area[~exact])), 'd5_median_exact_um': float(d5[exact].median()), 'd5_median_nonexact_um': float(d5[~exact].median()), 'd5_log_smd': float(smd(log_d5[exact], log_d5[~exact]))}
    print('exact:', rec['n_exact'], '/', rec['n_cells'], f"= {rec['exact_fraction']:.4%}")
    print('cell area median exact/nonexact:', f"{rec['area_median_exact']:.3f}", '/', f"{rec['area_median_nonexact']:.3f}", 'um^2')
    print('log-area SMD:', f"{rec['area_log_smd']:.4f}")
    print('5NN distance median exact/nonexact:', f"{rec['d5_median_exact_um']:.3f}", '/', f"{rec['d5_median_nonexact_um']:.3f}", 'um')
    print('log-5NN SMD:', f"{rec['d5_log_smd']:.4f}")
    x['_area_q'] = pd.qcut(area.rank(method='first'), 4, labels=False)
    x['_d5_q'] = pd.qcut(d5.rank(method='first'), 4, labels=False)
    print('\nExact fraction by CELL-AREA quartile:')
    print(x.groupby('_area_q')['annotation_status'].apply(lambda z: z.eq('unique_nucleus_inside_cell').mean()).to_string())
    print('\nExact fraction by 5NN-DISTANCE quartile:')
    print(x.groupby('_d5_q')['annotation_status'].apply(lambda z: z.eq('unique_nucleus_inside_cell').mean()).to_string())
    xr = (x['x_um'] - x['x_um'].min()) / max(1e-12, x['x_um'].max() - x['x_um'].min())
    yr = (x['y_um'] - x['y_um'].min()) / max(1e-12, x['y_um'].max() - x['y_um'].min())
    x['_xb'] = np.minimum((xr * 10).astype(int), 9)
    x['_yb'] = np.minimum((yr * 10).astype(int), 9)
    grid = x.assign(_exact=exact.astype(int)).groupby(['_xb', '_yb'])['_exact'].agg(['size', 'mean'])
    grid = grid[grid['size'] >= 200]
    rec['spatial_bins_ge200'] = int(len(grid))
    rec['spatial_exact_q10'] = float(grid['mean'].quantile(0.1))
    rec['spatial_exact_median'] = float(grid['mean'].median())
    rec['spatial_exact_q90'] = float(grid['mean'].quantile(0.9))
    print('\nSpatial-bin exact-rate q10/median/q90:')
    print(f"{rec['spatial_exact_q10']:.4f}", f"{rec['spatial_exact_median']:.4f}", f"{rec['spatial_exact_q90']:.4f}", f'(bins={len(grid)})')
    rows.append(rec)
out = pd.DataFrame(rows)
print('\n' + '=' * 90)
print('CONFIRMATORY7 SELECTION-BIAS SUMMARY')
print('=' * 90)
print(out.to_string(index=False))
out_path = ANN / 'mapping_selection_bias_confirmatory7.csv'
out.to_csv(out_path, index=False)
print('\nSAVED:', out_path)
