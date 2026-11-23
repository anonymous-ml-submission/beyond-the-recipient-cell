from pathlib import Path
import hashlib
import json
import sys
import tempfile
import shutil
import textwrap
import numpy as np
import pandas as pd

ROOT = Path.cwd()
SRC = ROOT / 'scripts/audit_actual_vs_local_decoy_source_confirmatory7.py'
BASE = ROOT / 'runs/independent_histo'
FROZEN = BASE / 'actual_vs_local_decoy_confirmatory7_source_map.csv'
GEOM = BASE / 'actual_vs_local_decoy_confirmatory7_geometry_v1.csv'
OUT = BASE / 'actual_vs_local_decoy_distance5_sensitivity_v1'
for p in (SRC, FROZEN, GEOM):
    assert p.is_file(), f'MISSING {p}'
assert not OUT.exists(), f'STOP: output exists {OUT}'

g = pd.read_csv(GEOM, dtype={'actual_source_cell_id': str})
frozen = pd.read_csv(FROZEN, dtype={'actual_source_cell_id': str})
KEY = ['sample', 'contrast', 'actual_source_cell_id']
assert len(g) == len(frozen) == 60393
assert not g.duplicated(KEY).any()
assert not frozen.duplicated(KEY).any()
assert set(map(tuple, g[KEY].to_numpy())) == set(map(tuple, frozen[KEY].to_numpy()))
assert g.groupby(['sample', 'contrast']).ngroups == 14

state = {'geometry': g, 'gene_rows': [], 'coverage': [], 'seen': set()}
code = SRC.read_text()
start = "        role_c = roles[roles['contrast'].eq(cname)]\n"
end = "        print('genes evaluated           =', len(needed_genes))\n"
final = 'decoy_map = pd.concat(map_parts, ignore_index=True)\n'
assert code.count(start) == code.count(end) == code.count(final) == 1, 'STOP: original source structure changed'
a = code.index(start)
b = code.index(end, a) + len(end)

replacement = '''
_key = (str(sid), str(cname))
assert _key not in _S['seen'], _key
_S['seen'].add(_key)
if str(cname) != 'Connective_to_Inflammatory':
    print('REVERSE_CONTRAST_NOT_REFITTED', sid, flush=True)
    continue

_geom = _S['geometry']
_geom = _geom.loc[
    (_geom['sample'] == sid) & (_geom['contrast'] == cname)
].set_index('actual_source_cell_id')
_ids = map_df['actual_source_cell_id'].astype(str).to_numpy()
assert len(_geom) == len(map_df)
assert set(_ids) == set(_geom.index)
_geom = _geom.loc[_ids]
assert np.array_equal(
    _geom['n_recipients'].to_numpy(int),
    map_df['n_recipients'].to_numpy(int)
)
assert np.allclose(
    _geom['mean_recipient_distance_um'].to_numpy(float),
    map_df['mean_recipient_distance_um'].to_numpy(float),
    rtol=1e-7, atol=1e-5
)
_gaps = _geom[[f'decoy_{k}_excess_um' for k in range(1, 6)]].to_numpy(float)
assert _gaps.shape == (len(sk), 5)
assert np.isfinite(_gaps).all() and np.all(_gaps >= -1e-5)
_gaps = np.maximum(_gaps, 0.)
_area_log = np.abs(np.log(source_area[sk][:, None] / source_area[decoy_local]))
_d5_log = np.abs(np.log((source_d5[sk][:, None] + 1e-6) /
                         (source_d5[decoy_local] + 1e-6)))
assert np.isfinite(_area_log).all() and np.isfinite(_d5_log).all()

_tiers = {
    'distance5': _gaps <= 5.,
    'distance5_area_density2x': ((_gaps <= 5.) &
                                 (_area_log <= np.log(2.)) &
                                 (_d5_log <= np.log(2.)))
}
_role_c = roles[roles['contrast'].eq(cname)]
_needed = list(_role_c['gene'].drop_duplicates())
for _tier, _eligible in _tiers.items():
    _cost = np.where(_eligible, _gaps, np.inf)
    _rank = _cost.argmin(axis=1)
    _best = _cost[np.arange(len(sk)), _rank]
    _keep = np.isfinite(_best)
    _sel = np.flatnonzero(_keep)
    _di = decoy_local[_sel, _rank[_sel]]
    _ak = sk[_sel]
    _cov = {name: np.asarray(value)[_sel] for name, value in covariates.items()}
    _min_n = len(_sel)
    if _tier == 'distance5':
        assert _min_n >= 50, f'{sid}/{_tier}: insufficient matched source units: {_min_n}'
    _source_area_gap = _area_log[_sel, _rank[_sel]]
    _source_d5_gap = _d5_log[_sel, _rank[_sel]]
    _S['coverage'].append({
        'sample': sid, 'donor': donor, 'contrast': cname, 'tier': _tier,
        'n_all_sources': int(len(sk)), 'n_selected': int(_min_n),
        'selected_fraction': float(_min_n / len(sk)),
        'fit_supported_n_ge_50': bool(_min_n >= 50),
        'median_excess_um': float(np.median(_best[_keep])) if _min_n else np.nan,
        'p90_excess_um': float(np.quantile(_best[_keep], 0.9)) if _min_n else np.nan,
        'median_abs_log_area_ratio': float(np.median(_source_area_gap)) if _min_n else np.nan,
        'median_abs_log_d5_ratio': float(np.median(_source_d5_gap)) if _min_n else np.nan,
    })
    if _min_n < 50:
        print('TIER_INSUFFICIENT_SUPPORT_SKIP_FIT', sid, _tier, _min_n, flush=True)
        continue
    for _gene in _needed:
        assert _gene in genes, f'MISSING FROZEN GENE {_gene}'
        _gv = dense_col(a.X, genes.get_loc(_gene))
        _source_counts = _gv[source_idx]
        _x_a = 10000.0 * _source_counts[_ak] / source_lib[_ak]
        _x_d = 10000.0 * _source_counts[_di] / source_lib[_di]
        _recipient_counts = _gv[ridx]
        _rec_sum = np.bincount(nearest, weights=_recipient_counts, minlength=n_source)
        _y = 10000.0 * _rec_sum[_ak] / rec_lib_sum[_ak]
        _ba, _na = adjusted_beta(_x_a, _y, _cov)
        _bd, _nd = adjusted_beta(_x_d, _y, _cov)
        _role = _role_c[_role_c['gene'].eq(_gene)]
        assert len(_role) == 1, f'AMBIGUOUS GENE ROLE {sid}/{_gene}'
        _role = _role.iloc[0]
        _S['gene_rows'].append({
            'sample': sid, 'donor': donor, 'contrast': cname, 'tier': _tier,
            'gene': _gene, 'gene_role': str(_role['gene_role']),
            'matched_to_marker': str(_role['matched_to_marker']),
            'n_source_units': int(_min_n),
            'n_model_actual': int(_na), 'n_model_decoy': int(_nd),
            'actual_beta': float(_ba), 'matched_decoy_beta': float(_bd),
            'identity_delta_beta': float(_ba - _bd),
            'actual_spearman': float(safe_spearman(_x_a, _y)),
            'matched_decoy_spearman': float(safe_spearman(_x_d, _y))
        })
    print('DISTANCE_RESTRICTED_FIT', sid, _tier,
          'SOURCE_UNITS', _min_n, 'GENES', len(_needed), flush=True)
'''
code = code[:a] + textwrap.indent(textwrap.dedent(replacement).strip('\n') + '\n', '        ') + code[b:]

finish = r'''
assert len(_S['seen']) == 14, f'Unexpected source groups: {_S["seen"]}'
_ge = pd.DataFrame(_S['gene_rows'])
_cv = pd.DataFrame(_S['coverage'])
assert len(_cv) == 14, len(_cv)
assert _cv['sample'].nunique() == 7
assert not _ge.duplicated(['sample', 'tier', 'gene']).any()
_marker = []
for (_sid, _donor, _tier, _marker_name), _t in _ge.groupby(
    ['sample', 'donor', 'tier', 'matched_to_marker'], sort=True
):
    _m = _t[_t['gene_role'] == 'marker']
    _c = _t[_t['gene_role'] == 'control']
    assert len(_m) == 1 and len(_c) == 3, (_sid, _tier, _marker_name)
    _v = float(_m.iloc[0]['identity_delta_beta'])
    _ctrl = float(_c['identity_delta_beta'].median())
    _marker.append({'sample': _sid, 'donor': _donor, 'tier': _tier,
                    'marker': _marker_name, 'n_source_units': int(_m.iloc[0]['n_source_units']),
                    'identity_delta_beta': _v,
                    'control_median_identity_delta_beta': _ctrl,
                    'marker_minus_control_identity_delta_beta': _v - _ctrl})
_mk = pd.DataFrame(_marker)
assert len(_mk) % 5 == 0 and len(_mk) >= 7 * 5
_summary = _mk.groupby(['sample', 'donor', 'tier'], as_index=False).agg(
    n_markers=('marker', 'nunique'),
    median_identity_delta_beta=('identity_delta_beta', 'median'),
    n_positive_markers=('identity_delta_beta', lambda v: int((v > 0).sum())),
    median_marker_minus_control=('marker_minus_control_identity_delta_beta', 'median'),
    n_marker_gt_control=('marker_minus_control_identity_delta_beta', lambda v: int((v > 0).sum()))
)
assert len(_summary) >= 7 and (_summary['n_markers'] == 5).all()
assert (_summary[_summary['tier'] == 'distance5']['sample'].nunique() == 7)
_out = _OUTPUT
assert not _out.exists(), f'STOP: output exists {_out}'
_tmp = Path(tempfile.mkdtemp(prefix='.distance5_work_', dir=_out.parent))
try:
    _cv.to_csv(_tmp / 'coverage_and_balance.csv', index=False)
    _ge.to_csv(_tmp / 'gene_coefficients.csv', index=False)
    _mk.to_csv(_tmp / 'marker_vs_locked_controls.csv', index=False)
    _summary.to_csv(_tmp / 'section_summary.csv', index=False)
    (_tmp / 'provenance.json').write_text(json.dumps({
        'original_source_sha256': hashlib.sha256(_ORIGINAL.read_bytes()).hexdigest(),
        'original_source_map_sha256': hashlib.sha256(_FROZEN.read_bytes()).hexdigest(),
        'measured_geometry_sha256': hashlib.sha256(_GEOM.read_bytes()).hexdigest(),
        'scope': 'seven confirmatory sections; primary connective-to-inflammatory only',
        'analysis_status': 'post-hoc restricted sensitivity; not predeclared confirmatory test',
        'design': 'single nearest-in-recipient-distance decoy per source, same retained source units and original covariates',
        'distance_caliper_um': 5.0,
        'extra_area_density_caliper': 'each source-to-decoy ratio within 2-fold (sensitivity tier)',
        'limitations': 'nearest-source assignment; incomplete matched coverage; matching does not prove transcript transfer',
    }, indent=2))
    _tmp.rename(_out)
except BaseException:
    shutil.rmtree(_tmp, ignore_errors=True)
    raise
print('\n===== SOURCE-IDENTITY DISTANCE-SENSITIVITY SUMMARY =====')
print(_cv.to_string(index=False))
print(_summary.to_string(index=False))
print('\nDISTANCE5_SOURCE_DECOY_SENSITIVITY_COMPLETE')
print('OUTPUT', _out)
print('ORIGINAL_SCRIPT_AND_FROZEN_RESULTS_UNCHANGED')
raise SystemExit(0)
'''
assert code.count(final) == 1
code = code.replace(final, textwrap.dedent(finish).lstrip('\n'), 1)
compiled = compile(code, str(SRC), 'exec')
namespace = {'__name__': '__main__', '__file__': str(SRC),
             '_S': state, '_OUTPUT': OUT, '_ORIGINAL': SRC,
             '_FROZEN': FROZEN, '_GEOM': GEOM}
sys.argv = [str(SRC)]
namespace.update(
    tempfile=tempfile,
    shutil=shutil,
    json=json,
    hashlib=hashlib,
    Path=Path,
)
exec(compiled, namespace)
raise RuntimeError('STOP: original script unexpectedly reached its end')

