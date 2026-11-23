#!/usr/bin/env python3
"""Read-only CVPR Primary2025 context x evaluation-target audit.

Raw-trained context models (24/64/128 um), each evaluated against the
same raw and estimated-corrected targets. Strict donor x gene pairing.
This is not an independent experiment and does not establish leakage causality.

No source model, configuration, predictions or metrics are modified.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

CONTEXTS = (24, 64, 128)
TARGETS = ("raw", "clean")
CONTRASTS = ((24, 64), (24, 128), (64, 128))
EPS = 1e-6
SEED = 20261030


def require(condition, message):
    if not condition:
        raise RuntimeError("STOP: " + message)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def input_files(root):
    return {
        (ctx, target): root / f"A_raw_ctx{ctx}" / "eval" / f"metrics_{target}_by_donor.csv"
        for ctx in CONTEXTS
        for target in TARGETS
    }


def load_inputs(root, expected_genes=280, verbose=True):
    require(root.is_dir(), f"missing Primary2025 run root: {root}")
    paths = input_files(root)
    missing = [str(p) for p in paths.values() if not p.is_file() or p.stat().st_size == 0]
    require(not missing, "missing required donor-level metric files (no gene-only fallback):\n" + "\n".join(missing))

    tables, manifest = {}, []
    index_sets = {}
    for (ctx, target), path in paths.items():
        df = pd.read_csv(path, dtype={"donor_id": "string", "gene": "string"})
        required = {"donor_id", "gene", "pcc"}
        require(required.issubset(df.columns), f"{path}: required {sorted(required)}; got {df.columns.tolist()}")
        require(df["donor_id"].notna().all() and df["gene"].notna().all(), f"{path}: missing donor_id/gene")
        df["donor_id"] = df["donor_id"].str.strip()
        df["gene"] = df["gene"].str.strip()
        require(df["donor_id"].ne("").all() and df["gene"].ne("").all(), f"{path}: blank donor_id/gene")
        require(not df.duplicated(["donor_id", "gene"]).any(), f"{path}: duplicate donor-gene rows")
        numeric = pd.to_numeric(df["pcc"], errors="coerce")
        require(not np.isinf(numeric.to_numpy(dtype=float)).any(), f"{path}: infinite PCC")
        valid = numeric.notna()
        require((numeric[valid].abs() <= 1 + 1e-8).all(), f"{path}: PCC outside [-1, 1]")
        df["pcc"] = numeric
        tab = df.set_index(["donor_id", "gene"])["pcc"].astype(float)
        tables[(ctx, target)] = tab
        index_sets[(ctx, target)] = set(tab.index.tolist())
        manifest.append({
            "context_um": ctx, "target": target, "path": str(path.resolve()),
            "sha256": sha256(path), "n_rows": int(len(df)),
            "n_valid_pcc": int(valid.sum()),
            "n_donors": int(df.donor_id.nunique()),
            "n_genes": int(df.gene.nunique()),
            "donors": sorted(df.donor_id.unique().tolist()),
        })
        if verbose:
            print(f"FILE_PASS ctx={ctx} target={target} rows={len(df)} valid_pcc={valid.sum()} "
                  f"donors={df.donor_id.nunique()} genes={df.gene.nunique()}", flush=True)

    base = index_sets[(24, "raw")]
    for k, inds in index_sets.items():
        if inds != base:
            only_ref = sorted(base - inds)[:5]
            only_other = sorted(inds - base)[:5]
            raise RuntimeError(f"STOP: donor-gene key mismatch for {k}: "
                               f"missing_from_file={len(base-inds)} examples={only_ref}; "
                               f"extra={len(inds-base)} examples={only_other}")

    genes = {gene for _, gene in base}
    donors = sorted({donor for donor, _ in base})
    require(len(genes) == expected_genes,
            f"gene universe has {len(genes)}, expected {expected_genes}. "
            "Do not silently alter the prespecified denominator.")
    require(len(donors) >= 1, "no donors")
    by_donor_counts = pd.Series([d for d, _ in base]).value_counts().to_dict()
    if verbose:
        print("KEY_IDENTITY_PASS donors=", donors, "genes=", len(genes),
              "keys_per_donor=", by_donor_counts, flush=True)
        if len(donors) == 1:
            print("INFERENCE_LIMIT: ONLY ONE HELD-OUT PATIENT; no patient-level CI or p value", flush=True)

    wide = pd.concat({f"{target}_{ctx}": tables[(ctx, target)] for ctx in CONTEXTS for target in TARGETS}, axis=1)
    wide.columns = wide.columns.astype(str)
    incomplete = wide.isna().any(axis=1)
    if verbose:
        print(f"COMPLETE_CASE_GATE kept={int((~incomplete).sum())}/{len(wide)} "
              f"excluded_any_nonfinite={int(incomplete.sum())}", flush=True)
    require((~incomplete).sum() > 0, "no paired complete-case gene-donor observations")
    require(int((~incomplete).sum()) >= max(1, math.ceil(0.75 * len(wide))),
            "more than 25% paired observations missing; investigate evaluation before interpreting")

    valid = wide.loc[~incomplete].copy()
    for col in valid.columns:
        valid[f"z_{col}"] = np.arctanh(valid[col].clip(-1 + EPS, 1 - EPS))
    return valid, manifest, {
        "n_key_pairs_total": len(wide),
        "n_key_pairs_complete": len(valid),
        "n_key_pairs_excluded": int(incomplete.sum()),
        "n_donors_all": len(donors),
        "donors_all": donors,
        "n_genes_all": len(genes),
        "n_donors_complete": len(valid.index.get_level_values("donor_id").unique()),
        "n_genes_complete": len(valid.index.get_level_values("gene").unique()),
        "missing_by_metric": {col: int(wide[col].isna().sum()) for col in wide.columns},
        "gene_per_donor": by_donor_counts,
    }


def donor_cluster_bootstrap(values, rng, n_boot=10000):
    """Equal-weight donors; report interval only for >=5 independent donors."""
    vals = np.asarray(values, dtype=float)
    if len(vals) < 5:
        return float("nan"), float("nan")
    picks = rng.integers(0, len(vals), size=(n_boot, len(vals)))
    replicates = vals[picks].mean(axis=1)
    return tuple(float(x) for x in np.quantile(replicates, [0.025, 0.975]))


def calculate(wide, qc, n_boot=10000):
    rng = np.random.default_rng(SEED)
    long = wide.reset_index()
    # Every effect uses Fisher-z of PCC; higher PCC is better.
    # All six metrics are measured on an identical complete-case donor-gene set.
    donor_gain_rows, gene_gain_rows, summ_rows = [], [], []
    for small, large in CONTRASTS:
        contrast = f"{small}_to_{large}"
        raw_gain = long[f"z_raw_{large}"] - long[f"z_raw_{small}"]
        clean_gain = long[f"z_clean_{large}"] - long[f"z_clean_{small}"]
        interaction = raw_gain - clean_gain
        frame = long[["donor_id", "gene"]].copy()
        frame["contrast"] = contrast
        frame["gain_raw_z"] = raw_gain
        frame["gain_clean_z"] = clean_gain
        frame["interaction_z"] = interaction
        gene_gain_rows.append(frame)
        donor = frame.groupby("donor_id", as_index=False).agg(
            n_genes=("gene", "size"),
            raw_gain_mean_z=("gain_raw_z", "mean"),
            clean_gain_mean_z=("gain_clean_z", "mean"),
            interaction_mean_z=("interaction_z", "mean"),
            genes_positive_interaction=("interaction_z", lambda s: int((s > 0).sum())),
        )
        donor["contrast"] = contrast
        donor_gain_rows.append(donor)
        lo, hi = donor_cluster_bootstrap(donor["interaction_mean_z"], rng, n_boot=n_boot)
        summ_rows.append({
            "contrast": contrast, "n_independent_donors": len(donor),
            "n_genes_complete_case": frame.gene.nunique(),
            "n_donor_gene_observations": len(frame),
            "mean_donor_raw_context_gain_z": float(donor.raw_gain_mean_z.mean()),
            "mean_donor_clean_context_gain_z": float(donor.clean_gain_mean_z.mean()),
            "mean_donor_interaction_z": float(donor.interaction_mean_z.mean()),
            "median_donor_interaction_z": float(donor.interaction_mean_z.median()),
            "n_donors_positive_interaction": int((donor.interaction_mean_z > 0).sum()),
            "donor_bootstrap_95ci_low": lo,
            "donor_bootstrap_95ci_high": hi,
            "patient_inference": "descriptive_only" if len(donor) < 5 else "cluster_bootstrap_exploratory",
        })

    donor_gain = pd.concat(donor_gain_rows, ignore_index=True)
    gene_gain = pd.concat(gene_gain_rows, ignore_index=True)
    summary = pd.DataFrame(summ_rows)

    # Rank the three contexts within each target, equal weight per donor.
    # The model ranking compares *raw-trained context models*, not a corrected-trained method.
    donor_model_rows = []
    for donor, group in long.groupby("donor_id", sort=True):
        for ctx in CONTEXTS:
            donor_model_rows.append({
                "donor_id": donor, "context_um": ctx,
                "raw_mean_z": float(group[f"z_raw_{ctx}"].mean()),
                "clean_mean_z": float(group[f"z_clean_{ctx}"].mean()),
                "n_genes": len(group),
            })
    donor_models = pd.DataFrame(donor_model_rows)
    models = donor_models.groupby("context_um", as_index=False)[["raw_mean_z", "clean_mean_z"]].mean()
    models["raw_equivalent_pcc"] = np.tanh(models.raw_mean_z)
    models["clean_equivalent_pcc"] = np.tanh(models.clean_mean_z)
    models["raw_rank_best_1"] = models.raw_mean_z.rank(ascending=False, method="min").astype(int)
    models["clean_rank_best_1"] = models.clean_mean_z.rank(ascending=False, method="min").astype(int)

    reversals = []
    for a, b in itertools.combinations(CONTEXTS, 2):
        x = models.set_index("context_um")
        raw_diff = float(x.loc[a, "raw_mean_z"] - x.loc[b, "raw_mean_z"])
        clean_diff = float(x.loc[a, "clean_mean_z"] - x.loc[b, "clean_mean_z"])
        tol = 1e-12
        reversal = raw_diff * clean_diff < -(tol ** 2) and abs(raw_diff) > tol and abs(clean_diff) > tol
        reversals.append({
            "model_a_context_um": a, "model_b_context_um": b,
            "raw_a_minus_b_z": raw_diff,
            "clean_a_minus_b_z": clean_diff,
            "raw_clean_ranking_inversion": bool(reversal),
            "has_tie": bool(abs(raw_diff) <= tol or abs(clean_diff) <= tol),
        })
    pairs = pd.DataFrame(reversals)
    n_inversions = int(pairs.raw_clean_ranking_inversion.sum())
    valid_pairs = int((~pairs.has_tie).sum())
    rank_rho = float(models.raw_mean_z.rank().corr(models.clean_mean_z.rank(), method="pearson"))
    rank_summary = {
        "n_context_models": len(CONTEXTS),
        "n_independent_donors": qc["n_donors_complete"],
        "n_model_pairs_without_ties": valid_pairs,
        "n_raw_clean_ranking_inversions": n_inversions,
        "fraction_pairs_inverted": float(n_inversions / valid_pairs) if valid_pairs else None,
        "spearman_context_ranking_raw_vs_clean": rank_rho if math.isfinite(rank_rho) else None,
        "rank_inference": "descriptive; three model variants; do not interpret as cross-patient replication",
    }
    return {
        "donor_gene_metrics": long,
        "donor_gene_context_gain": gene_gain,
        "donor_context_gain": donor_gain,
        "interaction_summary": summary,
        "model_metric_by_donor": donor_models,
        "model_ranking": models,
        "ranking_pairs": pairs,
        "ranking_summary": rank_summary,
    }


def emit(out, result, qc, manifest):
    require(not out.exists(), f"output already exists; refusing to overwrite: {out}")
    require(out.parent.is_dir(), f"output parent missing: {out.parent}")
    tmp = Path(tempfile.mkdtemp(prefix=".cvpr_context_target_audit_tmp_", dir=out.parent))
    try:
        for key in ["donor_gene_metrics", "donor_gene_context_gain", "donor_context_gain",
                    "interaction_summary", "model_metric_by_donor", "model_ranking", "ranking_pairs"]:
            result[key].to_csv(tmp / f"{key}.csv", index=False, float_format="%.12g")
        prov = {
            "analysis": "Primary2025 raw-trained contexts x evaluated target audit v1",
            "metric": "PCC (higher is better)",
            "primary_scale": "Fisher-z of PCC; mean equally across genes then independent donors",
            "formula": "(z_PCC_large_raw - z_PCC_small_raw) - (z_PCC_large_clean - z_PCC_small_clean)",
            "contexts_um": list(CONTEXTS),
            "evaluation_targets": list(TARGETS),
            "complete_case_policy": "Only donor-gene keys with finite metrics in all six input files; report exclusions",
            "CI_policy": "No patient-level CI if fewer than five independent donors; bootstrap donors otherwise",
            "bootstrap_seed": SEED,
            "noncausal_limitations": [
                "corrected target is an estimated reference, not absolute ground truth",
                "only raw-trained context variants are compared",
                "PCC summaries cannot verify identical cell-level evaluation masks",
                "gene and section counts are not independent patient replication",
                "patient-level generalization requires multiple genuinely held-out donors",
            ],
            "qc": qc, "input_manifest": manifest,
            "ranking_summary": result["ranking_summary"],
        }
        (tmp / "audit_provenance.json").write_text(json.dumps(prov, indent=2, allow_nan=False) + "\n")
        (tmp / "READ_ONLY_ANALYSIS_PASS.txt").write_text("Analysis complete; source inputs not modified.\n")
        require(not out.exists(), "output appeared during analysis; refusing overwrite")
        os.rename(tmp, out)
    finally:
        if tmp.exists():
            # Temp folder only; never delete user source or completed output.
            import shutil
            shutil.rmtree(tmp)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("check", "analyze"))
    ap.add_argument("--root", type=Path, default=Path("runs/primary2025_context_v1"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--expected-genes", type=int, default=280)
    ap.add_argument("--bootstrap", type=int, default=10000)
    a = ap.parse_args()
    require(a.expected_genes >= 1, "expected genes must be positive")
    require(a.bootstrap >= 100, "bootstrap count must be >=100")
    valid, manifest, qc = load_inputs(a.root, a.expected_genes)
    if a.command == "check":
        print("PREFLIGHT_PASS: donor-gene keys and six input files verified; NO FILES WRITTEN")
        return
    out = a.out or a.root / "cvpr_context_target_interaction_audit_v1"
    result = calculate(valid, qc, n_boot=a.bootstrap)
    emit(out, result, qc, manifest)
    print("\nCONTEXT x TARGET INTERACTION (FISHER Z)")
    print(result["interaction_summary"].to_string(index=False))
    print("\nRAW-TRAINED CONTEXT MODEL RANKING")
    print(result["model_ranking"].to_string(index=False))
    print("\nRANKING REVERSALS")
    print(result["ranking_pairs"].to_string(index=False))
    print("\nPATIENT REPLICATION LIMIT:", qc["n_donors_complete"],
          "independent held-out donor(s)")
    print("AUDIT_COMPLETE", out)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, KeyError) as e:
        raise SystemExit(str(e))
