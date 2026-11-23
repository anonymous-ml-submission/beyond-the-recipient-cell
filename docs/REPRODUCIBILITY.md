# Reproducibility and evidence boundaries

## Paper-to-code map

| Supplement | Task | Clean reviewer entry point / implementation |
|---|---|---|
| S1 | Xenium cell universe and H&E mapping | `leakage_vso/`, `experiments/train.py` |
| S2–S3 | Biological source localization/dose and negative controls | `scripts/audit_same_recipient_nuclear_localization_confirmatory7.py`, `scripts/audit_source_cell_dose_gate_confirmatory7.py` and other scientific scripts in `scripts/` |
| S4 | Raw/corrected targets | `experiments/estimate_leakage.py`, `experiments/prepare_corrected_targets.py` |
| S5 | CTX24/64/128 and patient rotations | `experiments/train.py`, `scripts/run_ablation_matrix.py`, `scripts/ranking_reversal.py` |
| S6 | Morphology and adapted GHIST | `cvpr_fair_baselines_v1/`, `experiments/ghist_training.py`, `external_methods/GHIST/` |
| S7 | Original source/control edits | `scripts/freeze_donor_intervention_pairs_v1.py`, `experiments/evaluate_interventions.py` |
| S8 | Outcome-blind rank-1 matching, edited predictions, patient effects | `experiments/match_controls.py`, `experiments/evaluate_interventions.py`, `experiments/aggregate_patient_effects.py`, `experiments/verify_results.py` |
| S8 | Post-hoc central-square geometry sensitivity | `experiments/geometry_sensitivity.py` |
| S9 | Interpretation and shared-source cancellation | Main paper equations; `experiments/aggregate_patient_effects.py` |
| S10 | Frozen output consistency and auditing | `experiments/verify_results.py`, original manifests (not bundled) |

## Exact matching design

The rank-1 candidate must have the same broad H&E class as the source, be fully visible in CTX128, and satisfy distance ≤ 40 μm, recipient-distance discrepancy ≤ 5 μm, absolute log polygon-area ratio ≤ 0.7, and absolute log source/candidate local five-nearest-neighbor ratio ≤ 0.7.

The original outcome-blind selector ranks eligible candidates using `distance_difference/5 + abs(log_area_ratio)/0.7 + abs(log_5NN_ratio)/0.7`, breaking exact ties by the string-form candidate cell ID. Saved selected IDs, rather than any new prediction-conditioned search, define the paired record universe.

## Frozen numerical checkpoints

| Check | Recorded value |
|---|---:|
| Original P2–P4 records | 38,695 |
| Rank-1 matched records | 8,365 |
| Geometry-clear subset | 5,005 |
| Original-control matched mean | 0.00698903 |
| Same-class-control matched mean | 0.00179142 |
| Descriptive attenuation | 74.3681% |
| Geometry-clear attenuation | 92.8435% |

Results are aggregated across three separately held-out patients, using the frozen patient/marker/operator hierarchy, **not** a flat pooled gene–pair mean.

## Verified vs. not verified

**Verified on original compute systems:** original S8 source hashes (5/5), donor intervention input hashes (8/8), H&E geometry hashes (32/32), saved candidate ordering for rank 1/2/3, patient-level paired arithmetic and geometry sensitivity. Other prior validations cover S2 biology, S5 donor metrics and S6 morphology.

**Verified on a source-only worktree:** Python syntax and the available S8 synthetic matching, uncertainty and frozen-checkpoint/replay safeguards.

**Not established by a source checkout:** original-data download/preparation, fresh full-cohort training, checkpoint availability, full edited-image GPU inference, and complete runtime equivalence of renamed artifact-path defaults. A no-data synthetic PASS must not be advertised as production numerical reproduction.

## External dependencies and provenance

GHIST files retain their original attribution and license. Vendored DeLeakage source is deliberately **not** shipped here until the exact upstream version, license and redistribution terms have been reviewed; correction integration in `leakage_vso/` is preserved. Input data and original model weights are not redistributed. Original server-specific archives and original-to-release hash ledgers remain private.

## Code entry points

The clean scripts under `experiments/` forward CLI options to byte-identical implementations under `scripts/`. Original versioned filenames are preserved to avoid silently breaking frozen provenance; reviewers can use the clean names for all primary operations. A private run directory may require supplying local `--project`, `--original-root`, or checkpoint arguments. Do not substitute generated test data for frozen experimental inputs.
