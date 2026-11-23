#!/usr/bin/env python3
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from leakage_vso.prepared import PreparedSample

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data/prepared_primary2025"
MARKERS = PROJECT / "p1_frozen_intervention_marker_contract.csv"
GENERATOR = PROJECT / "scripts/make_source_intervention_pairs.py"
OUT = PROJECT / "runs/independent_histo/donor_source_control_pairs_v1"

SECTIONS = {
    "P2": ["TENX195", "TENX196", "TENX197"],
    "P3": ["TENX198", "TENX199"],
    "P4": ["TENX200", "TENX201", "TENX202"],
}

PARAMS = {
    "max_distance_um": 40,
    "distance_caliper_um": 5,
    "max_log_area_ratio": 0.7,
    "max_per_marker": 500,
    "k_controls": 128,
}

GENES = {
    "MMP2", "SFRP4", "CCDC80", "CXCL12", "FBLN1",
    "TRAC", "CD3E", "IL7R", "CD8A", "CCL5",
}

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

assert MARKERS.is_file(), "STOP: P1 marker contract not transferred"
assert GENERATOR.is_file()
assert not OUT.exists(), "STOP: frozen pair directory already exists"

m = pd.read_csv(MARKERS)
required = {
    "gene", "source_cell_type",
    "recipient_cell_type", "control_cell_type"
}
assert required.issubset(m.columns)
assert len(m) == 10
assert m.gene.is_unique
assert set(m.gene) == GENES
assert m[list(required)].notna().all().all()

OUT.parent.mkdir(parents=True, exist_ok=True)

with tempfile.TemporaryDirectory(
    prefix=".donor_pairs_build_",
    dir=OUT.parent
) as temp:
    tmp = Path(temp)
    manifest = {
        "protocol": "donor_source_control_v1",
        "source_marker_contract_sha256": sha256(MARKERS),
        "pair_generator_sha256": sha256(GENERATOR),
        "parameters": PARAMS,
        "sections": {},
    }

    for patient, sections in SECTIONS.items():
        for sid in sections:
            sample = PreparedSample.load(str(DATA), sid)
            cells = sample.cells.copy()

            assert "cell_id" in cells
            # Join existing HoVer-Net/Xenium cell mapping.
            # Labels are H&E-derived; no expression-based
            # classification is introduced.
            ann_path = (
                PROJECT / "runs/independent_histo"
                / f"{sid}_cell_annotations.parquet"
            )
            ann = pd.read_parquet(
                ann_path,
                columns=[
                    "cell_id",
                    "cell_type",
                    "annotation_status",
                ],
            )

            cells["cell_id"] = cells["cell_id"].astype(str)
            ann["cell_id"] = ann["cell_id"].astype(str)

            assert cells.cell_id.is_unique, (
                sid, "DUPLICATE_PREPARED_CELL_ID"
            )
            assert ann.cell_id.is_unique, (
                sid, "DUPLICATE_ANNOTATION_CELL_ID"
            )
            assert "cell_type" not in cells.columns, (
                sid, "UNEXPECTED_EXISTING_CELL_TYPE"
            )

            n_before = len(cells)

            cells = cells.merge(
                ann,
                on="cell_id",
                how="left",
                validate="one_to_one",
            )

            assert cells.annotation_status.notna().all(), (
                sid, "INCOMPLETE_ANNOTATION_JOIN"
            )

            # Same exact nucleus-to-cell containment rule
            # used in the existing independent H&E analysis.
            cells = cells.loc[
                cells.annotation_status.eq(
                    "unique_nucleus_inside_cell"
                )
            ].copy()

            assert len(cells) > 0, (
                sid, "NO_EXACT_MAPPED_CELLS"
            )
            assert cells.cell_type.notna().all(), (
                sid, "MISSING_HISTO_CELL_TYPE"
            )

            print(
                f"{sid}: H&E exact-mapped cells "
                f"{len(cells)}/{n_before}",
                flush=True,
            )
            assert {"x_px", "y_px"}.issubset(cells.columns)

            cells["cell_id"] = cells.cell_id.astype(str)
            assert cells.cell_id.is_unique

            mpp = float(sample.pixel_size_um)
            assert np.isfinite(mpp) and mpp > 0

            if "x_um" not in cells:
                cells["x_um"] = cells.x_px.astype(float) * mpp
            if "y_um" not in cells:
                cells["y_um"] = cells.y_px.astype(float) * mpp

            assert np.isfinite(
                cells[["x_um", "y_um", "x_px", "y_px"]]
                .to_numpy(dtype=float)
            ).all()
            assert cells.cell_type.notna().all()

            available_types = set(cells.cell_type.astype(str))
            required_types = (
                set(m.source_cell_type.astype(str))
                | set(m.recipient_cell_type.astype(str))
                | set(m.control_cell_type.astype(str))
            )
            # HoVer-Net class names have HISTO_ prefixes.
            # The frozen marker contract uses unprefixed names.
            # Reconcile only these two known equivalent classes.
            aliases = {
                "HISTO_Connective": "Connective",
                "HISTO_Inflammatory": "Inflammatory",
            }

            for original, expected in aliases.items():
                if expected in required_types and original in available_types:
                    assert expected not in available_types, (
                        sid, "MIXED_CELL_TYPE_CONVENTIONS", expected
                    )
                    cells.loc[
                        cells.cell_type.eq(original), "cell_type"
                    ] = expected

            available_types = set(cells.cell_type.astype(str))

            print(
                sid,
                "MATCHING_CLASSES",
                sorted(required_types),
                "AVAILABLE",
                sorted(required_types & available_types),
                flush=True,
            )

            assert required_types.issubset(available_types), (
                sid, "MISSING_CELL_TYPES",
                sorted(required_types - available_types)
            )

            polygon = (
                DATA / "samples" / sid / "cell_polygons.parquet"
            )
            assert polygon.is_file(), f"Missing polygons: {sid}"

            cell_csv = tmp / f"{sid}_classified_cells.csv"
            pair_csv = tmp / f"{sid}_pairs.csv"

            cells.to_csv(cell_csv, index=False)

            cmd = [
                sys.executable,
                str(GENERATOR),
                "--cells", str(cell_csv),
                "--markers", str(MARKERS),
                "--polygons", str(polygon),
                "--out", str(pair_csv),
                "--max-distance-um", "40",
                "--distance-caliper-um", "5",
                "--max-log-area-ratio", "0.7",
                "--max-per-marker", "500",
                "--k-controls", "128",
            ]

            subprocess.run(cmd, cwd=PROJECT, check=True)

            pairs = pd.read_csv(pair_csv)
            assert len(pairs) > 0, f"NO_PAIRS: {sid}"
            assert set(pairs.gene) == GENES, (
                sid, "MISSING_MARKER_PAIRS",
                sorted(GENES - set(pairs.gene))
            )
            assert pairs.groupby("gene").size().le(500).all()
            assert (pairs.source_distance_um <= 40 + 1e-6).all()
            assert (pairs.distance_error_um <= 5 + 1e-6).all()
            assert np.isfinite(pairs.log_area_ratio_abs).all()
            assert (pairs.log_area_ratio_abs <= 0.7 + 1e-6).all()

            assert (
                pairs.recipient_cell_id.astype(str)
                != pairs.source_cell_id.astype(str)
            ).all()
            assert (
                pairs.recipient_cell_id.astype(str)
                != pairs.control_cell_id.astype(str)
            ).all()
            assert (
                pairs.source_cell_id.astype(str)
                != pairs.control_cell_id.astype(str)
            ).all()

            # Freeze the original intervention's complete
            # 128 um polygon-visibility eligibility upfront.
            poly = pd.read_parquet(
                polygon, columns=["cell_id", "x_px", "y_px"]
            )
            poly["cell_id"] = poly.cell_id.astype(str)

            bounds = poly.groupby("cell_id").agg(
                xmin=("x_px", "min"),
                xmax=("x_px", "max"),
                ymin=("y_px", "min"),
                ymax=("y_px", "max"),
            )

            coords = cells.set_index("cell_id")[["x_px", "y_px"]]
            half = 64.0 / mpp

            rec = coords.reindex(
                pairs.recipient_cell_id.astype(str)
            )
            assert rec.notna().all().all()

            eligible = np.ones(len(pairs), dtype=bool)

            for field in ("source_cell_id", "control_cell_id"):
                b = bounds.reindex(pairs[field].astype(str))
                assert b.notna().all().all(), (
                    sid, "MISSING_PAIR_POLYGON", field
                )

                eligible &= (
                    (b.xmin.to_numpy() >= rec.x_px.to_numpy() - half)
                    & (b.xmax.to_numpy() < rec.x_px.to_numpy() + half)
                    & (b.ymin.to_numpy() >= rec.y_px.to_numpy() - half)
                    & (b.ymax.to_numpy() < rec.y_px.to_numpy() + half)
                )

            n_before = len(pairs)
            pairs = pairs.loc[eligible].copy()

            assert len(pairs) > 0, f"NO_VISIBLE_PAIRS: {sid}"
            assert set(pairs.gene) == GENES, (
                sid, "MARKER_LOST_AFTER_VISIBILITY_FILTER"
            )

            pairs.to_csv(pair_csv, index=False)

            manifest["sections"][sid] = {
                "patient": patient,
                "pairs_before_visibility": n_before,
                "pairs_after_visibility": len(pairs),
                "gene_counts": {
                    str(k): int(v)
                    for k, v in pairs.groupby("gene").size().items()
                },
                "pairs_sha256": sha256(pair_csv),
                "polygon_sha256": sha256(polygon),
                "mpp": mpp,
            }

            # The intermediate cell table is only needed
            # during pair construction, not intervention.
            cell_csv.unlink()

            print(
                f"{sid} {patient}: "
                f"{len(pairs)}/{n_before} visible pairs",
                flush=True
            )

    (tmp / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    # Atomically publish a complete frozen set.
    os.rename(tmp, OUT)

print("ALL_EIGHT_PAIR_SETS_FROZEN")
print("OUTPUT:", OUT)
