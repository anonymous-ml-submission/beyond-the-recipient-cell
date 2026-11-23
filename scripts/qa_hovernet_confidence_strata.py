#!/usr/bin/env python3

from pathlib import Path
import argparse
import json
import re

import matplotlib.pyplot as plt
import numpy as np
import openslide
import pandas as pd
from shapely import wkb


BIO_CLASSES = [
    "Neoplastic",
    "Non-Neoplastic Epithelial",
    "Inflammatory",
    "Connective",
]


def safe_name(x):
    return re.sub(r"[^A-Za-z0-9]+", "_", x).strip("_")


def draw_geom(ax, geom, x0, y0):
    if geom is None or geom.is_empty:
        return

    if geom.geom_type == "Polygon":
        xy = np.asarray(geom.exterior.coords)
        ax.plot(
            xy[:, 0] - x0,
            xy[:, 1] - y0,
            linewidth=1.6,
            color="orange",
        )
        return

    if geom.geom_type in {"MultiPolygon", "GeometryCollection"}:
        for g in geom.geoms:
            draw_geom(ax, g, x0, y0)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--sample-dir", required=True)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--nuclei", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-per-stratum", type=int, default=8)
    ap.add_argument("--context-um", type=float, default=70.0)
    ap.add_argument("--seed", type=int, default=2026)

    a = ap.parse_args()

    sdir = Path(a.sample_dir)
    sid = sdir.name
    outdir = Path(a.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    image_path = Path(
        (sdir / "image_path.txt").read_text().strip()
    )

    mpp = float(
        (sdir / "pixel_size_um.txt").read_text().strip()
    )

    cells = pd.read_parquet(
        sdir / "cells.parquet",
        columns=["cell_id", "x_px", "y_px"],
    )

    verts = pd.read_parquet(
        sdir / "cell_polygons.parquet"
    )

    ann = pd.read_parquet(a.annotations)

    nuclei = pd.read_parquet(
        a.nuclei,
        columns=[
            "nucleus_id",
            "geometry_wkb",
        ],
    )

    cells["cell_id"] = cells["cell_id"].astype(str)
    verts["cell_id"] = verts["cell_id"].astype(str)
    ann["cell_id"] = ann["cell_id"].astype(str)
    nuclei["nucleus_id"] = nuclei["nucleus_id"].astype(str)

    ann["hovernet_prob"] = pd.to_numeric(
        ann["hovernet_prob"],
        errors="coerce",
    )

    ann["hovernet_nucleus_id"] = (
        ann["hovernet_nucleus_id"]
        .astype("string")
    )

    # Only the already-proven strict one-to-one mappings.
    exact = ann[
        ann["annotation_status"].eq(
            "unique_nucleus_inside_cell"
        )
        & ann["hovernet_type"].isin(BIO_CLASSES)
        & ann["hovernet_prob"].notna()
    ].copy()

    exact = exact.merge(
        cells,
        on="cell_id",
        how="inner",
        validate="one_to_one",
    )

    exact = exact.merge(
        nuclei,
        left_on="hovernet_nucleus_id",
        right_on="nucleus_id",
        how="inner",
        validate="one_to_one",
    )

    if "vertex_order" in verts.columns:
        verts = verts.sort_values(
            ["cell_id", "vertex_order"]
        )

    patch_px = int(round(a.context_um / mpp))
    if patch_px % 2:
        patch_px += 1

    slide = openslide.OpenSlide(str(image_path))

    all_selected = []
    threshold_record = {}

    for class_idx, typ in enumerate(BIO_CLASSES):

        g = exact[
            exact["hovernet_type"].eq(typ)
        ].copy()

        if len(g) < 100:
            raise RuntimeError(
                f"{typ}: unexpectedly few exact matches: {len(g)}"
            )

        # Empirical strata. No arbitrary probability threshold.
        q20 = float(g["hovernet_prob"].quantile(0.20))
        q40 = float(g["hovernet_prob"].quantile(0.40))
        q60 = float(g["hovernet_prob"].quantile(0.60))
        q80 = float(g["hovernet_prob"].quantile(0.80))

        threshold_record[typ] = {
            "n_exact": int(len(g)),
            "q20": q20,
            "q40": q40,
            "q60": q60,
            "q80": q80,
        }

        strata = {
            "LOW (bottom 20%)": g[
                g["hovernet_prob"] <= q20
            ],
            "MID (40-60%)": g[
                (g["hovernet_prob"] >= q40)
                & (g["hovernet_prob"] <= q60)
            ],
            "HIGH (top 20%)": g[
                g["hovernet_prob"] >= q80
            ],
        }

        selected = []

        for stratum_idx, (label, q) in enumerate(strata.items()):

            if len(q) < a.n_per_stratum:
                raise RuntimeError(
                    f"{typ}/{label}: only {len(q)} candidates"
                )

            z = q.sample(
                n=a.n_per_stratum,
                replace=False,
                random_state=(
                    a.seed
                    + 100 * class_idx
                    + stratum_idx
                ),
            ).copy()

            z["qa_class"] = typ
            z["qa_stratum"] = label

            selected.append(z)

        selected = pd.concat(
            selected,
            ignore_index=True,
        )

        all_selected.append(selected)

        ids = set(selected["cell_id"])

        vp = verts[
            verts["cell_id"].isin(ids)
        ].copy()

        polygon_lookup = {
            cid: q[["x_px", "y_px"]].to_numpy(float)
            for cid, q in vp.groupby("cell_id", sort=False)
        }

        strata_order = [
            "LOW (bottom 20%)",
            "MID (40-60%)",
            "HIGH (top 20%)",
        ]

        fig, axes = plt.subplots(
            3,
            a.n_per_stratum,
            figsize=(a.n_per_stratum * 2.45, 7.6),
            squeeze=False,
        )

        for ri, label in enumerate(strata_order):

            q = selected[
                selected["qa_stratum"].eq(label)
            ].reset_index(drop=True)

            for ci, row in q.iterrows():

                ax = axes[ri, ci]

                cx = float(row["x_px"])
                cy = float(row["y_px"])

                x0 = int(round(cx - patch_px / 2))
                y0 = int(round(cy - patch_px / 2))

                patch = slide.read_region(
                    (x0, y0),
                    0,
                    (patch_px, patch_px),
                ).convert("RGB")

                ax.imshow(patch)

                # Cyan = Xenium cell boundary
                xy = polygon_lookup[row["cell_id"]]

                ax.plot(
                    xy[:, 0] - x0,
                    xy[:, 1] - y0,
                    linewidth=1.5,
                    color="cyan",
                )

                # Orange = HoVer nucleus
                geom = wkb.loads(
                    bytes(row["geometry_wkb"])
                )

                draw_geom(
                    ax,
                    geom,
                    x0,
                    y0,
                )

                ax.set_title(
                    f"p={row['hovernet_prob']:.3f}",
                    fontsize=8,
                )

                if ci == 0:
                    ax.set_ylabel(
                        label,
                        fontsize=9,
                    )

                ax.set_xticks([])
                ax.set_yticks([])

        fig.suptitle(
            f"{sid} — {typ} — confidence-stratified morphology QA\n"
            "cyan=Xenium cell, orange=independent HoVer nucleus",
            fontsize=12,
        )

        fig.tight_layout()

        png = outdir / (
            f"{sid}_{safe_name(typ)}_confidence_QA.png"
        )

        fig.savefig(
            png,
            dpi=220,
            bbox_inches="tight",
        )

        plt.close(fig)

        print(
            f"{typ}: n={len(g)} "
            f"q20={q20:.4f} "
            f"q40={q40:.4f} "
            f"q60={q60:.4f} "
            f"q80={q80:.4f}"
        )

        print("SAVED:", png)

    slide.close()

    all_selected = pd.concat(
        all_selected,
        ignore_index=True,
    )

    all_selected[
        [
            "cell_id",
            "hovernet_nucleus_id",
            "hovernet_type",
            "hovernet_prob",
            "qa_class",
            "qa_stratum",
        ]
    ].to_csv(
        outdir / f"{sid}_confidence_QA_cells.csv",
        index=False,
    )

    (
        outdir / f"{sid}_confidence_thresholds.json"
    ).write_text(
        json.dumps(
            threshold_record,
            indent=2,
        )
    )

    print("\nTOTAL QA CELLS:", len(all_selected))
    print(
        "SAVED:",
        outdir / f"{sid}_confidence_QA_cells.csv",
    )
    print(
        "SAVED:",
        outdir / f"{sid}_confidence_thresholds.json",
    )
    print("\nCONFIDENCE-STRATIFIED QA GENERATION: PASS")


if __name__ == "__main__":
    main()
