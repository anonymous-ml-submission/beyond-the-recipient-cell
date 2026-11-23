#!/usr/bin/env python3

from pathlib import Path
import argparse
import json
import warnings

import anndata as ad
import numpy as np
import pandas as pd

from scipy.stats import spearmanr

import statsmodels.formula.api as smf

from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from shapely.validation import make_valid


SAMPLES = [
    "TENX191",
    "TENX192",
    "TENX193",
]

IDCOLS = [
    "recipient_cell_id",
    "source_cell_id",
    "control_cell_id",
]

REQUIRED_RESULT = [
    "gene",
    "intervention",
    "recipient_cell_id",
    "source_cell_id",
    "control_cell_id",
    "delta_source_minus_control",
]


def polygon_parts(g):

    if g is None or g.is_empty:
        return

    if g.geom_type == "Polygon":
        if g.area > 0:
            yield g
        return

    if g.geom_type in {
        "MultiPolygon",
        "GeometryCollection",
    }:
        for z in g.geoms:
            yield from polygon_parts(z)


def polygonal_only(g):

    parts = list(
        polygon_parts(g)
    )

    if not parts:
        return None

    q = unary_union(parts)

    if (
        q.is_empty
        or q.area <= 0
    ):
        return None

    return q


def build_polygon_lookup(
    path,
    needed_ids,
):

    v = pd.read_parquet(
        path
    )

    need = {
        "cell_id",
        "x_px",
        "y_px",
    }

    if not need.issubset(
        v.columns
    ):
        raise RuntimeError(
            f"{path}: polygon schema = "
            f"{list(v.columns)}"
        )

    v["cell_id"] = (
        v["cell_id"]
        .astype(str)
    )

    v = v[
        v["cell_id"].isin(
            needed_ids
        )
    ].copy()

    if "vertex_order" in v:
        v = v.sort_values(
            [
                "cell_id",
                "vertex_order",
            ]
        )

    lookup = {}

    for cid, g in v.groupby(
        "cell_id",
        sort=False,
    ):

        xy = g[
            ["x_px", "y_px"]
        ].to_numpy(float)

        xy = xy[
            np.isfinite(xy).all(
                axis=1
            )
        ]

        if len(xy) < 3:
            continue

        q = Polygon(xy)

        if not q.is_valid:
            q = make_valid(q)

        q = polygonal_only(q)

        if q is not None:
            lookup[str(cid)] = q

    missing = (
        set(needed_ids)
        -
        set(lookup)
    )

    if missing:
        raise RuntimeError(
            f"{path}: missing valid polygons "
            f"for {len(missing)} requested cells; "
            f"first={sorted(missing)[:5]}"
        )

    return lookup


def dense_col(
    X,
    rows,
    gene_index,
):

    z = X[
        rows,
        gene_index,
    ]

    if hasattr(
        z,
        "toarray",
    ):
        z = z.toarray()

    return (
        np.asarray(
            z,
            dtype=np.float32,
        )
        .reshape(-1)
    )


def safe_spearman(
    x,
    y,
):

    x = np.asarray(
        x,
        float,
    )

    y = np.asarray(
        y,
        float,
    )

    keep = (
        np.isfinite(x)
        &
        np.isfinite(y)
    )

    x = x[keep]
    y = y[keep]

    if (
        len(x) < 20
        or np.unique(x).size < 2
        or np.unique(y).size < 2
    ):
        return np.nan, len(x)

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore"
        )

        r = spearmanr(
            x,
            y,
        ).statistic

    return float(r), len(x)


def zscore(
    s,
):

    x = pd.to_numeric(
        s,
        errors="coerce",
    ).astype(float)

    sd = x.std(
        ddof=0
    )

    if (
        not np.isfinite(sd)
        or sd <= 0
    ):
        return pd.Series(
            np.nan,
            index=s.index,
        )

    return (
        x - x.mean()
    ) / sd


def attach_targets(
    d,
    sample_dir,
):

    cells = pd.read_parquet(
        sample_dir
        / "cells.parquet"
    )

    cells["cell_id"] = (
        cells["cell_id"]
        .astype(str)
    )

    if cells[
        "cell_id"
    ].duplicated().any():
        raise RuntimeError(
            "duplicated cell IDs"
        )

    raw = ad.read_h5ad(
        sample_dir
        / "raw.h5ad"
    )

    clean = ad.read_h5ad(
        sample_dir
        / "clean.h5ad"
    )

    if raw.shape != clean.shape:
        raise RuntimeError(
            "raw/clean shape mismatch"
        )

    if not raw.var_names.equals(
        clean.var_names
    ):
        raise RuntimeError(
            "raw/clean gene-axis mismatch"
        )

    if len(cells) != raw.n_obs:
        raise RuntimeError(
            "cells/raw N mismatch"
        )

    obs = np.asarray(
        raw.obs_names.astype(str)
    )

    cid = cells[
        "cell_id"
    ].astype(str).to_numpy()

    if not np.array_equal(
        obs,
        cid,
    ):
        raise RuntimeError(
            "raw obs order != cells.cell_id"
        )

    pos = pd.Series(
        np.arange(
            len(cells),
            dtype=np.int64,
        ),
        index=cid,
    )

    d = d.copy()

    d["raw_target"] = np.nan
    d["clean_target"] = np.nan
    d["leak_proxy"] = np.nan

    for gene, labels in d.groupby(
        "gene",
        sort=False,
    ).groups.items():

        if gene not in raw.var_names:
            raise RuntimeError(
                f"gene missing: {gene}"
            )

        labels = np.asarray(
            list(labels),
            dtype=int,
        )

        rec = d.loc[
            labels,
            "recipient_cell_id",
        ].astype(str)

        missing = (
            set(rec)
            -
            set(pos.index)
        )

        if missing:
            raise RuntimeError(
                f"recipient IDs missing: "
                f"{list(missing)[:5]}"
            )

        rows = pos.loc[
            rec
        ].to_numpy(
            dtype=np.int64
        )

        gi = int(
            raw.var_names.get_loc(
                gene
            )
        )

        rv = dense_col(
            raw.X,
            rows,
            gi,
        )

        cv = dense_col(
            clean.X,
            rows,
            gi,
        )

        leak = np.maximum(
            rv - cv,
            0.0,
        )

        d.loc[
            labels,
            "raw_target",
        ] = rv

        d.loc[
            labels,
            "clean_target",
        ] = cv

        d.loc[
            labels,
            "leak_proxy",
        ] = leak

    return d


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--prepared-root",
        required=True,
    )

    ap.add_argument(
        "--intervention-dir",
        required=True,
    )

    ap.add_argument(
        "--out",
        required=True,
    )

    a = ap.parse_args()

    prepared = Path(
        a.prepared_root
    )

    idir = Path(
        a.intervention_dir
    )

    out = Path(
        a.out
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    frames = []
    eligibility_rows = []

    for sid in SAMPLES:

        print(
            "\n==========",
            sid,
            "==========",
            flush=True,
        )

        rp = (
            idir
            / f"{sid}_ctx128_intervention.csv"
        )

        d = pd.read_csv(
            rp
        )

        missing = [
            c
            for c in REQUIRED_RESULT
            if c not in d.columns
        ]

        if missing:
            raise RuntimeError(
                f"{rp}: missing columns "
                f"{missing}; "
                f"available={list(d.columns)}"
            )

        for c in IDCOLS:
            d[c] = d[c].astype(
                str
            )

        d["gene"] = d[
            "gene"
        ].astype(str)

        d[
            "delta_source_minus_control"
        ] = pd.to_numeric(
            d[
                "delta_source_minus_control"
            ],
            errors="coerce",
        )

        d["section"] = sid

        sdir = (
            prepared
            / "samples"
            / sid
        )

        cells = pd.read_parquet(
            sdir
            / "cells.parquet",
            columns=[
                "cell_id",
                "x_px",
                "y_px",
            ],
        )

        cells["cell_id"] = (
            cells["cell_id"]
            .astype(str)
        )

        cells = cells.set_index(
            "cell_id",
            drop=False,
        )

        mpp = float(
            (
                sdir
                / "pixel_size_um.txt"
            )
            .read_text()
            .strip()
        )

        triples = d[
            IDCOLS
        ].drop_duplicates()

        needed = set(
            triples[
                "source_cell_id"
            ]
        ) | set(
            triples[
                "control_cell_id"
            ]
        )

        polygons = (
            build_polygon_lookup(
                sdir
                / "cell_polygons.parquet",
                needed,
            )
        )

        half24_px = (
            12.0 / mpp
        )

        half128_px = (
            64.0 / mpp
        )

        geom = []

        for r in triples.itertuples(
            index=False
        ):

            rec = cells.loc[
                str(
                    r.recipient_cell_id
                )
            ]

            cx = float(
                rec.x_px
            )

            cy = float(
                rec.y_px
            )

            b24 = box(
                cx - half24_px,
                cy - half24_px,
                cx + half24_px,
                cy + half24_px,
            )

            b128 = box(
                cx - half128_px,
                cy - half128_px,
                cx + half128_px,
                cy + half128_px,
            )

            sp = polygons[
                str(
                    r.source_cell_id
                )
            ]

            cp = polygons[
                str(
                    r.control_cell_id
                )
            ]

            source_inside128 = (
                b128.covers(sp)
            )

            control_inside128 = (
                b128.covers(cp)
            )

            # Strict definition:
            # neither polygon touches or overlaps
            # the central 24-um square.
            source_outside24 = (
                sp.disjoint(b24)
            )

            control_outside24 = (
                cp.disjoint(b24)
            )

            eligible = (
                source_inside128
                and control_inside128
                and source_outside24
                and control_outside24
            )

            geom.append({
                "recipient_cell_id":
                    str(
                        r.recipient_cell_id
                    ),

                "source_cell_id":
                    str(
                        r.source_cell_id
                    ),

                "control_cell_id":
                    str(
                        r.control_cell_id
                    ),

                "source_inside128":
                    source_inside128,

                "control_inside128":
                    control_inside128,

                "source_outside24":
                    source_outside24,

                "control_outside24":
                    control_outside24,

                "extra_context_eligible":
                    eligible,
            })

        geom = pd.DataFrame(
            geom
        )

        d = d.merge(
            geom,
            on=IDCOLS,
            how="left",
            validate="many_to_one",
        )

        if d[
            "extra_context_eligible"
        ].isna().any():
            raise RuntimeError(
                f"{sid}: geometry join failure"
            )

        eligibility_rows.append({
            "section":
                sid,

            "all_rows":
                int(len(d)),

            "all_unique_triples":
                int(
                    d[
                        IDCOLS
                    ].drop_duplicates().shape[0]
                ),

            "eligible_rows":
                int(
                    d[
                        "extra_context_eligible"
                    ].sum()
                ),

            "eligible_unique_triples":
                int(
                    d.loc[
                        d[
                            "extra_context_eligible"
                        ],
                        IDCOLS,
                    ]
                    .drop_duplicates()
                    .shape[0]
                ),
        })

        d = d[
            d[
                "extra_context_eligible"
            ]
        ].copy()

        d = d.reset_index(
            drop=True
        )

        if len(d) == 0:
            raise RuntimeError(
                f"{sid}: zero strict extra-context rows"
            )

        d = attach_targets(
            d,
            sdir,
        )

        frames.append(d)

    D = pd.concat(
        frames,
        ignore_index=True,
    )

    D[
        "specific_effect"
    ] = pd.to_numeric(
        D[
            "delta_source_minus_control"
        ],
        errors="coerce",
    )

    D["log_leak_proxy"] = (
        np.log1p(
            np.maximum(
                D["leak_proxy"],
                0.0,
            )
        )
    )

    D["log_clean_proxy"] = (
        np.log1p(
            np.maximum(
                D["clean_target"],
                0.0,
            )
        )
    )

    D["triple_id"] = (
        D["section"].astype(str)
        + "|"
        + D["recipient_cell_id"]
        + "|"
        + D["source_cell_id"]
        + "|"
        + D["control_cell_id"]
    )

    eligibility = pd.DataFrame(
        eligibility_rows
    )

    eligibility.to_csv(
        out
        / "extra_context_geometry_counts.csv",
        index=False,
    )

    D.to_csv(
        out
        / "extra_context_intervention_with_targets.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # Scale-robust pair-level correlation:
    # within section x gene x intervention.
    # ----------------------------------------------------------

    corr_rows = []

    for (
        section,
        gene,
        mode,
    ), g in D.groupby(
        [
            "section",
            "gene",
            "intervention",
        ],
        sort=True,
    ):

        r_leak, n1 = safe_spearman(
            g["specific_effect"],
            g["log_leak_proxy"],
        )

        r_clean, n2 = safe_spearman(
            g["specific_effect"],
            g["log_clean_proxy"],
        )

        corr_rows.append({
            "section":
                section,

            "gene":
                gene,

            "intervention":
                mode,

            "n":
                min(n1, n2),

            "rho_effect_vs_leak":
                r_leak,

            "rho_effect_vs_clean":
                r_clean,

            "rho_leak_minus_clean":
                (
                    r_leak - r_clean
                    if (
                        np.isfinite(r_leak)
                        and
                        np.isfinite(r_clean)
                    )
                    else np.nan
                ),
        })

    corr = pd.DataFrame(
        corr_rows
    )

    corr.to_csv(
        out
        / "within_section_gene_correlations.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # Primary adjusted within-gene/section comparison.
    #
    # Standardize within section x gene before pooling.
    # This prevents high-expression genes from dominating.
    # Cluster SE by matched source-recipient-control triple.
    # ----------------------------------------------------------

    summaries = {}

    optional_covariates = [
        c
        for c in [
            "source_distance_um",
            "distance_error_um",
            "log_area_ratio_abs",
        ]
        if c in D.columns
    ]

    for mode in [
        "blur",
        "inpaint",
    ]:

        q = D[
            D[
                "intervention"
            ].eq(mode)
        ].copy()

        group = [
            "section",
            "gene",
        ]

        q["effect_z"] = (
            q.groupby(
                group
            )[
                "specific_effect"
            ]
            .transform(zscore)
        )

        q["leak_z"] = (
            q.groupby(
                group
            )[
                "log_leak_proxy"
            ]
            .transform(zscore)
        )

        q["clean_z"] = (
            q.groupby(
                group
            )[
                "log_clean_proxy"
            ]
            .transform(zscore)
        )

        cov_z = []

        for c in optional_covariates:

            zname = (
                c + "_z"
            )

            q[zname] = (
                q.groupby(
                    group
                )[c]
                .transform(zscore)
            )

            cov_z.append(
                zname
            )

        needed = [
            "effect_z",
            "leak_z",
            "clean_z",
            "triple_id",
            *cov_z,
        ]

        q = q.dropna(
            subset=needed
        ).copy()

        if len(q) < 100:
            raise RuntimeError(
                f"{mode}: too few usable rows: "
                f"{len(q)}"
            )

        formula = (
            "effect_z ~ "
            "leak_z + clean_z"
        )

        if cov_z:
            formula += (
                " + "
                + " + ".join(
                    cov_z
                )
            )

        fit = smf.ols(
            formula,
            data=q,
        ).fit(
            cov_type="cluster",
            cov_kwds={
                "groups":
                    q[
                        "triple_id"
                    ]
            },
        )

        contrast = fit.t_test(
            "leak_z = clean_z"
        )

        ci = fit.conf_int()

        cm = corr[
            corr[
                "intervention"
            ].eq(mode)
        ][
            "rho_leak_minus_clean"
        ].dropna()

        summaries[mode] = {
            "n_rows":
                int(len(q)),

            "n_clusters":
                int(
                    q[
                        "triple_id"
                    ].nunique()
                ),

            "n_section_gene_correlations":
                int(len(cm)),

            "mean_rho_leak_minus_clean":
                (
                    float(
                        cm.mean()
                    )
                    if len(cm)
                    else None
                ),

            "median_rho_leak_minus_clean":
                (
                    float(
                        cm.median()
                    )
                    if len(cm)
                    else None
                ),

            "fraction_rho_difference_positive":
                (
                    float(
                        (
                            cm > 0
                        ).mean()
                    )
                    if len(cm)
                    else None
                ),

            "beta_leak_z":
                float(
                    fit.params[
                        "leak_z"
                    ]
                ),

            "se_leak_z":
                float(
                    fit.bse[
                        "leak_z"
                    ]
                ),

            "p_leak_z":
                float(
                    fit.pvalues[
                        "leak_z"
                    ]
                ),

            "ci95_leak_z": [
                float(
                    ci.loc[
                        "leak_z",
                        0,
                    ]
                ),
                float(
                    ci.loc[
                        "leak_z",
                        1,
                    ]
                ),
            ],

            "beta_clean_z":
                float(
                    fit.params[
                        "clean_z"
                    ]
                ),

            "se_clean_z":
                float(
                    fit.bse[
                        "clean_z"
                    ]
                ),

            "p_clean_z":
                float(
                    fit.pvalues[
                        "clean_z"
                    ]
                ),

            "beta_leak_minus_clean":
                float(
                    fit.params[
                        "leak_z"
                    ]
                    -
                    fit.params[
                        "clean_z"
                    ]
                ),

            "p_leak_equals_clean":
                float(
                    np.asarray(
                        contrast.pvalue
                    ).reshape(-1)[0]
                ),

            "formula":
                formula,
        }

        (
            out
            / f"clustered_regression_{mode}.txt"
        ).write_text(
            fit.summary().as_text()
            + "\n\n"
            + str(
                contrast
            )
            + "\n"
        )

    summary = {
        "analysis":
            "source intervention vs DeLeakage-derived contamination proxy",

        "restriction":
            (
                "source and matched control polygons "
                "fully inside 128um square and "
                "strictly disjoint from central 24um square"
            ),

        "test_patient":
            "Patient 1",

        "sections":
            SAMPLES,

        "important_limitation":
            (
                "three sections are one patient; "
                "this is within-patient mechanistic evidence, "
                "not independent-patient replication"
            ),

        "operators":
            summaries,
    }

    (
        out
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n"
    )

    print()
    print(
        "=============================================="
    )

    print(
        "STRICT EXTRA-CONTEXT GEOMETRY"
    )

    print(
        "=============================================="
    )

    print(
        eligibility.to_string(
            index=False
        )
    )

    print()
    print(
        "=============================================="
    )

    print(
        "SOURCE-LINK SUMMARY"
    )

    print(
        "=============================================="
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print()
    print(
        "INTERPRETATION:"
    )

    print(
        "Strongest support requires BOTH blur and inpaint "
        "to show beta_leak_z > beta_clean_z, "
        "with positive rho differences."
    )

    print(
        "A positive result is correction-dependent evidence "
        "for contamination-linked model dependence; "
        "it is not proof of transcript origin."
    )


if __name__ == "__main__":
    main()
