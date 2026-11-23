#!/usr/bin/env python3

from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


SAMPLES = [
    ("TENX191", "Patient 1"),
    ("TENX195", "Patient 2"),
    ("TENX197", "Patient 3"),
    ("TENX200", "Patient 4"),
]

ROOT = Path("data/prepared_primary2025/samples")
ANN = Path("runs/independent_histo")

CONTRASTS = [
    {
        "name": "Connective_to_Inflammatory",
        "source": "Connective",
        "recipient": "Inflammatory",
        "genes": [
            "MMP2",
            "SFRP4",
            "CCDC80",
            "CXCL12",
            "FBLN1",
        ],
    },
    {
        "name": "Inflammatory_to_Connective",
        "source": "Inflammatory",
        "recipient": "Connective",
        "genes": [
            "TRAC",
            "CD3E",
            "IL7R",
            "CD8A",
            "CCL5",
        ],
    },
]


def col(X, j):
    z = X[:, j]
    if sp.issparse(z):
        return np.asarray(z.toarray()).ravel()
    return np.asarray(z).ravel()


rows = []

for sid, donor in SAMPLES:

    a = ad.read_h5ad(ROOT / sid / "raw.h5ad")

    ann = pd.read_parquet(
        ANN / f"{sid}_cell_annotations.parquet"
    )

    ann["cell_id"] = ann["cell_id"].astype(str)
    ann = ann.set_index("cell_id").reindex(
        a.obs_names.astype(str)
    )

    if ann["annotation_status"].isna().any():
        raise RuntimeError(
            f"{sid}: incomplete annotation join"
        )

    exact = ann["annotation_status"].eq(
        "unique_nucleus_inside_cell"
    ).to_numpy()

    genes = pd.Index(a.var_names.astype(str))

    lib = np.asarray(
        a.X.sum(axis=1)
    ).ravel().astype(float)

    print("\n" + "=" * 90)
    print(sid, donor)
    print("=" * 90)

    for c in CONTRASTS:

        source = (
            exact
            & ann["hovernet_type"]
              .eq(c["source"])
              .to_numpy()
        )

        recipient = (
            exact
            & ann["hovernet_type"]
              .eq(c["recipient"])
              .to_numpy()
        )

        ns = int(source.sum())
        nr = int(recipient.sum())

        print(
            f"\n{c['name']}: "
            f"source={ns:,} recipient={nr:,}"
        )

        source_lib = lib[source].sum()
        recipient_lib = lib[recipient].sum()

        for gene in c["genes"]:

            if gene not in genes:
                raise RuntimeError(
                    f"{sid}: missing locked gene {gene}"
                )

            v = col(
                a.X,
                genes.get_loc(gene),
            ).astype(float)

            sv = v[source]
            rv = v[recipient]

            sdet = float(np.mean(sv > 0))
            rdet = float(np.mean(rv > 0))

            scount = float(sv.sum())
            rcount = float(rv.sum())

            scpt = (
                1e4 * scount / source_lib
                if source_lib > 0 else np.nan
            )

            rcpt = (
                1e4 * rcount / recipient_lib
                if recipient_lib > 0 else np.nan
            )

            eps = 1e-8

            log2_ratio = np.log2(
                (scpt + eps)
                /
                (rcpt + eps)
            )

            rows.append({
                "sample": sid,
                "donor": donor,
                "contrast": c["name"],
                "source_class": c["source"],
                "recipient_class": c["recipient"],
                "gene": gene,

                "n_source": ns,
                "n_recipient": nr,

                "source_detection": sdet,
                "recipient_detection": rdet,

                "source_cpt10k": scpt,
                "recipient_cpt10k": rcpt,

                "log2_source_recipient_ratio":
                    log2_ratio,

                "source_raw_counts": scount,
                "recipient_raw_counts": rcount,
            })

            print(
                f"{gene:8s} "
                f"src_det={sdet:.4f} "
                f"rec_det={rdet:.4f} "
                f"log2ratio={log2_ratio:+.3f} "
                f"src_counts={scount:.0f}"
            )


res = pd.DataFrame(rows)

detail = (
    ANN
    / "source_recipient_specificity_4donor_detail.csv"
)

res.to_csv(detail, index=False)

summary = (
    res.groupby(
        ["contrast", "gene"],
        as_index=False,
    )
    .agg(
        donors=("sample", "nunique"),
        min_source_detection=(
            "source_detection",
            "min",
        ),
        median_source_detection=(
            "source_detection",
            "median",
        ),
        median_recipient_detection=(
            "recipient_detection",
            "median",
        ),
        median_log2_ratio=(
            "log2_source_recipient_ratio",
            "median",
        ),
        min_log2_ratio=(
            "log2_source_recipient_ratio",
            "min",
        ),
        donors_positive=(
            "log2_source_recipient_ratio",
            lambda x: int(
                np.sum(np.asarray(x) > 0)
            ),
        ),
        total_source_counts=(
            "source_raw_counts",
            "sum",
        ),
    )
)

summary_path = (
    ANN
    / "source_recipient_specificity_4donor_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False,
)

print("\n" + "=" * 110)
print("SOURCE -> RECIPIENT SPECIFICITY SUMMARY")
print("=" * 110)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print("\nSAVED:", detail)
print("SAVED:", summary_path)
print("\nSOURCE/RECIPIENT SPECIFICITY AUDIT: COMPLETE")
