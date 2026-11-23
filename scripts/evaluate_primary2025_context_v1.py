#!/usr/bin/env python3

from pathlib import Path
import argparse

import torch

from train_primary2025_context_v1 import RawContextDataset

from leakage_vso.dataset import _dense_rows
from leakage_vso.evaluate import evaluate_checkpoint


TEST_IDS = [
    "TENX191",
    "TENX192",
    "TENX193",
]

EXPECTED_TEST_N = 839476


class RawContextTestDataset(
    RawContextDataset
):
    """
    Same exact outer image path as frozen raw-context training.

    Evaluation additionally returns the independently
    DeLeakage-corrected target for scoring the SAME predictions.
    """

    def __getitem__(
        self,
        idx,
    ):

        item = super().__getitem__(
            idx
        )

        si, li = self.rows[idx]

        s, gidx = self.samples[si]

        if s.clean is None:
            raise RuntimeError(
                f"{s.sample_id}: "
                "clean.h5ad missing"
            )

        clean = _dense_rows(
            s.clean.X,
            [li],
            gidx,
        )[0]

        item["clean"] = (
            torch.from_numpy(
                clean
            )
        )

        return item


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--checkpoint",
        required=True,
    )

    ap.add_argument(
        "--out",
        required=True,
    )

    ap.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    ap.add_argument(
        "--workers",
        type=int,
        default=6,
    )

    args = ap.parse_args()

    ck = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    cfg = ck["cfg"]

    genes = list(
        ck["genes"]
    )

    frozen_genes = [
        x.strip()
        for x in Path(
            "runs/frozen_primary2025/"
            "gene_axis_280.txt"
        ).read_text().splitlines()
        if x.strip()
    ]

    assert genes == frozen_genes
    assert len(genes) == 280

    assert (
        cfg["model"]["method"]
        == "context"
    )

    assert (
        cfg["data"]["target"]
        == "raw"
    )

    context = int(
        cfg["data"]["context_um"]
    )

    assert context in (
        24,
        64,
        128,
    )

    assert float(
        cfg["data"]["canvas_um"]
    ) == 128.0

    assert float(
        cfg["data"]["model_mpp"]
    ) == 0.5

    ds = RawContextTestDataset(
        prepared_root=
            "data/prepared_primary2025",

        sample_ids=
            TEST_IDS,

        genes=
            genes,

        # Enforce existence of clean targets at construction.
        target=
            "both",

        context_um=
            context,

        center_um=
            cfg["data"]["center_um"],

        model_mpp=
            cfg["data"]["model_mpp"],

        canvas_um=
            cfg["data"]["canvas_um"],

        split=
            "test",

        augment=
            False,

        ring_shuffle=
            False,
    )

    if len(ds) != EXPECTED_TEST_N:
        raise RuntimeError(
            f"test N changed: "
            f"{len(ds)} != "
            f"{EXPECTED_TEST_N}"
        )

    print(
        "=============================================================="
    )

    print(
        "PRIMARY2025 HELD-OUT TEST"
    )

    print(
        "context_um =",
        context,
    )

    print(
        "test IDs =",
        TEST_IDS,
    )

    print(
        "test N =",
        len(ds),
    )

    print(
        "genes =",
        len(genes),
    )

    print(
        "prediction source = "
        "RAW-TRAINED ContextRegressor"
    )

    print(
        "evaluation targets = "
        "RAW + DELEAKAGE-CORRECTED"
    )

    print(
        "=============================================================="
    )

    evaluate_checkpoint(
        checkpoint=
            args.checkpoint,

        dataset=
            ds,

        outdir=
            args.out,

        batch_size=
            args.batch_size,

        workers=
            args.workers,

        device=
            "auto",
    )

    out = Path(
        args.out
    )

    required = [
        "metrics_raw.csv",
        "metrics_raw_by_donor.csv",
        "metrics_clean.csv",
        "metrics_clean_by_donor.csv",
        "predictions.npz",
    ]

    for name in required:
        p = out / name
        if not p.is_file():
            raise RuntimeError(
                f"missing evaluation artifact: {p}"
            )

    print(
        "HELD-OUT TEST EVALUATION: PASS"
    )


if __name__ == "__main__":
    main()
