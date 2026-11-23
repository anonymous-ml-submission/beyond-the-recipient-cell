#!/usr/bin/env python3

from pathlib import Path
import argparse
import random

import torch
import yaml

from torch.utils.data import DataLoader, Subset
from torchvision.transforms import functional as TF

from leakage_vso.dataset import (
    CellImageDataset,
    _dense_rows,
    _resize_physical,
    _zero_outside_center,
)

from leakage_vso.train_engine import train_model


TRAIN_IDS = [
    "TENX195",
    "TENX196",
    "TENX197",
    "TENX198",
    "TENX199",
]

VAL_IDS = [
    "TENX200",
    "TENX201",
    "TENX202",
]

EXPECTED_TRAIN_N = 1571084
EXPECTED_VAL_N = 730562


class RawContextDataset(CellImageDataset):
    """
    Exact outer/raw-only path for the ordinary raw-label ContextRegressor.

    It deliberately does not construct center/ring views and does not
    retrieve corrected targets, because none are consumed by this model.
    """

    def __getitem__(self, idx):

        si, li = self.rows[idx]

        s, gidx = self.samples[si]

        row = s.cells.iloc[li]

        wsi = self._get_wsi(s)

        x = float(row.x_px)
        y = float(row.y_px)

        canvas_px = max(
            8,
            int(
                round(
                    self.canvas_um
                    / s.pixel_size_um
                )
            ),
        )

        outer = wsi.crop_px(
            x,
            y,
            canvas_px,
        )

        if self.augment:

            if random.random() < 0.5:
                outer = TF.hflip(outer)

            if random.random() < 0.5:
                outer = TF.vflip(outer)

            angle = random.uniform(
                0.0,
                360.0,
            )

            outer = TF.rotate(
                outer,
                angle,
                fill=(124, 116, 104),
            )

        outer = _resize_physical(
            outer,
            self.canvas_um,
            self.model_mpp,
        )

        outer_t = _zero_outside_center(
            self.to_tensor(outer),
            self.context_um,
            self.canvas_um,
        )

        raw = _dense_rows(
            s.raw.X,
            [li],
            gidx,
        )[0]

        item = {
            "outer":
                outer_t,

            "raw":
                torch.from_numpy(raw),

            "cell_id":
                str(row.cell_id),

            "sample_id":
                s.sample_id,

            "donor_id":
                str(
                    row.get(
                        "donor_id",
                        s.sample_id,
                    )
                ),

            "x_um":
                float(
                    row.get(
                        "x_um",
                        float("nan"),
                    )
                ),

            "y_um":
                float(
                    row.get(
                        "y_um",
                        float("nan"),
                    )
                ),
        }

        if "cell_type" in row.index:
            item["cell_type"] = str(
                row.cell_type
            )

        return item


def load_cfg(path):

    cfg = yaml.safe_load(
        Path(path).read_text()
    )

    assert cfg["seed"] == 2026

    assert cfg["data"]["train_sample_ids"] == TRAIN_IDS
    assert cfg["data"]["val_sample_ids"] == VAL_IDS

    assert cfg["data"]["target"] == "raw"

    assert float(
        cfg["data"]["canvas_um"]
    ) == 128.0

    assert float(
        cfg["data"]["center_um"]
    ) == 24.0

    assert float(
        cfg["data"]["model_mpp"]
    ) == 0.5

    assert int(
        cfg["data"]["context_um"]
    ) in {
        24,
        64,
        128,
    }

    assert cfg["model"]["method"] == "context"
    assert cfg["model"]["backbone"] == "resnet18"
    assert cfg["model"]["pretrained"] is True
    assert cfg["model"]["freeze_backbone"] is False

    assert int(
        cfg["training"]["epochs"]
    ) == 30

    assert int(
        cfg["training"]["batch_size"]
    ) == 48

    assert int(
        cfg["training"]["workers"]
    ) == 6

    return cfg


def genes_from_cfg(cfg):

    genes = [
        x.strip()
        for x in Path(
            cfg["data"]["genes"]
        ).read_text().splitlines()
        if x.strip()
    ]

    if len(genes) != 280:
        raise RuntimeError(
            f"expected 280 genes; got {len(genes)}"
        )

    if len(set(genes)) != 280:
        raise RuntimeError(
            "duplicate genes"
        )

    return genes


def make_dataset(
    cfg,
    genes,
    sample_ids,
    split,
    augment,
):

    return RawContextDataset(
        prepared_root=
            cfg["data"]["prepared_root"],

        sample_ids=
            sample_ids,

        genes=
            genes,

        target=
            "raw",

        context_um=
            cfg["data"]["context_um"],

        center_um=
            cfg["data"]["center_um"],

        model_mpp=
            cfg["data"]["model_mpp"],

        canvas_um=
            cfg["data"]["canvas_um"],

        split=
            split,

        augment=
            augment,

        ring_shuffle=
            False,
    )


def worker_preflight(
    cfg,
    genes,
):

    ds = make_dataset(
        cfg,
        genes,
        TRAIN_IDS,
        "train",
        True,
    )

    if len(ds) != EXPECTED_TRAIN_N:
        raise RuntimeError(
            f"train N changed: {len(ds)}"
        )

    # Important: do not touch ds[i] in the parent before workers start.
    idx = [
        (
            i * 104729
            + 17
        )
        % len(ds)
        for i in range(
            48 * 6
        )
    ]

    dl = DataLoader(
        Subset(
            ds,
            idx,
        ),
        batch_size=48,
        shuffle=False,
        num_workers=6,
        pin_memory=True,
    )

    n = 0

    for b in dl:

        if b["outer"].shape[1:] != (
            3,
            256,
            256,
        ):
            raise RuntimeError(
                f"bad image tensor {b['outer'].shape}"
            )

        if b["raw"].shape[1] != 280:
            raise RuntimeError(
                f"bad raw tensor {b['raw'].shape}"
            )

        n += int(
            b["outer"].shape[0]
        )

    if n != len(idx):
        raise RuntimeError(
            f"worker preflight N={n}"
        )

    print(
        "6-WORKER OUTER-ONLY PREFLIGHT: PASS",
        n,
        "cells",
        flush=True,
    )


def main():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--config",
        required=True,
    )

    p.add_argument(
        "--out",
        default=None,
    )

    p.add_argument(
        "--preflight-only",
        action="store_true",
    )

    a = p.parse_args()

    cfg = load_cfg(
        a.config
    )

    genes = genes_from_cfg(
        cfg
    )

    if a.preflight_only:

        worker_preflight(
            cfg,
            genes,
        )

        return

    if not a.out:
        raise RuntimeError(
            "--out required for training"
        )

    tr = make_dataset(
        cfg,
        genes,
        TRAIN_IDS,
        "train",
        True,
    )

    va = make_dataset(
        cfg,
        genes,
        VAL_IDS,
        "val",
        False,
    )

    if len(tr) != EXPECTED_TRAIN_N:
        raise RuntimeError(
            f"train N changed: {len(tr)}"
        )

    if len(va) != EXPECTED_VAL_N:
        raise RuntimeError(
            f"val N changed: {len(va)}"
        )

    print(
        "PRIMARY2025 RAW CONTEXT TRAINING",
        flush=True,
    )

    print(
        "context_um =",
        cfg["data"]["context_um"],
        flush=True,
    )

    print(
        "train =",
        len(tr),
        "val =",
        len(va),
        "genes =",
        len(genes),
        flush=True,
    )

    print(
        "train samples =",
        TRAIN_IDS,
        flush=True,
    )

    print(
        "val samples =",
        VAL_IDS,
        flush=True,
    )

    print(
        "test samples = NONE",
        flush=True,
    )

    train_model(
        cfg,
        tr,
        va,
        genes,
        a.out,
    )


if __name__ == "__main__":
    main()
