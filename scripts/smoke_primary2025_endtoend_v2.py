#!/usr/bin/env python3

from pathlib import Path
import hashlib
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from leakage_vso.dataset import CellImageDataset
from leakage_vso.models import ContextRegressor
from leakage_vso.losses import poisson_loss


ROOT = "data/prepared_primary2025"

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

TRAIN_N = 1571084
VAL_N = 730562

GENES = [
    x.strip()
    for x in Path(
        "runs/frozen_primary2025/gene_axis_280.txt"
    ).read_text().splitlines()
    if x.strip()
]

assert len(GENES) == 280

BATCH = 48
WORKERS = 0
SEED = 2026

# Same 128-um physical canvas for every context; therefore all use
# identical 256x256 tensors.
CONTEXTS = [24, 64, 128]

WARMUP_BATCHES = 4
MEASURE_BATCHES = 32
NEEDED = (
    WARMUP_BATCHES
    +
    MEASURE_BATCHES
) * BATCH


def sha(p):
    h = hashlib.sha256()

    with open(p, "rb") as f:
        for b in iter(
            lambda: f.read(1 << 20),
            b"",
        ):
            h.update(b)

    return h.hexdigest()


def subset_indices(
    n,
    seed,
):
    rng = np.random.default_rng(seed)

    idx = rng.choice(
        n,
        size=NEEDED,
        replace=False,
    )

    return np.asarray(
        idx,
        dtype=np.int64,
    )


def make_ds(
    ids,
    split,
    context,
    augment,
):
    return CellImageDataset(
        prepared_root=ROOT,
        sample_ids=ids,
        genes=GENES,
        target="raw",
        context_um=float(context),
        center_um=24.0,
        model_mpp=0.5,
        canvas_um=128.0,
        split=split,
        augment=augment,
        ring_shuffle=False,
    )


def make_loader(
    ds,
    idx,
    shuffle,
    seed,
):
    g = torch.Generator()
    g.manual_seed(seed)

    return DataLoader(
        Subset(
            ds,
            idx.tolist(),
        ),
        batch_size=BATCH,
        shuffle=shuffle,
        num_workers=WORKERS,
        pin_memory=True,
        drop_last=True,
        generator=g,
    )


if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA unavailable"
    )

device = torch.device(
    "cuda:0"
)

torch.manual_seed(
    SEED
)

np.random.seed(
    SEED
)

results = []


for context in CONTEXTS:

    print()
    print("=" * 78)
    print(
        f"CONTEXT {context} um — TRUE END-TO-END"
    )
    print("=" * 78)

    train_ds = make_ds(
        TRAIN_IDS,
        "train",
        context,
        True,
    )

    val_ds = make_ds(
        VAL_IDS,
        "val",
        context,
        False,
    )

    assert len(train_ds) == TRAIN_N
    assert len(val_ds) == VAL_N

    train_idx = subset_indices(
        len(train_ds),
        SEED + context,
    )

    val_idx = subset_indices(
        len(val_ds),
        SEED + 1000 + context,
    )

    train_loader = make_loader(
        train_ds,
        train_idx,
        True,
        SEED + 2000 + context,
    )

    val_loader = make_loader(
        val_ds,
        val_idx,
        False,
        SEED + 3000 + context,
    )

    print(
        "loading pretrained ResNet18..."
    )

    model = ContextRegressor(
        n_genes=280,
        backbone="resnet18",
        pretrained=True,
        freeze_backbone=False,
    ).to(
        device
    )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4,
    )

    # --------------------------------------------------------------
    # TRAIN: timer includes DataLoader/TIFF/augmentation + H2D +
    # forward/backward/optimizer.
    # --------------------------------------------------------------

    model.train()

    it = iter(
        train_loader
    )

    for _ in range(
        WARMUP_BATCHES
    ):
        b = next(it)

        x = b["outer"].to(
            device,
            non_blocking=True,
        )

        y = b["raw"].to(
            device,
            non_blocking=True,
        )

        opt.zero_grad(
            set_to_none=True
        )

        pred = model(x)

        loss = poisson_loss(
            pred,
            y,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            5.0,
        )

        opt.step()

    torch.cuda.synchronize()

    t0 = time.perf_counter()

    train_cells = 0

    for _ in range(
        MEASURE_BATCHES
    ):
        # next(it) is deliberately INSIDE timing.
        b = next(it)

        x = b["outer"].to(
            device,
            non_blocking=True,
        )

        y = b["raw"].to(
            device,
            non_blocking=True,
        )

        opt.zero_grad(
            set_to_none=True
        )

        pred = model(x)

        loss = poisson_loss(
            pred,
            y,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            5.0,
        )

        opt.step()

        train_cells += int(
            x.shape[0]
        )

    torch.cuda.synchronize()

    train_seconds = (
        time.perf_counter()
        - t0
    )

    train_cps = (
        train_cells
        /
        train_seconds
    )

    # --------------------------------------------------------------
    # VAL: timer includes TIFF loading + H2D + forward + CPU copy,
    # matching the important cost of current predict().
    # --------------------------------------------------------------

    model.eval()

    it = iter(
        val_loader
    )

    with torch.no_grad():

        for _ in range(
            WARMUP_BATCHES
        ):
            b = next(it)

            p = model(
                b["outer"].to(
                    device,
                    non_blocking=True,
                )
            )

            _ = p.cpu().numpy()

        torch.cuda.synchronize()

        t0 = time.perf_counter()

        val_cells = 0

        for _ in range(
            MEASURE_BATCHES
        ):
            b = next(it)

            p = model(
                b["outer"].to(
                    device,
                    non_blocking=True,
                )
            )

            # Current train_engine.predict() stores predictions on CPU.
            _ = p.cpu().numpy()

            # Also materialize raw target as current validation does.
            _ = b["raw"].numpy()

            val_cells += int(
                p.shape[0]
            )

        torch.cuda.synchronize()

        val_seconds = (
            time.perf_counter()
            - t0
        )

    val_cps = (
        val_cells
        /
        val_seconds
    )

    full_train_sec = (
        TRAIN_N
        /
        train_cps
    )

    full_val_sec = (
        VAL_N
        /
        val_cps
    )

    full_epoch_sec = (
        full_train_sec
        +
        full_val_sec
    )

    result = {
        "context_um":
            context,

        "train_cells_per_second_endtoend":
            train_cps,

        "val_cells_per_second_endtoend":
            val_cps,

        "projected_full_train_hours":
            full_train_sec / 3600.0,

        "projected_full_val_hours":
            full_val_sec / 3600.0,

        "projected_full_epoch_hours":
            full_epoch_sec / 3600.0,

        "projected_30_epoch_hours":
            30.0
            * full_epoch_sec
            / 3600.0,

        "batch_size":
            BATCH,

        "workers":
            WORKERS,

        "pretrained":
            True,

        "augment_train":
            True,

        "tensor_px":
            256,

        "canvas_um":
            128,

        "model_mpp":
            0.5,
    }

    results.append(
        result
    )

    print(
        "train end-to-end cells/sec =",
        train_cps,
    )

    print(
        "val end-to-end cells/sec =",
        val_cps,
    )

    print(
        "projected full epoch hours =",
        result[
            "projected_full_epoch_hours"
        ],
    )

    print(
        "projected 30 epochs hours =",
        result[
            "projected_30_epoch_hours"
        ],
    )

    del model
    del opt
    del train_loader
    del val_loader
    del train_ds
    del val_ds

    torch.cuda.empty_cache()


payload = {
    "status":
        "PASS",

    "purpose":
        "end_to_end_runtime_measurement_before_training_budget_freeze",

    "scientific_training":
        False,

    "test_data_used":
        False,

    "results":
        results,

    "provenance": {
        "script_sha256":
            sha(
                Path(__file__)
            ),

        "train_py_sha256":
            sha(
                Path(
                    "scripts/train.py"
                )
            ),

        "dataset_py_sha256":
            sha(
                Path(
                    "leakage_vso/dataset.py"
                )
            ),

        "train_engine_py_sha256":
            sha(
                Path(
                    "leakage_vso/train_engine.py"
                )
            ),
    },
}

OUT = Path(
    "runs/model_preflight/"
    "primary2025_endtoend_v2.json"
)

OUT.write_text(
    json.dumps(
        payload,
        indent=2,
    )
    + "\n"
)

print()
print("=" * 78)
print("TRUE END-TO-END MODEL TIMING: PASS")
print("=" * 78)

for r in results:

    print(
        "context",
        r["context_um"],
        "epoch_h=",
        round(
            r[
                "projected_full_epoch_hours"
            ],
            3,
        ),
        "30epoch_h=",
        round(
            r[
                "projected_30_epoch_hours"
            ],
            2,
        ),
    )

print(
    "output SHA256 =",
    sha(OUT),
)
print("=" * 78)
