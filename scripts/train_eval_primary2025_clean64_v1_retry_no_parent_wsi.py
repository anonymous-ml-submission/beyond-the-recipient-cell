#!/usr/bin/env python3

from pathlib import Path
import argparse

import torch
import yaml

from train_primary2025_context_v1 import (
    RawContextDataset,
    TRAIN_IDS,
    VAL_IDS,
)

from leakage_vso.dataset import (
    _dense_rows,
)

from leakage_vso.train_engine import (
    train_model,
)

from leakage_vso.evaluate import (
    evaluate_checkpoint,
)


TEST_IDS = [
    "TENX191",
    "TENX192",
    "TENX193",
]

EXPECTED = {
    "train": 1571084,
    "val": 730562,
    "test": 839476,
}


class CleanContextDataset(
    RawContextDataset
):

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
                f"{s.sample_id}: missing clean target"
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


def close_dataset(ds):

    for w in ds._wsi.values():
        try:
            w.close()
        except Exception:
            pass


def read_cfg(path):

    cfg = yaml.safe_load(
        Path(path).read_text()
    )

    assert cfg["seed"] == 2026
    assert cfg["model"]["method"] == "context"
    assert cfg["model"]["backbone"] == "resnet18"
    assert cfg["data"]["target"] == "clean"

    assert int(
        cfg["data"]["context_um"]
    ) == 64

    assert float(
        cfg["data"]["canvas_um"]
    ) == 128.0

    assert float(
        cfg["data"]["center_um"]
    ) == 24.0

    assert float(
        cfg["data"]["model_mpp"]
    ) == 0.5

    assert (
        cfg["data"]["train_sample_ids"]
        == TRAIN_IDS
    )

    assert (
        cfg["data"]["val_sample_ids"]
        == VAL_IDS
    )

    return cfg


def genes(cfg):

    g = [
        x.strip()
        for x in Path(
            cfg["data"]["genes"]
        ).read_text().splitlines()
        if x.strip()
    ]

    assert len(g) == 280
    assert len(set(g)) == 280

    return g


def make(
    cfg,
    genes,
    sample_ids,
    split,
    augment,
    target,
):

    return CleanContextDataset(
        prepared_root=
            cfg["data"]["prepared_root"],

        sample_ids=
            sample_ids,

        genes=
            genes,

        target=
            target,

        context_um=
            64,

        center_um=
            24,

        model_mpp=
            0.5,

        canvas_um=
            128,

        split=
            split,

        augment=
            augment,

        ring_shuffle=
            False,
    )


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        required=True,
    )

    ap.add_argument(
        "--run-dir",
        required=True,
    )

    ap.add_argument(
        "--stage",
        required=True,
        choices=[
            "train",
            "eval",
        ],
    )

    a = ap.parse_args()

    cfg = read_cfg(
        a.config
    )

    gene_axis = genes(
        cfg
    )

    run = Path(
        a.run_dir
    )

    if a.stage == "train":

        out = (
            run
            / "train"
        )

        if out.exists():
            raise RuntimeError(
                f"refusing overwrite: {out}"
            )

        tr = make(
            cfg,
            gene_axis,
            TRAIN_IDS,
            "train",
            True,
            "clean",
        )

        va = make(
            cfg,
            gene_axis,
            VAL_IDS,
            "val",
            False,
            "clean",
        )

        assert (
            len(tr)
            ==
            EXPECTED["train"]
        )

        assert (
            len(va)
            ==
            EXPECTED["val"]
        )

        # Do not open any parent-process WSI before forking workers.
        # This was explicitly required by the successful raw baseline.
        from torch.utils.data import DataLoader, Subset

        assert not tr._wsi and not va._wsi
        probe_ids = [
            (i * 104729 + 17) % len(tr)
            for i in range(48 * 6)
        ]
        loader = DataLoader(
            Subset(tr, probe_ids),
            batch_size=48,
            shuffle=False,
            num_workers=6,
            pin_memory=True,
        )
        checked = 0
        for batch in loader:
            assert tuple(batch["outer"].shape[1:]) == (3, 256, 256)
            assert batch["raw"].shape[1] == 280
            assert batch["clean"].shape[1] == 280
            checked += len(batch["outer"])
        assert checked == len(probe_ids), (checked, len(probe_ids))
        del loader
        assert not tr._wsi and not va._wsi

        print(
            "CLEAN64 INPUT CONTRACT: PASS",
            flush=True,
        )

        train_model(
            cfg,
            tr,
            va,
            gene_axis,
            out,
        )

        close_dataset(tr)
        close_dataset(va)

        assert (
            out
            / "best.pt"
        ).is_file()

        assert (
            out
            / "history.json"
        ).is_file()

        print(
            "CLEAN64 TRAINING: PASS",
            flush=True,
        )

    else:

        ck = (
            run
            / "train"
            / "best.pt"
        )

        if not ck.is_file():
            raise RuntimeError(
                f"missing checkpoint: {ck}"
            )

        out = (
            run
            / "eval"
        )

        if out.exists():
            raise RuntimeError(
                f"refusing overwrite: {out}"
            )

        te = make(
            cfg,
            gene_axis,
            TEST_IDS,
            "test",
            False,
            "both",
        )

        assert (
            len(te)
            ==
            EXPECTED["test"]
        )

        evaluate_checkpoint(
            checkpoint=
                ck,

            dataset=
                te,

            outdir=
                out,

            batch_size=
                64,

            workers=
                6,

            device=
                "auto",
        )

        close_dataset(te)

        for name in [
            "metrics_raw.csv",
            "metrics_raw_by_donor.csv",
            "metrics_clean.csv",
            "metrics_clean_by_donor.csv",
            "predictions.npz",
        ]:
            assert (
                out
                / name
            ).is_file()

        print(
            "CLEAN64 HELD-OUT EVAL: PASS",
            flush=True,
        )


if __name__ == "__main__":
    main()
