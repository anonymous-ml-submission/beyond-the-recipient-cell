from __future__ import annotations
import hashlib, json, os, random
from pathlib import Path
from typing import Any
import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def stable_int_hash(*parts: object, mod: int = 2**31 - 1) -> int:
    s = "||".join(map(str, parts)).encode("utf-8")
    return int(hashlib.sha256(s).hexdigest()[:16], 16) % mod


def atomic_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    os.replace(tmp, path)


def device_from_string(x: str) -> torch.device:
    if x == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(x)
