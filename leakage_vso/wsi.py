from __future__ import annotations
from pathlib import Path
import re
import numpy as np
from PIL import Image
import tifffile


def _physical_size_from_ome_xml(xml: str | None) -> tuple[float | None, float | None]:
    if not xml:
        return None, None
    mx = re.search(r'PhysicalSizeX="([0-9.eE+-]+)"(?:\s+PhysicalSizeXUnit="([^"]+)")?', xml)
    my = re.search(r'PhysicalSizeY="([0-9.eE+-]+)"(?:\s+PhysicalSizeYUnit="([^"]+)")?', xml)
    def conv(m):
        if not m:
            return None
        val = float(m.group(1)); unit = (m.group(2) or "µm").lower()
        if unit in {"µm", "um", "micrometer", "micrometre"}: return val
        if unit in {"nm", "nanometer", "nanometre"}: return val / 1000.0
        if unit in {"mm", "millimeter", "millimetre"}: return val * 1000.0
        return None
    return conv(mx), conv(my)


class TiffWSI:
    """Lazy-ish TIFF/OME-TIFF reader with coordinate-safe cropping.

    It uses tifffile's zarr backend when possible; otherwise falls back to a memmap/array.
    Coordinates passed to crop_px are level-0 pixels.
    """
    def __init__(self, path: str | Path, level: int = 0):
        self.path = str(path)
        self.level = int(level)
        self._tf = tifffile.TiffFile(self.path)
        self.pixel_size_um_x, self.pixel_size_um_y = _physical_size_from_ome_xml(self._tf.ome_metadata)
        self._arr = None
        try:
            import zarr
            store = self._tf.aszarr(series=0, level=self.level)
            self._arr = zarr.open(store, mode="r")
        except Exception:
            self._arr = self._tf.series[0].levels[self.level].asarray()
        self.shape = tuple(self._arr.shape)
        self._channels_last = self._infer_channels_last(self.shape)
        self.height, self.width = self._hw(self.shape)

    @staticmethod
    def _infer_channels_last(shape):
        if len(shape) == 2: return True
        if len(shape) == 3:
            if shape[-1] in (3,4): return True
            if shape[0] in (3,4): return False
        raise ValueError(f"Unsupported TIFF array shape {shape}; expected grayscale or RGB(A)")

    @staticmethod
    def _hw(shape):
        if len(shape) == 2: return int(shape[0]), int(shape[1])
        if shape[-1] in (3,4): return int(shape[0]), int(shape[1])
        return int(shape[1]), int(shape[2])

    def close(self):
        self._tf.close()

    def crop_px(self, x_px: float, y_px: float, size_px: int) -> Image.Image:
        size_px = int(size_px)
        x0 = int(round(x_px - size_px/2)); y0 = int(round(y_px - size_px/2))
        x1 = x0 + size_px; y1 = y0 + size_px
        out = np.full((size_px, size_px, 3), 255, dtype=np.uint8)
        sx0, sy0 = max(0,x0), max(0,y0)
        sx1, sy1 = min(self.width,x1), min(self.height,y1)
        if sx1 <= sx0 or sy1 <= sy0:
            return Image.fromarray(out)
        if len(self.shape) == 2:
            sub = np.asarray(self._arr[sy0:sy1, sx0:sx1])
            sub = np.repeat(sub[...,None], 3, axis=-1)
        elif self._channels_last:
            sub = np.asarray(self._arr[sy0:sy1, sx0:sx1, :3])
        else:
            sub = np.asarray(self._arr[:3, sy0:sy1, sx0:sx1]).transpose(1,2,0)
        if sub.dtype != np.uint8:
            sub = self._to_u8(sub)
        dy0, dx0 = sy0-y0, sx0-x0
        out[dy0:dy0+sub.shape[0], dx0:dx0+sub.shape[1]] = sub[...,:3]
        return Image.fromarray(out)

    @staticmethod
    def _to_u8(a: np.ndarray) -> np.ndarray:
        a = np.asarray(a)
        if np.issubdtype(a.dtype, np.integer):
            mx = np.iinfo(a.dtype).max
            return np.clip(a.astype(np.float32) / max(mx,1) * 255,0,255).astype(np.uint8)
        lo, hi = np.nanpercentile(a, [0.5,99.5])
        if hi <= lo: hi = lo + 1
        return np.clip((a-lo)/(hi-lo)*255,0,255).astype(np.uint8)
