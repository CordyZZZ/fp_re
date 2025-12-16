"""FoldPath dataset wrapper.

FoldPath expects a *set of paths* (variable cardinality, variable length).
MaskPlanner's PaintNet loader returns a flattened trajectory plus stroke_ids.

This wrapper:
  - reuses `PaintNetDataloader` for file I/O and normalization;
  - splits the trajectory into per-stroke paths via stroke_ids;
  - resamples each path to T points using linear interpolation on the index axis,
    following the FoldPath paper's use of scalars s in [-1, 1];
  - pads / truncates the path set to `num_queries`.

The wrapper is intentionally conservative: it does not change the underlying
data definitions, only repackages them for FoldPath.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .paintnet import PaintNetDataloader


def _split_by_stroke_ids(traj: np.ndarray, stroke_ids: np.ndarray) -> List[np.ndarray]:
    """Split (N,6) trajectory into list of (Li,6) paths by integer stroke_ids."""
    paths: List[np.ndarray] = []
    if traj.shape[0] == 0:
        return paths
    # stroke_ids can include padding; remove invalid
    valid = stroke_ids >= 0
    traj = traj[valid]
    stroke_ids = stroke_ids[valid]
    if traj.shape[0] == 0:
        return paths

    unique_ids = np.unique(stroke_ids)
    for sid in unique_ids:
        pts = traj[stroke_ids == sid]
        if pts.shape[0] >= 2:
            paths.append(pts)
    return paths


def _interp_path(path: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Resample a path to len(s) points.

    Args:
        path: (L,6)
        s:    (T,) values in [-1,1]

    Returns:
        (T,6)
    """
    L = path.shape[0]
    if L == 1:
        return np.repeat(path, repeats=len(s), axis=0)

    # map s in [-1,1] to t in [0, L-1]
    t = (s + 1.0) * 0.5 * (L - 1)
    t0 = np.floor(t).astype(np.int64)
    t1 = np.clip(t0 + 1, 0, L - 1)
    w = (t - t0).astype(np.float32)
    p0 = path[t0]
    p1 = path[t1]
    out = (1.0 - w)[:, None] * p0 + w[:, None] * p1

    # re-normalize orientation to unit vectors (defensive)
    v = out[:, 3:6]
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-8
    out[:, 3:6] = v / n
    return out


@dataclass
class FoldPathDatasetConfig:
    dataset: str
    roots: List[str]
    split: str = "train"
    pc_points: int = 5120
    normalization: str = "per-mesh"
    data_scale_factor: float | None = None
    augmentations: List[str] | None = None

    # FoldPath-specific
    num_queries: int = 40
    T: int = 64
    sampling: str = "uniform"  # "uniform" (train) or "equispaced" (test)


class FoldPathDataset(Dataset):
    def __init__(self, cfg: FoldPathDatasetConfig):
        super().__init__()
        self.cfg = cfg
        self.base = PaintNetDataloader(
            roots=cfg.roots,
            dataset=cfg.dataset,
            pc_points=cfg.pc_points,
            traj_points=5000,  # keep high; we split by stroke ids anyway
            lambda_points=1,
            overlapping=0,
            split=cfg.split,
            stroke_pred=False,
            extra_data=("orientnorm",),  # ensure 6D (pos + orientation vector)
            normalization=cfg.normalization,
            data_scale_factor=cfg.data_scale_factor,
            augmentations=cfg.augmentations or [],
        )

    def __len__(self) -> int:
        return len(self.base)

    def _sample_s(self) -> np.ndarray:
        if self.cfg.sampling == "equispaced":
            return np.linspace(-1.0, 1.0, self.cfg.T, dtype=np.float32)
        # uniform noise in [-1,1]
        return (np.random.rand(self.cfg.T).astype(np.float32) * 2.0) - 1.0

    def __getitem__(self, idx: int) -> dict:
        pc, traj, _traj_as_pc, stroke_ids, dirname = self.base[idx]

        # base loader may include padding; remove it conservatively
        traj = traj.reshape(-1, 6)
        stroke_ids = stroke_ids.reshape(-1)

        paths = _split_by_stroke_ids(traj, stroke_ids)
        s = self._sample_s()

        Q = self.cfg.num_queries
        T = self.cfg.T
        y = np.zeros((Q, T, 6), dtype=np.float32)
        f = np.zeros((Q,), dtype=np.float32)

        n = min(len(paths), Q)
        for i in range(n):
            y[i] = _interp_path(paths[i].astype(np.float32), s)
            f[i] = 1.0

        sample = {
            "pc": torch.from_numpy(pc).float(),          # (N,3)
            "s": torch.from_numpy(s).float().view(T, 1),  # (T,1)
            "y": torch.from_numpy(y).float(),           # (Q,T,6)
            "f": torch.from_numpy(f).float(),           # (Q,)
            "name": dirname,
        }
        return sample


def foldpath_collate(batch: List[dict]) -> dict:
    pc = torch.stack([b["pc"] for b in batch], dim=0)  # (B,N,3)
    s = torch.stack([b["s"] for b in batch], dim=0)    # (B,T,1)
    y = torch.stack([b["y"] for b in batch], dim=0)    # (B,Q,T,6)
    f = torch.stack([b["f"] for b in batch], dim=0)    # (B,Q)
    name = [b["name"] for b in batch]
    return {"pc": pc, "s": s, "y": y, "f": f, "name": name}
