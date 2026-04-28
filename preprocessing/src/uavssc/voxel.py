from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, Tuple

import numpy as np

from .constants import IGNORE_SEMANTIC_ID


Index3 = tuple[int, int, int]


@dataclass
class GridSpec:
    voxel_size: float
    origin_world: np.ndarray   # (3,)
    grid_size_xyz: tuple[int, int, int]  # (nx, ny, nz)

    @property
    def max_world(self) -> np.ndarray:
        return self.origin_world + self.voxel_size * np.asarray(self.grid_size_xyz, dtype=np.float64)



def world_to_voxel_idx(points_world: np.ndarray, spec: GridSpec) -> np.ndarray:
    rel = (np.asarray(points_world, dtype=np.float64) - spec.origin_world[None, :]) / spec.voxel_size
    idx = np.floor(rel).astype(np.int32)
    return idx



def voxel_idx_to_world_center(indices_xyz: np.ndarray, spec: GridSpec) -> np.ndarray:
    indices_xyz = np.asarray(indices_xyz, dtype=np.float64)
    return spec.origin_world[None, :] + (indices_xyz + 0.5) * spec.voxel_size



def within_grid(indices_xyz: np.ndarray, spec: GridSpec) -> np.ndarray:
    idx = np.asarray(indices_xyz, dtype=np.int32)
    nx, ny, nz = spec.grid_size_xyz
    ok = (
        (idx[:, 0] >= 0) & (idx[:, 0] < nx) &
        (idx[:, 1] >= 0) & (idx[:, 1] < ny) &
        (idx[:, 2] >= 0) & (idx[:, 2] < nz)
    )
    return ok


class SparseVoxelVotes:
    """Sparse scene-level vote accumulation.

    Stores per-voxel:
    - occupancy semantic votes: Counter(class_id -> count)
    - free-space count: int
    """

    def __init__(self, voxel_size: float):
        self.voxel_size = float(voxel_size)
        self.occ_votes: Dict[Index3, Counter] = defaultdict(Counter)
        self.free_counts: Dict[Index3, int] = defaultdict(int)
        self.min_idx = np.array([10**9, 10**9, 10**9], dtype=np.int64)
        self.max_idx = np.array([-10**9, -10**9, -10**9], dtype=np.int64)

    def _update_bounds(self, idx: np.ndarray) -> None:
        self.min_idx = np.minimum(self.min_idx, idx)
        self.max_idx = np.maximum(self.max_idx, idx)

    def add_occupied(self, indices_xyz: np.ndarray, class_ids: np.ndarray) -> None:
        indices_xyz = np.asarray(indices_xyz, dtype=np.int32)
        class_ids = np.asarray(class_ids, dtype=np.int32).reshape(-1)
        assert indices_xyz.shape[0] == class_ids.shape[0]
        for idx, cid in zip(indices_xyz, class_ids):
            key = (int(idx[0]), int(idx[1]), int(idx[2]))
            self.occ_votes[key][int(cid)] += 1
            self._update_bounds(idx)

    def add_free(self, indices_xyz: np.ndarray) -> None:
        indices_xyz = np.asarray(indices_xyz, dtype=np.int32)
        for idx in indices_xyz:
            key = (int(idx[0]), int(idx[1]), int(idx[2]))
            self.free_counts[key] += 1
            self._update_bounds(idx)

    def save_npz(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        occ_idx = []
        occ_cls = []
        occ_cnt = []
        for idx, counter in self.occ_votes.items():
            for cls_id, cnt in counter.items():
                occ_idx.append(idx)
                occ_cls.append(int(cls_id))
                occ_cnt.append(int(cnt))

        free_idx = []
        free_cnt = []
        for idx, cnt in self.free_counts.items():
            free_idx.append(idx)
            free_cnt.append(int(cnt))

        np.savez_compressed(
            path,
            voxel_size=np.array([self.voxel_size], dtype=np.float32),
            occ_idx=np.asarray(occ_idx, dtype=np.int32),
            occ_cls=np.asarray(occ_cls, dtype=np.int16),
            occ_cnt=np.asarray(occ_cnt, dtype=np.int32),
            free_idx=np.asarray(free_idx, dtype=np.int32),
            free_cnt=np.asarray(free_cnt, dtype=np.int32),
            min_idx=self.min_idx.astype(np.int32),
            max_idx=self.max_idx.astype(np.int32),
        )

    @staticmethod
    def load_npz(path: str | Path) -> 'SparseVoxelVotes':
        data = np.load(path, allow_pickle=False)
        obj = SparseVoxelVotes(float(data['voxel_size'][0]))
        occ_idx = data['occ_idx']
        occ_cls = data['occ_cls']
        occ_cnt = data['occ_cnt']
        free_idx = data['free_idx']
        free_cnt = data['free_cnt']
        for idx, cls_id, cnt in zip(occ_idx, occ_cls, occ_cnt):
            key = tuple(int(x) for x in idx)
            obj.occ_votes[key][int(cls_id)] += int(cnt)
        for idx, cnt in zip(free_idx, free_cnt):
            key = tuple(int(x) for x in idx)
            obj.free_counts[key] += int(cnt)
        obj.min_idx = data['min_idx'].astype(np.int64)
        obj.max_idx = data['max_idx'].astype(np.int64)
        return obj



def point_to_index(points_world: np.ndarray, voxel_size: float) -> np.ndarray:
    return np.floor(np.asarray(points_world, dtype=np.float64) / voxel_size).astype(np.int32)



def unique_rows_with_majority_label(idx_xyz: np.ndarray, class_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Deduplicate multiple points landing in same voxel.

    Returns:
        unique_idx: (M,3)
        majority_cls: (M,)
    """
    buckets: Dict[Index3, Counter] = defaultdict(Counter)
    for idx, cid in zip(idx_xyz, class_ids):
        key = (int(idx[0]), int(idx[1]), int(idx[2]))
        buckets[key][int(cid)] += 1

    out_idx = []
    out_cls = []
    for key, counter in buckets.items():
        out_idx.append(key)
        out_cls.append(counter.most_common(1)[0][0])
    return np.asarray(out_idx, dtype=np.int32), np.asarray(out_cls, dtype=np.int32)



def ray_voxel_indices(
    origin_world: np.ndarray,
    hit_world: np.ndarray,
    voxel_size: float,
    step_ratio: float = 0.5,
    include_endpoint: bool = False,
) -> np.ndarray:
    """Simple ray sampling by fixed world steps.

    Correct-first implementation. Good for dataset construction debugging.
    Later you may replace this with Amanatides-Woo for speed.
    """
    origin_world = np.asarray(origin_world, dtype=np.float64).reshape(3)
    hit_world = np.asarray(hit_world, dtype=np.float64).reshape(3)
    d = hit_world - origin_world
    dist = float(np.linalg.norm(d))
    if dist < 1e-9:
        return np.empty((0, 3), dtype=np.int32)
    direction = d / dist
    step = max(voxel_size * step_ratio, 1e-3)
    n = max(1, int(np.floor(dist / step)))
    ts = np.linspace(0.0, dist, n + 1, endpoint=include_endpoint)
    samples = origin_world[None, :] + ts[:, None] * direction[None, :]
    idx = point_to_index(samples, voxel_size)
    if idx.shape[0] == 0:
        return idx
    idx_unique = np.unique(idx, axis=0)
    # Exclude the endpoint voxel from free space by default.
    if (not include_endpoint) and idx_unique.shape[0] > 0:
        hit_idx = point_to_index(hit_world[None, :], voxel_size)[0]
        keep = np.any(idx_unique != hit_idx[None, :], axis=1)
        idx_unique = idx_unique[keep]
    return idx_unique



def resolve_voxel_state(
    occ_counter: Counter | None,
    free_count: int,
    min_occ_votes: int,
    min_free_votes: int,
    occ_free_ratio: float,
) -> tuple[int, int]:
    """Resolve voxel to state and semantic.

    Returns:
        state: 255 unknown, 0 free, 1 occupied
        sem: semantic class for occupied else IGNORE_SEMANTIC_ID
    """
    occ_votes = 0 if occ_counter is None else int(sum(occ_counter.values()))
    if occ_votes == 0 and free_count == 0:
        return 255, IGNORE_SEMANTIC_ID
    if occ_votes >= min_occ_votes and occ_votes >= free_count * occ_free_ratio:
        sem = int(occ_counter.most_common(1)[0][0]) if occ_counter is not None else IGNORE_SEMANTIC_ID
        return 1, sem
    if free_count >= min_free_votes:
        return 0, IGNORE_SEMANTIC_ID
    return 255, IGNORE_SEMANTIC_ID



def dense_local_grid_from_sparse(
    sparse_votes: SparseVoxelVotes,
    center_world: np.ndarray,
    grid_size_m_xyz: tuple[float, float, float],
    voxel_size: float,
    min_occ_votes: int = 1,
    min_free_votes: int = 1,
    occ_free_ratio: float = 1.0,
) -> dict[str, np.ndarray]:
    center_world = np.asarray(center_world, dtype=np.float64).reshape(3)
    size_m = np.asarray(grid_size_m_xyz, dtype=np.float64)
    nxyz = np.ceil(size_m / voxel_size).astype(np.int32)
    nx, ny, nz = [int(v) for v in nxyz]
    origin_world = center_world - 0.5 * size_m

    occ_mask = np.zeros((nx, ny, nz), dtype=np.uint8)
    free_mask = np.zeros((nx, ny, nz), dtype=np.uint8)
    known_mask = np.zeros((nx, ny, nz), dtype=np.uint8)
    sem_label = np.full((nx, ny, nz), IGNORE_SEMANTIC_ID, dtype=np.uint8)

    # Candidate index range in global sparse integer grid.
    idx_min = np.floor(origin_world / voxel_size).astype(np.int32)
    idx_max = np.floor((origin_world + size_m) / voxel_size).astype(np.int32)

    def global_to_local(key: Index3) -> Index3:
        gx, gy, gz = key
        return gx - idx_min[0], gy - idx_min[1], gz - idx_min[2]

    # Iterate over union of sparse keys in region.
    keys = set()
    for key in sparse_votes.occ_votes.keys():
        k = np.asarray(key, dtype=np.int32)
        if np.all(k >= idx_min) and np.all(k < idx_max):
            keys.add(key)
    for key in sparse_votes.free_counts.keys():
        k = np.asarray(key, dtype=np.int32)
        if np.all(k >= idx_min) and np.all(k < idx_max):
            keys.add(key)

    for key in keys:
        lx, ly, lz = global_to_local(key)
        if not (0 <= lx < nx and 0 <= ly < ny and 0 <= lz < nz):
            continue
        state, sem = resolve_voxel_state(
            sparse_votes.occ_votes.get(key),
            int(sparse_votes.free_counts.get(key, 0)),
            min_occ_votes=min_occ_votes,
            min_free_votes=min_free_votes,
            occ_free_ratio=occ_free_ratio,
        )
        if state == 255:
            continue
        known_mask[lx, ly, lz] = 1
        if state == 1:
            occ_mask[lx, ly, lz] = 1
            sem_label[lx, ly, lz] = sem
        elif state == 0:
            free_mask[lx, ly, lz] = 1

    return {
        'origin_world': origin_world.astype(np.float32),
        'voxel_size': np.array([voxel_size], dtype=np.float32),
        'grid_size_xyz': np.array([nx, ny, nz], dtype=np.int32),
        'occ_mask': occ_mask,
        'free_mask': free_mask,
        'known_mask': known_mask,
        'sem_label': sem_label,
    }
