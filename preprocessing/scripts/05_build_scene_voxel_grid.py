from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from rich import print

from uavssc.utils import load_yaml
from uavssc.voxel import SparseVoxelVotes, resolve_voxel_state



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=str, required=True, help='Directory containing *_sparse_votes.npz')
    ap.add_argument('--config', type=str, default='configs/default.yaml')
    ap.add_argument('--output', type=str, required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    min_occ_votes = int(cfg['voxel']['min_occ_votes'])
    min_free_votes = int(cfg['voxel']['min_free_votes'])
    occ_free_ratio = float(cfg['voxel']['occ_free_ratio'])

    in_root = Path(args.input)
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    for path in sorted(in_root.glob('*_sparse_votes.npz')):
        sparse = SparseVoxelVotes.load_npz(path)
        keys = set(sparse.occ_votes.keys()) | set(sparse.free_counts.keys())
        idxs = []
        states = []
        sems = []
        for key in keys:
            state, sem = resolve_voxel_state(
                sparse.occ_votes.get(key),
                int(sparse.free_counts.get(key, 0)),
                min_occ_votes=min_occ_votes,
                min_free_votes=min_free_votes,
                occ_free_ratio=occ_free_ratio,
            )
            if state == 255:
                continue
            idxs.append(key)
            states.append(state)
            sems.append(sem)

        out_path = out_root / path.name.replace('_sparse_votes.npz', '_resolved_voxels.npz')
        np.savez_compressed(
            out_path,
            idx=np.asarray(idxs, dtype=np.int32),
            state=np.asarray(states, dtype=np.uint8),
            sem=np.asarray(sems, dtype=np.uint8),
            voxel_size=np.array([sparse.voxel_size], dtype=np.float32),
            min_idx=sparse.min_idx.astype(np.int32),
            max_idx=sparse.max_idx.astype(np.int32),
        )
        n_occ = int(np.sum(np.asarray(states) == 1))
        n_free = int(np.sum(np.asarray(states) == 0))
        print(f'[green]Saved[/green] {out_path} | occ={n_occ} free={n_free}')


if __name__ == '__main__':
    main()
