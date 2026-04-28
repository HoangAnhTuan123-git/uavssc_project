from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from .constants import IGNORE_SEMANTIC_ID


class LocalGridDataset(Dataset):
    def __init__(self, root: str | Path, input_key: str = 'input_occ_lidar') -> None:
        self.root = Path(root)
        self.files = sorted(self.root.rglob('*.npz'))
        self.input_key = input_key
        if not self.files:
            raise FileNotFoundError(f'No .npz local grids found under {self.root}')

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        data = np.load(self.files[index], allow_pickle=False)
        x = data[self.input_key].astype(np.float32)
        occ = data['occ_mask'].astype(np.float32)
        known = data['known_mask'].astype(np.float32)
        sem = data['sem_label'].astype(np.int64)

        # Dense 3D conv expects (C, X, Y, Z) or (C, D, H, W). We keep xyz here.
        sample = {
            'x': torch.from_numpy(x[None, ...]),
            'occ': torch.from_numpy(occ[None, ...]),
            'known': torch.from_numpy(known[None, ...]),
            'sem': torch.from_numpy(sem),
        }
        return sample
