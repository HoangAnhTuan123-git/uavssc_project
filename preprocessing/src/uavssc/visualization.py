from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def save_rgb(path: str | Path, img_rgb: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), img_bgr)



def save_semantic_slice_png(path: str | Path, sem_label: np.ndarray, z_index: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sl = sem_label[:, :, z_index]
    plt.figure(figsize=(6, 6))
    plt.imshow(sl.T, interpolation='nearest')
    plt.title(f'sem_label z={z_index}')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
