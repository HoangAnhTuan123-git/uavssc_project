from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

COLORS = np.array([
    [0,0,0], [70,70,70], [128,64,128], [128,128,128], [0,0,255], [0,120,255],
    [150,100,100], [100,100,40], [180,180,180], [255,255,0], [107,142,35],
    [80,120,40], [255,0,255], [255,120,0], [180,220,220], [100,100,100], [160,160,160],
], dtype=np.float32) / 255.0
CMAP = ListedColormap(COLORS)

def topdown_xyz(target_xyz):
    vol = np.transpose(target_xyz, (2,1,0))
    occ = (vol > 0) & (vol != 255)
    zdim, ydim, xdim = vol.shape
    td = np.zeros((ydim, xdim), dtype=np.int64)
    for z in range(zdim - 1, -1, -1):
        m = occ[z]
        td[m] = vol[z][m]
    return td

def variants(a):
    return {
        "identity": a,
        "flipud": np.flipud(a),
        "fliplr": np.fliplr(a),
        "rot180": np.rot90(a, 2),
        "transpose": a.T,
        "transpose_flipud": np.flipud(a.T),
        "transpose_fliplr": np.fliplr(a.T),
        "transpose_rot180": np.rot90(a.T, 2),
    }

ap = argparse.ArgumentParser()
ap.add_argument("--npz", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()
z = np.load(args.npz, allow_pickle=True)
td = topdown_xyz(z["target"])
fig, axes = plt.subplots(2,4,figsize=(16,8))
for ax, (name, img) in zip(axes.reshape(-1), variants(td).items()):
    ax.imshow(img, cmap=CMAP, vmin=0, vmax=16, interpolation="nearest")
    ax.set_title(name); ax.axis("off")
fig.tight_layout()
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.out, dpi=150)
print("saved", args.out)
