from pathlib import Path
import argparse
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", type=str)
    args = ap.parse_args()
    arr = np.load(args.npz, allow_pickle=False)
    print("keys:", sorted(arr.files))
    for k in sorted(arr.files):
        v = arr[k]
        if isinstance(v, np.ndarray):
            print(k, v.shape, v.dtype)

if __name__ == "__main__":
    main()
