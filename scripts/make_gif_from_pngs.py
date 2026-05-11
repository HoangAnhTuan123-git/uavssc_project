#!/usr/bin/env python
from pathlib import Path
import argparse
import imageio.v2 as imageio

ap=argparse.ArgumentParser()
ap.add_argument('--input-dir', required=True)
ap.add_argument('--output', required=True)
ap.add_argument('--limit', type=int, default=100)
ap.add_argument('--duration', type=float, default=0.12)
args=ap.parse_args()
files=sorted(Path(args.input_dir).glob('*.png'))[:args.limit]
if not files:
    raise SystemExit('no png files found')
frames=[imageio.imread(f) for f in files]
Path(args.output).parent.mkdir(parents=True, exist_ok=True)
imageio.mimsave(args.output, frames, duration=args.duration, loop=0)
print('saved', args.output)
