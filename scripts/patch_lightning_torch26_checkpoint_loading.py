from pathlib import Path
import pytorch_lightning  # noqa: F401
import torch  # noqa: F401

p = Path('/usr/local/lib/python3.10/dist-packages/pytorch_lightning/utilities/cloud_io.py')
s = p.read_text()
backup = p.with_suffix('.py.bak_weights_only_fix')
if not backup.exists():
    backup.write_text(s)
if 'weights_only=False' not in s:
    s = s.replace('return torch.load(f, map_location=map_location)',
                  'return torch.load(f, map_location=map_location, weights_only=False)')
    p.write_text(s)
    print('patched', p)
else:
    print('already patched', p)
