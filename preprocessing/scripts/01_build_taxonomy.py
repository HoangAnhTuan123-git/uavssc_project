from __future__ import annotations

import argparse

from uavssc.constants import DEFAULT_DYNAMIC_RAW_IDS, DEFAULT_IGNORE_RAW_IDS, RAW_CMAP
from uavssc.utils import save_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=str, required=True)
    args = ap.parse_args()

    taxonomy = {
        'raw_classes': RAW_CMAP,
        'ignore_raw_ids': DEFAULT_IGNORE_RAW_IDS,
        'dynamic_raw_ids': DEFAULT_DYNAMIC_RAW_IDS,
        'notes': [
            'Edit this taxonomy before training.',
            'Unnamed raw IDs are ignored by default.',
            'You may remap some classes or collapse similar drivable/walkable classes later.',
        ],
    }
    save_json(taxonomy, args.output)
    print(f'Saved taxonomy to {args.output}')


if __name__ == '__main__':
    main()
