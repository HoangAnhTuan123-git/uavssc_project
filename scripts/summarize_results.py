#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def flatten(prefix: str, obj, out: Dict[str, object]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            flatten(f"{prefix}.{k}" if prefix else str(k), v, out)
    else:
        out[prefix] = obj


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize JSON metrics into a CSV table.")
    ap.add_argument("--metrics-root", type=Path, default=Path("results/metrics"))
    ap.add_argument("--out-csv", type=Path, default=Path("results/tables/metrics_summary.csv"))
    args = ap.parse_args()

    rows: List[Dict[str, object]] = []
    for p in sorted(args.metrics_root.rglob("*.json")):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] could not parse {p}: {e}")
            continue
        row: Dict[str, object] = {"path": str(p)}
        flatten("", obj, row)
        rows.append(row)

    if not rows:
        print("No JSON metrics found.")
        return

    fieldnames = sorted({k for row in rows for k in row.keys()})
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
