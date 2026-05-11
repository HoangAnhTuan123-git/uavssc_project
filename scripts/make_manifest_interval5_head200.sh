#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
SCENE="${SCENE:-interval5_AMtown01}"
N="${N:-200}"
OUT="${OUT:-data/index/manifest_interval5_${SCENE}_head${N}.parquet}"
python - <<PYINNER
from pathlib import Path
import pandas as pd
scene=${SCENE@Q}
n=int(${N@Q})
manifest=Path('data/index/manifest_interval5.parquet')
df=pd.read_parquet(manifest)
print('available scenes:', sorted(df['scene'].dropna().unique())[:20])
sdf=df[df['scene']==scene].head(n).copy()
Path(${OUT@Q}).parent.mkdir(parents=True, exist_ok=True)
sdf.to_parquet(${OUT@Q})
print('wrote', ${OUT@Q}, 'rows', len(sdf))
PYINNER
