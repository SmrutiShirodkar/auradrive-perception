"""Small, dependency-light Parquet read/write helpers.

Deliberately uses pandas + pyarrow (already a core dependency) instead of
pulling in Spark/Delta Lake for the nuScenes-mini scale. The Bronze/Silver
naming mirrors the medallion layout from the report; swapping this for a
real Delta Lake writer later is a drop-in change since the DataFrame shapes
don't need to change.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def read_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)
