"""
End-to-end CLI entrypoint: nuScenes-mini raw JSON -> Bronze sweeps ->
Silver fused world model + quarantine, written as Parquet.

    python -m auradrive.pipeline --nuscenes-root v1.0-mini --out data

Kept intentionally thin: no Spark session, no Azure SDK calls. Everything
runs in-process on pandas/pyarrow, which is enough for nuScenes-mini's
~31k sweep rows and keeps the repo runnable on a laptop with no cloud
dependency.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from auradrive.fusion.world_model import build_fused_world_model
from auradrive.ingest.nuscenes_loader import load_bronze_sweeps
from auradrive.utils.io import write_parquet


def run_pipeline(nuscenes_root: str | Path, out_dir: str | Path, threshold_ms: float = 10.0) -> None:
    out_dir = Path(out_dir)
    t0 = time.perf_counter()

    bronze = load_bronze_sweeps(nuscenes_root)
    bronze_path = write_parquet(bronze, out_dir / "bronze" / "sweeps.parquet")
    print(f"[bronze] {len(bronze):>6} sweep rows -> {bronze_path}")

    fused = build_fused_world_model(bronze, threshold_ms=threshold_ms)
    fused_path = write_parquet(fused.fused_long, out_dir / "silver" / "fused_world_model.parquet")
    quarantine_path = write_parquet(fused.quarantined, out_dir / "silver" / "quarantine.parquet")

    print(
        f"[silver] {len(fused.fused_long):>6} fused rows, "
        f"{len(fused.quarantined):>6} quarantined "
        f"(pass rate {fused.pass_rate:.1%}, threshold {threshold_ms}ms)"
    )
    print(f"[silver] -> {fused_path}")
    print(f"[silver] -> {quarantine_path}")
    print(f"done in {time.perf_counter() - t0:.2f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AuraDrive Bronze->Silver pipeline.")
    parser.add_argument("--nuscenes-root", required=True, help="Path to v1.0-mini root directory")
    parser.add_argument("--out", default="data", help="Output directory for bronze/silver parquet")
    parser.add_argument("--threshold-ms", type=float, default=10.0, help="Temporal residual quality gate threshold")
    args = parser.parse_args()

    run_pipeline(args.nuscenes_root, args.out, args.threshold_ms)


if __name__ == "__main__":
    main()
