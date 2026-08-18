# AuraDrive Perception

A code first implementation of the data engineering and machine learning problem described in the AuraDrive Technologies Big Data consultancy report (COMP40711). The report proposes an Azure architecture for fusing multi modal autonomous vehicle sensor data (camera, LiDAR, radar) at fleet scale. This repository implements the same core logic in runnable Python, on the real nuScenes mini dataset, and extends it with an object detection model and serving layer that the report only described conceptually.

## Relationship to the report

The report is a full Azure product level design (Event Hubs, Azure Batch, ADLS Gen2, Databricks, Cosmos DB, Synapse, AKS) sized for a 200 vehicle fleet generating roughly 500 TB per day. Reproducing that literally would mean provisioning expensive cloud services around a 4 GB sample dataset, which would demonstrate portal configuration rather than engineering.

This repository instead implements the parts of the report that are genuinely about engineering logic, at a scale that runs on a laptop, and is honest about what is and is not proven here.

| Report component | This repository | Status |
|---|---|---|
| Landing and ingest of multiplexed sensor files | `auradrive.ingest.nuscenes_loader` | Implemented, runs on real data |
| Extraction and demultiplexing (Azure Batch and Docker) | Not needed; nuScenes mini ships pre extracted | Out of scope |
| Partitioned storage with schema enforcement (ADLS Gen2, Delta Lake) | Parquet Bronze and Silver layers via `auradrive.utils.io` | Implemented at reduced scale |
| Sensor fusion and temporal quality gate (Databricks and Spark) | `auradrive.fusion.world_model`, `auradrive.quality.temporal_gate` | Implemented, same 10 ms residual logic, pandas instead of Spark |
| Perception model training and MLOps loop | `auradrive.training.*`, MLflow tracking | New. The report only describes this in one paragraph; this repository actually builds it |
| Over the air model deployment | `auradrive.serving.app` (FastAPI) | New, same role as the report's deployment step, served as an HTTP API instead of a fleet push |
| Cosmos DB metadata hub, Synapse, AKS, Event Hubs | Not built | Out of scope, low value at this data scale |
| Pipeline health and drift monitoring (Application Insights, Grafana) | `auradrive.quality.monitoring` | Implemented as a standalone module producing metrics any dashboard could consume |

The Big Data challenge the report selects as its focus is Variety: the structural incompatibility between binary sensor blobs and relational JSON metadata, and Veracity: calibration drift and temporal misalignment across sensors. Both are addressed here directly, on the same nuScenes dataset the report references. Volume at 200 vehicle, 500 TB per day scale and Velocity as real time streaming ingestion are not tested by this repository; those remain the report's proposed design, not something a laptop scale project can honestly claim to validate.

## Pipeline overview

```
nuScenes mini raw JSON tables
        |
        v
+----------------------+
|  Bronze layer         |   auradrive.ingest.nuscenes_loader
|  one row per sweep     |   joins sample_data, ego_pose,
|  file, full context    |   calibrated_sensor, sensor, sample, scene
+----------------------+
        |
        v
+----------------------+
|  Temporal quality gate|   auradrive.quality.temporal_gate
|  nearest sweep per     |   residual = |target_ts - nearest_ts|
|  channel, per key frame|   pass if residual <= 10 ms, else quarantine
+----------------------+
        |
   pass |    quarantine
        v         v
+----------------------+   +----------------------+
|  Silver layer          |   |  Quarantine table     |
|  fused world model     |   |  audit trail with       |
|  one row per sample,    |   |  diagnostic tag         |
|  channel                |   +----------------------+
+----------------------+
        |
        v
+----------------------+
|  Gold layer             |   auradrive.training.labels
|  2D detection boxes     |   projects 3D annotations into
|  per CAM_FRONT frame    |   pixel space using calibration
+----------------------+
        |
        v
+----------------------+
|  Training               |   auradrive.training.train
|  fine tune Faster R-CNN |   MLflow tracked params, metrics,
|  MobileNetV3 backbone   |   checkpoint artifact
+----------------------+
        |
        v
+----------------------+
|  Serving                |   auradrive.serving.app
|  FastAPI /predict       |   loads checkpoint, returns
|  and /health             |   detections and latency
+----------------------+
        |
        v
+----------------------+
|  Monitoring              |   auradrive.quality.monitoring
|  per channel quarantine  |   rate and PSI based drift
|  rate, drift severity    |   report on live traffic
+----------------------+
```

## Repository layout

```
src/auradrive/
  ingest/       raw nuScenes JSON tables -> Bronze sweep table
  quality/      temporal residual quality gate, drift and health monitoring
  fusion/       Bronze + quality gate -> Silver fused world model
  training/     3D to 2D box projection, dataset, detector fine tuning
  serving/      FastAPI inference app, model loading
  utils/        Parquet read and write helpers
  pipeline.py   CLI entrypoint, Bronze -> Silver end to end
tests/          pytest suite, unit tests plus tests gated on real data presence
data/           bronze, silver, gold parquet output (not committed)
```

## Data flow at a glance

| Stage | Input | Output | Row count on nuScenes mini |
|---|---|---|---|
| Bronze ingest | 13 raw JSON tables | one row per sensor sweep | 31,206 |
| Temporal quality gate | Bronze sweeps | residual per sample, channel | 4,848 |
| Silver fusion | passed residuals | fused world model rows | 1,662 (pass rate 34.3% at 10 ms threshold) |
| Gold labels | Silver CAM_FRONT rows + 3D annotations | 2D detection boxes | 1,412 boxes across 98 key frames |

The 34.3% pass rate is expected, not a bug: cameras run near 12 Hz and LiDAR near 20 Hz, so most non key frame sweeps genuinely fall outside a strict 10 ms window of the key frame timestamp. This mirrors the report's own claim that a naive fusion pipeline needs an explicit quality gate rather than assuming every sweep is usable.

## Dataset

This project runs on the nuScenes mini split, about 4 GB, which is not included in this repository. Download it from the official nuScenes site:

https://www.nuscenes.org/nuscenes#download

Select the "Mini" split under Full dataset (v1.0), then extract it so the folder sits at the repository root as `v1.0-mini/`, matching the layout used by every command below:

```
AuraDrive-Perception/
  v1.0-mini/
    v1.0-mini/       (JSON metadata tables)
    samples/
    sweeps/
    maps/
  src/
  tests/
  ...
```

The dataset is distributed under its own nuScenes license terms; see the LICENSE file inside the downloaded archive.

## Running it

Install dependencies (CPU only, no GPU or Azure account required):

```
pip install -e ".[ml,serve,dev]"
```

Run the Bronze to Silver pipeline against the local nuScenes mini copy:

```
python -m auradrive.pipeline --nuscenes-root v1.0-mini --out data
```

Build Gold layer detection labels and train the detector:

```
python -m auradrive.training.train --nuscenes-root v1.0-mini \
    --labels data/gold/cam_front_labels.parquet --epochs 5
```

Serve the trained model:

```
uvicorn auradrive.serving.app:app --reload
curl -F "file=@some_image.jpg" http://localhost:8000/predict
```

Run tests:

```
pytest tests/ -v
```

## Design choices worth calling out

Pandas and pyarrow are used for the Bronze and Silver layers instead of Spark and Delta Lake. At nuScenes mini's scale (31,206 sweeps, well under a gigabyte) a Spark cluster would add dependency weight without changing the result. The DataFrame shapes match what a Spark job would produce, so swapping the storage backend later is a localized change, not a rewrite.

The detection model uses a MobileNetV3 backbone (about 20 MB) rather than a larger ResNet based detector, to keep training and inference fast on CPU. The same training script runs unchanged on an Azure ML GPU compute instance; only the `--mlflow-uri` flag needs to point at a remote tracking server instead of the local file store.

The temporal quality gate's 10 ms threshold and quarantine behaviour are taken directly from the report's section on integrated sensor fusion and quality control, not chosen independently.

## Known caveats

The RADAR_FRONT channel shows a 0 ms residual for every sample in the current test run, which stands out against every other channel. This is most likely because that channel's own key frame timestamp is used as its nearest match by construction, rather than a genuine calibration issue. It is left as is here and flagged rather than silently adjusted, since resolving it needs a closer look at how key frame membership interacts with the nearest neighbour search per channel.

This project validates the report's Variety and Veracity solution end to end on a small, real dataset. It does not attempt to prove Volume at fleet scale or Velocity as real time streaming ingestion, both of which remain the report's proposed design rather than something demonstrated in code here.
