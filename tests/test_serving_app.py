from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from auradrive.serving.app import app

CHECKPOINT = Path(__file__).resolve().parents[1] / "data" / "models" / "cam_front_fasterrcnn.pt"


def test_health_reports_unloaded_model_gracefully(monkeypatch):
    monkeypatch.setattr(
        "auradrive.serving.app.DEFAULT_CHECKPOINT", Path("does/not/exist.pt")
    )
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["model_loaded"] is False


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="no trained checkpoint present")
def test_predict_rejects_invalid_image():
    with TestClient(app) as client:
        r = client.post(
            "/predict", files={"file": ("bad.jpg", b"not an image", "image/jpeg")}
        )
        assert r.status_code == 400


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="no trained checkpoint present")
def test_predict_returns_well_formed_response():
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color="gray").save(buf, format="JPEG")
    buf.seek(0)

    with TestClient(app) as client:
        r = client.post("/predict", files={"file": ("blank.jpg", buf, "image/jpeg")})
        assert r.status_code == 200
        body = r.json()
        assert "detections" in body
        assert "inference_ms" in body
        assert isinstance(body["detections"], list)
