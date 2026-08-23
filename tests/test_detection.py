from __future__ import annotations

import sahi

from traffic_agent.detection import DetectorConfig, RoadUserDetector


def test_sahi_model_receives_configured_image_size(monkeypatch) -> None:
    captured: dict[str, object] = {}
    sentinel = object()

    def fake_from_pretrained(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        sahi.AutoDetectionModel,
        "from_pretrained",
        staticmethod(fake_from_pretrained),
    )

    detector = RoadUserDetector(
        DetectorConfig(
            model_path="weights.pt",
            device="0",
            confidence=0.07,
            image_size=960,
        )
    )

    assert detector._load_sahi_model() is sentinel
    assert captured["model_type"] == "ultralytics"
    assert captured["model_path"] == "weights.pt"
    assert captured["device"] == "cuda:0"
    assert captured["confidence_threshold"] == 0.07
    assert captured["image_size"] == 960
