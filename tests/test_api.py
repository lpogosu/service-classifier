"""HTTP-контракт: сервер поднимается офлайн и отдаёт согласованные ответы."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from api.server import app

SPECIALIST_DESCRIPTION = (
    "Мастер с собственным инструментом.\n"
    "- укладка плитки и керамогранита\n"
    "- штукатурка стен по маякам\n"
    "Каждый вид работ считаю отдельно, цена от 900 руб. за м2."
)

TILING = 105
PLASTER = 108


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    """Сервер поднимается без ключа — проверяем именно offline-режим."""
    patcher = pytest.MonkeyPatch()
    patcher.delenv("OPENROUTER_API_KEY", raising=False)
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        patcher.undo()


def test_health_reports_offline_mode(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["llm"] == "offline"
    assert body["categoriesLoaded"] > 0
    assert body["splitMode"] in {"classifier", "llm", "heuristic"}


def test_detect_endpoint_needs_no_llm(client: TestClient) -> None:
    response = client.post(
        "/detect",
        json={"itemId": 1, "mcId": 101, "description": SPECIALIST_DESCRIPTION},
    )
    body = response.json()
    assert response.status_code == 200
    assert TILING in body["detectedMcIds"]
    assert body["details"][str(TILING)]["matchedPhrases"]


def test_process_returns_drafts_for_split_ads(client: TestClient) -> None:
    response = client.post(
        "/process",
        json={"itemId": 1, "mcId": 101, "description": SPECIALIST_DESCRIPTION},
    )
    body = response.json()
    assert response.status_code == 200
    assert set(body["detectedMcIds"]) >= {TILING, PLASTER}
    assert body["shouldSplit"] is True
    assert {draft["mcId"] for draft in body["drafts"]} == set(body["detectedMcIds"])
    assert all(draft["fallback"] for draft in body["drafts"])


def test_details_flag_adds_explanations(client: TestClient) -> None:
    response = client.post(
        "/process?details=true",
        json={"itemId": 1, "mcId": 101, "description": SPECIALIST_DESCRIPTION},
    )
    body = response.json()
    assert body["detectionDetails"]
    assert all(detail["matchedPhrases"] for detail in body["detectionDetails"])
    assert body["removedByVerification"] == []


def test_empty_description_is_rejected(client: TestClient) -> None:
    response = client.post("/process", json={"itemId": 1, "mcId": 101, "description": ""})
    assert response.status_code == 422


def test_batch_returns_one_result_per_item(client: TestClient) -> None:
    response = client.post(
        "/batch",
        json={
            "items": [
                {"itemId": 1, "mcId": 101, "description": SPECIALIST_DESCRIPTION},
                {"itemId": 2, "mcId": 101, "description": "Продаю велосипед."},
            ]
        },
    )
    body = response.json()
    assert len(body["results"]) == 2
    assert body["results"][1]["detectedMcIds"] == []
    assert body["processingTimeSec"] >= 0
