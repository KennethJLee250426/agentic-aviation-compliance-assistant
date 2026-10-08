from fastapi.testclient import TestClient

from app_api import app

client = TestClient(app)


def test_health_live_endpoint():
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_query_endpoint_success(monkeypatch):
    monkeypatch.setattr("app_api.optional_auth", lambda: {"user": "test"})
    monkeypatch.setattr("app_api.settings.AUTH_REQUIRED", False)
    monkeypatch.setattr("app_api.os.path.exists", lambda p: True)
    monkeypatch.setattr(
        "app_api.run_compliance_rag",
        lambda *args, **kwargs: {
            "final_answer": "ok",
            "answer_status": "COMPLETED",
            "context_text": "ctx",
            "needs_review": False,
            "verification": {"passed": True},
            "retry_count": 0,
            "sub_queries": ["q"],
            "all_relevant_authorities": ["EASA"],
            "corpus_version": "v1",
            "authority_findings": {},
            "retrieval_stats": {"returned_after_threshold": 1},
        },
    )

    response = client.post("/api/query", json={"question": "What are EASA tool calibration rules?"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "ok"
    assert "request_id" in payload
    assert "processing_ms" in payload


def test_query_endpoint_missing_vector_db(monkeypatch):
    monkeypatch.setattr("app_api.settings.AUTH_REQUIRED", False)
    monkeypatch.setattr("app_api.os.path.exists", lambda p: False)

    response = client.post("/api/query", json={"question": "What are EASA tool calibration rules?"})
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "request_id" in detail
