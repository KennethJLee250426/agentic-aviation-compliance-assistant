from types import SimpleNamespace

import pytest

import graph


class DummyCollection:
    def __init__(self):
        self.query_kwargs = None

    def count(self):
        return 1

    def peek(self, limit=1):
        return {"embeddings": [[0.1, 0.2, 0.3]]}

    def query(self, **kwargs):
        self.query_kwargs = kwargs
        return {
            "documents": [["doc-1", "doc-2"]],
            "metadatas": [[{"authority": "EASA"}, {"authority": "CAAS"}]],
            "distances": [[0.1, 0.8]],
        }


class DummyClient:
    def __init__(self, collection):
        self.collection = collection

    def get_or_create_collection(self, name):
        return self.collection


def test_vector_db_query_applies_authority_filter_and_threshold(monkeypatch):
    collection = DummyCollection()
    monkeypatch.setattr(graph.chromadb, "PersistentClient", lambda path: DummyClient(collection))
    monkeypatch.setattr(
        graph,
        "embedding",
        lambda **kwargs: SimpleNamespace(data=[{"embedding": [0.2, 0.3, 0.4]}]),
    )

    docs, metadatas, stats = graph.query_vector_db(
        "fuel requirements",
        top_k=2,
        authority="EASA",
        similarity_threshold=0.75,
    )

    assert docs == ["doc-1"]
    assert metadatas == [{"authority": "EASA"}]
    assert collection.query_kwargs["where"] == {"authority": "EASA"}
    assert stats["returned_before_threshold"] == 2
    assert stats["returned_after_threshold"] == 1


def test_rag_pipeline_execution_with_mocks(monkeypatch):
    monkeypatch.setattr(
        graph,
        "query_vector_db",
        lambda *args, **kwargs: (
            ["sample context"],
            [{"authority": "EASA"}],
            {"returned_after_threshold": 1},
        ),
    )
    monkeypatch.setattr(
        graph,
        "completion",
        lambda **kwargs: SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Mocked answer"))]
        ),
    )

    result = graph.run_compliance_rag(query="What are basic aviation safety rules?", authority="EASA")

    assert result["final_answer"] == "Mocked answer"
    assert result["answer_status"] == "COMPLETED"
    assert result["all_relevant_authorities"] == ["EASA"]
    assert result["retrieval_stats"]["returned_after_threshold"] == 1
