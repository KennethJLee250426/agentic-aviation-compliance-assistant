from graph import query_vector_db, run_compliance_rag


def test_vector_db_query():
    """Verify vector retrieval returns list structures."""
    docs, metadatas = query_vector_db("fuel requirements", top_k=2)
    assert isinstance(docs, list)
    assert isinstance(metadatas, list)


def test_rag_pipeline_execution():
    """Verify RAG pipeline returns expected keys using default model."""
    result = run_compliance_rag(query="What are basic aviation safety rules?")
    assert "final_answer" in result
    assert "answer_status" in result
    assert "context_text" in result
    assert "sub_queries" in result
    assert "all_relevant_authorities" in result