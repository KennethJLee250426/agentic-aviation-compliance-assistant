import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from graph import run_query, vectorstore

app = FastAPI(title="Aviation Regulatory Agentic RAG POC")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://compliance.example.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)


class QueryRequest(BaseModel):
    question: str
    authority: str = "ALL"
    skip_verification: bool = False  # opt-in fast path; see graph.py


@app.get("/", response_class=HTMLResponse)
async def read_index():
    if os.path.exists("templates/index.html"):
        with open("templates/index.html", "r", encoding="utf-8") as f:
            return f.read()
    return "<h3>Error: templates/index.html not found!</h3>"


@app.post("/api/query")
async def query_rag(req: QueryRequest):
    if vectorstore is None:
        raise HTTPException(
            status_code=400,
            detail="Vector database not found. Please run ingest.py first.",
        )

    try:
        result = run_query(req.question, req.authority, req.skip_verification)
        return {
            "answer": result["final_answer"],
            "sources": result["context_text"],
            "needs_review": result["needs_review"],
            "verification": result["verification"],
            "retry_count": result["retry_count"],
            "sub_queries": result["sub_queries"],
            "relevant_authorities": result["all_relevant_authorities"],
            "authority_findings": {
                authority: finding["draft"]
                for authority, finding in result["authority_findings"].items()
            },
        }
    import logging
import uuid

logger = logging.getLogger(__name__)

try:
    result = run_query(req.question, req.authority, req.skip_verification)
    return {...}
except Exception:
    error_id = str(uuid.uuid4())
    logger.exception("Query failed: %s", error_id)
    raise HTTPException(
        status_code=500,
        detail=f"Request failed. Reference ID: {error_id}"
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app_api:app", host="0.0.0.0", port=8000, reload=True)
