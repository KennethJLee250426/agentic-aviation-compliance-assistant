import logging
import os
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, field_validator

from graph import run_query, vectorstore

app = FastAPI(title="Aviation Regulatory Agentic RAG POC")

allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:8000").split(",")
allowed_origins = [origin.strip() for origin in allowed_origins if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=4000)
    authority: str = "ALL"
    skip_verification: bool = False  # opt-in fast path; see graph.py

    @field_validator("authority")
    @classmethod
    def validate_authority(cls, value: str) -> str:
        normalized = value.upper()
        allowed = {"ALL", "EASA", "CAAS", "CAAC"}
        if normalized not in allowed:
            raise ValueError("Unsupported authority")
        return normalized


@app.get("/", response_class=HTMLResponse)
async def read_index():
    if os.path.exists("templates/index.html"):
        with open("templates/index.html", "r", encoding="utf-8") as f:
            return f.read()
    return "<h3>Error: templates/index.html not found!</h3>"


@app.get("/health")
async def health_check():
    return {"status": "ok", "vectorstore_loaded": vectorstore is not None}


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
            "answer": result.get("final_answer", ""),
            "sources": result.get("context_text", ""),
            "needs_review": bool(result.get("needs_review", False)),
            "verification": result.get("verification", {}),
            "retry_count": result.get("retry_count", 0),
            "sub_queries": result.get("sub_queries", []),
            "relevant_authorities": result.get("all_relevant_authorities", []),
            "authority_findings": {
                authority: finding.get("draft", "")
                for authority, finding in result.get("authority_findings", {}).items()
            },
        }
    except Exception:
        error_id = str(uuid.uuid4())
        logging.exception("Query failed: %s", error_id)
        raise HTTPException(
            status_code=500,
            detail=f"Request failed. Reference ID: {error_id}",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app_api:app", host="0.0.0.0", port=8000, reload=False)
