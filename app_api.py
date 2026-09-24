import logging
import os
import uuid

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, field_validator

from auth import optional_auth
from config import settings
from graph import run_query, vectorstore

app = FastAPI(title="Aviation Regulatory Agentic RAG POC")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=4000)
    authority: str = "ALL"
    skip_verification: bool = False

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


@app.get("/health/live")
async def health_live():
    return {"status": "ok"}


@app.get("/health/ready")
async def health_ready():
    return {
        "status": "ok",
        "vectorstore_loaded": vectorstore is not None,
        "auth_required": settings.auth_required,
        "corpus_version": settings.corpus_version,
    }


@app.post("/api/query")
async def query_rag(req: QueryRequest, auth=Depends(optional_auth)):
    if settings.auth_required and auth is None:
        raise HTTPException(status_code=401, detail="Authentication required")

    if vectorstore is None:
        raise HTTPException(
            status_code=400,
            detail="Vector database not found. Please run ingest.py first.",
        )

    try:
        result = run_query(req.question, req.authority, req.skip_verification)
        return {
            "answer": result.get("final_answer", ""),
            "answer_status": result.get("answer_status", "UNKNOWN"),
            "sources": result.get("context_text", ""),
            "needs_review": bool(result.get("needs_review", False)),
            "verification": result.get("verification", {}),
            "retry_count": result.get("retry_count", 0),
            "sub_queries": result.get("sub_queries", []),
            "relevant_authorities": result.get("all_relevant_authorities", []),
            "corpus_version": result.get("corpus_version", settings.corpus_version),
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
