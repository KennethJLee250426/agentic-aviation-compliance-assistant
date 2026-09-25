import os
import asyncio
import logging

from fastapi import Depends, FastAPI, HTTPException, status, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, field_validator

from auth import optional_auth
from config import settings
from graph import run_query, vectorstore

logger = logging.getLogger(__name__)

app = FastAPI(title="Aviation Regulatory Agentic RAG POC")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

# PRODUCTION NOTE: Concurrency Controls
# Fallback to defaults if these aren't defined in your config.py
MAX_CONCURRENT_QUERIES = getattr(settings, 'max_concurrent_queries', 5)
QUERY_TIMEOUT_SECONDS = getattr(settings, 'query_timeout_seconds', 120.0)

# Global semaphore limits the number of active graph executions
query_semaphore = asyncio.Semaphore(MAX_CONCURRENT_QUERIES)

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
    if vectorstore is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store is not loaded",
        )

    return {
        "status": "ready",
        "vectorstore_loaded": True,
        "auth_required": getattr(settings, 'auth_required', False),
        "corpus_version": getattr(settings, 'corpus_version', "UNKNOWN"),
    }


@app.post("/api/query")
async def query_rag(req: QueryRequest, request: Request, auth=Depends(optional_auth)):
    if getattr(settings, 'auth_required', False) and auth is None:
        raise HTTPException(status_code=401, detail="Authentication required")

    if vectorstore is None:
        raise HTTPException(
            status_code=400,
            detail="Vector database not found. Please run ingest.py first.",
        )

    # 1. Enforce Global Concurrency Limits via Semaphore
    if query_semaphore.locked():
        logger.warning("Server at maximum concurrent capacity. Request queued.")
        
    async with query_semaphore:
        try:
            # 2. Check for client disconnection before starting heavy work
            if await request.is_disconnected():
                logger.info("Client disconnected before query execution.")
                return Response(status_code=499) # Client Closed Request

            # 3. Prevent Event Loop Blocking & Enforce Timeouts
            # run_query is synchronous/blocking. to_thread pushes it to a worker thread.
            result = await asyncio.wait_for(
                asyncio.to_thread(run_query, req.question, req.authority, req.skip_verification),
                timeout=QUERY_TIMEOUT_SECONDS
            )

            return {
                "answer": result.get("final_answer", ""),
                "answer_status": result.get("answer_status", "UNKNOWN"),
                "sources": result.get("context_text", ""),
                "needs_review": bool(result.get("needs_review", False)),
                "verification": result.get("verification", {}),
                "retry_count": result.get("retry_count", 0),
                "sub_queries": result.get("sub_queries", []),
                "relevant_authorities": result.get("all_relevant_authorities", []),
                "corpus_version": result.get("corpus_version", getattr(settings, 'corpus_version', "UNKNOWN")),
                "authority_findings": {
                    authority: finding.get("draft", "")
                    for authority, finding in result.get("authority_findings", {}).items()
                },
            }

        except asyncio.TimeoutError:
            logger.error(f"Query timed out after {QUERY_TIMEOUT_SECONDS} seconds.")
            raise HTTPException(status_code=504, detail="Request timed out while generating response.")
        
        except asyncio.CancelledError:
            logger.warning("Request cancelled by client mid-execution.")
            raise  # FastAPI handles CancelledError internally
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Request failed. See server logs.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_api:app", host="0.0.0.0", port=8000, reload=False)
