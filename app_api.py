import asyncio
import logging
import os

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from auth import optional_auth
from config import settings
from graph import run_compliance_rag, vector_store_is_ready

logger = logging.getLogger(__name__)
app = FastAPI(title="Aviation Regulatory Agentic RAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

query_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_QUERIES)


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., min_length=3, max_length=4000)
    authority: str = "ALL"
    skip_verification: bool = False

    @field_validator("authority")
    @classmethod
    def validate_authority(cls, value: str) -> str:
        normalized = value.upper()
        if normalized not in {"ALL", "EASA", "CAAS", "CAAC"}:
            raise ValueError("Unsupported authority")
        return normalized


def _read_template_file(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as source:
        return source.read()


@app.get("/", response_class=HTMLResponse)
async def read_index():
    if os.path.exists("templates/index.html"):
        return await asyncio.to_thread(_read_template_file, "templates/index.html")
    raise HTTPException(status_code=404, detail="Web interface not found.")


@app.get("/health/live")
async def health_live():
    return {"status": "ok"}


@app.get("/health/ready")
async def health_ready():
    if not vector_store_is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector collection is missing or empty.",
        )
    return {
        "status": "ready",
        "vectorstore_loaded": True,
        "auth_required": settings.auth_required,
        "corpus_version": settings.corpus_version,
    }


@app.post("/api/query")
async def query_rag(
    req: QueryRequest,
    request: Request,
    auth=Depends(optional_auth),  # noqa: B008
):
    if not os.path.exists(settings.VECTOR_DB_PATH) or not vector_store_is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector database is missing or empty. Build the index first.",
        )

    try:
        await asyncio.wait_for(
            query_semaphore.acquire(),
            timeout=settings.QUEUE_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Query capacity is full. Retry shortly.",
            headers={"Retry-After": "5"},
        )

    task = None
    try:
        if await request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client disconnected.")

        task = asyncio.create_task(
            asyncio.to_thread(
                run_compliance_rag,
                req.question,
                req.authority,
                req.skip_verification,
            )
        )
        # Keep the capacity slot reserved until the worker really finishes,
        # even if the HTTP request times out or is cancelled.
        task.add_done_callback(lambda completed: query_semaphore.release())
        result = await asyncio.wait_for(
            asyncio.shield(task),
            timeout=settings.QUERY_TIMEOUT_SECONDS,
        )
        return {
            "answer": result.get("final_answer", ""),
            "answer_status": result.get("answer_status", "UNKNOWN"),
            "sources": result.get("context_text", ""),
            "needs_review": bool(result.get("needs_review", True)),
            "verification": result.get("verification", {"passed": False}),
            "retry_count": result.get("retry_count", 0),
            "sub_queries": result.get("sub_queries", []),
            "relevant_authorities": result.get("all_relevant_authorities", []),
            "corpus_version": result.get("corpus_version", settings.corpus_version),
            "authority_findings": result.get("authority_findings", {}),
        }
    except asyncio.TimeoutError:
        logger.error("Query exceeded %.1f seconds.", settings.QUERY_TIMEOUT_SECONDS)
        raise HTTPException(status_code=504, detail="Query timed out.")
    except asyncio.CancelledError:
        logger.warning("Request cancelled while query worker may still be running.")
        raise
    except HTTPException:
        raise
    except Exception:
        logger.exception("Query execution failed.")
        raise HTTPException(status_code=500, detail="Request failed. See server logs.")
    finally:
        if task is None:
            query_semaphore.release()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app_api:app", host="0.0.0.0", port=8000, reload=False)
