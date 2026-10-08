import asyncio
import logging
import os
import time
import uuid

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, field_validator

from auth import optional_auth
from config import settings, validate_auth_settings
from graph import run_compliance_rag

logger = logging.getLogger(__name__)

app = FastAPI(title="Aviation Regulatory Agentic RAG POC")


@app.on_event("startup")
async def startup_validation() -> None:
    validate_auth_settings()


allowed_origins = getattr(
    settings,
    "allowed_origins",
    getattr(settings, "ALLOWED_ORIGINS", ["*"])
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
)

MAX_CONCURRENT_QUERIES = getattr(settings, "MAX_CONCURRENT_QUERIES", getattr(settings, "max_concurrent_queries", 5))
QUERY_TIMEOUT_SECONDS = getattr(settings, "QUERY_TIMEOUT_SECONDS", getattr(settings, "query_timeout_seconds", 120.0))

query_semaphore = asyncio.Semaphore(MAX_CONCURRENT_QUERIES)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=4000)
    authority: str = "ALL"
    skip_verification: bool = False
    provider: str | None = Field(default=None, description="Preferred LLM provider (e.g. gemini, openai, anthropic, ollama)")
    model: str | None = Field(default=None, description="Specific model string (e.g. gpt-4o, claude-3-5-sonnet-20241022, gemini-2.5-flash)")

    @field_validator("authority")
    @classmethod
    def validate_authority(cls, value: str) -> str:
        normalized = value.upper()
        allowed = {"ALL", "EASA", "CAAS", "CAAC"}
        if normalized not in allowed:
            raise ValueError("Unsupported authority")
        return normalized


def _read_template_file(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def _request_id(request: Request) -> str:
    return request.headers.get("X-Request-ID") or str(uuid.uuid4())


@app.get("/", response_class=HTMLResponse)
async def read_index():
    if os.path.exists("templates/index.html"):
        content = await asyncio.to_thread(_read_template_file, "templates/index.html")
        return content
    return "<h3>Error: templates/index.html not found!</h3>"


@app.get("/health/live")
async def health_live():
    return {"status": "ok"}


@app.get("/health/ready")
async def health_ready():
    db_exists = os.path.exists(settings.VECTOR_DB_PATH)
    if not db_exists:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store is not loaded. Please run ingest.py first.",
        )

    return {
        "status": "ready",
        "vectorstore_loaded": True,
        "auth_required": getattr(settings, "auth_required", False),
        "corpus_version": getattr(settings, "corpus_version", "UNKNOWN"),
    }


@app.post("/api/query")
async def query_rag(req: QueryRequest, request: Request, auth=Depends(optional_auth)):  # noqa: B008
    req_id = _request_id(request)
    started = time.perf_counter()

    if getattr(settings, "auth_required", False) and auth is None:
        raise HTTPException(status_code=401, detail={"message": "Authentication required", "request_id": req_id})

    if not os.path.exists(settings.VECTOR_DB_PATH):
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Vector database not found. Please run ingest.py first.",
                "request_id": req_id,
            },
        )

    if query_semaphore.locked():
        logger.warning("Server at maximum concurrent capacity. Request queued. request_id=%s", req_id)

    async with query_semaphore:
        try:
            if await request.is_disconnected():
                logger.info("Client disconnected before query execution. request_id=%s", req_id)
                return Response(status_code=499)

            result = await asyncio.wait_for(
                asyncio.to_thread(
                    run_compliance_rag,
                    req.question,
                    req.authority,
                    req.skip_verification,
                    req.provider,
                    req.model,
                ),
                timeout=QUERY_TIMEOUT_SECONDS,
            )

            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            response_payload = {
                "request_id": req_id,
                "processing_ms": elapsed_ms,
                "answer": result.get("final_answer", ""),
                "answer_status": result.get("answer_status", "UNKNOWN"),
                "sources": result.get("context_text", ""),
                "needs_review": bool(result.get("needs_review", False)),
                "verification": result.get("verification", {}),
                "retry_count": result.get("retry_count", 0),
                "sub_queries": result.get("sub_queries", []),
                "relevant_authorities": result.get("all_relevant_authorities", []),
                "corpus_version": result.get("corpus_version", getattr(settings, "corpus_version", "UNKNOWN")),
                "retrieval": result.get("retrieval_stats", {}),
                "authority_findings": {
                    authority: finding.get("draft", "") if isinstance(finding, dict) else str(finding)
                    for authority, finding in result.get("authority_findings", {}).items()
                },
            }
            logger.info(
                "Query completed request_id=%s authority=%s ms=%s",
                req_id,
                req.authority,
                elapsed_ms,
            )
            return response_payload

        except asyncio.TimeoutError as exc:
            logger.error("Query timed out after %s seconds. request_id=%s", QUERY_TIMEOUT_SECONDS, req_id)
            raise HTTPException(
                status_code=504,
                detail={
                    "message": "Request timed out while generating response.",
                    "request_id": req_id,
                },
            ) from exc

        except asyncio.CancelledError:
            logger.warning("Request cancelled by client mid-execution. request_id=%s", req_id)
            raise

        except ValueError as exc:
            logger.exception("Validation error during query execution. request_id=%s", req_id)
            raise HTTPException(
                status_code=422,
                detail={
                    "message": str(exc),
                    "request_id": req_id,
                },
            ) from exc

        except Exception as exc:  # noqa: BLE001
            logger.exception("Query execution failed. request_id=%s", req_id)
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Request failed. See server logs.",
                    "request_id": req_id,
                },
            ) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app_api:app", host="0.0.0.0", port=8000, reload=False)
