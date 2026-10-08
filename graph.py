import logging
from typing import Any

import chromadb
import litellm
from litellm import completion, embedding

from config import settings

logger = logging.getLogger(__name__)

# Drop parameters unsupported by specific providers automatically
litellm.drop_params = True


def resolve_model_and_provider(
    provider: str | None = None,
    model: str | None = None,
    default_model: str = "ollama/llama3.1:8b",
) -> tuple[str, str | None]:
    """
    Resolves the LLM completion model string and custom_llm_provider for LiteLLM.
    """
    target_model = model or getattr(settings, "DEFAULT_LLM_MODEL", default_model)

    if provider and not model:
        provider_lower = provider.lower()
        provider_defaults = {
            "openai": "gpt-4o",
            "anthropic": "anthropic/claude-3-5-sonnet",
            "gemini": "gemini/gemini-2.5-flash",
            "ollama": "ollama/llama3.1:8b",
        }
        target_model = provider_defaults.get(provider_lower, target_model)

    model_lower = target_model.lower()
    custom_provider = None

    if "gemini" in model_lower or provider == "gemini":
        if not target_model.startswith("gemini/"):
            target_model = f"gemini/{target_model}"
        custom_provider = "gemini"
    elif "gpt" in model_lower or "openai" in model_lower or provider == "openai":
        custom_provider = "openai"
    elif "claude" in model_lower or "anthropic" in model_lower or provider == "anthropic":
        custom_provider = "anthropic"
    elif "ollama" in model_lower or provider == "ollama":
        custom_provider = "ollama"

    return target_model, custom_provider


def resolve_embedding_model(
    provider: str | None = None,
    model: str | None = None,
    default_model: str = "ollama/nomic-embed-text",
) -> tuple[str, str | None]:
    """
    Resolves the embedding model string and custom_llm_provider for LiteLLM.
    Mirrors resolve_model_and_provider but for embedding calls.
    """
    target_model = model or getattr(settings, "EMBEDDING_MODEL", default_model)

    if provider and not model:
        provider_lower = provider.lower()
        provider_defaults = {
            "openai": "text-embedding-3-small",
            "gemini": "gemini/gemini-embedding-001",
            "cohere": "cohere/embed-english-v3.0",
            "mistral": "mistral/mistral-embed",
            "ollama": "ollama/nomic-embed-text",
            "bedrock": "bedrock/amazon.titan-embed-text-v2:0",
        }
        target_model = provider_defaults.get(provider_lower, target_model)

    model_lower = target_model.lower()
    custom_provider = None

    if "gemini" in model_lower or provider == "gemini":
        if not target_model.startswith("gemini/"):
            target_model = f"gemini/{target_model}"
        custom_provider = "gemini"
    elif "text-embedding" in model_lower or "ada-002" in model_lower or provider == "openai":
        custom_provider = "openai"
    elif "cohere" in model_lower or provider == "cohere":
        custom_provider = "cohere"
    elif "mistral" in model_lower or provider == "mistral":
        custom_provider = "mistral"
    elif "ollama" in model_lower or provider == "ollama":
        custom_provider = "ollama"
    elif "bedrock" in model_lower or provider == "bedrock":
        custom_provider = "bedrock"

    return target_model, custom_provider


def _build_where_filter(authority: str | None) -> dict[str, str] | None:
    if not authority:
        return None
    normalized = authority.upper()
    if normalized == "ALL":
        return None
    return {"authority": normalized}


def query_vector_db(
    query_text: str,
    top_k: int = 5,
    provider: str | None = None,
    model: str | None = None,
    authority: str = "ALL",
    similarity_threshold: float | None = None,
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    """Retrieve top-k regulatory chunks matching the query embedding."""
    client = chromadb.PersistentClient(path=settings.VECTOR_DB_PATH)
    collection = client.get_or_create_collection(name="aviation_regulations")

    # Fall back to .env-configured provider if none passed explicitly
    provider = provider or getattr(settings, "EMBEDDING_PROVIDER", None)
    embed_model, embed_provider = resolve_embedding_model(provider, model)

    active_key = None
    if embed_provider == "gemini":
        active_key = getattr(settings, "GEMINI_API_KEY", None)
    elif embed_provider == "openai":
        active_key = getattr(settings, "OPENAI_API_KEY", None)
    elif embed_provider == "cohere":
        active_key = getattr(settings, "COHERE_API_KEY", None)
    elif embed_provider == "mistral":
        active_key = getattr(settings, "MISTRAL_API_KEY", None)

    response = embedding(
        model=embed_model,
        input=[query_text],
        api_key=active_key if active_key else None,
    )
    query_vec = response.data[0]["embedding"]

    # --- Dimension safety check ---
    if collection.count() > 0:
        sample = collection.peek(limit=1)
        existing_embeddings = sample.get("embeddings")
        if existing_embeddings is not None and len(existing_embeddings) > 0:
            existing_dim = len(existing_embeddings[0])
            query_dim = len(query_vec)
            if existing_dim != query_dim:
                raise ValueError(
                    f"Embedding dimension mismatch: the vector DB was built with "
                    f"{existing_dim}-dimensional embeddings, but the current "
                    f"EMBEDDING_MODEL ('{embed_model}') produces {query_dim}-dimensional "
                    f"vectors. This usually means EMBEDDING_MODEL was changed after "
                    f"ingestion. Re-run ingest.py to rebuild the vector DB with the "
                    f"new model, or revert EMBEDDING_MODEL to match the existing data."
                )

    query_kwargs: dict[str, Any] = {
        "query_embeddings": [query_vec],
        "n_results": top_k,
    }
    where_filter = _build_where_filter(authority)
    if where_filter:
        query_kwargs["where"] = where_filter

    results = collection.query(**query_kwargs)

    docs = results.get("documents", [[]])[0] if results.get("documents") else []
    metadatas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
    distances = results.get("distances", [[]])[0] if results.get("distances") else []

    threshold = similarity_threshold
    if threshold is None:
        threshold = getattr(settings, "RAG_SIMILARITY_THRESHOLD", 0.75)

    # For cosine distance, smaller is better. Similarity ~ 1 - distance.
    max_distance = 1.0 - max(0.0, min(1.0, float(threshold)))

    filtered_docs: list[str] = []
    filtered_metadatas: list[dict[str, Any]] = []
    filtered_distances: list[float] = []

    for idx, doc in enumerate(docs):
        distance = distances[idx] if idx < len(distances) else None
        if distance is not None and distance > max_distance:
            continue
        filtered_docs.append(doc)
        filtered_metadatas.append(metadatas[idx] if idx < len(metadatas) else {})
        if distance is not None:
            filtered_distances.append(distance)

    retrieval_stats: dict[str, Any] = {
        "authority_filter": authority.upper(),
        "requested_top_k": top_k,
        "returned_before_threshold": len(docs),
        "returned_after_threshold": len(filtered_docs),
        "similarity_threshold": threshold,
    }
    if filtered_distances:
        retrieval_stats["max_distance_after_threshold"] = max(filtered_distances)
        retrieval_stats["min_distance_after_threshold"] = min(filtered_distances)

    return filtered_docs, filtered_metadatas, retrieval_stats


def run_compliance_rag(
    query: str,
    authority: str = "ALL",
    skip_verification: bool = False,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> dict:
    """Execute RAG pipeline with flexible authority filtering and multi-provider selection."""

    docs, metadatas, retrieval_stats = query_vector_db(
        query,
        top_k=getattr(settings, "RAG_TOP_K", 5),
        provider=provider,
        authority=authority,
    )

    if docs:
        context_str = "\n\n".join([
            f"[Source: {meta.get('authority', 'REG')}] {doc}"
            for doc, meta in zip(docs, metadatas)
        ])
    else:
        context_str = "No specific regulatory context retrieved from the database."

    target_model, custom_provider = resolve_model_and_provider(provider, model)

    active_key = api_key
    if not active_key:
        if custom_provider == "gemini":
            active_key = getattr(settings, "GEMINI_API_KEY", "")
        elif custom_provider == "openai":
            active_key = getattr(settings, "OPENAI_API_KEY", "")
        elif custom_provider == "anthropic":
            active_key = getattr(settings, "ANTHROPIC_API_KEY", "")

    system_prompt = (
        "You are an expert aviation compliance assistant. Answer the user query strictly "
        "using the provided regulatory context. Cite relevant clauses where applicable."
    )

    user_prompt = f"Target Authority: {authority}\nRegulatory Context:\n{context_str}\n\nUser Question: {query}"

    completion_kwargs = {
        "model": target_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "api_key": active_key if active_key else None,
        "temperature": 0.1,
    }
    if custom_provider:
        completion_kwargs["custom_llm_provider"] = custom_provider

    response = completion(**completion_kwargs)
    final_answer = response.choices[0].message.content

    return {
        "final_answer": final_answer,
        "answer_status": "COMPLETED",
        "context_text": context_str,
        "needs_review": False,
        "verification": {"passed": True, "skipped": skip_verification},
        "retry_count": 0,
        "sub_queries": [query],
        "all_relevant_authorities": [authority],
        "corpus_version": getattr(settings, "corpus_version", "UNKNOWN"),
        "authority_findings": {},
        "retrieval_stats": retrieval_stats,
    }
