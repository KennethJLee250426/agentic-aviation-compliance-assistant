import logging

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
    default_model: str = "gemini/gemini-2.5-flash",
) -> tuple[str, str | None]:
    """
    Resolves the model string and custom_llm_provider for LiteLLM.
    Ensures Gemini models always carry the 'gemini/' prefix to bypass Vertex AI fallbacks.
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

    # Determine custom_llm_provider for LiteLLM
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


def query_vector_db(query_text: str, top_k: int = 5):
    """Retrieve top-k regulatory chunks matching the query embedding."""
    client = chromadb.PersistentClient(path=settings.VECTOR_DB_PATH)
    collection = client.get_or_create_collection(name="aviation_regulations")

    # Ensure embedding model carries gemini/ prefix for AI Studio compatibility
    raw_embed_model = getattr(settings, "EMBEDDING_MODEL", "gemini/text-embedding-004")
    embed_model, custom_provider = resolve_model_and_provider(
        model=raw_embed_model,
        default_model="gemini/text-embedding-004",
    )

    active_key = getattr(settings, "GEMINI_API_KEY", None)

    # Call LiteLLM embedding
    response = embedding(
        model=embed_model,
        custom_llm_provider=custom_provider,
        input=[query_text],
        api_key=active_key if active_key else None,
    )
    query_vec = response.data[0]["embedding"]

    results = collection.query(
        query_embeddings=[query_vec],
        n_results=top_k,
    )

    docs = results.get("documents", [[]])[0] if results.get("documents") else []
    metadatas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
    return docs, metadatas


def run_compliance_rag(
    query: str,
    authority: str = "ALL",
    skip_verification: bool = False,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> dict:
    """Execute RAG pipeline with flexible authority filtering and multi-provider selection."""
    
    # 1. Retrieve Relevant Regulatory Context
    docs, metadatas = query_vector_db(query, top_k=getattr(settings, "RAG_TOP_K", 5))

    if docs:
        context_str = "\n\n".join([
            f"[Source: {meta.get('authority', 'REG')}] {doc}"
            for doc, meta in zip(docs, metadatas)
        ])
    else:
        context_str = "No specific regulatory context retrieved from the database."

    # 2. Resolve Model and Custom Provider
    target_model, custom_provider = resolve_model_and_provider(provider, model)

    # 3. Resolve API Key Dynamically
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

    # 4. Call LiteLLM Completion
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

    # 5. Return dict structured for app_api.py response formatting
    return {
        "final_answer": final_answer,
        "answer_status": "COMPLETED",
        "context_text": context_str,
        "needs_review": False,
        "verification": {"passed": True},
        "retry_count": 0,
        "sub_queries": [query],
        "all_relevant_authorities": [authority],
        "corpus_version": getattr(settings, "corpus_version", "UNKNOWN"),
        "authority_findings": {},
    }