"""
Agentic RAG pipeline for aviation regulatory compliance.

Flow:
    plan -> authority_agents (fan out over EASA/CAAS/CAAC as needed)
         -> aggregate -> verify -> (retry authority_agents | finalize)

Agents:
- Planner: decides sub-queries and which authorities (EASA/CAAS/CAAC) are
  relevant to the question.                                    llama3.1:8b
- Authority specialist agents: one per relevant authority. Each retrieves
  ONLY from that authority's documents and drafts a scoped finding, or
  states plainly that its corpus has no relevant material.      llama3.1:8b
- Aggregator: merges the per-authority findings into one answer, explicitly
  calling out where authorities agree, conflict, or where one is silent.
                                                                  llama3.1:8b
- Verifier: checks the merged answer against all retrieved context, can
  send the graph back for another retrieval pass with a refined query.
                                                                deepseek-r1:8b

Sequential execution (not concurrent) is intentional: on an 8GB-VRAM laptop
GPU, only one 8B model can be resident at a time anyway (see model config
below), so there is no benefit to parallelizing the authority agents.
"""

import json
import os
import re
from typing import Dict, List, Optional, TypedDict

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

MAX_RETRIES = 2
DB_DIR = "./regulatory_chroma_db"
AUTHORITIES = ["EASA", "CAAS", "CAAC"]  # must match ingest.py's authority.upper()

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
# Hardware note: an 8GB-VRAM GPU (e.g. RTX 4060 laptop) cannot hold two
# resident 8B models at once (~4.7GB + ~4.9GB at Q4). Let Ollama fully
# unload one model and load the other on each transition rather than
# pinning either to CPU. To keep that swap clean (not a slow partial-VRAM
# fit), set on the Ollama server itself, before `ollama serve`:
#
#   export OLLAMA_MAX_LOADED_MODELS=1
#
worker_llm = ChatOllama(model="llama3.1:8b", temperature=0.0, keep_alive="10m")
# num_predict caps DeepSeek-R1's <think> + answer length. 600 was too tight
# in practice — R1's reasoning trace alone was often exceeding it, cutting
# the response off before it ever reached the JSON verdict, which made
# extract_json() fail and every answer get incorrectly flagged as
# "needs review". 2000 gives real headroom; tune down only if you confirm
# via testing that your questions consistently finish well under that.
verifier_llm = ChatOllama(
    model="deepseek-r1:8b", temperature=0.0, keep_alive="10m", num_predict=2000
)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore: Optional[Chroma] = (
    Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    if os.path.exists(DB_DIR)
    else None
)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class AuthorityFinding(TypedDict):
    context: str
    draft: str
    has_docs: bool


class RAGState(TypedDict):
    question: str
    authority: str  # user-requested filter: "ALL" or one of AUTHORITIES
    sub_queries: List[str]
    all_relevant_authorities: List[str]  # fixed set decided at planning time
    relevant_authorities: List[str]  # working set for this pass (narrows on retry)
    authority_findings: Dict[str, AuthorityFinding]
    context_text: str  # combined, across relevant authorities
    draft_answer: str  # aggregated answer
    verification: dict
    final_answer: str
    needs_review: bool
    retry_count: int
    skip_verification: bool  # opt-in fast path, see run_query()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def strip_think(text: str) -> str:
    """Remove DeepSeek-R1's <think>...</think> reasoning block from output
    shown to end users. Keep the raw text elsewhere (e.g. logs) if you want
    an audit trail of the verifier's reasoning."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_json(text: str) -> dict:
    """Best-effort JSON extraction: models sometimes wrap JSON in prose or
    markdown fences despite instructions. Falls back to a safe default.
    Tries the <think>-stripped text first; if that finds nothing (e.g. the
    <think> tag was left unclosed because generation was cut short), also
    tries the raw text in case a JSON block appears after it anyway."""
    for candidate in (strip_think(text), text):
        cleaned = re.sub(r"```json|```", "", candidate).strip()
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            continue
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
    return {}


def format_docs(docs: List[Document]) -> str:
    return "\n\n---\n\n".join(
        f"[{doc.metadata.get('authority', 'UNKNOWN')} - "
        f"{os.path.basename(doc.metadata.get('source', 'Unknown'))}]\n{doc.page_content}"
        for doc in docs
    )


def dedupe_docs(docs: List[Document]) -> List[Document]:
    seen = set()
    unique = []
    for d in docs:
        key = (d.metadata.get("source"), d.page_content[:200])
        if key not in seen:
            seen.add(key)
            unique.append(d)
    return unique


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------
# Cheap heuristic to avoid an LLM call for the common case: a straightforward
# single-lookup question. Only questions that look like they need comparing
# or cross-checking multiple things get the (slower) LLM decomposition call.
_MULTIHOP_MARKERS = (
    "compare", "comparison", "versus", " vs ", "difference between",
    "conflict", "contradict", "both", "satisfy", "does our", "cross-check",
    "cross check", "align with", "consistent with",
)


def _looks_multihop(question: str) -> bool:
    q = question.lower()
    return any(marker in q for marker in _MULTIHOP_MARKERS)


DECOMPOSE_PROMPT = ChatPromptTemplate.from_template(
    """You are a query planner for an aviation regulatory compliance system.
This question looks like it may need multiple retrieval passes (comparing
requirements, or checking whether one satisfies another). Break it into
2-4 focused sub-queries.

Respond ONLY with JSON, no other text:
{{"sub_queries": ["query 1", "query 2"]}}

Question: {question}
"""
)


def plan_node(state: RAGState) -> RAGState:
    # Authority relevance is a fixed rule, not an LLM guess: for compliance
    # content, silently having the planner exclude an authority it judged
    # "irrelevant" is a real risk if it judges wrong. Default to checking
    # all authorities unless the user explicitly filtered to one.
    if state["authority"] != "ALL" and state["authority"] in AUTHORITIES:
        relevant_authorities = [state["authority"]]
    else:
        relevant_authorities = list(AUTHORITIES)

    if _looks_multihop(state["question"]):
        resp = worker_llm.invoke(
            DECOMPOSE_PROMPT.invoke({"question": state["question"]})
        )
        parsed = extract_json(resp.content)
        sub_queries = parsed.get("sub_queries") or [state["question"]]
    else:
        sub_queries = [state["question"]]

    return {
        **state,
        "sub_queries": sub_queries,
        "all_relevant_authorities": relevant_authorities,
        "relevant_authorities": relevant_authorities,
        "retry_count": 0,
    }


# ---------------------------------------------------------------------------
# Authority specialist agents
# ---------------------------------------------------------------------------
AUTHORITY_AGENT_PROMPT = ChatPromptTemplate.from_template(
    """You are the {authority} regulatory compliance specialist agent for an
aviation MRO QEHS system. You only have access to {authority} source
documents — you do not speak for any other authority.

Source excerpts from {authority} documents:
{context}

Question: {question}

Using ONLY the excerpts above, write a short finding (a few sentences) that
answers the question from {authority}'s perspective, citing the specific
document/section referenced. If the excerpts do not contain material
relevant to the question, say plainly: "No relevant {authority} material
found for this question in the current corpus." Do not guess.
"""
)


def _retrieve_for_authority(authority: str, sub_queries: List[str]) -> List[Document]:
    if vectorstore is None:
        return []
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4, "filter": {"authority": authority}},
    )
    docs: List[Document] = []
    for q in sub_queries:
        docs.extend(retriever.invoke(q))
    return dedupe_docs(docs)


def authority_agents_node(state: RAGState) -> RAGState:
    findings: Dict[str, AuthorityFinding] = dict(state.get("authority_findings") or {})

    for authority in state["relevant_authorities"]:
        docs = _retrieve_for_authority(authority, state["sub_queries"])
        context = format_docs(docs)

        if not docs:
            findings[authority] = {
                "context": "",
                "draft": f"No relevant {authority} material found for this "
                f"question in the current corpus.",
                "has_docs": False,
            }
            continue

        resp = worker_llm.invoke(
            AUTHORITY_AGENT_PROMPT.invoke(
                {
                    "authority": authority,
                    "context": context,
                    "question": state["question"],
                }
            )
        )
        findings[authority] = {
            "context": context,
            "draft": resp.content,
            "has_docs": True,
        }

    return {**state, "authority_findings": findings}


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------
AGGREGATE_PROMPT = ChatPromptTemplate.from_template(
    """You are the lead QEHS regulatory compliance specialist. Specialist
agents for each relevant authority have reported their findings below.
Combine them into a single answer for a QA/EHS engineer:

- State the answer clearly, citing authority and document/section for
  every claim (only from what the specialists reported).
- Explicitly note where authorities AGREE, where they CONFLICT, and where
  one authority has no relevant material.
- Do not invent anything beyond what the specialist findings state.
- Each authority uses its own role names and terminology (e.g. "Compliance
  Monitoring Manager", "Accountable Manager", "Quality Manager", "Part-IS
  Manager"). These are DIFFERENT roles unless a specialist finding
  explicitly states they are the same. Never write "(also referred to as
  X)" or treat two named roles as interchangeable unless that equivalence
  is stated in the findings below — keep each authority's terminology
  separate rather than merging distinct roles into one umbrella term.

Question: {question}

Specialist findings:
{findings}
"""
)


def aggregate_node(state: RAGState) -> RAGState:
    findings = state["authority_findings"]
    authorities = state["all_relevant_authorities"]

    combined_context = "\n\n===\n\n".join(
        f"### {authority} source excerpts\n{findings[authority]['context']}"
        for authority in authorities
        if authority in findings and findings[authority]["has_docs"]
    )

    # Nothing to reconcile across authorities — skip the LLM call and use
    # the single specialist's draft directly.
    if len(authorities) == 1 and authorities[0] in findings:
        return {
            **state,
            "context_text": combined_context,
            "draft_answer": findings[authorities[0]]["draft"],
        }

    findings_block = "\n\n".join(
        f"### {authority}\n{findings[authority]['draft']}"
        for authority in authorities
        if authority in findings
    )

    resp = worker_llm.invoke(
        AGGREGATE_PROMPT.invoke(
            {"question": state["question"], "findings": findings_block}
        )
    )

    return {
        **state,
        "context_text": combined_context,
        "draft_answer": resp.content,
    }


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------
VERIFY_PROMPT = ChatPromptTemplate.from_template(
    """You are a critical reviewer checking a draft regulatory compliance
answer against its source context, before it goes to a QA/EHS engineer.

Context provided to the answering agents:
{context}

Draft answer:
{draft_answer}

Authorities involved: {authorities}

Check:
1. Does every specific claim (clause numbers, requirements, dates) in the
   draft actually appear in the context? Flag anything that looks invented.
2. Is there a conflict between authorities or chunks (e.g. an older vs
   newer revision) that the draft ignored or misrepresented?
3. Is the context sufficient to answer the question at all, or is it thin?
4. Does the draft claim two differently-named roles, terms, or documents
   are "the same as" or "also referred to as" each other? If so, does the
   context actually state that equivalence anywhere, or did the draft
   assume it? Different authorities using similar-sounding role names
   (e.g. Compliance Monitoring Manager vs Accountable Manager vs Part-IS
   Manager) are NOT automatically the same role — treat an unstated
   equivalence as an invented claim, same as check 1.

Keep your reasoning brief and focused — a few sentences per check above is
enough, you do not need to restate the full context or draft back to
yourself. Then respond ONLY with JSON, no other text:
{{"approved": true/false, "reason": "short explanation", "retry_query": "a refined search query if approved is false, else empty string", "retry_authority": "which single authority from {authorities} most needs re-checking, or ALL if unclear, or empty string if approved"}}
"""
)


def verify_node(state: RAGState) -> RAGState:
    resp = verifier_llm.invoke(
        VERIFY_PROMPT.invoke(
            {
                "context": state["context_text"],
                "draft_answer": state["draft_answer"],
                "authorities": ", ".join(state["all_relevant_authorities"]),
            }
        )
    )
    parsed = extract_json(resp.content)
    verification = {
        "approved": parsed.get("approved", False),
        "reason": parsed.get("reason", "Could not parse verifier output."),
        "retry_query": parsed.get("retry_query", ""),
        "retry_authority": parsed.get("retry_authority", "ALL"),
    }
    return {**state, "verification": verification}


def route_after_authority_agents(state: RAGState) -> str:
    # Opt-in fast path...
    single_authority = state["all_relevant_authorities"]
    is_low_risk = (
        len(single_authority) == 1
        and state["authority_findings"].get(single_authority[0], {}).get("has_docs")
        and not _looks_multihop(state["question"])
    )
    if state["skip_verification"] and is_low_risk:
        return "finalize_unverified"
    return "aggregate"


def finalize_unverified_node(state: RAGState) -> RAGState:
    authority = state["all_relevant_authorities"][0]
    finding = state["authority_findings"][authority]
    return {
        **state,
        "context_text": finding["context"],
        "draft_answer": finding["draft"],
        "final_answer": finding["draft"],
        "verification": {
            "approved": None,
            "reason": "Verification skipped (fast mode). Human review required."
        },
        "needs_review": True,
    }


def route_after_verify(state: RAGState) -> str:
    if state["verification"].get("approved"):
        return "finalize"
    if state["retry_count"] >= MAX_RETRIES:
        return "finalize"
    return "retry"


def prep_retry_node(state: RAGState) -> RAGState:
    retry_query = state["verification"].get("retry_query") or state["question"]
    retry_authority = state["verification"].get("retry_authority", "ALL")
    findings = state["authority_findings"]

    if retry_authority in state["all_relevant_authorities"]:
        # The verifier confidently named one authority — only it gets
        # re-processed; the others' cached findings are kept as-is.
        narrowed = [retry_authority]
    else:
        # Ambiguous verifier response ("ALL" or unparseable). Do NOT blindly
        # re-run every authority with the modified query — that would
        # silently overwrite authorities that already found correct,
        # relevant material with whatever a different query happens to
        # retrieve, which can make a good finding worse. Instead, only
        # re-run authorities that came back empty last time (the ones
        # actually likely responsible for "context is thin"). If every
        # authority already found something, fall back to re-running all
        # of them, since we genuinely don't know which one is at fault.
        empty_authorities = [
            a
            for a in state["all_relevant_authorities"]
            if not findings.get(a, {}).get("has_docs", False)
        ]
        narrowed = empty_authorities or list(state["all_relevant_authorities"])

    # Search with BOTH the verifier's refined query and the original
    # question — not just the refined one. The refined query isn't
    # guaranteed to retrieve better than the original phrasing did; adding
    # it as an extra query (not a replacement) increases recall instead of
    # gambling away a previously-good match.
    retry_sub_queries = [state["question"]]
    if retry_query != state["question"]:
        retry_sub_queries.append(retry_query)

    return {
        **state,
        "sub_queries": retry_sub_queries,
        "relevant_authorities": narrowed,
        "retry_count": state["retry_count"] + 1,
    }


def finalize_node(state: RAGState) -> RAGState:
    approved = state["verification"].get("approved", False)
    answer = state["draft_answer"]
    if not approved:
        reason = state["verification"].get("reason", "unspecified")
        answer = (
            "⚠️ **Needs human review** — automated verification could not "
            f"confirm this answer against source documents ({reason}).\n\n"
            + answer
        )
    return {**state, "final_answer": answer, "needs_review": not approved}


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(RAGState)
    graph.add_node("plan", plan_node)
    graph.add_node("authority_agents", authority_agents_node)
    graph.add_node("aggregate", aggregate_node)
    graph.add_node("verify", verify_node)
    graph.add_node("prep_retry", prep_retry_node)
    graph.add_node("finalize", finalize_node)
    graph.add_node("finalize_unverified", finalize_unverified_node)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "authority_agents")
    graph.add_conditional_edges(
        "authority_agents",
        route_after_authority_agents,
        {"aggregate": "aggregate", "finalize_unverified": "finalize_unverified"},
    )
    graph.add_edge("aggregate", "verify")
    graph.add_conditional_edges(
        "verify", route_after_verify, {"retry": "prep_retry", "finalize": "finalize"}
    )
    graph.add_edge("prep_retry", "authority_agents")
    graph.add_edge("finalize", END)
    graph.add_edge("finalize_unverified", END)

    return graph.compile()


compiled_graph = build_graph()


def run_query(
    question: str, authority: str = "ALL", skip_verification: bool = False
) -> RAGState:
    initial_state: RAGState = {
        "question": question,
        "authority": authority,
        "sub_queries": [],
        "all_relevant_authorities": [],
        "relevant_authorities": [],
        "authority_findings": {},
        "context_text": "",
        "draft_answer": "",
        "verification": {},
        "final_answer": "",
        "needs_review": False,
        "retry_count": 0,
        "skip_verification": skip_verification,
    }
    return compiled_graph.invoke(initial_state)
