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
from pydantic import BaseModel, Field, ValidationError, field_validator

from config import settings

MAX_RETRIES = 2
DB_DIR = settings.vector_db_path
AUTHORITIES = ["EASA", "CAAS", "CAAC"]

worker_llm = ChatOllama(
    model=settings.worker_model,
    base_url=settings.ollama_host,
    temperature=0.0,
    keep_alive="10m",
)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore: Optional[Chroma] = (
    Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    if os.path.exists(DB_DIR)
    else None
)

verifier_llm = ChatOllama(
    model=settings.verifier_model,
    base_url=settings.ollama_host,
    temperature=0.0,
    keep_alive="10m",
    num_predict=2000,
)


class VerificationResult(BaseModel):
    approved: bool
    reason: str = Field(..., min_length=1)
    retry_query: str = ""
    retry_authority: str = "ALL"

    @field_validator("retry_authority")
    @classmethod
    def validate_retry_authority(cls, value: str) -> str:
        normalized = (value or "ALL").upper()
        allowed = {"ALL", "EASA", "CAAS", "CAAC"}
        if normalized not in allowed:
            return "ALL"
        return normalized


class AuthorityFinding(TypedDict):
    context: str
    draft: str
    has_docs: bool


class RAGState(TypedDict):
    question: str
    authority: str
    sub_queries: List[str]
    all_relevant_authorities: List[str]
    relevant_authorities: List[str]
    authority_findings: Dict[str, AuthorityFinding]
    context_text: str
    draft_answer: str
    verification: dict
    final_answer: str
    needs_review: bool
    retry_count: int
    skip_verification: bool
    answer_status: str
    corpus_version: str


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_json(text: str) -> dict:
    for candidate in (strip_think(text), text):
        cleaned = re.sub(r"```json|```", "", candidate).strip()
        match = re.search(r"\{.*?\}", cleaned, flags=re.DOTALL)
        if not match:
            continue
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
    return {}


def extract_verification_result(text: str) -> dict:
    parsed = extract_json(text)
    if not parsed:
        return {}

    try:
        result = VerificationResult.model_validate(parsed)
        return result.model_dump()
    except ValidationError:
        reason = str(parsed.get("reason") or "Could not parse verifier output.")
        retry_query = str(parsed.get("retry_query") or "")
        retry_authority = str(parsed.get("retry_authority") or "ALL").upper()
        if retry_authority not in {"ALL", "EASA", "CAAS", "CAAC"}:
            retry_authority = "ALL"

        return {
            "approved": bool(parsed.get("approved", False)),
            "reason": reason,
            "retry_query": retry_query,
            "retry_authority": retry_authority,
        }


def format_docs(docs: List[Document]) -> str:
    if not docs:
        return ""
    return "\n\n---\n\n".join(
        f"[{doc.metadata.get('authority', 'UNKNOWN')} - "
        f"{os.path.basename(doc.metadata.get('source', 'Unknown'))}]\n"
        f"{doc.page_content}"
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
This question looks like it may need multiple retrieval passes. Break it into
2-4 focused sub-queries.

Respond ONLY with JSON, no other text:
{{"sub_queries": ["query 1", "query 2"]}}

Question: {question}
"""
)


def plan_node(state: RAGState) -> RAGState:
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
        "sub_queries": sub_queries[:4],
        "all_relevant_authorities": relevant_authorities,
        "relevant_authorities": relevant_authorities,
        "retry_count": 0,
        "answer_status": "PENDING",
    }


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


AGGREGATE_PROMPT = ChatPromptTemplate.from_template(
    """You are the lead QEHS regulatory compliance specialist. Specialist
agents for each relevant authority have reported their findings below.
Combine them into a single answer for a QA/EHS engineer:

- State the answer clearly, citing authority and document/section for
  every claim (only from what the specialists reported).
- Explicitly note where authorities AGREE, where they CONFLICT, and where
  one authority has no relevant material.
- Do not invent anything beyond what the specialist findings state.

Question: {question}

Specialist findings:
{findings}
"""
)


def aggregate_node(state: RAGState) -> RAGState:
    findings = state["authority_findings"]
    authorities = state["all_relevant_authorities"]

    if not authorities or not any(authority in findings for authority in authorities):
        return {
            **state,
            "context_text": "",
            "draft_answer": "No relevant material found in the current corpus.",
        }

    combined_context = "\n\n===\n\n".join(
        f"### {authority} source excerpts\n{findings[authority]['context']}"
        for authority in authorities
        if authority in findings and findings[authority]["has_docs"]
    )

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
        AGGREGATE_PROMPT.invoke({"question": state["question"], "findings": findings_block})
    )

    return {
        **state,
        "context_text": combined_context,
        "draft_answer": resp.content,
    }


VERIFY_PROMPT = ChatPromptTemplate.from_template(
    """You are a critical reviewer checking a draft regulatory compliance
answer against its source context.

Context:
{context}

Draft answer:
{draft_answer}

Authorities involved:
{authorities}

Check whether the claims are supported by the context, whether conflicts
were ignored, and whether the context is sufficient.

Respond ONLY with JSON:
{{"approved": true, "reason": "short explanation", "retry_query": "", "retry_authority": "ALL"}}
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
    parsed = extract_verification_result(resp.content)
    verification = {
        "approved": parsed.get("approved", False),
        "reason": parsed.get("reason", "Could not parse verifier output."),
        "retry_query": parsed.get("retry_query", ""),
        "retry_authority": parsed.get("retry_authority", "ALL"),
    }
    return {**state, "verification": verification}


def route_after_authority_agents(state: RAGState) -> str:
    authorities = state.get("all_relevant_authorities") or []
    is_low_risk = (
        len(authorities) == 1
        and state["authority_findings"].get(authorities[0], {}).get("has_docs")
        and not _looks_multihop(state["question"])
    )
    if state.get("skip_verification") and is_low_risk:
        return "finalize_unverified"
    return "aggregate"


def finalize_unverified_node(state: RAGState) -> RAGState:
    authorities = state.get("all_relevant_authorities") or []
    if not authorities:
        return {
            **state,
            "final_answer": "No relevant authority was selected for this question.",
            "verification": {
                "approved": None,
                "reason": "Verification skipped (fast mode). Human review required.",
            },
            "needs_review": True,
            "answer_status": "REQUIRES_HUMAN_REVIEW",
            "corpus_version": state.get("corpus_version", settings.corpus_version),
        }

    authority = authorities[0]
    finding = state["authority_findings"].get(authority, {})
    final_draft = finding.get("draft", "No relevant material found in the current corpus.")
    return {
        **state,
        "context_text": finding.get("context", ""),
        "draft_answer": final_draft,
        "final_answer": final_draft,
        "verification": {
            "approved": None,
            "reason": "Verification skipped (fast mode). Human review required.",
        },
        "needs_review": True,
        "answer_status": "REQUIRES_HUMAN_REVIEW",
        "corpus_version": state.get("corpus_version", settings.corpus_version),
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
        narrowed = [retry_authority]
    else:
        empty_authorities = [
            a for a in state["all_relevant_authorities"]
            if not findings.get(a, {}).get("has_docs", False)
        ]
        narrowed = empty_authorities or list(state["all_relevant_authorities"])

    retry_sub_queries = [state["question"]]
    if retry_query != state["question"]:
        retry_sub_queries.append(retry_query)

    return {
        **state,
        "sub_queries": retry_sub_queries[:4],
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
        return {
            **state,
            "final_answer": answer,
            "needs_review": True,
            "answer_status": "REQUIRES_HUMAN_REVIEW",
            "corpus_version": state.get("corpus_version", settings.corpus_version),
        }

    return {
        **state,
        "final_answer": answer,
        "needs_review": False,
        "answer_status": "VERIFIED",
        "corpus_version": state.get("corpus_version", settings.corpus_version),
    }


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
        "answer_status": "PENDING",
        "corpus_version": settings.corpus_version,
    }
    return compiled_graph.invoke(initial_state)
