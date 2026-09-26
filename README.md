# Aviation Regulatory Agentic RAG (Prototype)

A prototype question-answering system for aviation regulatory documents from EASA, CAAS, and CAAC. It uses a planner, regulator-specific retrieval agents, an answer synthesizer, and an AI verifier to draft answers from an indexed document collection.

## Data and model-provider behavior

The default configuration uses Gemini for both chat and embeddings. Prompts and retrieved document excerpts are sent to the configured model provider. Ollama can be configured for local inference; the application is not local-only by default.

The generated vector index is stored locally. Keep source documents and generated indexes out of Git unless you have permission and a specific reason to publish them.

## Requirements

- Python 3.10 or newer
- Regulatory source documents that you are authorized to use
- A model provider and its API key, or a configured local Ollama installation
- Ollama is optional when using a cloud provider

## 1. Install the project

Open a terminal in the project directory and create a virtual environment:

    python -m venv .venv

Activate it:

    # Windows PowerShell
    .venv\Scripts\Activate.ps1

    # macOS or Linux
    source .venv/bin/activate

Install dependencies:

    pip install -r requirements.txt

## 2. Configure the application

Copy the example environment file:

    # Windows PowerShell
    Copy-Item .env.example .env

    # macOS or Linux
    cp .env.example .env

Set a unique AUTH_TOKEN with at least 32 characters. One way to generate one is:

    python -c "import secrets; print(secrets.token_urlsafe(32))"

Put the generated value in .env. The app refuses to start with an empty, short, or example authentication token when AUTH_REQUIRED is enabled.

The default models use Gemini. Set GEMINI_API_KEY in .env to use them. For a different provider, configure a compatible chat model, embedding provider/model, and required credentials. The chat model and embedding model are separate settings.

Optional role-specific model settings are available:

- PLANNER_LLM_MODEL
- SPECIALIST_LLM_MODEL
- AGGREGATOR_LLM_MODEL
- VERIFIER_LLM_MODEL

Leave these blank to use DEFAULT_LLM_MODEL for every role. You can also set ALLOWED_ORIGINS, MAX_CONCURRENT_QUERIES, QUERY_TIMEOUT_SECONDS, and QUEUE_TIMEOUT_SECONDS in .env.

## 3. Add regulatory documents

Place documents in the matching authority folder:

    regulations/
    ├── easa/
    ├── caas/
    └── caac/

The ingestion script supports PDF, DOCX, XML, TXT, and Markdown files. It extracts text and splits it into overlapping passages before generating embeddings. Scanned PDFs without extractable text need OCR before ingestion.

## 4. Build the search index

Run:

    python ingest.py

The index is written to indexes/<CORPUS_VERSION>, and indexes/current.txt is updated after a successful build. A manifest records the documents and chunk counts.

Each corpus version is built once. If the index path already exists, set a new CORPUS_VERSION in .env before rebuilding. If a document fails to load or embed, the build does not become active; check the staging manifest reported by the script.

The application uses the embedding provider and model configured during ingestion. Keep those settings the same when querying. If you change the embedding model, build a new corpus version with that model.

## 5. Start the API and web interface

Start the server:

    python app_api.py

Open http://localhost:8000. Enter the configured AUTH_TOKEN in the access-token field before submitting a question.

The API provides:

- GET /health/live — process liveness
- GET /health/ready — checks that the configured Chroma collection exists and contains data
- POST /api/query — generates an answer from the indexed documents

## 6. Ask questions through the API

Example using curl on macOS or Linux:

    curl -X POST http://localhost:8000/api/query \
      -H 'Content-Type: application/json' \
      -H 'Authorization: Bearer YOUR_AUTH_TOKEN' \
      -d '{"question":"What records are required for this maintenance activity?","authority":"EASA"}'

In Windows PowerShell, use `Invoke-RestMethod`:

    $headers = @{ Authorization = "Bearer YOUR_AUTH_TOKEN" }
    $body = @{ question = "What records are required for this maintenance activity?"; authority = "EASA" } | ConvertTo-Json
    Invoke-RestMethod -Method Post -Uri http://localhost:8000/api/query -Headers $headers -ContentType "application/json" -Body $body

Request fields:

| Field | Required | Description |
|---|---:|---|
| question | Yes | Question to answer; 3 to 4,000 characters |
| authority | No | ALL, EASA, CAAS, or CAAC; defaults to ALL |
| skip_verification | No | Skips the AI verifier when true; the answer still requires human review |

The API also requires an Authorization: Bearer token header when AUTH_REQUIRED is enabled.

Responses include the answer, retrieved source excerpts, authorities searched, specialist findings, retry count, and verifier result. The answer status is AGENT_REVIEWED when the AI verifier approves the draft, or NEEDS_HUMAN_REVIEW otherwise. needs_review remains true for every answer.

## 7. Agent workflow

Each request follows this sequence:

1. **Planner:** creates focused retrieval queries and selects relevant authorities. Comparison questions search all three authorities.
2. **Regulator specialists:** each selected authority agent searches only its own indexed passages and writes a finding with source IDs.
3. **Aggregator:** combines the findings, preserving disagreements and gaps in the evidence.
4. **Verifier:** checks claims and citations against the retrieved passages. If the verifier rejects the draft, the specialists and aggregator receive feedback for one revision.

By default, the roles use the same model. Set the role-specific model settings in .env to route particular roles to different compatible models.

## 8. Review and limitations

The verifier is an automated model check. It does not certify compliance, replace a qualified reviewer, or guarantee that the index contains the latest or complete regulations. Check the cited source excerpts and the underlying official material before using an answer in compliance work.

The quality of answers depends on the accuracy, currency, completeness, and extraction quality of the documents you index. This prototype does not establish regulatory applicability to a particular aircraft, organization, maintenance event, or jurisdiction.

Setting skip_verification to true bypasses the verifier for that request. The answer remains marked as requiring human review.

## Troubleshooting

**The app refuses to start because of AUTH_TOKEN**  
Set a unique token of at least 32 characters in .env. Do not use the example value.

**The vector collection is missing or empty**  
Check that supported documents are in the authority folders and run python ingest.py. If a corpus version already exists, choose a new CORPUS_VERSION.

**Ingestion reports failed documents**  
Review the staging manifest named in the error. Check file readability, extractable text, provider credentials, and provider limits.

**Embedding dimension mismatch**  
The embedding model used by the API must match the model that built the active index. Restore the previous embedding configuration or build a new corpus version.

**A query times out or takes a long time**  
The request may require several model calls: planning, retrieval and specialist work, synthesis, and verification. Check model-provider availability and adjust the timeout or model settings if needed.

## License

The software is licensed under the MIT License; see LICENSE. The license does not grant rights to redistribute regulatory source documents. Those remain subject to the terms of their issuing authorities.
