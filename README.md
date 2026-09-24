# Aviation Regulatory Agentic RAG (POC)

A local, privacy-preserving question-answering system for aviation
regulatory documents (EASA, CAAS, CAAC). Everything runs on your own
machine — no data leaves your laptop, no API keys, no cloud calls.

Instead of a single AI model answering from a single search, this system
uses several small AI models working together as "agents": one plans the
search, one specialist per regulator checks its own documents, one
combines their findings, and one double-checks the final answer before
you see it.

---

## 1. What you need before you start

You don't need to know Python or AI to get this running — just follow
the steps in order.

| Requirement | Why | Notes |
|---|---|---|
| A computer with a decent GPU (8GB+ VRAM recommended) | The AI models run locally and need graphics memory | Works on CPU only too, but much slower |
| [Python 3.10 or newer](https://www.python.org/downloads/) | Runs the application code | Check with `python --version` |
| [Ollama](https://ollama.com/download) | Runs the AI models locally | Free, install like any normal app |
| Your regulatory documents (PDF, XML, Docx, etc) | The content the system answers questions from | See folder structure below |

---

## 2. Install Ollama and download the models

Ollama is the program that actually runs the AI models on your machine.
Once it's installed, open a terminal (Command Prompt / PowerShell on
Windows, Terminal on Mac/Linux) and download the two models this project
uses:

```bash
ollama pull llama3.1:8b
ollama pull deepseek-r1:8b
```

This downloads about 5GB per model, so it may take a while depending on
your internet connection. You only need to do this once.

### If you have a smaller GPU (8GB VRAM or less)

Both models together are too large to fit in VRAM at the same time on an
8GB card. That's fine — the system is built to work around this — but
you should tell Ollama to fully swap out one model before loading the
other, rather than trying (and struggling) to fit both:

**Mac/Linux**, before starting Ollama:
```bash
export OLLAMA_MAX_LOADED_MODELS=1
ollama serve
```

**Windows**: set `OLLAMA_MAX_LOADED_MODELS` to `1` under
*System Properties → Environment Variables*, then restart the Ollama
service (or restart your computer).

You can check this is working by running `ollama ps` while a question is
being answered — it should only ever show one model loaded at a time.

---

## 3. Set up the project

Download/clone this project folder, then open a terminal inside it and
run:

```bash
# create an isolated Python environment (recommended, avoids conflicts)
python -m venv venv

# activate it
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# install the required Python packages
pip install -r requirements.txt
```

---

## 4. Add your documents

Create this folder structure inside the project (if it doesn't already
exist) and drop your regulatory documents (PDF or XML) into the matching
authority's folder:

```
regulations/
├── easa/
│   └── (EASA PDFs/XML files go here)
├── caas/
│   └── (CAAS PDFs/XML files go here)
└── caac/
    └── (CAAC PDFs/XML files go here)
```

Then build the searchable database from these documents:

```bash
python ingest.py
```

You'll see progress messages as it reads, splits, and indexes your
documents. This creates a `regulatory_chroma_db/` folder — that's your
local database. Re-run `ingest.py` any time you add or change documents.

---

## 5. Start the app

Make sure Ollama is running in the background, then:

```bash
python app_api.py
```

You should see a message that the server is running. Open your browser
to:

```
http://localhost:8000
```

If a `templates/index.html` file exists in the project, you'll see a
simple chat-style web page with:
- an **authority filter** dropdown (ALL, EASA, CAAS, or CAAC only)
- a **fast mode** checkbox — ticks `skip_verification` on for that
  question (see Section 8)
- answers that highlight in yellow with a "⚠️ Needs human review" badge
  whenever the verifier couldn't fully confirm them
- a collapsible **"Per-authority findings"** section under each answer,
  showing what each regulator's specialist found individually before
  being combined

If there's no `templates/index.html`, you can still send questions
directly via the API (see below) — the web page is just a convenience.

### Stopping the app

Go back to the terminal where it's running and press `Ctrl+C`. This shuts
down the server cleanly. This does **not** stop Ollama or unload the AI
models — Ollama keeps running in the background as its own service.

---

## 6. Asking questions

If there's no web page, or you want to script/test it directly, send a
question to the API. From a terminal:

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the record-keeping requirements for engine component repair?"}'
```

Or from Python:

```python
import requests

response = requests.post(
    "http://localhost:8000/api/query",
    json={"question": "What are the record-keeping requirements for engine component repair?"},
)
print(response.json()["answer"])
```

### Request options

| Field | Default | What it does |
|---|---|---|
| `question` | *(required)* | Your question, in plain English |
| `authority` | `"ALL"` | Limit the search to one regulator: `"EASA"`, `"CAAS"`, or `"CAAC"` |
| `skip_verification` | `false` | Skip the double-check step for faster (but unverified) answers — see [Section 8](#8-understanding-the-response) |

### Example response

```json
{
  "answer": "...",
  "sources": "...",
  "needs_review": false,
  "verification": {"approved": true, "reason": "...", "retry_query": "", "retry_authority": ""},
  "retry_count": 0,
  "sub_queries": ["..."],
  "relevant_authorities": ["EASA", "CAAS", "CAAC"],
  "authority_findings": {"EASA": "...", "CAAS": "...", "CAAC": "..."}
}
```

---

## 7. How it works (the "agentic" part)

Rather than one AI model doing everything in one pass, your question goes
through several stages, each handled by a specialised step:

```
Your question
     │
     ▼
 1. Planner ──────────► decides if this is a simple question or needs
     │                   breaking into multiple search queries, and which
     │                   regulators (EASA/CAAS/CAAC) are relevant
     ▼
 2. Specialist agents ─► one per relevant regulator — each searches ONLY
     │                   that regulator's documents and writes a finding
     │                   scoped to what it found (or says plainly if it
     │                   found nothing relevant)
     ▼
 3. Aggregator ───────► combines the specialists' findings into one
     │                   answer, calling out where regulators agree,
     │                   disagree, or are silent
     ▼
 4. Verifier ─────────► double-checks the combined answer actually
     │                   matches the source documents, catching invented
     │                   citations or missed conflicts
     ▼
  If the verifier isn't satisfied, it sends the process back to step 2
  with a more targeted search (up to 2 retries) before finalizing.
     ▼
  Final answer, with a "⚠️ Needs human review" flag if verification
  was never able to fully confirm it.
```

Two different AI models are used for this:
- **Llama 3.1** handles the planning, searching, and drafting (steps 1–3)
- **DeepSeek-R1** handles the verification (step 4), since it's tuned for
  careful, step-by-step reasoning

---

## 8. Understanding the response

- **`needs_review: true`** means the automated verifier couldn't fully
  confirm the answer against the source documents — treat it as a
  starting point, not a final answer, and check the cited sources
  yourself.
- **`sources`** shows the exact document excerpts the answer was based
  on — always worth a quick look for anything going into real compliance
  work.
- **`authority_findings`** lets you see what each regulator's specialist
  found individually, before they were combined — useful for spotting
  exactly where a conflict between regulators comes from.
- **`skip_verification: true`** skips step 4 entirely for a faster
  response, but only when the system judges it low-risk (a single
  regulator, with matching documents found, and a simple non-comparison
  question). For anything you plan to rely on, leave this off.

---

## 9. Troubleshooting

**"Vector database not found" error** — you haven't run `python
ingest.py` yet, or it failed. Check that your documents are in the
`regulations/easa|caas|caac/` folders and re-run it.

**Answers are slow** — this is expected on a single consumer GPU running
two 8B models locally; each question can take anywhere from several
seconds to over a minute depending on complexity and your hardware.
Setting `"skip_verification": true` for casual/exploratory questions will
speed things up.

**Answers say "No relevant [authority] material found"** — that
regulator's folder has no documents covering that topic, or the wording
of your question doesn't match the documents closely enough. Try
rephrasing, or confirm the relevant PDF/XML is actually in that folder
and was picked up by `ingest.py`.

**Ollama seems to be using a lot of memory / running slowly** — confirm
`OLLAMA_MAX_LOADED_MODELS=1` is set (see Section 2) so it isn't trying to
keep both models loaded at once on a small GPU.

---

## 10. License

The code in this repository is licensed under the MIT License (see
`LICENSE`). This covers the software only — it does **not** cover the
regulatory documents you supply in `regulations/`, which remain the
property of their respective issuing authorities (EASA, CAAS, CAAC) under
their own terms. Check each authority's terms of use before redistributing
their documents; some permit reproduction with attribution, others require
written permission.

