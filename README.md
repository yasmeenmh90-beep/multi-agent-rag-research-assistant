# Multi-Agent RAG — Grounded Research Assistant

A retrieval-augmented generation system built with **LangGraph** that routes
queries across multiple domain-specific retrievers, verifies every answer
against retrieved evidence with a critic agent, and only shows an answer
once it's actually backed by the source material — or honestly says it
doesn't know.

Unlike a single RAG chain, this is a genuine 7-agent pipeline:

```
                ┌───────────────┐
   query ──────►│ Contextualizer│  rewrites follow-up questions into
                └──────┬────────┘  standalone questions using chat history
                       ▼
                ┌─────────────┐
   (session)    │   Router    │  classifies query → picks domain(s)
   history ────►└──────┬──────┘
                       ▼
                ┌─────────────┐
                │   Planner   │  decomposes multi-part questions into
                └──────┬──────┘  focused sub-questions (or leaves simple
                       │         questions unchanged)
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
  ┌───────────┐  ┌───────────┐  ┌───────────┐
  │  Hybrid   │  │  Hybrid   │  │  Hybrid   │   dense + BM25 fused via RRF,
  │ Retriever │  │ Retriever │  │ Retriever │   then cross-encoder reranked
  │  domain A │  │  domain B │  │  domain C │
  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘
        └──────────────┼──────────────┘
                       ▼
                ┌─────────────┐
                │ Synthesizer │  drafts answer w/ citations
                └──────┬──────┘
                       ▼
                ┌─────────────┐
                │   Critic    │  checks answer is grounded in
                │  (verifier) │  retrieved chunks
                └──────┬──────┘
              grounded │ not grounded (max 1 retry)
                       │        │
                       │        ▼
                       │  ┌─────────────┐
                       │  │  Rewriter   │  reformulates the question using
                       │  └──────┬──────┘  the critic's own feedback, then
                       │         │          loops back to retrieval
                       │         └──────────────┐
                       ▼                        │
                final answer + sources ◄─────────┘ (after retry, either way)
```

## Why this exists

Most "AI research assistant" demos are a chatbot bolted onto a PDF — no way
to know if the answer is actually true or just plausible-sounding. This
project makes the AI justify its own answer before showing it to you: the
critic agent checks the draft against the retrieved evidence, and if it
isn't well-supported, a rewriter reformulates and retries instead of
returning an unverified answer.

## Primary interface: Django dashboard

The main way to use this project is the **Django web dashboard**
(`django_dashboard/`) — a full UI on top of the FastAPI backend below.

- **Chat** — ask questions, see answers stream in live, with a
  GROUNDED / UNGROUNDED badge, response time, a copy button, and an
  expandable **Agent trace** (domains routed, planner's sub-questions,
  source PDFs — each one clickable to open/download)
- **Explain Simply** — a toggle that rewrites the same grounded, cited
  answer in plain, beginner-friendly language
- **Corpus browser** — see every ingested document, grouped by domain
- **Upload documents** — add your own PDFs into a new or existing domain;
  they're embedded and queryable immediately, no restart needed
- **History** — every past conversation, searchable, renameable,
  deletable, and exportable as Markdown
- **Trending topics / popular papers** — which domains and source papers
  get asked about most, built from real usage
- **Dashboard stats** — a live grounded rate calculated across every real
  conversation, not a one-time benchmark
- Mobile-responsive, with an animated startup screen

A minimal Gradio UI (`ui/app_gradio.py`) also exists for quick local testing
directly against the FastAPI backend, without the dashboard.

## Stack

- **LangGraph** — agent orchestration / state graph
- **LangChain** — retrievers, prompt templates, output parsers
- **Chroma** — vector store
- **FastAPI** — API layer, streaming responses
- **Django** — primary web dashboard
- **Gradio** — minimal secondary UI for quick testing
- **Docker** — packaging

## Project layout

```
app/
  agents/
    contextualizer.py  # resolves follow-up questions using chat history
    router.py           # classifies query, picks which domain retriever(s) to call
    planner.py          # decomposes multi-part questions into sub-questions
    retriever.py         # builds domain-scoped hybrid retriever agents
    reranker.py           # cross-encoder reranking of hybrid search results
    hybrid_search.py       # BM25 + dense search fused via RRF
    synthesizer.py          # produces the final cited answer
    critic.py                 # grounds-checks the draft answer against retrieved chunks
    rewriter.py                # reformulates the question on a failed grounding check
    graph.py                    # wires agents together into a LangGraph StateGraph
  ingestion/
    fetch_arxiv.py       # bulk-downloads papers from arXiv by search query
    ingest.py              # loads, chunks, and embeds documents into Chroma
  vectorstore/
    store.py               # Chroma collection management, one collection per domain
  main.py                    # FastAPI app: /query, /query/stream, /corpus/upload
  config.py                    # settings (model names, chunk sizes, domains)
ui/
  app_gradio.py                 # minimal chat UI hitting the FastAPI backend directly
django_dashboard/
  chat/                          # the main dashboard app - views, models, templates
  dashboard/                      # Django project settings
data/
  documents/                       # source docs, organized by domain subfolder
```

## Setup

### 1. FastAPI backend

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in your OpenAI API key
```

Add your corpus into `data/documents/<domain_name>/`, e.g.:

```
data/documents/
  ai_papers/
  multi_agent/
  general/
```

Each subfolder becomes its own retriever domain. You can also fetch a
corpus in bulk directly from arXiv:

```bash
python -m app.ingestion.fetch_arxiv --query "retrieval augmented generation" --domain ai_papers --max 100
```

Then ingest whatever's in `data/documents/`:

```bash
python -m app.ingestion.ingest
```

Run the API:

```bash
uvicorn app.main:app --reload
```

### 2. Django dashboard

In a second terminal:

```bash
cd django_dashboard
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # set FASTAPI_BASE_URL to point at the API above
python manage.py migrate
python manage.py runserver 8001
```

Open `http://127.0.0.1:8001` — the FastAPI backend must already be running
on port 8000 for the dashboard to work, since it proxies every query there.

### 3. (Optional) Gradio quick-test UI

```bash
python ui/app_gradio.py
```

## Docker

```bash
docker compose up --build
```

## Hybrid retrieval + cross-encoder reranking

Hybrid search (dense + BM25) pulls candidates per domain via Reciprocal
Rank Fusion, then a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`
by default) re-scores each (query, passage) pair jointly and keeps only
the top results. RRF's fused rank is a cheap positional heuristic; the
cross-encoder actually reads the query against each passage together,
which is what production RAG systems use to fix cases where a
semantically-similar-but-irrelevant chunk outranks the one that actually
answers the question.

The reranker model (~90MB) downloads once from Hugging Face on first run
and is cached under `~/.cache/huggingface`; the server loads it at startup
so the first query isn't stuck waiting on it.

## Streaming

`POST /query/stream` returns Server-Sent Events instead of a single JSON
blob — the answer streams in token by token as the synthesizer generates
it, instead of waiting for the whole pipeline to finish. If the critic
rejects the draft and the rewriter retries, a `restart` event tells the
client to clear the partial answer before the second attempt streams in.
The Django dashboard uses this endpoint by default.

## Conversation memory

The API keeps a short server-side history per `session_id` (last 6 turns,
in-memory — resets on restart). The Django dashboard tracks this
automatically per conversation; the contextualizer agent uses it to
resolve follow-up questions like "what are its limitations?" without
you having to repeat context.

## Tracing (LangSmith)

Every agent call in this graph is a LangChain runnable, so LangSmith tracing
works with zero code changes — just environment variables:

```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__your_key_here
LANGCHAIN_PROJECT=multi-agent-rag
```

Each query shows up as a full trace: which domain the router picked, what
sub-questions the planner produced, which chunks the hybrid retriever
pulled, the synthesizer's draft, the critic's verdict, and — when it
fires — the rewriter's reformulated query on retry. `GET /health` reports
whether tracing is currently active.

## Notes

- The router + per-domain retrievers is what makes this "multi-agent"
  rather than a single RAG chain.
- The critic agent is the interesting part: it re-checks the draft answer
  against the actual retrieved text and can trigger a re-retrieval loop
  instead of just returning an ungrounded answer. Tested directly: asking
  something entirely outside the corpus produces an honest "the provided
  context does not contain this information" instead of a fabricated
  answer — and that honest refusal is itself correctly marked grounded.
- The Django dashboard's live grounded-rate stat is calculated from real
  usage across every conversation, not a one-time benchmark.
