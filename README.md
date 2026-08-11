# Multi-Agent RAG System

A retrieval-augmented generation system built with **LangGraph** that routes
queries across multiple domain-specific retrievers, verifies answers against
retrieved evidence with a critic agent, and synthesizes a final cited response.

Unlike a single RAG chain, this is a genuine multi-agent pipeline:

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

Built as a portfolio project to demonstrate multi-agent orchestration,
retrieval grounding, and evaluation — not just a "chatbot over a PDF."

## Stack

- **LangGraph** — agent orchestration / state graph
- **LangChain** — retrievers, prompt templates, output parsers
- **Chroma** — vector store
- **FastAPI** — API layer
- **Gradio** — demo UI
- **RAGAS** — retrieval/answer quality evaluation
- **Docker** — packaging

## Project layout

```
app/
  agents/
    router.py        # classifies query, picks which domain retriever(s) to call
    retriever.py      # builds domain-scoped retriever agents
    critic.py          # grounds-check the draft answer against retrieved chunks
    synthesizer.py     # produces the final cited answer
    graph.py            # wires agents together into a LangGraph StateGraph
  ingestion/
    ingest.py          # loads, chunks, and embeds documents into Chroma
  vectorstore/
    store.py            # Chroma collection management, one collection per domain
  eval/
    ragas_eval.py       # faithfulness / context precision evaluation
  main.py               # FastAPI app exposing /query and /ingest
  config.py             # settings (model names, chunk sizes, domains)
ui/
  app_gradio.py         # minimal chat UI hitting the FastAPI backend
data/
  documents/            # put your source docs here, organized by domain subfolder
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in your API key
```

### 1. Add your corpus

Drop documents into `data/documents/<domain_name>/`, e.g.:

```
data/documents/
  research_papers/
  policy_docs/
  general/
```

Each subfolder becomes its own retriever agent / domain. Aim for 100+ docs
total if you want this to genuinely back the "120+ documents, 8,000+ chunks"
claim — this is not hardcoded, it depends on what you ingest.

### 2. Ingest

```bash
python -m app.ingestion.ingest
```

This chunks every file under `data/documents/` and writes embeddings into a
Chroma collection per domain, printing the resulting document/chunk counts.

### 3. Run the API

```bash
uvicorn app.main:app --reload
```

### 4. Run the demo UI

```bash
python ui/app_gradio.py
```

### 5. Evaluate

```bash
python -m app.eval.ragas_eval
```

## Docker

```bash
docker compose up --build
```

## Cross-encoder reranking

Hybrid search (dense + BM25) pulls 12 candidates per domain via Reciprocal
Rank Fusion, then a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`
by default) re-scores each (query, passage) pair jointly and keeps only
the top 4. RRF's fused rank is a cheap positional heuristic; the
cross-encoder actually reads the query against each passage together,
which is what production RAG systems use to fix cases where a
semantically-similar-but-irrelevant chunk outranks the one that actually
answers the question.

The model (~90MB) downloads once from Hugging Face on first run and is
cached under `~/.cache/huggingface`; the server loads it at startup
(`Loading cross-encoder reranker model...` in the logs) so the first
query isn't stuck waiting on it. Swap `RERANK_MODEL` in `.env` for a
different cross-encoder if you want to trade accuracy for speed or vice
versa.

## Streaming

`POST /query/stream` returns Server-Sent Events instead of a single JSON
blob - the answer streams in token by token as the synthesizer generates
it, instead of waiting 10-20s for the whole pipeline to finish. Only the
synthesizer's own tokens are streamed (not the router/planner/critic's
internal LLM calls); if the critic rejects the draft and the rewriter
retries, a `restart` event tells the client to clear the partial answer
before the second attempt streams in.

```bash
curl -N -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "What is hybrid retrieval in RAG?"}'
```

Event types: `{"type": "token", "content": "..."}`, `{"type": "restart"}`,
and a final `{"type": "done", "session_id": ..., "sources": [...], ...}`
carrying the same metadata the non-streaming `/query` endpoint returns.
The Gradio UI uses this endpoint by default.

## Tracing (LangSmith)

Every agent call in this graph is a LangChain runnable, so LangSmith tracing
works with zero code changes - just environment variables:

1. Get a free API key at [smith.langchain.com](https://smith.langchain.com)
2. In `.env`, set:
   ```
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=ls__your_key_here
   LANGCHAIN_PROJECT=multi-agent-rag
   ```
3. Restart `uvicorn app.main:app --reload`
4. Run any query, then open the project on [smith.langchain.com](https://smith.langchain.com)

Each query shows up as a full trace: which domain the router picked, what
sub-questions the planner produced, exactly which chunks the hybrid
retriever pulled (and their RRF-fused rank), the synthesizer's draft, the
critic's grounded/not-grounded verdict, and - when it fires - the
rewriter's reformulated query on retry. `GET /health` reports whether
tracing is currently active.

This is the single most useful thing to pull up in an interview: instead of
describing the architecture, you can point at a real trace and show exactly
what each agent decided on a real question.

## Conversation memory

The API keeps a short server-side history per `session_id` (last 6 turns,
in-memory - resets on restart). Send `session_id` back on follow-up
requests to get contextual answers:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is hybrid retrieval in RAG?"}'
# -> {"answer": "...", "session_id": "abc-123", ...}

curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are its main limitations?", "session_id": "abc-123"}'
# the contextualizer agent rewrites this into a standalone question
# using the previous turn before the rest of the pipeline runs
```

The Gradio UI handles this automatically (session_id is tracked per browser
tab); use the "New conversation" button to start fresh.

## Notes for the CV / interview story

- The router + per-domain retrievers is what makes this "multi-agent" rather
  than a single RAG chain — be ready to explain that distinction.
- The critic agent is the interesting part: it re-checks the draft answer
  against the actual retrieved text and can trigger a re-retrieval loop
  instead of just returning an ungrounded answer. Walk through this loop in
  interviews; it is the most defensible, non-generic part of the project.
- Run the RAGAS eval on a held-out question set and keep the numbers
  (faithfulness, context precision/recall) — a number beats "it works well."
