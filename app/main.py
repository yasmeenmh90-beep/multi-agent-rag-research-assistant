import json
import shutil
import uuid

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.agents.graph import run_query, app_graph
from app.agents.hybrid_search import warm_bm25_cache
from app.agents.reranker import warm_reranker
from app.config import discover_domains, TRACING_ENABLED, LANGCHAIN_PROJECT, DOCUMENTS_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from app.ingestion.ingest import ingest_domain
from app.eval.ragas_eval import run_evaluation
from langchain.text_splitter import RecursiveCharacterTextSplitter

app = FastAPI(title="Multi-Agent RAG API")

SESSIONS: dict[str, list[dict]] = {}
MAX_HISTORY_TURNS = 6


@app.on_event("startup")
def _warm_caches():
    domains = discover_domains()
    if domains:
        print(f"Warming BM25 index for domains: {domains} ...")
        warm_bm25_cache(domains)
        print("BM25 index warm-up done.")

    print("Loading cross-encoder reranker model...")
    warm_reranker()
    print("Reranker model loaded.")


class QueryRequest(BaseModel):
    question: str
    session_id: str | None = None
    explain_simply: bool = False


class QueryResponse(BaseModel):
    answer: str
    sub_questions: list[str]
    domains_used: list[str]
    sources: list[str]
    is_grounded: bool
    session_id: str


class UploadResponse(BaseModel):
    domain: str
    files_saved: int
    files_failed: list[str]
    chunks_added: int


class EvalQuestionResult(BaseModel):
    question: str
    faithfulness: float
    context_precision: float


class EvalResponse(BaseModel):
    num_questions: int
    faithfulness_avg: float
    context_precision_avg: float
    per_question: list[EvalQuestionResult]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "domains": discover_domains(),
        "tracing_enabled": TRACING_ENABLED,
        "langsmith_project": LANGCHAIN_PROJECT if TRACING_ENABLED else None,
    }


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    session_id = req.session_id or str(uuid.uuid4())
    history = SESSIONS.get(session_id, [])

    result = run_query(
        req.question,
        chat_history=history,
        session_id=session_id,
        explain_simply=req.explain_simply,
    )

    answer = result.get("final_answer") or ""

    history = history + [{"question": req.question, "answer": answer}]
    SESSIONS[session_id] = history[-MAX_HISTORY_TURNS:]

    return QueryResponse(
        answer=answer,
        sub_questions=result.get("sub_questions") or [req.question],
        domains_used=result.get("domains") or [],
        sources=result.get("sources") or [],
        is_grounded=bool(result.get("is_grounded")),
        session_id=session_id,
    )


@app.post("/session/{session_id}/reset")
def reset_session(session_id: str):
    SESSIONS.pop(session_id, None)
    return {"status": "cleared", "session_id": session_id}


@app.post("/corpus/upload", response_model=UploadResponse)
async def upload_documents(
    domain: str = Form(...),
    files: list[UploadFile] = File(...),
):
    """
    Lets a user add their own PDFs to a domain (existing or brand new) and
    ingests them immediately - the domain becomes queryable right after
    this call returns, no separate script run needed.
    """
    domain = domain.strip().lower().replace(" ", "_")
    if not domain:
        raise HTTPException(status_code=400, detail="domain must not be empty")

    domain_dir = DOCUMENTS_DIR / domain
    domain_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    failed: list[str] = []

    for upload in files:
        if not upload.filename.lower().endswith((".pdf", ".txt", ".md")):
            failed.append(upload.filename)
            continue
        dest = domain_dir / upload.filename
        try:
            with open(dest, "wb") as f:
                shutil.copyfileobj(upload.file, f)
            saved += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  failed to save {upload.filename}: {exc}")
            failed.append(upload.filename)

    chunks_added = 0
    if saved:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        _, chunks_added = ingest_domain(domain, splitter)
        warm_bm25_cache([domain])

    return UploadResponse(
        domain=domain,
        files_saved=saved,
        files_failed=failed,
        chunks_added=chunks_added,
    )


@app.post("/eval/run", response_model=EvalResponse)
def eval_run():
    """
    Runs the RAGAS evaluation (app/eval/ragas_eval.py) against the fixed
    EVAL_QUESTIONS set and returns faithfulness/context_precision scores.
    This is a synchronous, CPU/LLM-bound call - FastAPI runs regular `def`
    endpoints in a threadpool automatically, so it won't block other
    requests, but it can take a minute or two (5 questions, each running
    the full agent pipeline plus RAGAS scoring).
    """
    result = run_evaluation()
    return EvalResponse(**result)


@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    """
    Server-Sent Events endpoint. Streams only the synthesizer agent's
    tokens (not the router/planner/critic's internal LLM calls) as they're
    generated, so the client sees the answer appear word-by-word.

    If the critic rejects the draft and the rewriter triggers a retry, the
    synthesizer runs a second time - we detect that (a new LLM run_id for
    the same node) and emit a "restart" event so the client clears the
    partial answer instead of showing two answers stitched together.

    Event types sent as `data: {...}\\n\\n`:
      - {"type": "token", "content": "..."}   - one streamed token
      - {"type": "restart"}                    - discard streamed text so far
      - {"type": "done", ...}                  - final metadata (sources,
                                                  domains, grounded, etc.)
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    session_id = req.session_id or str(uuid.uuid4())
    history = SESSIONS.get(session_id, [])

    async def event_generator():
        initial_state = {
            "question": req.question,
            "chat_history": history,
            "retry_count": 0,
            "explain_simply": req.explain_simply,
        }
        trace_config = {
            "run_name": "multi_agent_rag_query",
            "tags": ["multi-agent-rag"],
            "metadata": {"session_id": session_id},
        }

        current_synth_run_id = None
        final_state = None

        async for event in app_graph.astream_events(
            initial_state, version="v2", config=trace_config
        ):
            kind = event["event"]
            node = (event.get("metadata") or {}).get("langgraph_node")

            if kind == "on_chat_model_start" and node == "synthesizer":
                run_id = event["run_id"]
                if current_synth_run_id is not None and run_id != current_synth_run_id:
                    yield f"data: {json.dumps({'type': 'restart'})}\n\n"
                current_synth_run_id = run_id

            elif kind == "on_chat_model_stream" and node == "synthesizer":
                chunk = event["data"]["chunk"]
                token = getattr(chunk, "content", "") or ""
                if token:
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

            elif kind == "on_chain_end" and node == "finalize":
                final_state = event["data"]["output"]

        answer = (final_state or {}).get("final_answer") or ""

        history_new = history + [{"question": req.question, "answer": answer}]
        SESSIONS[session_id] = history_new[-MAX_HISTORY_TURNS:]

        done_payload = {
            "type": "done",
            "session_id": session_id,
            "sub_questions": (final_state or {}).get("sub_questions") or [req.question],
            "domains_used": (final_state or {}).get("domains") or [],
            "sources": (final_state or {}).get("sources") or [],
            "is_grounded": bool((final_state or {}).get("is_grounded")),
        }
        yield f"data: {json.dumps(done_payload)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)