import asyncio
import io
import json
import os
import shutil
import uuid
import zipfile

from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.agents.graph import run_query, app_graph
from app.agents.hybrid_search import warm_bm25_cache
from app.agents.reranker import warm_reranker
from app.config import discover_domains, TRACING_ENABLED, LANGCHAIN_PROJECT, DOCUMENTS_DIR, SEED_DOCUMENTS_DIR, CHROMA_PERSIST_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from app.ingestion.ingest import ingest_domain
from app.eval.ragas_eval import run_evaluation
from app.ingestion.fetch_and_ingest_topic import fetch_and_ingest_topic
from app.agents.lit_review import generate_literature_review
from app.citations import format_bibliography
from langchain.text_splitter import RecursiveCharacterTextSplitter

app = FastAPI(title="Multi-Agent RAG API")

SESSIONS: dict[str, list[dict]] = {}
MAX_HISTORY_TURNS = 6

ALLOWED_UPLOAD_EXTENSIONS = (".pdf", ".txt", ".md")
MAX_UPLOAD_FILE_SIZE_BYTES = int(os.getenv("MAX_UPLOAD_FILE_SIZE_MB", "2048")) * 1024 * 1024
# Env-driven: Render can set MAX_UPLOAD_FILE_SIZE_MB=150 to stay safe on its
# 512MB instance; local dev defaults to 2048MB (2GB) if the var is unset.
MAX_ZIP_MEMBERS = 300  # guard against a maliciously huge/zip-bomb archive
MAX_ZIP_TOTAL_EXTRACTED_BYTES = MAX_UPLOAD_FILE_SIZE_BYTES  # same env-driven cap
# applies to a zip's *total* extracted content - without this, the per-file
# allowance times 300 members could add up to hundreds of GB from one small
# compressed upload (the classic "zip bomb")


def _seed_and_ingest_if_needed():
    """First-boot-only (per persistent disk): if DOCUMENTS_DIR is empty -
    a fresh disk, or a deploy with no persistent disk at all - copy the
    corpus baked into the Docker image and fully ingest it, so Chroma
    isn't empty just because the disk was. A sentinel file on the same
    disk marks this done, so later restarts (where the disk already has
    documents *and* embeddings from a prior boot) skip straight to the
    fast BM25-only warm-up instead of re-ingesting everything - and re-
    embedding real files every restart, unlike a sentinel check, would
    also mean real OpenAI embedding-API cost on every single restart."""
    sentinel = Path(CHROMA_PERSIST_DIR) / ".seeded"
    if sentinel.exists():
        return

    documents_dir = Path(DOCUMENTS_DIR)
    documents_dir.mkdir(parents=True, exist_ok=True)
    is_empty = not any(documents_dir.iterdir())

    if is_empty and SEED_DOCUMENTS_DIR.exists() and SEED_DOCUMENTS_DIR.resolve() != documents_dir.resolve():
        print(f"First boot on this disk: seeding {documents_dir} from the image's baked-in corpus at {SEED_DOCUMENTS_DIR} ...")
        for domain_folder in SEED_DOCUMENTS_DIR.iterdir():
            if domain_folder.is_dir():
                shutil.copytree(domain_folder, documents_dir / domain_folder.name, dirs_exist_ok=True)

    domains = discover_domains()
    if domains:
        print(f"First boot on this disk: ingesting {domains} into Chroma ...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        for domain in domains:
            try:
                ingest_domain(domain, splitter)
            except Exception as exc:  # noqa: BLE001
                print(f"  failed to ingest {domain}: {exc}")

    Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
    sentinel.touch()
    print("Seeding + ingest complete for this disk - future restarts will skip this step.")


def _warm_caches_sync():
    _seed_and_ingest_if_needed()

    domains = discover_domains()
    if domains:
        print(f"Warming BM25 index for domains: {domains} ...")
        warm_bm25_cache(domains)
        print("BM25 index warm-up done.")

    print("Loading cross-encoder reranker model...")
    warm_reranker()
    print("Reranker model loaded.")


@app.on_event("startup")
async def _warm_caches():
    # Run in a background thread instead of blocking the startup event -
    # on memory-capped instances, waiting here for BM25/reranker to finish
    # loading before uvicorn opens its port is what causes Render's port
    # scanner to time out and kill the deploy as "no open ports detected".
    # Letting the server start accepting connections immediately means the
    # deploy succeeds even while warm-up finishes in the background.
    asyncio.create_task(asyncio.to_thread(_warm_caches_sync))


def _format_size(num_bytes: int) -> str:
    """Human-readable size for error messages - MB below 1GB, GB above."""
    if num_bytes >= 1024 ** 3:
        gb = num_bytes / 1024 ** 3
        return f"{gb:.0f}GB" if gb == int(gb) else f"{gb:.1f}GB"
    return f"{num_bytes // (1024 ** 2)}MB"


class QueryRequest(BaseModel):
    question: str
    session_id: str | None = None
    explain_simply: bool = False


class QueryResponse(BaseModel):
    answer: str
    sub_questions: list[str]
    domains_used: list[str]
    sources: list[str]
    source_details: list[dict] = []
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


class LitReviewRequest(BaseModel):
    topic: str
    max_papers_per_source: int = 15
    citation_style: str = "apa"  # "apa" or "ieee"


class LitReviewPaper(BaseModel):
    title: str
    authors: list[str]
    year: str
    venue: str
    ingested: bool
    citation: str


class LitReviewResponse(BaseModel):
    topic: str
    domain: str
    review_text: str
    num_found: int
    num_ingested: int
    papers: list[LitReviewPaper]
    bibliography: list[str]


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
        source_details=result.get("source_details") or [],
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
    Lets a user add their own PDFs/text/markdown to a domain (existing or
    brand new) and ingests them immediately - the domain becomes queryable
    right after this call returns, no separate script run needed.

    Also accepts .zip archives: each allowed member file inside is
    extracted and treated exactly like an individually-uploaded file.
    Member filenames are reduced to their basename only (no directory
    components) before being written to disk, which prevents a
    "zip-slip" archive from writing outside the domain folder via a
    path like "../../etc/passwd" inside the zip entry name.
    """
    domain = domain.strip().lower().replace(" ", "_")
    if not domain:
        raise HTTPException(status_code=400, detail="domain must not be empty")

    domain_dir = DOCUMENTS_DIR / domain
    domain_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    failed: list[str] = []

    def _write(filename: str, data: bytes) -> None:
        nonlocal saved
        if len(data) > MAX_UPLOAD_FILE_SIZE_BYTES:
            failed.append(f"{filename} (too large, max {_format_size(MAX_UPLOAD_FILE_SIZE_BYTES)})")
            return
        dest = domain_dir / filename
        try:
            with open(dest, "wb") as f:
                f.write(data)
            saved += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  failed to save {filename}: {exc}")
            failed.append(filename)

    for upload in files:
        name_lower = upload.filename.lower()
        raw = await upload.read()

        if name_lower.endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                    members = [m for m in zf.infolist() if not m.is_dir()]
            except zipfile.BadZipFile:
                failed.append(f"{upload.filename} (not a valid zip)")
                continue

            total_extracted = 0
            for member in members[:MAX_ZIP_MEMBERS]:
                member_name = os.path.basename(member.filename)  # zip-slip guard
                if not member_name or not member_name.lower().endswith(ALLOWED_UPLOAD_EXTENSIONS):
                    continue
                # member.file_size is the *declared* uncompressed size straight
                # from the zip header - checking it here rejects an oversized
                # or zip-bomb member without spending memory/CPU actually
                # inflating it first.
                if member.file_size > MAX_UPLOAD_FILE_SIZE_BYTES:
                    failed.append(f"{member_name} (too large, max {_format_size(MAX_UPLOAD_FILE_SIZE_BYTES)})")
                    continue
                total_extracted += member.file_size
                if total_extracted > MAX_ZIP_TOTAL_EXTRACTED_BYTES:
                    failed.append(f"{upload.filename}: remaining members skipped (archive exceeds {_format_size(MAX_ZIP_TOTAL_EXTRACTED_BYTES)} total extracted)")
                    break
                try:
                    data = zf.read(member)
                except Exception as exc:  # noqa: BLE001
                    print(f"  failed to read {member_name} from {upload.filename}: {exc}")
                    failed.append(member_name)
                    continue
                _write(member_name, data)
            continue

        if not name_lower.endswith(ALLOWED_UPLOAD_EXTENSIONS):
            failed.append(upload.filename)
            continue
        _write(upload.filename, raw)

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


@app.post("/literature-review", response_model=LitReviewResponse)
def literature_review(req: LitReviewRequest):
    """
    Given a topic: searches arXiv + Semantic Scholar, downloads whatever
    open-access PDFs are available, ingests them into a fresh domain, and
    generates a structured literature review with per-claim citations and
    a formatted bibliography.

    This is a slow, synchronous endpoint (search + downloads + ingestion +
    LLM synthesis can take a couple of minutes for 15-30 papers) - FastAPI
    runs regular `def` endpoints in a threadpool, so it won't block other
    requests, but the client should expect a long wait, not a quick response.
    """
    if req.citation_style not in ("apa", "ieee"):
        raise HTTPException(status_code=400, detail="citation_style must be 'apa' or 'ieee'")

    fetch_result = fetch_and_ingest_topic(req.topic, max_per_source=req.max_papers_per_source)
    papers = fetch_result["papers"]

    review = {"review_text": "No papers were successfully ingested for this topic.", "sources_used": []}
    if fetch_result["num_ingested"] > 0:
        review = generate_literature_review(req.topic, fetch_result["domain"])

    bibliography = format_bibliography(papers, style=req.citation_style)

    paper_responses = [
        LitReviewPaper(
            title=p["title"],
            authors=p.get("authors", []),
            year=p.get("year", ""),
            venue=p.get("venue", ""),
            ingested=p["ingested"],
            citation=bibliography[i],
        )
        for i, p in enumerate(papers)
    ]

    return LitReviewResponse(
        topic=req.topic,
        domain=fetch_result["domain"],
        review_text=review["review_text"],
        num_found=fetch_result["num_found"],
        num_ingested=fetch_result["num_ingested"],
        papers=paper_responses,
        bibliography=bibliography,
    )


@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    """
    Server-Sent Events endpoint.

    Streams:
      - agent/node trace events
      - synthesizer tokens
      - final metadata
      - errors
    """

    if not req.question.strip():
        raise HTTPException(
            status_code=400,
            detail="question must not be empty",
        )

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
            "metadata": {
                "session_id": session_id,
            },
        }

        final_state = {}
        current_synth_run_id = None

        try:
            # ---------------------------------------------------------
            # Tell frontend that connection is alive
            # ---------------------------------------------------------
            yield ": connected\n\n"

            # ---------------------------------------------------------
            # Stream LangGraph events
            # ---------------------------------------------------------
            async for event in app_graph.astream_events(
                initial_state,
                version="v2",
                config=trace_config,
            ):

                kind = event.get("event")
                metadata = event.get("metadata") or {}
                data = event.get("data") or {}

                node = (
                    metadata.get("langgraph_node")
                    or metadata.get("node")
                    or metadata.get("langgraph_step")
                )

                # -----------------------------------------------------
                # DEBUG LOGGING
                # -----------------------------------------------------
                print(
                    f"[TRACE] event={kind} node={node}",
                    flush=True,
                )

                # -----------------------------------------------------
                # SEND AGENT TRACE TO FRONTEND
                # -----------------------------------------------------
                if node:
                    trace_payload = {
                        "type": "agent_trace",
                        "node": node,
                        "event": kind,
                    }

                    yield (
                        "data: "
                        + json.dumps(trace_payload)
                        + "\n\n"
                    )

                # -----------------------------------------------------
                # SYNTHESIZER START
                # -----------------------------------------------------
                if (
                    kind == "on_chat_model_start"
                    and node == "synthesizer"
                ):

                    run_id = event.get("run_id")

                    if (
                        current_synth_run_id is not None
                        and run_id != current_synth_run_id
                    ):
                        yield (
                            "data: "
                            + json.dumps({
                                "type": "restart"
                            })
                            + "\n\n"
                        )

                    current_synth_run_id = run_id

                # -----------------------------------------------------
                # SYNTHESIZER STREAMING TOKENS
                # -----------------------------------------------------
                elif (
                    kind == "on_chat_model_stream"
                    and node == "synthesizer"
                ):

                    chunk = data.get("chunk")

                    token = ""

                    if chunk is not None:
                        token = getattr(
                            chunk,
                            "content",
                            "",
                        ) or ""

                    if token:

                        yield (
                            "data: "
                            + json.dumps({
                                "type": "token",
                                "content": token,
                            })
                            + "\n\n"
                        )

                # -----------------------------------------------------
                # FINAL GRAPH STATE
                # -----------------------------------------------------
                elif (
                    kind == "on_chain_end"
                    and node == "finalize"
                ):

                    final_state = data.get("output") or {}

            # ---------------------------------------------------------
            # FINAL ANSWER
            # ---------------------------------------------------------
            answer = final_state.get("final_answer") or ""

            # ---------------------------------------------------------
            # SAVE CHAT HISTORY
            # ---------------------------------------------------------
            history_new = history + [
                {
                    "question": req.question,
                    "answer": answer,
                }
            ]

            SESSIONS[session_id] = (
                history_new[-MAX_HISTORY_TURNS:]
            )

            # ---------------------------------------------------------
            # DONE EVENT
            # ---------------------------------------------------------
            done_payload = {
                "type": "done",
                "session_id": session_id,

                "sub_questions": (
                    final_state.get("sub_questions")
                    or [req.question]
                ),

                "domains_used": (
                    final_state.get("domains")
                    or []
                ),

                "sources": (
                    final_state.get("sources")
                    or []
                ),

                "source_details": (
                    final_state.get("source_details")
                    or []
                ),

                "is_grounded": bool(
                    final_state.get("is_grounded")
                ),
            }

            yield (
                "data: "
                + json.dumps(done_payload)
                + "\n\n"
            )

            # ---------------------------------------------------------
            # END OF STREAM
            # ---------------------------------------------------------
            yield "data: [DONE]\n\n"

        except Exception as exc:

            print(
                f"[query/stream] ERROR: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

            error_payload = {
                "type": "error",
                "message": str(exc),
            }

            yield (
                "data: "
                + json.dumps(error_payload)
                + "\n\n"
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
