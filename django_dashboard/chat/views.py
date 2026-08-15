import json
import os
import time
from collections import Counter

import requests
from django.conf import settings
from django.http import StreamingHttpResponse, JsonResponse, FileResponse, Http404, HttpResponse, HttpResponseNotAllowed
from django.shortcuts import render, get_object_or_404, redirect
from django.utils.html import escape
from django.utils.safestring import mark_safe
from django.views.decorators.csrf import csrf_exempt, ensure_csrf_cookie
from django.views.decorators.http import require_POST

from .models import Conversation, Message, UploadRecord, LiteratureReview


@ensure_csrf_cookie
def index(request):
    """Main dashboard: sidebar (stats + suggestions) + chat panel.
    If ?c=<id> is present, that conversation's messages are preloaded.
    Full conversation history lives on /history/."""
    active_id = request.GET.get("c")
    active_conversation = None
    messages = []
    if active_id:
        active_conversation = get_object_or_404(Conversation, pk=active_id)
        messages = active_conversation.messages.all()

    total_conversations = Conversation.objects.count()
    total_messages = Message.objects.count()
    grounded_count = Message.objects.filter(is_grounded=True).count()
    grounded_rate = round(100 * grounded_count / total_messages) if total_messages else 0

    domain_counter = Counter()
    source_counter = Counter()
    for m in Message.objects.only("domains_used", "sources"):
        for d in (m.domains_used or []):
            domain_counter[d] += 1
        for s in (m.sources or []):
            source_counter[s] += 1

    top_domains = domain_counter.most_common(5)
    top_sources = source_counter.most_common(5)

    corpus_dir = getattr(settings, "CORPUS_DIR", None)
    if not corpus_dir:
        corpus_dir = os.path.normpath(
            os.path.join(settings.BASE_DIR, "..", "data", "documents")
        )
    corpus_total_docs = 0
    corpus_domain_count = 0
    if os.path.isdir(corpus_dir):
        for name in os.listdir(corpus_dir):
            domain_path = os.path.join(corpus_dir, name)
            if os.path.isdir(domain_path):
                n = len([f for f in os.listdir(domain_path) if f.lower().endswith(".pdf")])
                if n:
                    corpus_domain_count += 1
                    corpus_total_docs += n

    context = {
        "active_conversation": active_conversation,
        "messages": messages,
        "stats": {
            "total_conversations": total_conversations,
            "total_messages": total_messages,
            "grounded_rate": grounded_rate,
        },
        "top_domains": top_domains,
        "top_sources": top_sources,
        "corpus_total_docs": corpus_total_docs,
        "corpus_domain_count": corpus_domain_count,
    }
    return render(request, "chat/index.html", context)


def history(request):
    """Full conversation history, newest first."""
    conversations = Conversation.objects.all()
    return render(request, "chat/history.html", {"conversations": conversations})


@require_POST
def rename_conversation(request, pk):
    """Rename a conversation's title from the History page."""
    conversation = get_object_or_404(Conversation, pk=pk)
    new_title = (request.POST.get("title") or "").strip()
    if new_title:
        conversation.title = new_title[:200]
        conversation.save()
    return redirect("chat:history")


@require_POST
def delete_conversation(request, pk):
    """Delete a conversation (and its messages, via cascade) from History."""
    conversation = get_object_or_404(Conversation, pk=pk)
    conversation.delete()
    return redirect("chat:history")


def export_conversation(request, pk):
    """Download a conversation as a Markdown file - questions, answers,
    grounded status, and sources, in order."""
    conversation = get_object_or_404(Conversation, pk=pk)

    lines = [
        f"# {conversation.title or 'Conversation'}",
        "",
        f"*Exported {conversation.updated_at:%Y-%m-%d %H:%M}*",
        "",
    ]
    for m in conversation.messages.all():
        lines.append(f"## Q: {m.question}")
        lines.append("")
        lines.append(m.answer)
        lines.append("")
        grounded_line = "**Grounded:** " + ("Yes" if m.is_grounded else "No")
        if m.retried:
            grounded_line += " (rewriter retried)"
        lines.append(grounded_line)
        if m.sources:
            lines.append(f"**Sources:** {', '.join(m.sources)}")
        lines.append("")
        lines.append("---")
        lines.append("")

    content = "\n".join(lines)
    safe_title = (conversation.title or "conversation").strip()[:50]
    safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in safe_title)
    filename = f"{safe_title.replace(' ', '_') or 'conversation'}.md"

    resp = HttpResponse(content, content_type="text/markdown")
    resp["Content-Disposition"] = f'attachment; filename="{filename}"'
    return resp


@csrf_exempt
@require_POST
def set_feedback(request, pk):
    """Sets or clears thumbs up/down feedback on a message.
    POST body: {"feedback": "up" | "down" | ""}"""
    message = get_object_or_404(Message, pk=pk)
    body = json.loads(request.body or "{}")
    value = body.get("feedback", "")
    if value not in ("up", "down", ""):
        return JsonResponse({"error": "feedback must be 'up', 'down', or ''"}, status=400)
    message.feedback = value
    message.save(update_fields=["feedback"])
    return JsonResponse({"ok": True, "feedback": message.feedback})


def _review_text_to_html(text: str) -> str:
    """Converts the synthesizer's markdown-ish output (#### / ### headers,
    blank-line paragraphs, [source.pdf] citation markers) into real HTML
    instead of showing literal '### Heading' text on the page. Escapes the
    LLM-generated content first since it's untrusted for HTML purposes."""
    import re

    lines = text.split("\n")
    html_parts = []
    paragraph_buffer = []

    def flush_paragraph():
        if paragraph_buffer:
            joined = " ".join(paragraph_buffer)
            joined = re.sub(
                r"\[([^\]]+\.pdf)\]",
                r'<span class="citation-badge">\1</span>',
                joined,
            )
            html_parts.append(f"<p>{joined}</p>")
            paragraph_buffer.clear()

    for raw_line in lines:
        line = escape(raw_line.strip())
        # escape() ran before the citation regex substitution below, so
        # "[x.pdf]" survives escaping unchanged - safe to pattern-match on.
        if line.startswith("#### "):
            flush_paragraph()
            html_parts.append(f"<h4>{line[5:]}</h4>")
        elif line.startswith("### "):
            flush_paragraph()
            html_parts.append(f"<h3>{line[4:]}</h3>")
        elif line == "":
            flush_paragraph()
        else:
            paragraph_buffer.append(line)

    flush_paragraph()
    return mark_safe("\n".join(html_parts))


def literature_review(request):
    """GET: show the topic input form. POST: proxy to FastAPI's
    /literature-review endpoint (search + ingest + synthesize), which is
    slow (search + downloads + LLM synthesis) - timeout is generous."""
    if request.method == "POST":
        topic = (request.POST.get("topic") or "").strip()
        citation_style = request.POST.get("citation_style", "apa")
        max_papers = int(request.POST.get("max_papers", 15))

        if not topic:
            return render(request, "chat/lit_review.html", {"error": "Please enter a topic."})

        try:
            resp = requests.post(
                f"{settings.FASTAPI_BASE_URL}/literature-review",
                json={
                    "topic": topic,
                    "max_papers_per_source": max_papers,
                    "citation_style": citation_style,
                },
                timeout=600,  # search + downloads + synthesis can genuinely take minutes
            )
            resp.raise_for_status()
            result = resp.json()
        except requests.RequestException as exc:
            return render(request, "chat/lit_review.html", {
                "error": f"Literature review generation failed: {exc}",
                "topic": topic,
            })

        LiteratureReview.objects.create(
            topic=topic,
            domain=result.get("domain", ""),
            citation_style=citation_style,
            num_found=result.get("num_found", 0),
            num_ingested=result.get("num_ingested", 0),
            review_text=result.get("review_text", ""),
            papers=result.get("papers", []),
            bibliography=result.get("bibliography", []),
        )

        return render(request, "chat/lit_review.html", {
            "result": result,
            "review_html": _review_text_to_html(result.get("review_text", "")),
            "topic": topic,
            "citation_style": citation_style,
            "bibliography_json": json.dumps(result.get("bibliography", [])),
        })

    return render(request, "chat/lit_review.html", {})


def literature_review_detail(request, pk):
    """Reopens a past literature review from the DB - no FastAPI call, no
    re-search, just redisplays what was already generated and saved."""
    record = get_object_or_404(LiteratureReview, pk=pk)
    result = {
        "domain": record.domain,
        "num_found": record.num_found,
        "num_ingested": record.num_ingested,
        "review_text": record.review_text,
        "papers": record.papers,
        "bibliography": record.bibliography,
    }
    return render(request, "chat/lit_review.html", {
        "result": result,
        "review_html": _review_text_to_html(record.review_text),
        "topic": record.topic,
        "citation_style": record.citation_style,
        "bibliography_json": json.dumps(record.bibliography),
    })


def _sanitize_pdf_text(text: str) -> str:
    """fpdf2's core fonts don't cover every unicode character an LLM might
    output (smart quotes, em dashes, ellipsis) - swap them for ASCII
    equivalents so the PDF doesn't throw a font encoding error."""
    replacements = {
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "-", "\u2026": "...",
    }
    for a, b in replacements.items():
        text = text.replace(a, b)
    return text


def literature_review_download(request):
    """POST: formats the already-generated review (resubmitted via hidden
    form fields on the results page) as a downloadable PDF. Doesn't call
    FastAPI again - this is pure formatting of data the page already has."""
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])

    from fpdf import FPDF

    topic = request.POST.get("topic", "Literature Review")
    review_text = request.POST.get("review_text", "")
    citation_style = request.POST.get("citation_style", "apa")
    try:
        bibliography = json.loads(request.POST.get("bibliography_json", "[]"))
    except json.JSONDecodeError:
        bibliography = []

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 10, _sanitize_pdf_text(f"Literature Review: {topic}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.set_font("Helvetica", "", 11)
    for line in review_text.splitlines():
        clean = _sanitize_pdf_text(line)
        if clean.startswith("#### "):
            pdf.set_font("Helvetica", "B", 12)
            pdf.multi_cell(0, 8, clean[5:], new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 11)
        elif clean.startswith("### "):
            pdf.set_font("Helvetica", "B", 13)
            pdf.multi_cell(0, 9, clean[4:], new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 11)
        elif clean.strip() == "":
            pdf.ln(3)
        else:
            pdf.multi_cell(0, 6, clean, new_x="LMARGIN", new_y="NEXT")

    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 13)
    pdf.multi_cell(0, 9, f"Bibliography ({citation_style.upper()})", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for i, citation in enumerate(bibliography, 1):
        pdf.multi_cell(0, 6, _sanitize_pdf_text(f"{i}. {citation}"), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)

    pdf_bytes = bytes(pdf.output())
    response = HttpResponse(pdf_bytes, content_type="application/pdf")
    safe_topic = "".join(c if c.isalnum() or c in " -_" else "" for c in topic)[:60].strip()
    safe_topic = safe_topic.replace(" ", "_") or "literature_review"
    response["Content-Disposition"] = f'attachment; filename="{safe_topic}.pdf"'
    return response


def download_source(request):
    """Serves a source PDF for viewing/download, found by filename across
    the corpus's domain subfolders. Usage: /source/?file=<filename>"""
    filename = request.GET.get("file", "")
    if not filename or "/" in filename or "\\" in filename:
        raise Http404("Invalid filename")

    corpus_dir = getattr(settings, "CORPUS_DIR", None)
    if not corpus_dir:
        corpus_dir = os.path.normpath(
            os.path.join(settings.BASE_DIR, "..", "data", "documents")
        )

    if not os.path.isdir(corpus_dir):
        raise Http404("Corpus directory not found")

    for domain_name in os.listdir(corpus_dir):
        candidate = os.path.join(corpus_dir, domain_name, filename)
        if os.path.isfile(candidate):
            return FileResponse(open(candidate, "rb"), content_type="application/pdf")

    raise Http404("Source file not found")


def _known_domains():
    """Domain names seen so far, for the Upload page's domain suggestions -
    drawn from past upload records plus whatever's already on disk."""
    names = set(
        UploadRecord.objects.order_by().values_list("domain", flat=True).distinct()
    )
    corpus_dir = getattr(settings, "CORPUS_DIR", None)
    if not corpus_dir:
        corpus_dir = os.path.normpath(
            os.path.join(settings.BASE_DIR, "..", "data", "documents")
        )
    if os.path.isdir(corpus_dir):
        for name in os.listdir(corpus_dir):
            if os.path.isdir(os.path.join(corpus_dir, name)):
                names.add(name)
    return sorted(names)


MAX_UPLOAD_FILE_SIZE_BYTES = int(os.getenv("MAX_UPLOAD_FILE_SIZE_MB", "2048")) * 1024 * 1024
# Env-driven so Render (memory-constrained, 512MB total) can be configured
# with a low value like 150 while local development keeps the default
# 2048MB (2GB) "practically unlimited" behavior - same code, different
# MAX_UPLOAD_FILE_SIZE_MB per deployment.


def _format_size(num_bytes: int) -> str:
    """Human-readable size for the upload page's hint text and error
    messages - MB below 1GB, GB at or above, so the same wording works
    whether this deployment is configured for 150MB or 2GB."""
    if num_bytes >= 1024 ** 3:
        gb = num_bytes / 1024 ** 3
        return f"{gb:.0f}GB" if gb == int(gb) else f"{gb:.1f}GB"
    return f"{num_bytes // (1024 ** 2)}MB"


def _upload_base_context():
    """Context shared by every render() in upload_corpus - the upload
    history/known domains lists, plus the size limit info the template
    and its JS need to display and enforce client-side."""
    return {
        "upload_history": UploadRecord.objects.all()[:20],
        "known_domains": _known_domains(),
        "max_upload_bytes": MAX_UPLOAD_FILE_SIZE_BYTES,
        "max_upload_display": _format_size(MAX_UPLOAD_FILE_SIZE_BYTES),
    }


def upload_corpus(request):
    """GET: show the upload form. POST: proxy the files (or pasted text,
    wrapped as a virtual .txt file) to FastAPI's /corpus/upload endpoint,
    which saves + immediately ingests them, then save an UploadRecord so
    the page has a real upload history."""
    if request.method == "POST":
        domain = (request.POST.get("domain") or "").strip()
        files = request.FILES.getlist("files")
        pasted_text = (request.POST.get("pasted_text") or "").strip()
        pasted_title = (request.POST.get("pasted_title") or "").strip()

        if not domain or not (files or pasted_text):
            return render(request, "chat/upload.html", {
                "error": "Please provide a domain name, and at least one file or some pasted text.",
                **_upload_base_context(),
            })

        oversized = [f.name for f in files if f.size > MAX_UPLOAD_FILE_SIZE_BYTES]
        if oversized:
            return render(request, "chat/upload.html", {
                "error": f"These files exceed the {_format_size(MAX_UPLOAD_FILE_SIZE_BYTES)} limit: {', '.join(oversized)}",
                **_upload_base_context(),
            })

        files_payload = [
            ("files", (f.name, f.read(), f.content_type or "application/pdf"))
            for f in files
        ]

        if pasted_text:
            safe_title = "".join(c if c.isalnum() or c in " -_" else "" for c in pasted_title)[:60].strip()
            safe_title = safe_title.replace(" ", "_") or "pasted_text"
            files_payload.append(
                ("files", (f"{safe_title}.txt", pasted_text.encode("utf-8"), "text/plain"))
            )

        try:
            resp = requests.post(
                f"{settings.FASTAPI_BASE_URL}/corpus/upload",
                data={"domain": domain},
                files=files_payload,
                timeout=300,
            )
            resp.raise_for_status()
            result = resp.json()
        except requests.RequestException as exc:
            return render(request, "chat/upload.html", {
                "error": f"Upload failed: {exc}",
                **_upload_base_context(),
            })

        UploadRecord.objects.create(
            domain=domain,
            files_saved=result.get("files_saved", 0),
            chunks_added=result.get("chunks_added", 0),
            files_failed=result.get("files_failed", []),
        )

        return render(request, "chat/upload.html", {
            "result": result,
            **_upload_base_context(),
        })

    return render(request, "chat/upload.html", _upload_base_context())


def corpus(request):
    """Lists the ingested document corpus, grouped by domain subfolder."""
    corpus_dir = getattr(settings, "CORPUS_DIR", None)
    if not corpus_dir:
        corpus_dir = os.path.normpath(
            os.path.join(settings.BASE_DIR, "..", "data", "documents")
        )

    domains = []
    if os.path.isdir(corpus_dir):
        for name in sorted(os.listdir(corpus_dir)):
            domain_path = os.path.join(corpus_dir, name)
            if not os.path.isdir(domain_path):
                continue
            files = sorted(
                f for f in os.listdir(domain_path) if f.lower().endswith(".pdf")
            )
            domains.append({"name": name, "count": len(files), "files": files})

    total_docs = sum(d["count"] for d in domains)

    return render(request, "chat/corpus.html", {
        "domains": domains,
        "total_docs": total_docs,
        "corpus_dir": corpus_dir,
        "corpus_dir_found": os.path.isdir(corpus_dir),
    })


@require_POST
def new_conversation(request):
    """Create an empty conversation and redirect the client to it."""
    conversation = Conversation.objects.create()
    return JsonResponse({"id": conversation.id})


@csrf_exempt
@require_POST
def _strip_null_bytes(value):
    """Postgres text/jsonb columns reject the NUL byte (\\x00) outright -
    not just discourage it, a single one anywhere in the payload crashes
    the INSERT with 'unsupported Unicode escape sequence'. PDF text
    extraction occasionally produces one from a garbled/corrupted source
    file, and that one bad character would otherwise take down an
    entire message's save (and, since answer_text/sources/etc are one
    combined INSERT, the whole conversation turn) rather than just that
    one snippet. Recurses through strings, lists, and dicts so it's safe
    to call on the full final_event payload in one pass."""
    if isinstance(value, str):
        return value.replace("\x00", "")
    if isinstance(value, list):
        return [_strip_null_bytes(v) for v in value]
    if isinstance(value, dict):
        return {k: _strip_null_bytes(v) for k, v in value.items()}
    return value


def stream_chat(request):
    """
    Proxies to the FastAPI backend's /query/stream SSE endpoint, forwarding
    tokens straight through to the browser as they arrive, and - once the
    stream finishes - saves the finished Q&A turn (including how long it
    took) into this conversation.
    """
    body = json.loads(request.body or "{}")
    question = (body.get("question") or "").strip()
    conversation_id = body.get("conversation_id")
    explain_simply = bool(body.get("explain_simply", False))

    if not question:
        return JsonResponse({"error": "question must not be empty"}, status=400)

    conversation = get_object_or_404(Conversation, pk=conversation_id) if conversation_id else Conversation.objects.create()

    def event_stream():
        start_time = time.monotonic()

        payload = {"question": question, "explain_simply": explain_simply}
        if conversation.session_id:
            payload["session_id"] = conversation.session_id

        resp = requests.post(
            f"{settings.FASTAPI_BASE_URL}/query/stream",
            json=payload,
            stream=True,
            timeout=180,
        )

        answer_text = ""
        final_event = None
        saw_restart = False

        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue

            yield line + "\n\n"

            try:
                event = json.loads(line[len("data: "):])
            except json.JSONDecodeError:
                continue

            if event.get("type") == "restart":
                answer_text = ""
                saw_restart = True
            elif event.get("type") == "token":
                answer_text += event.get("content", "")
            elif event.get("type") == "done":
                final_event = event

        elapsed_s = round(time.monotonic() - start_time, 1)

        if final_event:
            if not conversation.session_id:
                conversation.session_id = final_event.get("session_id", "")
            if conversation.title in ("", "New conversation"):
                conversation.title = question[:80]
            conversation.save()

            message = Message.objects.create(
                conversation=conversation,
                question=_strip_null_bytes(question),
                answer=_strip_null_bytes(answer_text),
                sub_questions=_strip_null_bytes(final_event.get("sub_questions", [])),
                domains_used=_strip_null_bytes(final_event.get("domains_used", [])),
                sources=_strip_null_bytes(final_event.get("sources", [])),
                source_details=_strip_null_bytes(final_event.get("source_details", [])),
                is_grounded=bool(final_event.get("is_grounded")),
                retried=saw_restart,
                response_time_s=elapsed_s,
            )
            yield f"data: {json.dumps({'type': 'message_id', 'id': message.id})}\n\n"

        # Extra client-side event (our own, not from FastAPI) so the live
        # UI can show the timing without waiting for a page reload.
        yield f"data: {json.dumps({'type': 'response_time', 'seconds': elapsed_s})}\n\n"
        yield f"data: {json.dumps({'type': 'conversation_id', 'id': conversation.id})}\n\n"

    response = StreamingHttpResponse(event_stream(), content_type="text/event-stream")
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"
    return response

def stats_dashboard(request):
    """Stats overview: conversation/message counts, grounded rate, corpus
    size, a 14-day conversation trend, domain distribution, response times
    for the last 20 turns, recent conversations, trending domains, and
    the most-cited source documents. All computed from the real
    Conversation/Message tables - no mock data."""
    from datetime import timedelta
    from django.utils import timezone

    total_conversations = Conversation.objects.count()
    total_messages = Message.objects.count()
    grounded_count = Message.objects.filter(is_grounded=True).count()
    ungrounded_count = total_messages - grounded_count
    grounded_rate = round(100 * grounded_count / total_messages) if total_messages else 0

    domain_counter = Counter()
    source_counter = Counter()
    for m in Message.objects.only("domains_used", "sources"):
        for d in (m.domains_used or []):
            domain_counter[d] += 1
        for s in (m.sources or []):
            source_counter[s] += 1

    trending_domains = domain_counter.most_common(5)
    popular_sources = source_counter.most_common(5)
    domain_distribution = domain_counter.most_common(8)

    corpus_dir = getattr(settings, "CORPUS_DIR", None)
    if not corpus_dir:
        corpus_dir = os.path.normpath(
            os.path.join(settings.BASE_DIR, "..", "data", "documents")
        )
    corpus_total_docs = 0
    corpus_domain_count = 0
    if os.path.isdir(corpus_dir):
        for name in os.listdir(corpus_dir):
            domain_path = os.path.join(corpus_dir, name)
            if os.path.isdir(domain_path):
                n = len([f for f in os.listdir(domain_path) if f.lower().endswith(".pdf")])
                if n:
                    corpus_domain_count += 1
                    corpus_total_docs += n

    today = timezone.localdate()
    conv_trend = []
    for i in range(13, -1, -1):
        day = today - timedelta(days=i)
        count = Conversation.objects.filter(created_at__date=day).count()
        conv_trend.append({"date": day.strftime("%m-%d"), "count": count})

    response_times = list(
        Message.objects.exclude(response_time_s=0)
        .order_by("-created_at")[:20]
        .values_list("response_time_s", flat=True)
    )
    response_times.reverse()

    recent_conversations = Conversation.objects.all()[:6]
    lit_review_count = LiteratureReview.objects.count()
    recent_lit_reviews = LiteratureReview.objects.all()[:5]

    context = {
        "stats": {
            "total_conversations": total_conversations,
            "total_messages": total_messages,
            "grounded_rate": grounded_rate,
            "grounded_count": grounded_count,
            "ungrounded_count": ungrounded_count,
            "corpus_total_docs": corpus_total_docs,
            "corpus_domain_count": corpus_domain_count,
            "lit_review_count": lit_review_count,
        },
        "recent_conversations": recent_conversations,
        "recent_lit_reviews": recent_lit_reviews,
        "trending_domains": trending_domains,
        "popular_sources": popular_sources,
        # Passed as plain Python objects - the template feeds these to
        # Chart.js via the |json_script filter, which serializes and
        # escapes them safely into their own <script type="application/json">
        # tag, rather than splicing raw {{ }} tags into inline JS.
        "conv_trend": conv_trend,
        "domain_distribution": domain_distribution,
        "response_times": response_times,
    }
    return render(request, "chat/dashboard.html", context)
