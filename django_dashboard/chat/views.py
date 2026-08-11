import json
import os
import time
from collections import Counter

import requests
from django.conf import settings
from django.http import StreamingHttpResponse, JsonResponse, FileResponse, Http404, HttpResponse
from django.shortcuts import render, get_object_or_404, redirect
from django.views.decorators.csrf import csrf_exempt, ensure_csrf_cookie
from django.views.decorators.http import require_POST

from .models import Conversation, Message, UploadRecord


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


def upload_corpus(request):
    """GET: show the upload form. POST: proxy the files to FastAPI's
    /corpus/upload endpoint, which saves + immediately ingests them, then
    save an UploadRecord so the page has a real upload history."""
    if request.method == "POST":
        domain = (request.POST.get("domain") or "").strip()
        files = request.FILES.getlist("files")

        if not domain or not files:
            return render(request, "chat/upload.html", {
                "error": "Please provide a domain name and at least one file.",
                "upload_history": UploadRecord.objects.all()[:20],
                "known_domains": _known_domains(),
            })

        files_payload = [
            ("files", (f.name, f.read(), f.content_type or "application/pdf"))
            for f in files
        ]

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
                "upload_history": UploadRecord.objects.all()[:20],
                "known_domains": _known_domains(),
            })

        UploadRecord.objects.create(
            domain=domain,
            files_saved=result.get("files_saved", 0),
            chunks_added=result.get("chunks_added", 0),
            files_failed=result.get("files_failed", []),
        )

        return render(request, "chat/upload.html", {
            "result": result,
            "upload_history": UploadRecord.objects.all()[:20],
            "known_domains": _known_domains(),
        })

    return render(request, "chat/upload.html", {
        "upload_history": UploadRecord.objects.all()[:20],
        "known_domains": _known_domains(),
    })


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

            Message.objects.create(
                conversation=conversation,
                question=question,
                answer=answer_text,
                sub_questions=final_event.get("sub_questions", []),
                domains_used=final_event.get("domains_used", []),
                sources=final_event.get("sources", []),
                is_grounded=bool(final_event.get("is_grounded")),
                retried=saw_restart,
                response_time_s=elapsed_s,
            )

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

    context = {
        "stats": {
            "total_conversations": total_conversations,
            "total_messages": total_messages,
            "grounded_rate": grounded_rate,
            "grounded_count": grounded_count,
            "ungrounded_count": ungrounded_count,
            "corpus_total_docs": corpus_total_docs,
            "corpus_domain_count": corpus_domain_count,
        },
        "recent_conversations": recent_conversations,
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