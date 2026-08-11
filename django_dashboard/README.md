# Multi-Agent RAG — Django Dashboard

A full dashboard frontend for the [Multi-Agent RAG](../README.md) system:
chat, conversation history, and usage stats — built with Django instead of
the Gradio demo UI (`../ui/app_gradio.py`), which still works independently
if you want it too.

**This does not touch the agent pipeline at all.** Django is a thin
frontend + persistence layer; every question still goes through the exact
same FastAPI backend (`../app/main.py`) and the same 7-agent LangGraph
pipeline. Django's job is: render the dashboard, proxy chat requests to
FastAPI's `/query/stream`, and save the finished Q&A turns to its own
database so they show up in the sidebar history and stats panel.

```
Browser  ──fetch()──►  Django (this app)  ──proxy──►  FastAPI (../app)  ──►  LangGraph agents
                            │
                            └─ saves finished turns to SQLite for history/stats
```

## Setup

From this directory:

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
python manage.py migrate
```

Make sure the FastAPI backend is already running separately (see the main
project README) on the URL set in `.env` (`FASTAPI_BASE_URL`, defaults to
`http://localhost:8000`).

Then run the dashboard:

```bash
python manage.py runserver 8001
```

Open **http://127.0.0.1:8001**.

(Port 8001 because FastAPI is already using 8000 - if you'd rather free up
8000 for this, stop the Gradio UI and adjust ports as you like; the two
frontends don't conflict with each other.)

## What's in here

```
dashboard/            # Django project settings
  settings.py            # FASTAPI_BASE_URL, DB, static files config
  urls.py
chat/                  # the one app
  models.py              # Conversation, Message
  views.py                # index (dashboard page), new_conversation,
                           # stream_chat (the FastAPI streaming proxy)
  admin.py                 # conversations/messages browsable in /admin
  templates/chat/index.html
  static/chat/css/style.css
  static/chat/js/chat.js   # fetch() + ReadableStream SSE parsing
```

## Admin panel

```bash
python manage.py createsuperuser
```

Then visit `/admin` to browse every conversation and message stored,
including the metadata (sources, domains, grounded status) each answer was
saved with - useful for a screenshot showing this isn't just a chat UI,
it's backed by a real data model.

## Notes

- `stream_chat` uses `@csrf_exempt` since it's a same-origin POST driven by
  this page's own JS (the CSRF cookie is still set via `ensure_csrf_cookie`
  on the index view) - a stricter setup would validate `X-CSRFToken`
  explicitly instead of exempting the view. Worth mentioning if asked about
  it in an interview - it's a deliberate demo-scope shortcut, not an
  oversight.
- Conversation history is real (SQLite), not mocked - refreshing the page
  or restarting the server keeps every conversation.
- The `session_id` FastAPI hands back on each turn is stored on the
  `Conversation` row, so the agent pipeline's contextualizer can still
  resolve multi-turn follow-ups exactly like it does in the Gradio UI.
