"""
Gradio demo UI over the FastAPI backend. Run the API first:
    uvicorn app.main:app --reload
Then:
    python ui/app_gradio.py
"""
import json
import requests
import gradio as gr

API_URL = "http://localhost:8000/query/stream"

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root {
    --bg: #14161C;
    --panel: #1C1F29;
    --border: #2E323D;
    --text: #E8E6E1;
    --text-dim: #9A9FAE;
    --accent: #E3A857;
    --good: #7FA87F;
    --bad: #C46B4F;
}

.gradio-container { background: var(--bg) !important; font-family: 'IBM Plex Sans', sans-serif !important; }

#app-header { text-align: center; padding: 10px 0 2px 0; }
#app-header h1 {
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    letter-spacing: -0.01em;
    color: var(--text);
    font-size: 1.5rem;
    margin: 0;
}
#app-header p { color: var(--text-dim); font-size: 0.85rem; margin-top: 4px; }

#pipeline-trace {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: var(--text-dim);
    padding: 10px 0 18px 0;
}
.pt-node { padding: 4px 10px; border: 1px solid var(--border); border-radius: 5px; background: var(--panel); }
.pt-arrow { color: var(--border); }

#chatbot { background: var(--panel) !important; border: 1px solid var(--border) !important; border-radius: 10px !important; }

#ask-row { margin-top: 10px; }

footer { display: none !important; }
"""

PIPELINE_HTML = """
<div id="pipeline-trace">
  <span class="pt-node">router</span><span class="pt-arrow">&#8594;</span>
  <span class="pt-node">planner</span><span class="pt-arrow">&#8594;</span>
  <span class="pt-node">hybrid retriever</span><span class="pt-arrow">&#8594;</span>
  <span class="pt-node">synthesizer</span><span class="pt-arrow">&#8594;</span>
  <span class="pt-node">critic</span><span class="pt-arrow">&#8594;</span>
  <span class="pt-node">rewriter (retry)</span>
</div>
"""

HEADER_HTML = """
<div id="app-header">
  <h1>MULTI-AGENT RAG</h1>
  <p>Six-agent pipeline over a 150-document research corpus</p>
</div>
"""


def _format_meta(event: dict) -> str:
    meta_lines = []

    sub_qs = event.get("sub_questions", [])
    if len(sub_qs) > 1:
        meta_lines.append("**Sub-questions:** " + "; ".join(sub_qs))

    sources = event.get("sources", [])
    if sources:
        meta_lines.append("**Sources:** " + ", ".join(sources))

    domains = event.get("domains_used", [])
    badge = "\U0001F7E2 grounded" if event.get("is_grounded") else "\U0001F7E1 ungrounded"
    meta_lines.append(f"**Domains:** {', '.join(domains)} \u00b7 {badge}")

    return "\n\n---\n" + "\n\n".join(meta_lines)


def ask(question, history, session_id):
    """Generator: yields (history, textbox_value, session_id) repeatedly so
    Gradio streams the answer into the chat bubble token by token."""
    if not question or not question.strip():
        yield history, "", session_id
        return

    history = history + [[question, ""]]
    yield history, "", session_id

    payload = {"question": question}
    if session_id:
        payload["session_id"] = session_id

    try:
        resp = requests.post(API_URL, json=payload, stream=True, timeout=180)
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        history[-1][1] = (
            "Request timed out. The pipeline runs several agent steps (router, "
            "planner, hybrid retrieval, synthesizer, critic, and sometimes a "
            "rewrite retry), and the first query also builds the BM25 index - "
            "that first call can take a couple minutes. Try again; it should be "
            "faster afterward."
        )
        yield history, "", session_id
        return
    except Exception as exc:  # noqa: BLE001
        history[-1][1] = f"Error calling API: {exc}"
        yield history, "", session_id
        return

    answer_text = ""
    current_session_id = session_id

    try:
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue

            try:
                event = json.loads(line[len("data: "):])
            except json.JSONDecodeError:
                continue

            event_type = event.get("type")

            if event_type == "restart":
                # Critic rejected the first draft and the rewriter is
                # retrying - discard the partial answer and start clean
                # rather than showing two answers stitched together.
                answer_text = ""
                history[-1][1] = ""
                yield history, "", current_session_id

            elif event_type == "token":
                answer_text += event.get("content", "")
                history[-1][1] = answer_text
                yield history, "", current_session_id

            elif event_type == "done":
                current_session_id = event.get("session_id", session_id)
                history[-1][1] = answer_text + _format_meta(event)
                yield history, "", current_session_id

    except Exception as exc:  # noqa: BLE001
        history[-1][1] = (history[-1][1] or "") + f"\n\n[stream error: {exc}]"
        yield history, "", current_session_id


def new_conversation():
    return [], None


with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Base(), title="Multi-Agent RAG") as demo:
    gr.HTML(HEADER_HTML)
    gr.HTML(PIPELINE_HTML)

    session_state = gr.State(None)  # holds the session_id for this browser tab

    chatbot = gr.Chatbot(elem_id="chatbot", height=460, show_label=False)

    with gr.Row(elem_id="ask-row"):
        msg = gr.Textbox(
            placeholder="Ask something from the ingested corpus...",
            show_label=False,
            scale=8,
        )
        submit = gr.Button("Ask", variant="primary", scale=1)

    new_chat = gr.Button("New conversation", size="sm")

    msg.submit(ask, [msg, chatbot, session_state], [chatbot, msg, session_state])
    submit.click(ask, [msg, chatbot, session_state], [chatbot, msg, session_state])
    new_chat.click(new_conversation, None, [chatbot, session_state])

if __name__ == "__main__":
    demo.launch()
