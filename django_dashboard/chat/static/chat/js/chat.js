// Render single newlines as <br> (matches the old |linebreaksbr behavior)
if (window.marked) {
    marked.setOptions({ breaks: true });
}

function getCookie(name) {
    const match = document.cookie.match(new RegExp('(^| )' + name + '=([^;]+)'));
    return match ? decodeURIComponent(match[2]) : null;
}

const csrftoken = getCookie('csrftoken');

const form = document.getElementById('chat-form');
const input = document.getElementById('question-input');
const sendBtn = document.getElementById('send-btn');
const messagesEl = document.getElementById('messages');
const conversationIdField = document.getElementById('conversation-id');
const newConvBtn = document.getElementById('new-conversation');
const explainSimplyToggle = document.getElementById('explain-simply-toggle');
const toastContainer = document.getElementById('toast-container');

function showToast(message) {
    if (!toastContainer) return;
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.textContent = message;
    toastContainer.appendChild(toast);
    requestAnimationFrame(() => toast.classList.add('toast-show'));
    setTimeout(() => {
        toast.classList.remove('toast-show');
        setTimeout(() => toast.remove(), 300);
    }, 2200);
}

// ---------- Keyboard shortcuts ----------
document.addEventListener('keydown', (e) => {
    const isMod = e.metaKey || e.ctrlKey;
    if (isMod && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        input.focus();
    } else if (isMod && e.key === 'Enter') {
        e.preventDefault();
        if (document.activeElement === input) {
            form.requestSubmit();
        }
    }
});

document.querySelectorAll('.suggestion-chip').forEach((chip) => {
    chip.addEventListener('click', () => {
        input.value = chip.dataset.question || '';
        autoGrow();
        input.focus();
    });
});

// ---------- Markdown rendering for messages already in the page ----------
function renderMarkdownIn(el) {
    if (!window.marked) return;
    const raw = el.dataset.raw;
    if (raw === undefined) return;
    el.innerHTML = marked.parse(raw);
}
document.querySelectorAll('.answer-text[data-raw]').forEach(renderMarkdownIn);

function autoGrow() {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 140) + 'px';
}
input.addEventListener('input', autoGrow);

input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        form.requestSubmit();
    }
});

newConvBtn.addEventListener('click', async () => {
    const resp = await fetch(window.DJANGO_URLS.newConversation, {
        method: 'POST',
        headers: { 'X-CSRFToken': csrftoken },
    });
    const data = await resp.json();
    window.location.href = '?c=' + data.id;
});

function addUserBubble(text) {
    const row = document.createElement('div');
    row.className = 'msg-row user';
    row.innerHTML = `<div class="bubble user-bubble"></div>`;
    row.querySelector('.bubble').textContent = text;
    messagesEl.appendChild(row);
    return row;
}

function addAssistantBubble() {
    const row = document.createElement('div');
    row.className = 'msg-row assistant';
    row.innerHTML = `
        <div class="bubble assistant-bubble">
            <div class="agent-status"></div>
            <div class="answer-text"></div>
            <div class="badge-row"></div>
            <details class="trace-block">
                <summary>Agent trace</summary>
            </details>
        </div>`;
    messagesEl.appendChild(row);
    return row;
}

function copyBtn() {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'copy-btn';
    btn.title = 'Copy answer';
    btn.textContent = '⧉ Copy';
    btn.addEventListener('click', () => {
        const answerEl = btn.closest('.bubble').querySelector('.answer-text');
        const raw = answerEl.dataset.raw || answerEl.textContent;
        navigator.clipboard.writeText(raw).then(() => {
            btn.textContent = '✓ Copied';
            showToast('Answer copied to clipboard');
            setTimeout(() => { btn.textContent = '⧉ Copy'; }, 1500);
        });
    });
    return btn;
}

// Wire up copy buttons already rendered server-side (historical messages).
document.querySelectorAll('.badge-row .copy-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
        const answerEl = btn.closest('.bubble').querySelector('.answer-text');
        const raw = answerEl.dataset.raw || answerEl.textContent;
        navigator.clipboard.writeText(raw).then(() => {
            btn.textContent = '✓ Copied';
            showToast('Answer copied to clipboard');
            setTimeout(() => { btn.textContent = '⧉ Copy'; }, 1500);
        });
    });
});

// ---------- Live "which agent is running" status while waiting ----------
const AGENT_STAGES = [
    'Routing query…',
    'Planning sub-questions…',
    'Retrieving evidence…',
    'Synthesizing answer…',
    'Checking grounding…',
];

function startAgentStatus(statusEl) {
    let i = 0;
    statusEl.style.display = 'block';
    statusEl.textContent = AGENT_STAGES[0];
    const timer = setInterval(() => {
        i = (i + 1) % AGENT_STAGES.length;
        statusEl.textContent = AGENT_STAGES[i];
    }, 650);
    return timer;
}

function stopAgentStatus(statusEl, timer) {
    clearInterval(timer);
    statusEl.style.display = 'none';
}

function chip(text, cls) {
    const span = document.createElement('span');
    span.className = 'chip' + (cls ? ' ' + cls : '');
    span.textContent = text;
    return span;
}

function traceSection(label, contentEl) {
    const wrap = document.createElement('div');
    wrap.className = 'trace-section';
    const labelEl = document.createElement('div');
    labelEl.className = 'trace-label';
    labelEl.textContent = label;
    wrap.appendChild(labelEl);
    wrap.appendChild(contentEl);
    return wrap;
}

function listEl(items, cls) {
    const ul = document.createElement('ul');
    ul.className = 'trace-list' + (cls ? ' ' + cls : '');
    items.forEach((item) => {
        const li = document.createElement('li');
        li.textContent = item;
        ul.appendChild(li);
    });
    return ul;
}

function sourceListEl(items) {
    const ul = document.createElement('ul');
    ul.className = 'trace-list sources-list';
    items.forEach((item) => {
        const li = document.createElement('li');
        const a = document.createElement('a');
        a.href = `${window.DJANGO_URLS.downloadSource}?file=${encodeURIComponent(item)}`;
        a.target = '_blank';
        a.className = 'source-link';
        a.textContent = item;
        li.appendChild(a);
        ul.appendChild(li);
    });
    return ul;
}

function chipsEl(items) {
    const div = document.createElement('div');
    div.className = 'trace-chips';
    items.forEach((item) => div.appendChild(chip(item)));
    return div;
}

function renderMeta(bubbleRow, event, retried, responseTimeSeconds) {
    const answerEl = bubbleRow.querySelector('.answer-text');
    answerEl.dataset.raw = answerEl.textContent || '';

    const badgeRow = bubbleRow.querySelector('.badge-row');
    badgeRow.innerHTML = '';
    const groundedBadge = document.createElement('span');
    groundedBadge.className = 'grounded-badge ' + (event.is_grounded ? 'good' : 'warn');
    groundedBadge.textContent = event.is_grounded ? '● GROUNDED' : '● UNGROUNDED';
    badgeRow.appendChild(groundedBadge);

    if (retried) {
        const retryBadge = document.createElement('span');
        retryBadge.className = 'retry-badge';
        retryBadge.textContent = '↻ rewriter retried';
        badgeRow.appendChild(retryBadge);
    }

    if (responseTimeSeconds !== undefined && responseTimeSeconds !== null) {
        const timeBadge = document.createElement('span');
        timeBadge.className = 'response-time';
        timeBadge.textContent = `${responseTimeSeconds}s`;
        badgeRow.appendChild(timeBadge);
    }

    badgeRow.appendChild(copyBtn());

    const details = bubbleRow.querySelector('.trace-block');
    details.querySelectorAll('.trace-section').forEach((el) => el.remove());

    const domains = event.domains_used || [];
    const subQuestions = event.sub_questions || [];
    const sources = event.sources || [];

    if (domains.length) {
        details.appendChild(traceSection('Domains routed', chipsEl(domains)));
    }
    if (subQuestions.length) {
        details.appendChild(traceSection('Sub-questions (planner)', listEl(subQuestions)));
    }
    if (sources.length) {
        details.appendChild(traceSection(`Sources (${sources.length})`, sourceListEl(sources)));
    }
}

async function sendMessage(question) {
    const emptyState = messagesEl.querySelector('.empty-state');
    if (emptyState) emptyState.remove();

    addUserBubble(question);
    const assistantRow = addAssistantBubble();
    const answerEl = assistantRow.querySelector('.answer-text');
    const statusEl = assistantRow.querySelector('.agent-status');

    sendBtn.disabled = true;
    let statusTimer = startAgentStatus(statusEl);

    const explainSimply = explainSimplyToggle ? explainSimplyToggle.checked : false;

    let resp;
    try {
        resp = await fetch(window.DJANGO_URLS.stream, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': csrftoken,
            },
            body: JSON.stringify({
                question: question,
                conversation_id: conversationIdField.value || null,
                explain_simply: explainSimply,
            }),
        });
    } catch (err) {
        stopAgentStatus(statusEl, statusTimer);
        answerEl.textContent = 'Error calling backend: ' + err;
        sendBtn.disabled = false;
        return;
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let answerText = '';
    let retried = false;
    let responseTimeSeconds = null;
    let doneEvent = null;

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split('\n\n');
        buffer = parts.pop();

        for (const part of parts) {
            if (!part.startsWith('data: ')) continue;
            let event;
            try {
                event = JSON.parse(part.slice(6));
            } catch (e) {
                continue;
            }

            if (event.type === 'restart') {
                answerText = '';
                answerEl.innerHTML = '';
                retried = true;
                statusTimer = startAgentStatus(statusEl);
            } else if (event.type === 'token') {
                if (answerText === '') {
                    stopAgentStatus(statusEl, statusTimer);
                }
                answerText += event.content;
                answerEl.innerHTML = window.marked ? marked.parse(answerText) : answerText;
                messagesEl.scrollTop = messagesEl.scrollHeight;
            } else if (event.type === 'done') {
                stopAgentStatus(statusEl, statusTimer);
                doneEvent = event;
            } else if (event.type === 'response_time') {
                responseTimeSeconds = event.seconds;
                if (doneEvent) {
                    renderMeta(assistantRow, doneEvent, retried, responseTimeSeconds);
                }
            } else if (event.type === 'conversation_id') {
                conversationIdField.value = event.id;
                history.replaceState(null, '', '?c=' + event.id);
            }
        }
    }

    stopAgentStatus(statusEl, statusTimer);
    sendBtn.disabled = false;
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

form.addEventListener('submit', (e) => {
    e.preventDefault();
    const question = input.value.trim();
    if (!question) return;
    input.value = '';
    autoGrow();
    sendMessage(question);
});