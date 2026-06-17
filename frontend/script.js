(() => {
  const chatEl = document.getElementById('chat');
  const emptyStateEl = document.getElementById('empty-state');
  const formEl = document.getElementById('chat-form');
  const inputEl = document.getElementById('chat-input');
  const sendBtn = document.getElementById('send-btn');
  const sourcesToggle = document.getElementById('show-sources-toggle');
  const clearBtn = document.getElementById('clear-btn');

  // Each entry: { role: 'user' | 'assistant', content: string, sources: string[] | null }
  let messages = [];
  let showSources = false;
  let isStreaming = false;

  // ── Math normalizer ──────────────────────────────────────────────────────
  // Same transform as the original Streamlit app: KaTeX picks up $...$ and
  // $$...$$, but the LLM often writes \( ... \) and \[ ... \]. This rewrites
  // the latter into the former before markdown/KaTeX render the text.
  function normalizeMath(text) {
    text = text.replace(/\\\[([\s\S]*?)\\\]/g, (_, inner) => `$$${inner}$$`);
    text = text.replace(/\\\(([\s\S]*?)\\\)/g, (_, inner) => `$${inner}$`);
    return text;
  }

  function escapeHtml(str) {
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
  }

  function renderMarkdownInto(el, text) {
    const html = marked.parse(normalizeMath(text));
    el.innerHTML = DOMPurify.sanitize(html);
    if (window.renderMathInElement) {
      renderMathInElement(el, {
        delimiters: [
          { left: '$$', right: '$$', display: true },
          { left: '$', right: '$', display: false },
        ],
        throwOnError: false,
      });
    }
  }

  // ── Source excerpt renderer ──────────────────────────────────────────────
  function buildSourcesElement(sources) {
    if (!sources || sources.length === 0) return null;
    const details = document.createElement('details');
    details.className = 'sources';

    const summary = document.createElement('summary');
    const n = sources.length;
    summary.textContent = `${n} source excerpt${n !== 1 ? 's' : ''} referenced`;
    details.appendChild(summary);

    sources.forEach((chunk, idx) => {
      const wrap = document.createElement('div');
      wrap.className = 'source-excerpt';

      const label = document.createElement('strong');
      label.textContent = `Excerpt ${idx + 1}`;
      wrap.appendChild(label);

      const pre = document.createElement('pre');
      const code = document.createElement('code');
      const preview = chunk.slice(0, 450) + (chunk.length > 450 ? '…' : '');
      code.textContent = preview;
      pre.appendChild(code);
      wrap.appendChild(pre);

      details.appendChild(wrap);
    });

    return details;
  }

  // ── Message element builder ──────────────────────────────────────────────
  function buildMessageElement(msg) {
    const wrapper = document.createElement('div');
    wrapper.className = `message ${msg.role}`;

    const content = document.createElement('div');
    content.className = 'message-content';
    renderMarkdownInto(content, msg.content);
    wrapper.appendChild(content);

    if (showSources && msg.sources && msg.sources.length > 0) {
      const sourcesEl = buildSourcesElement(msg.sources);
      if (sourcesEl) wrapper.appendChild(sourcesEl);
    }

    return wrapper;
  }

  // ── Full re-render (used on toggle change, clear, and turn start) ────────
  function renderAll() {
    emptyStateEl.style.display = messages.length === 0 ? '' : 'none';
    chatEl.innerHTML = '';
    messages.forEach((msg) => {
      chatEl.appendChild(buildMessageElement(msg));
    });
    scrollToBottom();
  }

  function scrollToBottom() {
    chatEl.scrollTop = chatEl.scrollHeight;
  }

  // ── Sidebar controls ──────────────────────────────────────────────────────
  sourcesToggle.addEventListener('change', () => {
    showSources = sourcesToggle.checked;
    renderAll();
  });

  clearBtn.addEventListener('click', () => {
    messages = [];
    renderAll();
  });

  // ── Input box: auto-grow + Enter to send, Shift+Enter for a new line ─────
  inputEl.addEventListener('input', () => {
    inputEl.style.height = 'auto';
    inputEl.style.height = `${Math.min(inputEl.scrollHeight, 160)}px`;
  });

  inputEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      formEl.requestSubmit();
    }
  });

  // ── Submit: stream the assistant's reply ──────────────────────────────────
  formEl.addEventListener('submit', async (e) => {
    e.preventDefault();
    const text = inputEl.value.trim();
    if (!text || isStreaming) return;

    inputEl.value = '';
    inputEl.style.height = 'auto';

    messages.push({ role: 'user', content: text, sources: null });
    const assistantMsg = { role: 'assistant', content: '', sources: null };
    messages.push(assistantMsg);
    renderAll();

    const assistantEl = chatEl.lastElementChild;
    const contentEl = assistantEl.querySelector('.message-content');

    contentEl.innerHTML = showSources
      ? '<span class="thinking">Retrieving relevant passages…</span>'
      : '<span class="cursor">▌</span>';

    setStreaming(true);

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, show_sources: showSources }),
      });

      if (!res.ok || !res.body) {
        throw new Error(`Request failed (${res.status})`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let fullText = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split('\n');
        buffer = lines.pop();

        for (const line of lines) {
          if (!line.trim()) continue;
          const evt = JSON.parse(line);

          if (evt.type === 'sources') {
            assistantMsg.sources = evt.sources;
            contentEl.innerHTML = '<span class="cursor">▌</span>';
          } else if (evt.type === 'chunk') {
            fullText += evt.text;
            assistantMsg.content = fullText;
            renderMarkdownInto(contentEl, fullText);
            const cursor = document.createElement('span');
            cursor.className = 'cursor';
            cursor.textContent = ' ▌';
            contentEl.appendChild(cursor);
            scrollToBottom();
          } else if (evt.type === 'error') {
            contentEl.innerHTML = `<p class="error-text">${escapeHtml(evt.message)}</p>`;
          } else if (evt.type === 'done') {
            renderMarkdownInto(contentEl, fullText);
          }
        }
      }
    } catch (err) {
      contentEl.innerHTML = `<p class="error-text">${escapeHtml(err.message || 'Something went wrong.')}</p>`;
    } finally {
      setStreaming(false);
      renderAll();
    }
  });

  function setStreaming(state) {
    isStreaming = state;
    inputEl.disabled = state;
    sendBtn.disabled = state;
  }

  renderAll();
})();
