/**
 * EVIL-AI Chat Widget
 * Self-initializing, Shadow DOM-isolated chat widget for the EVIL-AI RAG system.
 * Loads via a single <script> tag. Communicates with the API via SSE over POST (fetch + ReadableStream).
 *
 * Usage:
 *   <script src="https://evil-ai-rag.rahtiapp.fi/static/evil-ai-chat.js" data-api-url="https://evil-ai-rag.rahtiapp.fi"></script>
 *
 * Or set window.EVIL_AI_API_URL before loading the script.
 */
(function () {
  'use strict';

  /* ============================================================
     SVG Icons
     ============================================================ */
  const ICON_CHAT = `<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm0 14H5.17L4 17.17V4h16v12z"/><path d="M7 9h2v2H7zm4 0h2v2h-2zm4 0h2v2h-2z"/></svg>`;
  const ICON_CLOSE = `<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/></svg>`;
  const ICON_SEND = `<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>`;
  const ICON_BOT = `<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M12 2a2 2 0 0 1 2 2c0 .74-.4 1.39-1 1.73V7h1a7 7 0 0 1 7 7h1a1 1 0 0 1 1 1v3a1 1 0 0 1-1 1h-1.07A7.001 7.001 0 0 1 7.07 19H6a1 1 0 0 1-1-1v-3a1 1 0 0 1 1-1h1a7 7 0 0 1 7-7h-1V5.73c-.6-.34-1-.99-1-1.73a2 2 0 0 1 2-2zM7.5 13a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3zm9 0a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3z"/></svg>`;
  const ICON_WARN = `<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z"/></svg>`;

  /* ============================================================
     EvilAIChat Class
     ============================================================ */
  class EvilAIChat {
    constructor() {
      this.apiUrl = this._getApiUrl();
      this.sessionId = this._getSessionId();
      this.history = this._loadHistory();
      this.isOpen = false;
      this.isStreaming = false;
      this._onlineHandler = null;
      this._offlineHandler = null;

      this._createWidget();
      this._attachEvents();
      this._restoreMessages();
    }

    /* ----------------------------------------------------------
       Configuration helpers
       ---------------------------------------------------------- */
    _getApiUrl() {
      const script = document.querySelector('script[src*="evil-ai-chat"]');
      if (script && script.dataset.apiUrl) {
        return script.dataset.apiUrl.replace(/\/+$/, '');
      }
      if (typeof window !== 'undefined' && window.EVIL_AI_API_URL) {
        return window.EVIL_AI_API_URL.replace(/\/+$/, '');
      }
      return 'https://evil-ai-rag.rahtiapp.fi';
    }

    _getSessionId() {
      let id = null;
      try {
        id = sessionStorage.getItem('evil-ai-session');
      } catch (_) { /* private browsing */ }
      if (!id) {
        id = Math.random().toString(36).substring(2, 10) +
             Date.now().toString(36);
        try {
          sessionStorage.setItem('evil-ai-session', id);
        } catch (_) { /* ignore */ }
      }
      return id;
    }

    /* ----------------------------------------------------------
       History (localStorage)
       ---------------------------------------------------------- */
    _loadHistory() {
      try {
        const raw = localStorage.getItem('evil-ai-chat-history');
        if (!raw) return [];
        const parsed = JSON.parse(raw);
        if (!Array.isArray(parsed)) return [];
        return parsed.slice(-50);
      } catch (_) {
        return [];
      }
    }

    _saveHistory() {
      try {
        localStorage.setItem(
          'evil-ai-chat-history',
          JSON.stringify(this.history.slice(-50))
        );
      } catch (_) { /* quota exceeded or private mode */ }
    }

    _clearHistory() {
      this.history = [];
      this._saveHistory();
      if (this.messagesEl) {
        this.messagesEl.innerHTML = '';
        this.messagesEl.appendChild(this.typingIndicator);
        this._showWelcome();
      }
    }

    /* ----------------------------------------------------------
       Widget Creation (Shadow DOM)
       ---------------------------------------------------------- */
    _createWidget() {
      // Host element
      this.container = document.createElement('div');
      this.container.id = 'evil-ai-chat-widget';
      this.shadow = this.container.attachShadow({ mode: 'open' });

      // CSS — load from same origin as JS
      const cssUrl = this.apiUrl + '/static/evil-ai-chat.css';
      const linkEl = document.createElement('link');
      linkEl.rel = 'stylesheet';
      linkEl.href = cssUrl;
      this.shadow.appendChild(linkEl);

      // Build HTML
      const wrapper = document.createElement('div');
      wrapper.className = 'evil-ai-wrapper';
      wrapper.innerHTML = `
        <div class="chat-panel hidden" role="dialog" aria-label="EVIL-AI Chat">
          <div class="chat-header">
            <span class="header-title">
              ${ICON_BOT}
              EVIL-AI Research Assistant
            </span>
            <button class="close-btn" aria-label="Close chat" title="Close">&times;</button>
          </div>
          <div class="offline-banner" role="alert">
            ${ICON_WARN}
            <span>Connection lost. Messages will be sent when reconnected.</span>
          </div>
          <div class="messages-container" aria-live="polite" aria-relevant="additions">
            <div class="typing-indicator hidden">
              <span></span><span></span><span></span>
            </div>
          </div>
          <div class="input-area">
            <textarea placeholder="Ask about EVIL-AI research..." rows="1" aria-label="Type your message"></textarea>
            <button class="send-btn" aria-label="Send message" title="Send">
              ${ICON_SEND}
            </button>
          </div>
          <div class="powered-by">Powered by EVIL-AI RAG</div>
        </div>
        <button class="toggle-btn" aria-label="Open chat assistant" title="Chat with EVIL-AI">
          <span class="icon-chat">${ICON_CHAT}</span>
          <span class="icon-close">${ICON_CLOSE}</span>
          <span class="unread-badge" aria-hidden="true"></span>
        </button>
      `;

      this.shadow.appendChild(wrapper);

      // Cache DOM references
      this.wrapperEl = wrapper;
      this.toggleBtn = wrapper.querySelector('.toggle-btn');
      this.chatPanel = wrapper.querySelector('.chat-panel');
      this.closeBtn = wrapper.querySelector('.close-btn');
      this.messagesEl = wrapper.querySelector('.messages-container');
      this.typingIndicator = wrapper.querySelector('.typing-indicator');
      this.textareaEl = wrapper.querySelector('textarea');
      this.sendBtn = wrapper.querySelector('.send-btn');
      this.offlineBanner = wrapper.querySelector('.offline-banner');
      this.unreadBadge = wrapper.querySelector('.unread-badge');

      document.body.appendChild(this.container);
    }

    /* ----------------------------------------------------------
       Event Binding
       ---------------------------------------------------------- */
    _attachEvents() {
      // Toggle open/close
      this.toggleBtn.addEventListener('click', () => this._togglePanel());
      this.closeBtn.addEventListener('click', () => this._closePanel());

      // Send
      this.sendBtn.addEventListener('click', () => this._handleSend());

      // Keyboard
      this.textareaEl.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
          e.preventDefault();
          this._handleSend();
        }
      });

      // Auto-resize textarea
      this.textareaEl.addEventListener('input', () => this._autoResize());

      // Online / Offline
      this._onlineHandler = () => {
        this.offlineBanner.classList.remove('visible');
      };
      this._offlineHandler = () => {
        this.offlineBanner.classList.add('visible');
      };
      window.addEventListener('online', this._onlineHandler);
      window.addEventListener('offline', this._offlineHandler);

      // Check initial state
      if (!navigator.onLine) {
        this.offlineBanner.classList.add('visible');
      }

      // Close on Escape
      this.shadow.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && this.isOpen) {
          this._closePanel();
        }
      });
    }

    /* ----------------------------------------------------------
       Panel open / close
       ---------------------------------------------------------- */
    _togglePanel() {
      if (this.isOpen) {
        this._closePanel();
      } else {
        this._openPanel();
      }
    }

    _openPanel() {
      this.isOpen = true;
      this.chatPanel.classList.remove('hidden');
      this.chatPanel.classList.add('visible');
      this.toggleBtn.classList.add('open');
      this.unreadBadge.classList.remove('visible');
      this.textareaEl.focus();
      this._scrollToBottom();
    }

    _closePanel() {
      this.isOpen = false;
      this.chatPanel.classList.remove('visible');
      this.chatPanel.classList.add('hidden');
      this.toggleBtn.classList.remove('open');
    }

    /* ----------------------------------------------------------
       Restore previous messages
       ---------------------------------------------------------- */
    _restoreMessages() {
      if (this.history.length === 0) {
        this._showWelcome();
        return;
      }

      for (const msg of this.history) {
        this._appendMessage(msg.role, msg.content, false);
        if (msg.sources && msg.sources.length > 0) {
          this._appendSources(msg.sources, false);
        }
      }
    }

    _showWelcome() {
      // Only show if no messages present (besides typing indicator)
      const existingMessages = this.messagesEl.querySelectorAll('.message');
      if (existingMessages.length > 0) return;

      const welcome = document.createElement('div');
      welcome.className = 'welcome-message';
      welcome.innerHTML = `
        <div class="welcome-icon">${ICON_BOT}</div>
        <h3>Welcome!</h3>
        <p>I'm the EVIL-AI Research Assistant. Ask me anything about the EVIL-AI project's research papers and findings.</p>
      `;
      this.messagesEl.insertBefore(welcome, this.typingIndicator);
    }

    _removeWelcome() {
      const welcome = this.messagesEl.querySelector('.welcome-message');
      if (welcome) welcome.remove();
    }

    /* ----------------------------------------------------------
       Auto-resize textarea
       ---------------------------------------------------------- */
    _autoResize() {
      const el = this.textareaEl;
      el.style.height = 'auto';
      // Clamp to 4 lines max (~96px)
      el.style.height = Math.min(el.scrollHeight, 96) + 'px';
    }

    /* ----------------------------------------------------------
       Send message
       ---------------------------------------------------------- */
    _handleSend() {
      const text = this.textareaEl.value.trim();
      if (!text || this.isStreaming) return;

      this._removeWelcome();

      // Reset textarea
      this.textareaEl.value = '';
      this.textareaEl.style.height = 'auto';

      // Add user message
      this._appendMessage('user', text);
      this.history.push({ role: 'user', content: text });
      this._saveHistory();

      // Send to API
      this._sendMessage(text);
    }

    /* ----------------------------------------------------------
       Append message to UI
       ---------------------------------------------------------- */
    _appendMessage(role, content, animate = true) {
      const msgEl = document.createElement('div');
      msgEl.className = `message ${role}`;
      if (!animate) msgEl.style.animation = 'none';

      const bubble = document.createElement('div');
      bubble.className = 'bubble';

      if (role === 'assistant') {
        bubble.innerHTML = this._renderMarkdown(content);
      } else {
        bubble.textContent = content;
      }

      const timestamp = document.createElement('div');
      timestamp.className = 'timestamp';
      timestamp.textContent = this._formatTime(new Date());

      msgEl.appendChild(bubble);
      msgEl.appendChild(timestamp);

      // Insert before typing indicator
      this.messagesEl.insertBefore(msgEl, this.typingIndicator);
      this._scrollToBottom();

      return msgEl;
    }

    /* ----------------------------------------------------------
       Append error message to UI
       ---------------------------------------------------------- */
    _appendError(text) {
      const msgEl = document.createElement('div');
      msgEl.className = 'message assistant error';

      const bubble = document.createElement('div');
      bubble.className = 'bubble';
      bubble.textContent = text;

      msgEl.appendChild(bubble);
      this.messagesEl.insertBefore(msgEl, this.typingIndicator);
      this._scrollToBottom();
    }

    /* ----------------------------------------------------------
       Update the last bot message (streaming)
       ---------------------------------------------------------- */
    _updateBotMessage(element, content) {
      const bubble = element.querySelector('.bubble');
      if (bubble) {
        bubble.innerHTML = this._renderMarkdown(content);
        this._scrollToBottom();
      }
    }

    /* ----------------------------------------------------------
       Append sources
       ---------------------------------------------------------- */
    _appendSources(sources, animate = true) {
      if (!sources || sources.length === 0) return;

      const details = document.createElement('details');
      details.className = 'sources';
      if (!animate) details.style.animation = 'none';

      const summary = document.createElement('summary');
      summary.textContent = `${sources.length} reference${sources.length !== 1 ? 's' : ''}`;
      details.appendChild(summary);

      const list = document.createElement('div');
      list.className = 'sources-list';

      for (const src of sources) {
        const item = document.createElement('div');
        item.className = 'source-item';

        const filename = src.filename || src.source || 'Unknown source';
        const page = src.page != null ? `Page ${src.page}` : '';
        const section = src.section ? `Section: ${src.section}` : '';
        const meta = [page, section].filter(Boolean).join(', ');
        const snippet = src.snippet || src.content || '';

        item.innerHTML = `
          <strong>${this._escapeHtml(filename)}</strong>
          ${meta ? `<span class="source-meta">${this._escapeHtml(meta)}</span>` : ''}
          ${snippet ? `<div class="source-snippet">${this._escapeHtml(snippet)}</div>` : ''}
        `;
        list.appendChild(item);
      }

      details.appendChild(list);
      this.messagesEl.insertBefore(details, this.typingIndicator);
      this._scrollToBottom();
    }

    /* ----------------------------------------------------------
       Typing indicator
       ---------------------------------------------------------- */
    _showTypingIndicator() {
      this.typingIndicator.classList.remove('hidden');
      this._scrollToBottom();
    }

    _hideTypingIndicator() {
      this.typingIndicator.classList.add('hidden');
    }

    /* ----------------------------------------------------------
       Send message to API (SSE via POST + ReadableStream)
       ---------------------------------------------------------- */
    async _sendMessage(text) {
      this.isStreaming = true;
      this.sendBtn.disabled = true;
      this._showTypingIndicator();

      const body = {
        question: text,
        session_id: this.sessionId,
        history: this.history.slice(-10).map(h => ({ role: h.role, content: h.content }))
      };

      let botMessageEl = null;
      let fullResponse = '';
      let sources = [];

      try {
        const response = await fetch(this.apiUrl + '/api/v1/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body)
        });

        if (!response.ok) {
          const errText = await response.text().catch(() => 'Unknown error');
          throw new Error(`Server error (${response.status}): ${errText}`);
        }

        // Check if it's a streaming response
        const contentType = response.headers.get('content-type') || '';

        if (contentType.includes('text/event-stream') || contentType.includes('text/plain')) {
          // SSE streaming response
          const reader = response.body.getReader();
          const decoder = new TextDecoder();
          let buffer = '';

          while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
              const trimmed = line.trim();
              if (!trimmed || !trimmed.startsWith('data: ')) continue;

              const dataStr = trimmed.slice(6);
              if (dataStr === '[DONE]') {
                this.isStreaming = false;
                this._hideTypingIndicator();
                continue;
              }

              let data;
              try {
                data = JSON.parse(dataStr);
              } catch (_) {
                continue;
              }

              if (data.type === 'token') {
                // First token — create bot message bubble and hide typing
                if (!botMessageEl) {
                  this._hideTypingIndicator();
                  botMessageEl = this._appendMessage('assistant', '');
                }
                fullResponse += data.content;
                this._updateBotMessage(botMessageEl, fullResponse);

              } else if (data.type === 'sources') {
                sources = data.data || data.sources || [];
                this._appendSources(sources);

              } else if (data.type === 'done') {
                this.isStreaming = false;
                this._hideTypingIndicator();

              } else if (data.type === 'error') {
                throw new Error(data.message || data.content || 'Stream error');
              }
            }
          }

          // Process any remaining buffer
          if (buffer.trim().startsWith('data: ')) {
            const dataStr = buffer.trim().slice(6);
            if (dataStr !== '[DONE]') {
              try {
                const data = JSON.parse(dataStr);
                if (data.type === 'token') {
                  if (!botMessageEl) {
                    this._hideTypingIndicator();
                    botMessageEl = this._appendMessage('assistant', '');
                  }
                  fullResponse += data.content;
                  this._updateBotMessage(botMessageEl, fullResponse);
                } else if (data.type === 'sources') {
                  sources = data.data || data.sources || [];
                  this._appendSources(sources);
                }
              } catch (_) { /* ignore parse error on final chunk */ }
            }
          }

        } else {
          // JSON response (non-streaming fallback)
          const data = await response.json();
          this._hideTypingIndicator();
          fullResponse = data.answer || data.response || data.content || JSON.stringify(data);
          botMessageEl = this._appendMessage('assistant', fullResponse);

          sources = data.sources || data.references || [];
          if (sources.length > 0) {
            this._appendSources(sources);
          }
        }

        // Save assistant response to history
        if (fullResponse) {
          const historyEntry = { role: 'assistant', content: fullResponse };
          if (sources.length > 0) historyEntry.sources = sources;
          this.history.push(historyEntry);
          this._saveHistory();
        }

        // Show unread badge if panel is closed
        if (!this.isOpen) {
          this.unreadBadge.classList.add('visible');
        }

      } catch (err) {
        console.error('[EVIL-AI Chat] Error:', err);
        this._hideTypingIndicator();
        this._appendError(
          `Sorry, something went wrong: ${err.message || 'Network error'}. Please try again.`
        );
      } finally {
        this.isStreaming = false;
        this.sendBtn.disabled = false;
        this._hideTypingIndicator();
        this.textareaEl.focus();
      }
    }

    /* ----------------------------------------------------------
       Minimal Markdown renderer
       ---------------------------------------------------------- */
    _renderMarkdown(text) {
      if (!text) return '';

      let html = this._escapeHtml(text);

      // Code blocks (``` ... ```)
      html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
        return `<pre><code>${code.trim()}</code></pre>`;
      });

      // Inline code
      html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

      // Bold
      html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

      // Italic
      html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

      // Links
      html = html.replace(
        /\[([^\]]+)\]\((https?:\/\/[^\)]+)\)/g,
        '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>'
      );

      // Unordered lists
      html = html.replace(/^[-*] (.+)$/gm, '<li>$1</li>');
      html = html.replace(/((?:<li>.*<\/li>\n?)+)/g, '<ul>$1</ul>');

      // Mark ordered list items with a data attribute to distinguish from unordered
      html = html.replace(/^\d+\.\s(.+)$/gm, '<li class="ol-item">$1</li>');
      html = html.replace(/((?:<li class="ol-item">.*?<\/li>\s*)+)/gs, '<ol>$1</ol>');
      // Clean up the class after wrapping
      html = html.replace(/<li class="ol-item">/g, '<li>');

      // Line breaks -> paragraphs
      html = html.replace(/\n\n/g, '</p><p>');
      html = html.replace(/\n/g, '<br>');

      // Wrap in paragraph if not already wrapped in a block element
      if (!html.startsWith('<')) {
        html = `<p>${html}</p>`;
      }

      return html;
    }

    /* ----------------------------------------------------------
       Utility helpers
       ---------------------------------------------------------- */
    _escapeHtml(str) {
      const div = document.createElement('div');
      div.textContent = str;
      return div.innerHTML;
    }

    _formatTime(date) {
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }

    _scrollToBottom() {
      requestAnimationFrame(() => {
        this.messagesEl.scrollTop = this.messagesEl.scrollHeight;
      });
    }
  }

  /* ============================================================
     Auto-initialize
     ============================================================ */
  function init() {
    // Avoid double initialization
    if (document.getElementById('evil-ai-chat-widget')) return;
    new EvilAIChat();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
