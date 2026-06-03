(function(){
  const dataTag = document.getElementById('war-room-data');
  const root = document.querySelector('[data-war-root]');
  if(!dataTag || !root) return;

  let payload = {};
  try { payload = JSON.parse(dataTag.textContent || '{}'); } catch (_) { payload = {}; }

  const allMessages = Array.isArray(payload.live_messages) ? payload.live_messages : (payload.feed || []);
  let messages = allMessages.slice();
  const stream = document.querySelector('[data-chat-stream]');
  const clock = document.querySelector('[data-war-clock]');
  const typingPanel = document.querySelector('[data-typing-panel]');
  const typingAvatar = document.querySelector('[data-typing-avatar]');
  const typingName = document.querySelector('[data-typing-name]');
  const typingState = document.querySelector('[data-typing-state]');
  const countdown = document.querySelector('[data-next-countdown]');
  const visibleCount = document.querySelector('[data-visible-count]');
  const queueProgress = document.querySelector('[data-queue-progress]');
  const speedButtons = document.querySelectorAll('[data-speed]');
  const revealAllButton = document.querySelector('[data-reveal-all]');
  const agentButtons = document.querySelectorAll('[data-agent-filter]');
  const sessionTabs = document.querySelectorAll('[data-session-filter]');

  let visible = 0;
  let mode = 'live';
  let timer = null;
  let nextAt = 0;
  let activeAgent = 'all';
  let activeSession = 'all';

  function escapeHtml(str){
    return String(str || '').replace(/[&<>'"]/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[ch]));
  }

  function tickClock(){
    if(!clock) return;
    const now = new Date();
    clock.textContent = `JST ${now.toLocaleTimeString('en-GB',{timeZone:'Asia/Tokyo',hour12:false})}`;
  }

  function delayFor(index){
    const msg = messages[index] || {};
    if(mode === 'demo') return 3600 + Math.floor(Math.random() * 2200);
    return Math.max(1000, Number(msg.delay_seconds || 210) * 1000);
  }

  function updateQueue(){
    if(visibleCount) visibleCount.textContent = String(visible);
    if(queueProgress){
      const pct = messages.length ? Math.min(100, (visible / messages.length) * 100) : 0;
      queueProgress.style.width = `${pct}%`;
    }
  }

  function updateTyping(nextMsg){
    if(!typingPanel || !nextMsg){
      if(typingPanel) typingPanel.classList.add('is-done');
      return;
    }
    typingPanel.classList.remove('is-done');
    typingPanel.style.setProperty('--agent-color', nextMsg.color || '#7DF9FF');
    if(typingAvatar) typingAvatar.src = nextMsg.avatar_image || '/assets/ai-arena/agents/daily_striker.png';
    if(typingName) typingName.textContent = nextMsg.agent_name || 'AGENT';
    if(typingState) typingState.textContent = `${nextMsg.state || 'thinking'}...`;
  }

  function renderEvidenceNumbers(msg){
    const nums = Array.isArray(msg.evidence_numbers) ? msg.evidence_numbers.filter(Boolean).slice(0, 4) : [];
    if(!nums.length) return '';
    return `<div class="evidence-numbers">${nums.map(x => `<span>${escapeHtml(x)}</span>`).join('')}</div>`;
  }

  function renderMessage(msg){
    if(!stream) return;
    const symbol = msg.linked_symbol ? `<em>${escapeHtml(msg.linked_symbol)}</em>` : '';
    const linkedName = msg.linked_name ? `<small>${escapeHtml(msg.linked_name)}</small>` : '';
    const reply = msg.reply_to_agent ? `<span class="reply-chip">Replying to ${escapeHtml(msg.reply_to_agent)}</span>` : '';
    const evidence = msg.evidence_label ? `<span class="chat-evidence">${escapeHtml(msg.evidence_label)}</span>` : '';
    const why = msg.why_it_matters ? `<div class="why-it-matters"><b>Why it matters</b>${escapeHtml(msg.why_it_matters)}</div>` : '';
    const node = document.createElement('article');
    node.className = 'live-chat-message';
    node.dataset.agentId = msg.agent_id || '';
    node.dataset.sessionId = msg.session_id || '';
    node.style.setProperty('--agent-color', msg.color || '#7DF9FF');
    node.innerHTML = `
      <img class="chat-avatar" src="${escapeHtml(msg.avatar_image || '')}" alt="${escapeHtml(msg.agent_name || 'Agent')} avatar" loading="lazy" />
      <div class="chat-bubble">
        <div class="chat-meta">
          <strong>${escapeHtml(msg.agent_name || 'AGENT')}</strong>
          <span>${escapeHtml(msg.state || 'THINKING')}</span>
          <i>${escapeHtml(String(msg.message_type || 'evidence').replace(/_/g, ' '))}</i>
          ${symbol}
          ${linkedName}
        </div>
        ${reply}
        <p>${escapeHtml(msg.body || '')}</p>
        ${renderEvidenceNumbers(msg)}
        ${why}
        <div class="chat-foot">
          ${evidence}
          <small>${escapeHtml(new Date().toLocaleTimeString('en-GB',{timeZone:'Asia/Tokyo',hour12:false}))} JST</small>
        </div>
      </div>`;
    stream.appendChild(node);
    requestAnimationFrame(() => node.classList.add('is-visible'));
    stream.scrollTo({ top: stream.scrollHeight, behavior: 'smooth' });
  }

  function applyAgentFilter(){
    document.querySelectorAll('.live-chat-message').forEach(node => {
      const okAgent = activeAgent === 'all' || node.dataset.agentId === activeAgent;
      const okSession = activeSession === 'all' || node.dataset.sessionId === activeSession;
      node.classList.toggle('is-muted-by-filter', !(okAgent && okSession));
    });
  }

  function revealNext(){
    if(visible >= messages.length){
      updateTyping(null);
      updateQueue();
      if(countdown) countdown.textContent = 'All queued thoughts are visible.';
      return;
    }
    const msg = messages[visible];
    visible += 1;
    renderMessage(msg);
    updateQueue();
    applyAgentFilter();
    scheduleNext();
  }

  function scheduleNext(){
    if(timer) clearTimeout(timer);
    if(visible >= messages.length){
      updateTyping(null);
      return;
    }
    const next = messages[visible];
    const delay = delayFor(visible);
    nextAt = Date.now() + delay;
    updateTyping(next);
    timer = setTimeout(revealNext, delay);
  }

  function tickCountdown(){
    if(!countdown || visible >= messages.length) return;
    const remain = Math.max(0, Math.ceil((nextAt - Date.now()) / 1000));
    const m = Math.floor(remain / 60);
    const s = String(remain % 60).padStart(2, '0');
    countdown.textContent = mode === 'demo' ? `Fast preview: next thought in ${remain}s` : `Next live thought in ${m}:${s}`;
  }

  function resetStream(newMessages){
    if(timer) clearTimeout(timer);
    messages = newMessages.slice();
    visible = 0;
    if(stream) stream.innerHTML = '';
    updateQueue();
    if(messages.length) revealNext();
    else if(stream) stream.innerHTML = '<p class="empty-chat">No GPT-4o conversation has been generated for this filter.</p>';
  }

  speedButtons.forEach(button => {
    button.addEventListener('click', () => {
      speedButtons.forEach(x => x.classList.remove('is-active'));
      button.classList.add('is-active');
      mode = button.dataset.speed || 'live';
      scheduleNext();
    });
  });

  if(revealAllButton){
    revealAllButton.addEventListener('click', () => {
      if(timer) clearTimeout(timer);
      while(visible < messages.length) revealNext();
    });
  }

  agentButtons.forEach(button => {
    button.addEventListener('click', () => {
      const wasActive = button.classList.contains('is-active');
      agentButtons.forEach(x => x.classList.remove('is-active'));
      if(wasActive){ activeAgent = 'all'; }
      else { button.classList.add('is-active'); activeAgent = button.dataset.agentFilter || 'all'; }
      applyAgentFilter();
    });
  });

  sessionTabs.forEach(button => {
    button.addEventListener('click', () => {
      sessionTabs.forEach(x => x.classList.remove('is-active'));
      button.classList.add('is-active');
      activeSession = button.dataset.sessionFilter || 'all';
      const nextMessages = activeSession === 'all' ? allMessages : allMessages.filter(m => m.session_id === activeSession);
      resetStream(nextMessages);
    });
  });

  const tape = document.querySelector('.pulse-track');
  if(tape) tape.innerHTML = `${tape.innerHTML}${tape.innerHTML}`;

  tickClock();
  setInterval(tickClock, 1000);
  setInterval(tickCountdown, 1000);

  if(messages.length){
    revealNext();
  } else if(stream){
    stream.innerHTML = '<p class="empty-chat">No GPT-4o conversation has been generated yet.</p>';
  }
})();
