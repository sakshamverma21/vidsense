/* VidSense customer workspace */
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const esc = (value = '') => String(value).replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[char]));

let activeCorpusId = localStorage.getItem('vidsense.activeCorpusId') || 'demo';
let activeCorpus = null;
let polling = null;
let streaming = false;
let activeView = 'workspace';

const modeCopy = {
  question: { kicker: 'Answer', title: 'Here’s what I found', loading: 'Looking through your videos' },
  summary: { kicker: 'Collection overview', title: 'The big picture', loading: 'Building an overview from your videos' },
  highlights: { kicker: 'Highlights', title: 'Moments worth your time', loading: 'Finding the strongest moments' },
};

function safeMarkdown(text) {
  return marked.parse(esc(text || ''), { headerIds: false, mangle: false });
}

async function api(url, options) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || data.error || 'Something went wrong. Please try again.');
  return data;
}

function statusLabel(corpus) {
  if (corpus.status === 'processing') return 'Preparing collection';
  if (corpus.status === 'partial') return 'Ready with notes';
  if (corpus.status === 'error') return 'Needs attention';
  return 'Ready to explore';
}

function progressLabel(progress = {}) {
  const { phase, current, total } = progress;
  if (phase === 'queued') return 'Your collection is queued.';
  if (phase === 'resolving_sources') return 'Checking your YouTube links…';
  if (phase === 'sources_resolved') return `Found ${total || current || 0} videos. Preparing captions…`;
  if (phase === 'fetching_transcript') return `Preparing captions for video ${current || 0} of ${total || 0}…`;
  if (phase === 'building_index') return 'Organizing the collection for questions…';
  return '';
}

function renderNotice(corpus) {
  const notices = [$('#collection-notice'), $('#library-notice')];
  let message = '';
  let kind = '';
  if (corpus.status === 'processing') {
    message = progressLabel(corpus.progress) || 'Preparing your collection…';
  } else if (corpus.status === 'error') {
    message = corpus.error || 'We could not prepare this collection. Check the source links and try again.';
    kind = 'error';
  } else if ((corpus.warnings || []).length || (corpus.skipped || []).length) {
    const parts = [...(corpus.warnings || [])];
    if (corpus.skipped?.length) parts.push(`${corpus.skipped.length} video${corpus.skipped.length === 1 ? '' : 's'} could not be used because captions were unavailable.`);
    message = parts.join(' ');
    kind = 'warning';
  }
  notices.forEach((notice) => {
    notice.textContent = message;
    notice.className = `notice ${kind}`;
    notice.classList.toggle('hidden', !message);
  });
}

function thumb(videoId) {
  return `https://img.youtube.com/vi/${encodeURIComponent(videoId)}/mqdefault.jpg`;
}

function renderSidebarSources(videos) {
  const target = $('#source-list');
  if (!videos.length) {
    target.innerHTML = '<p class="source-list-empty">No ready videos yet.</p>';
    return;
  }
  const sourceRow = (video) => `
    <a class="source-mini" href="${esc(video.youtube_link)}" target="_blank" rel="noopener">
      <img src="${thumb(video.video_id)}" alt="" loading="lazy" />
      <span class="source-mini-text"><span class="source-mini-title">${esc(video.title)}</span><span class="source-mini-meta">${esc(video.duration)}</span></span>
    </a>`;
  const visible = videos.slice(0, 4);
  const hidden = videos.slice(4);
  target.innerHTML = visible.map(sourceRow).join('') + (hidden.length ? `
    <details class="source-more"><summary>Show ${hidden.length} more video${hidden.length === 1 ? '' : 's'}</summary>
      <div class="source-more-list">${hidden.map(sourceRow).join('')}</div>
    </details>` : '');
}

function renderLibrary(videos) {
  const target = $('#video-grid');
  if (!videos.length) {
    target.innerHTML = '<div class="empty-citations">Videos will appear here once their captions are ready.</div>';
    return;
  }
  target.innerHTML = videos.map((video) => video.status === 'skipped' ? `
    <article class="video-card video-card-skipped">
      <div class="video-card-body">
        <div class="video-card-title">${esc(video.title || video.video_id)}</div>
        <div class="video-card-meta">Captions unavailable</div>
        <p class="video-card-reason">${esc(video.reason || 'This video could not be added to the collection.')}</p>
      </div>
    </article>` : `
    <article class="video-card">
      <img src="${thumb(video.video_id)}" alt="" loading="lazy" />
      <div class="video-card-body">
        <div class="video-card-title">${esc(video.title)}</div>
        <div class="video-card-meta">${esc(video.duration)}${video.source_kind === 'playlist' ? ' · Playlist' : ''}</div>
        <a href="${esc(video.youtube_link)}" target="_blank" rel="noopener">Open on YouTube</a>
      </div>
    </article>`).join('');
}

function renderCorpus(corpus) {
  activeCorpus = corpus;
  const ready = ['ready', 'partial'].includes(corpus.status);
  $('#collection-name').textContent = corpus.name;
  const status = $('#collection-status');
  status.className = `collection-status ${corpus.status}`;
  status.lastElementChild.textContent = statusLabel(corpus);
  $('#collection-count').textContent = ready
    ? `${corpus.video_count} video${corpus.video_count === 1 ? '' : 's'} in this collection`
    : progressLabel(corpus.progress) || 'This may take a moment.';
  renderSidebarSources(corpus.videos || []);
  renderLibrary(corpus.video_statuses || corpus.videos || []);
  renderNotice(corpus);

  const showWorkspace = ready || corpus.status === 'processing';
  $('#welcome').classList.toggle('hidden', showWorkspace);
  $('#workspace').classList.toggle('hidden', !showWorkspace);
  $('#workspace-title').textContent = ready ? 'What would you like to know?' : 'We’re getting your collection ready';
  $('#question').disabled = !ready;
  $$('.action-button, #ask').forEach((button) => { button.disabled = !ready || streaming; });
  if (!ready && corpus.status === 'error') $('#welcome').classList.remove('hidden');
}

function stopPolling() {
  if (polling) window.clearTimeout(polling);
  polling = null;
}

async function loadCorpus() {
  try {
    const corpus = await api(`/api/corpora/${encodeURIComponent(activeCorpusId)}`);
    renderCorpus(corpus);
    if (corpus.status === 'processing') {
      stopPolling();
      polling = window.setTimeout(loadCorpus, 1200);
    } else {
      stopPolling();
    }
  } catch (error) {
    if (activeCorpusId !== 'demo') {
      activeCorpusId = 'demo';
      localStorage.removeItem('vidsense.activeCorpusId');
      return loadCorpus();
    }
    renderCorpus({ corpus_id: 'demo', name: 'Starter collection', status: 'error', videos: [], video_count: 0, error: error.message });
  }
}

function setView(view) {
  activeView = view;
  $('#view-workspace').classList.toggle('hidden', view !== 'workspace');
  $('#view-library').classList.toggle('hidden', view !== 'library');
  $$('.nav-item').forEach((button) => button.classList.toggle('nav-item-active', button.dataset.view === view));
  $('#sidebar').classList.remove('open');
}

function openModal() {
  $('#source-modal').classList.remove('hidden');
  $('#source-error').classList.add('hidden');
  window.setTimeout(() => $('#source-input').focus(), 0);
}
function closeModal() { $('#source-modal').classList.add('hidden'); }

async function createCollection() {
  const raw = $('#source-input').value.trim();
  const error = $('#source-error');
  if (!raw) {
    error.textContent = 'Paste at least one YouTube video or playlist link.';
    error.classList.remove('hidden');
    return;
  }
  const button = $('#create-source');
  button.disabled = true;
  button.textContent = 'Starting…';
  try {
    const result = await api('/api/corpora', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name: $('#collection-input').value.trim() || 'My video collection',
        inputs: raw.split(/\r?\n/),
      }),
    });
    activeCorpusId = result.corpus_id;
    localStorage.setItem('vidsense.activeCorpusId', activeCorpusId);
    $('#source-input').value = '';
    $('#collection-input').value = '';
    closeModal();
    setView('workspace');
    renderCorpus(result);
    loadCorpus();
  } catch (err) {
    error.textContent = err.message;
    error.classList.remove('hidden');
  } finally {
    button.disabled = false;
    button.textContent = 'Build collection';
  }
}

function resetResult(mode) {
  const copy = modeCopy[mode];
  $('#result').classList.remove('hidden');
  $('#result-kicker').textContent = copy.kicker;
  $('#result-title').textContent = copy.title;
  $('#answer').innerHTML = '<p class="muted-answer">Your collection is being reviewed. This can take a moment for larger playlists.</p>';
  $('#citations').innerHTML = '';
  $('#trajectory').innerHTML = '';
  $('#researching').classList.remove('hidden');
}

function renderCitations(citations) {
  const target = $('#citations');
  if (!citations?.length) {
    target.innerHTML = '<div class="empty-citations">No transcript moments were returned for this response.</div>';
    return;
  }
  target.innerHTML = citations.map((citation) => `
    <a class="citation-card" href="${esc(citation.youtube_link)}" target="_blank" rel="noopener">
      <img src="${thumb(citation.video_id)}" alt="" loading="lazy" />
      <span class="citation-content">
        <span class="citation-time">${esc(citation.formatted_time)}</span>
        <span class="citation-title">${esc(citation.title)}</span>
        <span class="citation-snippet">${esc((citation.text || '').replace(/\s+/g, ' ').slice(0, 160))}</span>
      </span>
    </a>`).join('');
}

function addTrajectoryStep(record) {
  const detail = record.type === 'tool'
    ? 'Reviewed relevant transcript passages.'
    : record.type === 'synthesize'
      ? 'Prepared a cited response.'
      : record.type === 'verify'
        ? 'Checked the response against its evidence.'
        : 'Planned the next research step.';
  const item = document.createElement('li');
  item.textContent = detail;
  $('#trajectory').appendChild(item);
}

function finishResearch(eventSource) {
  if (eventSource) eventSource.close();
  streaming = false;
  $('#researching').classList.add('hidden');
  $$('.action-button, #ask').forEach((button) => { button.disabled = !['ready', 'partial'].includes(activeCorpus?.status); });
}

function runResearch(mode) {
  if (streaming || !['ready', 'partial'].includes(activeCorpus?.status)) return;
  const question = $('#question').value.trim();
  if (mode === 'question' && !question) {
    $('#question').focus();
    return;
  }
  streaming = true;
  $$('.action-button, #ask').forEach((button) => { button.disabled = true; });
  resetResult(mode);
  const params = new URLSearchParams({ corpus_id: activeCorpusId, mode, question: mode === 'question' ? question : '' });
  const stream = new EventSource(`/api/research/stream?${params.toString()}`);
  let complete = false;
  stream.addEventListener('step', (event) => addTrajectoryStep(JSON.parse(event.data)));
  stream.addEventListener('answer', (event) => {
    const answer = JSON.parse(event.data).answer || '';
    $('#answer').innerHTML = safeMarkdown(answer);
  });
  stream.addEventListener('done', (event) => {
    const result = JSON.parse(event.data);
    $('#answer').innerHTML = safeMarkdown(result.answer || 'No answer was returned.');
    renderCitations(result.citations);
    complete = true;
    finishResearch(stream);
  });
  stream.addEventListener('error', (event) => {
    if (complete) return;
    let message = 'We could not complete this request. Please try again.';
    try { message = JSON.parse(event.data).message || message; } catch (_) { /* native network event */ }
    $('#answer').innerHTML = `<p class="error-answer">${esc(message)}</p>`;
    finishResearch(stream);
  });
  stream.onerror = () => { if (!complete && streaming) finishResearch(stream); };
}

$$('[data-view]').forEach((button) => button.addEventListener('click', () => setView(button.dataset.view)));
['#open-source-modal', '#top-add-source', '#welcome-add-source', '#workspace-add-source', '#library-add-source'].forEach((selector) => $(selector).addEventListener('click', openModal));
$('#close-source-modal').addEventListener('click', closeModal);
$('#cancel-source').addEventListener('click', closeModal);
$('#source-modal').addEventListener('click', (event) => { if (event.target.id === 'source-modal') closeModal(); });
$('#create-source').addEventListener('click', createCollection);
$('#open-sidebar').addEventListener('click', () => $('#sidebar').classList.add('open'));
$('#close-sidebar').addEventListener('click', () => $('#sidebar').classList.remove('open'));
$('#ask').addEventListener('click', () => runResearch('question'));
$$('.action-button').forEach((button) => button.addEventListener('click', () => runResearch(button.dataset.mode)));
$('#question').addEventListener('keydown', (event) => { if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) runResearch('question'); });

loadCorpus();
