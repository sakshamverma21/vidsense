/* VidSense customer workspace. Account cookies are managed only by the server. */
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const esc = (value = '') => String(value ?? '').replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));

// Consume emailed tokens before any requests. Keep tokens only in memory.
let emailAction = null;
if (/^#(?:verify|reset)(?:=|&|$)/.test(window.location.hash)) {
  const fragment = window.location.hash.slice(1);
  window.history.replaceState(window.history.state, '', window.location.pathname + window.location.search);
  const params = new URLSearchParams(fragment);
  const mode = params.has('reset') ? 'reset' : 'verify';
  emailAction = { mode, token: params.get(mode) || '' };
}

let auth = { user: null, enabled: false, registration_available: false };
let authLoading = true;
let sessionError = '';
let authRevision = 0;
let sessionController = null;
let authController = null;
let authMode = 'login';
let authBusy = false;
let authComplete = false;
let signingOut = false;
let logoutFailed = false;
let collections = [];
let listState = 'idle';
let listError = '';
let listRevision = 0;
let activeCorpusId = 'demo';
let activeCorpus = null;
let activeView = 'workspace';
let generation = 0;
let contextController = new AbortController();
let researchController = null;
let polling = null;
let streaming = false;
let sourceBusy = false;
let retryBusy = false;
let deleteBusy = false;
let pendingDelete = null;

const modeCopy = {
  question: { kicker: 'Answer', title: 'Here’s what I found', loading: 'Looking through your videos' },
  summary: { kicker: 'Collection overview', title: 'The big picture', loading: 'Building an overview from your videos' },
  highlights: { kicker: 'Highlights', title: 'Moments worth your time', loading: 'Finding the strongest moments' },
};
const authCopy = {
  login: ['Welcome back', 'Sign in to open your saved collections and add YouTube sources.', 'Sign in'],
  register: ['Create your account', 'We’ll email a link to verify your address before you can import videos.', 'Create account'],
  forgot: ['Reset your password', 'Enter your email address. If the account is eligible, we’ll send a password reset link.', 'Send reset link'],
  resend: ['Verify your email', 'Enter your email address. If verification is needed, we’ll send a fresh link.', 'Send verification link'],
  verify: ['Verify your email', 'Use the button below to confirm the email address from your link.', 'Verify email'],
  reset: ['Choose a new password', 'Set a new password for the account from your emailed link.', 'Save new password'],
  account: ['Your account', 'Your collections are saved to your account.', ''],
};

function messageText(data, fallback = 'Something went wrong. Please try again.') {
  if (typeof data?.detail === 'string') return data.detail;
  if (Array.isArray(data?.detail)) return data.detail.map((item) => item.msg || 'Check your input.').join(' ');
  return [data?.error, data?.message].find((value) => typeof value === 'string' && value) || fallback;
}

function errorText(error) {
  return error instanceof TypeError ? 'Could not reach VidSense. Check your connection and try again.' : error.message || 'Something went wrong. Please try again.';
}

async function request(url, options = {}) {
  const method = (options.method || 'GET').toUpperCase();
  const headers = new Headers(options.headers);
  const mutation = !['GET', 'HEAD'].includes(method);
  if (mutation) {
    headers.set('X-VidSense-Request', '1');
    headers.set('Content-Type', 'application/json');
  }
  const response = await fetch(url, {
    ...options, method, headers, credentials: 'same-origin', cache: 'no-store',
    ...(mutation ? { body: options.body ?? '{}' } : {}),
  });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    const error = new Error(messageText(data));
    error.status = response.status;
    throw error;
  }
  return response;
}

async function api(url, options) {
  const response = await request(url, options);
  return response.json();
}

function safeMarkdown(text) {
  // Preserve source HTML as text, including inside code, then sanitize the parsed DOM.
  const source = String(text || '');
  let html = `<p>${esc(source).replace(/\n/g, '<br>')}</p>`;
  if (typeof marked !== 'undefined') {
    const renderer = new marked.Renderer();
    renderer.html = (raw) => esc(raw);
    renderer.image = (_href, _title, alt) => esc(alt);
    html = marked.parse(source, { renderer, headerIds: false, mangle: false });
  }
  const parsed = new DOMParser().parseFromString(html, 'text/html');
  const allowed = new Set(['P', 'BR', 'STRONG', 'EM', 'DEL', 'BLOCKQUOTE', 'UL', 'OL', 'LI', 'PRE', 'CODE', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'HR', 'TABLE', 'THEAD', 'TBODY', 'TR', 'TH', 'TD', 'A']);
  const clean = (node) => {
    if (node.nodeType === 3) return document.createTextNode(node.textContent);
    if (node.nodeType !== 1) return document.createDocumentFragment();
    const children = document.createDocumentFragment();
    [...node.childNodes].forEach((child) => children.appendChild(clean(child)));
    if (!allowed.has(node.tagName)) {
      if (node.tagName === 'IMG') children.appendChild(document.createTextNode(node.getAttribute('alt') || ''));
      return children;
    }
    const element = document.createElement(node.tagName.toLowerCase());
    if (node.tagName === 'A') {
      let url;
      try { url = new URL(node.getAttribute('href')); } catch (_) { return children; }
      if (!['http:', 'https:'].includes(url.protocol)) return children;
      element.setAttribute('href', url.href);
      element.setAttribute('target', '_blank');
      element.setAttribute('rel', 'noopener noreferrer');
      if (node.hasAttribute('title')) element.setAttribute('title', node.getAttribute('title'));
    }
    element.appendChild(children);
    return element;
  };
  const container = document.createElement('div');
  [...parsed.body.childNodes].forEach((node) => container.appendChild(clean(node)));
  return container.innerHTML;
}

function youtubeURL(videoId, timestamp = 0) {
  if (!/^[\w-]{11}$/.test(String(videoId || ''))) return '';
  const seconds = Number(timestamp);
  const time = Number.isFinite(seconds) ? Math.max(0, Math.floor(seconds)) : 0;
  return `https://www.youtube.com/watch?v=${videoId}${time ? `&t=${time}s` : ''}`;
}

function thumbnail(videoId) {
  return youtubeURL(videoId) ? `<img src="https://img.youtube.com/vi/${esc(videoId)}/mqdefault.jpg" alt="" loading="lazy" />` : '';
}

function videoLink(video, content, className = '', timestamp = 0) {
  const url = youtubeURL(video.video_id, timestamp);
  return url ? `<a class="${className}" href="${esc(url)}" target="_blank" rel="noopener noreferrer">${content}</a>` : `<div class="${className}">${content}</div>`;
}

function statusLabel(corpus) {
  if (corpus.status === 'deleting') return 'Deletion needs attention';
  if (corpus.status === 'processing') return corpus.progress?.phase === 'restoring_index' ? 'Reopening collection' : 'Preparing collection';
  if (corpus.status === 'partial') return 'Ready with notes';
  if (corpus.status === 'error') return 'Needs attention';
  return 'Ready to explore';
}

function progressLabel(progress = {}) {
  const { phase, current, total } = progress;
  if (phase === 'restoring_index') return 'Reopening your saved collection and rebuilding its search index…';
  if (phase === 'queued') return 'Your collection is queued.';
  if (phase === 'loading_demo') return 'Opening the starter collection…';
  if (phase === 'resolving_sources') return 'Checking your YouTube links…';
  if (phase === 'sources_resolved') return `Found ${total || current || 0} videos. Preparing captions…`;
  if (phase === 'fetching_transcript') return `Preparing captions for video ${current || 0} of ${total || 0}…`;
  if (phase === 'building_index') return 'Organizing the collection for questions…';
  return '';
}

function showMessage(selector, message, kind = '') {
  const target = $(selector);
  target.textContent = message;
  target.classList.toggle('hidden', !message);
  target.classList.toggle('error', kind === 'error');
  target.classList.toggle('warning', kind === 'warning');
}

function sourceButtons() {
  return ['#open-source-modal', '#top-add-source', '#welcome-add-source', '#workspace-add-source', '#library-add-source'].map($);
}

function renderAccount() {
  const user = auth.user;
  const unavailable = authLoading || !!sessionError || !auth.enabled || signingOut || logoutFailed;
  let summary = 'Sign in to save your collections.';
  let notice = 'The starter demo is public. Sign in and verify your email to add your own sources.';
  if (authLoading) summary = notice = 'Checking account availability. You can explore the starter demo.';
  else if (signingOut) summary = notice = 'Signing out. Your workspace has been cleared.';
  else if (logoutFailed) summary = notice = 'Your workspace is cleared, but sign-out could not be confirmed. Retry sign out to finish.';
  else if (sessionError) summary = notice = sessionError;
  else if (!auth.enabled) summary = notice = 'The starter demo is available. Imports are currently unavailable because accounts are not configured.';
  else if (user) {
    summary = `${user.name || user.email} · ${user.email_verified ? 'Email verified' : 'Email not verified'}`;
    notice = user.email_verified ? '' : 'Verify your email before adding sources. Open your account to resend the email or check verification status.';
  }
  if (!unavailable && !auth.registration_available) notice += `${notice ? ' ' : ''}Email setup is pending. New accounts and verification or password reset emails are currently unavailable. Existing users can sign in.`;
  $('#account-summary').textContent = summary;
  $('#sidebar-note').textContent = auth.enabled && !sessionError
    ? user ? 'Your personal collections are saved to your account. The starter collection is public.' : 'Explore the public starter collection, or sign in to keep your own video collections together.'
    : 'The public starter collection is available to everyone.';
  $('#account-notice-text').textContent = notice;
  $('#account-notice').classList.toggle('hidden', !notice);
  $('#account-notice').classList.toggle('error', !!sessionError || logoutFailed);
  $('#retry-session').classList.toggle('hidden', !sessionError || signingOut || logoutFailed);
  $('#retry-logout').classList.toggle('hidden', !logoutFailed);
  ['#top-account', '#sidebar-account'].forEach((selector) => {
    $(selector).classList.toggle('hidden', unavailable);
    $(selector).textContent = user ? 'Your account' : 'Sign in';
  });
  $('#sign-out').classList.toggle('hidden', !user || signingOut);
  $('#refresh-collections').classList.toggle('hidden', !user);
  $('#refresh-collections').disabled = listState === 'loading';
  sourceButtons().forEach((button) => { button.disabled = unavailable; button.title = unavailable ? notice : ''; });
  updateRecovery();
  if ($('#auth-modal').open) renderAuthForm();
}

function renderCollections() {
  const row = (corpus) => {
    const id = corpus.corpus_id;
    return `<div class="collection-row"><button class="collection-choice ${id === activeCorpusId ? 'selected' : ''}" data-corpus-id="${esc(id)}" aria-pressed="${id === activeCorpusId}" ${corpus.status === 'deleting' ? 'disabled title="Delete again to finish removing this collection"' : ''}><span>${esc(corpus.name)}</span><small>${id === 'demo' ? 'Public starter demo' : esc(statusLabel(corpus))}</small></button>${id === 'demo' ? '' : `<button class="collection-delete" data-delete-id="${esc(id)}" aria-label="Delete ${esc(corpus.name)}" title="Delete collection">×</button>`}</div>`;
  };
  $('#collection-list').innerHTML = row({ corpus_id: 'demo', name: 'Starter collection' }) + (auth.user ? collections.map(row).join('') : '');
  $('#collections-message').textContent = !auth.user ? 'Personal collections appear here after sign-in.' : listState === 'loading' ? 'Loading saved collections…' : listState === 'error' ? listError : !collections.length ? 'No saved collections yet. Add sources to get started.' : '';
  $('#refresh-collections').disabled = listState === 'loading';
}

function renderNotice(corpus) {
  let message = '';
  let kind = '';
  if (corpus.status === 'processing') message = progressLabel(corpus.progress) || 'Preparing your collection…';
  else if (corpus.status === 'error') { message = corpus.error || 'We could not prepare this collection. Check the source links or retry preparation.'; kind = 'error'; }
  else if ((corpus.warnings || []).length || (corpus.skipped || []).length) {
    const parts = [...(corpus.warnings || [])];
    if (corpus.skipped?.length) parts.push(`${corpus.skipped.length} video${corpus.skipped.length === 1 ? '' : 's'} could not be used because captions were unavailable.`);
    message = parts.join(' ');
    kind = 'warning';
  }
  ['#collection-notice', '#library-notice'].forEach((selector) => showMessage(selector, message, kind));
}

function renderSidebarSources(videos) {
  if (!videos.length) { $('#source-list').innerHTML = '<p class="source-list-empty">No ready videos yet.</p>'; return; }
  const row = (video) => videoLink(video, `${thumbnail(video.video_id)}<span class="source-mini-text"><span class="source-mini-title">${esc(video.title)}</span><span class="source-mini-meta">${esc(video.duration)}</span></span>`, 'source-mini');
  $('#source-list').innerHTML = videos.slice(0, 4).map(row).join('') + (videos.length > 4 ? `<details class="source-more"><summary>Show ${videos.length - 4} more videos</summary><div class="source-more-list">${videos.slice(4).map(row).join('')}</div></details>` : '');
}

function renderLibrary(videos) {
  $('#video-grid').innerHTML = videos.length ? videos.map((video) => `<article class="video-card ${video.status === 'skipped' ? 'video-card-skipped' : ''}">${video.status === 'skipped' ? '' : thumbnail(video.video_id)}<div class="video-card-body"><div class="video-card-title">${esc(video.title || video.video_id)}</div><div class="video-card-meta">${video.status === 'skipped' ? 'Captions unavailable' : `${esc(video.duration)}${video.source_kind === 'playlist' ? ' · Playlist' : ''}`}</div>${video.status === 'skipped' ? `<p class="video-card-reason">${esc(video.reason || 'This video could not be added to the collection.')}</p>` : youtubeURL(video.video_id) ? videoLink(video, 'Open on YouTube') : ''}</div></article>`).join('') : '<div class="empty-citations">Videos will appear here once their captions are ready.</div>';
}

function updateResearchControls() {
  const ready = ['ready', 'partial'].includes(activeCorpus?.status);
  $('#question').disabled = !ready || signingOut;
  $$('.action-button, #ask').forEach((button) => { button.disabled = !ready || streaming || signingOut; });
}

function updateRecovery() {
  const failed = activeCorpus?.status === 'error';
  $('#collection-recovery').classList.toggle('hidden', !failed);
  $('#retry-collection').classList.toggle('hidden', !failed || activeCorpusId === 'demo' || activeCorpus?._loadError);
  $('#retry-collection').disabled = retryBusy || !auth.user?.email_verified || !auth.enabled || authLoading;
  $('#retry-collection').textContent = retryBusy ? 'Restarting…' : 'Retry preparation';
  $('#retry-collection').title = auth.user && !auth.user.email_verified ? 'Verify your email before retrying preparation.' : '';
}

function renderCorpus(corpus) {
  activeCorpus = corpus;
  const ready = ['ready', 'partial'].includes(corpus.status);
  $('#collection-name').textContent = corpus.name || 'Video collection';
  $('#collection-status').className = `collection-status ${['processing', 'partial', 'error', 'ready'].includes(corpus.status) ? corpus.status : ''}`;
  $('#collection-status').lastElementChild.textContent = statusLabel(corpus);
  $('#collection-count').textContent = ready ? `${corpus.video_count || 0} video${corpus.video_count === 1 ? '' : 's'} in this collection` : progressLabel(corpus.progress) || 'This may take a moment.';
  renderSidebarSources(corpus.videos || []);
  renderLibrary(corpus.video_statuses || corpus.videos || []);
  renderNotice(corpus);
  $('#welcome').classList.toggle('hidden', ready || corpus.status === 'processing');
  $('#workspace').classList.toggle('hidden', !ready && corpus.status !== 'processing');
  $('#workspace-title').textContent = ready ? 'What would you like to know?' : corpus.progress?.phase === 'restoring_index' ? 'We’re reopening your collection' : 'We’re getting your collection ready';
  const index = collections.findIndex((item) => item.corpus_id === corpus.corpus_id);
  if (index >= 0 && !corpus._loadError) collections[index] = corpus;
  renderCollections();
  updateResearchControls();
  updateRecovery();
}

function stopPolling() {
  if (polling !== null) window.clearTimeout(polling);
  polling = null;
}

function clearResult() {
  $('#result').classList.add('hidden');
  ['#answer', '#citations', '#trajectory'].forEach((selector) => { $(selector).replaceChildren(); });
  $('#researching').classList.add('hidden');
  $('#details').open = false;
  $('#question').value = '';
}

function invalidateWorkspace() {
  generation += 1;
  listRevision += 1;
  contextController.abort();
  contextController = new AbortController();
  researchController?.abort();
  researchController = null;
  stopPolling();
  streaming = false;
  sourceBusy = false;
  retryBusy = false;
  deleteBusy = false;
  pendingDelete = null;
  closeDialog('#source-modal');
  closeDialog('#delete-modal');
  $('#create-source').disabled = false;
  $('#create-source').textContent = 'Build collection';
  clearResult();
  showMessage('#workspace-notice', '');
}

function switchCorpus(id, initialCorpus = null) {
  if (id !== 'demo' && (!auth.user || !collections.some((item) => item.corpus_id === id))) return;
  invalidateWorkspace();
  activeCorpusId = id;
  renderCorpus(initialCorpus || { corpus_id: id, name: collections.find((item) => item.corpus_id === id)?.name || 'Starter collection', status: 'processing', videos: [], progress: { phase: id === 'demo' ? 'loading_demo' : 'restoring_index' } });
  setView(activeView);
  void loadCorpus();
  if (auth.user) void refreshCollections();
}

function applySession(data, forceReset = false) {
  const user = data.enabled ? data.user || null : null;
  const changed = auth.user?.id !== user?.id || forceReset;
  auth = { user, enabled: data.enabled === true, registration_available: data.registration_available === true };
  authLoading = false;
  sessionError = '';
  if (changed) {
    collections = [];
    listState = 'idle';
    listError = '';
    $('#source-form').reset();
    switchCorpus('demo');
  }
  renderAccount();
  renderCollections();
}

async function checkSession() {
  const revision = ++authRevision;
  sessionController?.abort();
  sessionController = new AbortController();
  authLoading = true;
  renderAccount();
  try {
    const data = await api('/api/auth/session', { signal: sessionController.signal });
    if (revision !== authRevision) return;
    if (typeof data.enabled !== 'boolean') throw new Error('The account service is unavailable. You can still explore the starter demo.');
    applySession(data);
  } catch (error) {
    if (revision !== authRevision || error.name === 'AbortError') return;
    authLoading = false;
    sessionError = `${errorText(error)} You can still explore the starter demo.`;
    renderAccount();
  }
}

function sessionExpired(error, expectedGeneration) {
  if (error.status !== 401 || expectedGeneration !== generation || !auth.user) return false;
  authRevision += 1;
  sessionController?.abort();
  applySession({ ...auth, user: null }, true);
  showMessage('#workspace-notice', 'Your session expired. Sign in again to open your saved collections.');
  return true;
}

async function refreshCollections() {
  if (!auth.user || signingOut || logoutFailed) return;
  const expected = generation;
  const revision = ++listRevision;
  listState = 'loading';
  renderCollections();
  try {
    const data = await api('/api/corpora', { signal: contextController.signal });
    if (expected !== generation || revision !== listRevision) return;
    if (!Array.isArray(data.collections)) throw new Error('Saved collections could not be read. Please refresh the list.');
    collections = data.collections.filter((item) => typeof item.corpus_id === 'string' && item.corpus_id !== 'demo');
    listState = 'ready';
    listError = '';
  } catch (error) {
    if (expected !== generation || revision !== listRevision || error.name === 'AbortError') return;
    if (sessionExpired(error, expected)) return;
    listState = 'error';
    listError = `${errorText(error)} Use Refresh to try again.`;
  }
  renderCollections();
}

async function loadCorpus() {
  stopPolling();
  const expected = generation;
  const id = activeCorpusId;
  try {
    const corpus = await api(`/api/corpora/${encodeURIComponent(id)}`, { signal: contextController.signal });
    if (expected !== generation || id !== activeCorpusId) return;
    renderCorpus(corpus);
    if (corpus.status === 'processing') polling = window.setTimeout(() => { if (expected === generation) void loadCorpus(); }, 1200);
  } catch (error) {
    if (expected !== generation || error.name === 'AbortError') return;
    if (sessionExpired(error, expected)) return;
    if (id !== 'demo' && [403, 404].includes(error.status)) {
      collections = collections.filter((item) => item.corpus_id !== id);
      switchCorpus('demo');
      showMessage('#workspace-notice', errorText(error), 'error');
      return;
    }
    renderCorpus({ corpus_id: id, name: activeCorpus?.name || 'Starter collection', status: 'error', videos: [], video_count: 0, error: errorText(error), _loadError: true });
  }
}

function setView(view) {
  activeView = view;
  $('#view-workspace').classList.toggle('hidden', view !== 'workspace');
  $('#view-library').classList.toggle('hidden', view !== 'library');
  $$('[data-view]').forEach((button) => {
    button.classList.toggle('nav-item-active', button.dataset.view === view);
    if (button.dataset.view === view) button.setAttribute('aria-current', 'page');
    else button.removeAttribute('aria-current');
  });
  $('#sidebar').classList.remove('open');
  $('#open-sidebar').setAttribute('aria-expanded', 'false');
}

function closeDialog(selector) {
  if ($(selector).open) $(selector).close();
}

function openDialog(selector, focusSelector) {
  ['#source-modal', '#auth-modal', '#delete-modal'].filter((id) => id !== selector).forEach(closeDialog);
  if (!$(selector).open) $(selector).showModal();
  $(focusSelector)?.focus();
}

function openSources() {
  if (authLoading || !auth.enabled || sessionError || signingOut || logoutFailed) return;
  if (!auth.user) return openAuth('login');
  if (!auth.user.email_verified) return openAuth('account');
  showMessage('#source-error', '');
  openDialog('#source-modal', '#source-input');
}

async function loadLimits() {
  try {
    const health = await api('/api/health');
    const limit = Number(health.max_playlist_videos);
    $('#source-limits').textContent = Number.isInteger(limit) && limit > 0
      ? `Use one link per line. Public and accessible unlisted playlists are supported. Each playlist is limited to ${limit} videos.`
      : 'Use one link per line. Public and accessible unlisted playlists are supported. The server applies the configured video limit.';
  } catch (_) {
    $('#source-limits').textContent = 'Use one link per line. Public and accessible unlisted playlists are supported. The video limit could not be loaded; the server will apply its configured limit.';
  }
}

async function createCollection(event) {
  event.preventDefault();
  if (sourceBusy) return;
  if (!auth.user?.email_verified || !auth.enabled) return openSources();
  const raw = $('#source-input').value.trim();
  if (!raw) { showMessage('#source-error', 'Paste at least one YouTube video or playlist link.'); $('#source-input').focus(); return; }
  const expected = generation;
  sourceBusy = true;
  $('#create-source').disabled = true;
  $('#create-source').textContent = 'Starting…';
  showMessage('#source-error', '');
  try {
    const result = await api('/api/corpora', { method: 'POST', signal: contextController.signal, body: JSON.stringify({ name: $('#collection-input').value.trim() || 'My video collection', inputs: raw.split(/\r?\n/).map((line) => line.trim()).filter(Boolean) }) });
    if (expected !== generation) return;
    collections.unshift(result);
    $('#source-form').reset();
    activeView = 'workspace';
    switchCorpus(result.corpus_id, result);
  } catch (error) {
    if (expected !== generation || error.name === 'AbortError' || sessionExpired(error, expected)) return;
    showMessage('#source-error', errorText(error));
  } finally {
    if (expected === generation) { sourceBusy = false; $('#create-source').disabled = false; $('#create-source').textContent = 'Build collection'; }
  }
}

async function retryCollection() {
  if (retryBusy || activeCorpusId === 'demo' || !auth.user?.email_verified || activeCorpus?.status !== 'error') return;
  const expected = generation;
  retryBusy = true;
  updateRecovery();
  try {
    const result = await api(`/api/corpora/${encodeURIComponent(activeCorpusId)}/retry`, { method: 'POST', signal: contextController.signal });
    if (expected !== generation) return;
    switchCorpus(activeCorpusId, result);
  } catch (error) {
    if (expected !== generation || error.name === 'AbortError' || sessionExpired(error, expected)) return;
    showMessage('#workspace-notice', errorText(error), 'error');
  } finally {
    if (expected === generation) { retryBusy = false; updateRecovery(); }
  }
}

function openDelete(id) {
  const corpus = collections.find((item) => item.corpus_id === id);
  if (!corpus || !auth.user) return;
  pendingDelete = { id, generation };
  deleteBusy = false;
  $('#confirm-delete').disabled = false;
  $('#confirm-delete').textContent = 'Delete collection';
  $('#delete-copy').textContent = `Delete “${corpus.name}” and its saved sources? This cannot be undone.`;
  showMessage('#delete-error', '');
  openDialog('#delete-modal', '#cancel-delete');
}

async function deleteCollection() {
  if (!pendingDelete || pendingDelete.generation !== generation || deleteBusy || !auth.user) return;
  const { id, generation: expected } = pendingDelete;
  deleteBusy = true;
  $('#confirm-delete').disabled = true;
  $('#confirm-delete').textContent = 'Deleting…';
  showMessage('#delete-error', '');
  try {
    await api(`/api/corpora/${encodeURIComponent(id)}`, { method: 'DELETE', signal: contextController.signal });
    if (expected !== generation) return;
    listRevision += 1;
    collections = collections.filter((item) => item.corpus_id !== id);
    closeDialog('#delete-modal');
    if (activeCorpusId === id) switchCorpus('demo');
    else { renderCollections(); void refreshCollections(); }
  } catch (error) {
    if (expected !== generation || error.name === 'AbortError' || sessionExpired(error, expected)) return;
    showMessage('#delete-error', errorText(error));
    if (error.status === 409) void refreshCollections();
  } finally {
    if (expected === generation) { deleteBusy = false; $('#confirm-delete').disabled = false; $('#confirm-delete').textContent = 'Delete collection'; }
  }
}

function authModeAvailable() {
  if (authLoading || sessionError || !auth.enabled || signingOut || logoutFailed) return false;
  if (['register', 'forgot', 'resend'].includes(authMode) && !auth.registration_available) return false;
  return !(['verify', 'reset'].includes(authMode) && !emailAction?.token);
}

function renderAuthForm() {
  const [title, copy, submit] = authCopy[authMode];
  $('#auth-title').textContent = title;
  $('#auth-copy').textContent = authMode === 'account' && auth.user ? `${auth.user.email} · ${auth.user.email_verified ? 'Email verified. Your collections are saved to your account.' : 'Verify this email address before adding sources.'}` : copy;
  const fields = { name: authMode === 'register', email: ['login', 'register', 'forgot', 'resend'].includes(authMode), password: ['login', 'register', 'reset'].includes(authMode), confirm: ['register', 'reset'].includes(authMode) };
  Object.entries(fields).forEach(([name, visible]) => {
    $(`#auth-${name}-field`).classList.toggle('hidden', !visible);
    $(`#auth-${name}`).disabled = !visible;
    $(`#auth-${name}`).required = visible && name !== 'name';
  });
  $('#auth-password').autocomplete = authMode === 'login' ? 'current-password' : 'new-password';
  $('#auth-fields').disabled = authBusy || !authModeAvailable();
  $('#auth-form').classList.toggle('hidden', authMode === 'account' || authComplete);
  $('#auth-submit').textContent = authBusy ? 'Please wait…' : submit;
  $('#close-auth-modal').disabled = authBusy;
  let unavailable = '';
  if (authLoading) unavailable = 'Checking account availability…';
  else if (sessionError) unavailable = sessionError;
  else if (!auth.enabled) unavailable = 'Accounts are not configured. Imports are currently unavailable; you can explore the starter demo.';
  else if (!auth.registration_available) unavailable = 'Email setup is pending. New accounts and verification or password reset emails are currently unavailable. Existing users can sign in.';
  if (!authComplete && ['verify', 'reset'].includes(authMode) && !emailAction?.token) unavailable = 'This email link is incomplete. Request a fresh link using the options below.';
  showMessage('#auth-availability', unavailable, 'warning');
  $$('[data-auth-mode]').forEach((button) => {
    const mode = button.dataset.authMode;
    button.classList.toggle('hidden', (!auth.enabled && !authLoading) || mode === authMode || (mode === 'login' && !!auth.user && !authComplete) || (mode === 'register' && !!auth.user) || (mode === 'resend' && !!auth.user?.email_verified));
    button.disabled = authBusy || !auth.enabled || authLoading || !!sessionError || (mode !== 'login' && !auth.registration_available);
  });
  $('#refresh-account').classList.toggle('hidden', !auth.user || !!auth.user.email_verified);
  $('#refresh-account').disabled = authBusy || authLoading;
}

function openAuth(mode) {
  if (authBusy) return;
  authMode = mode || emailAction?.mode || (auth.user ? 'account' : 'login');
  authComplete = false;
  $('#auth-form').reset();
  $('#auth-confirm').setCustomValidity('');
  $('#auth-email').value = auth.user?.email || '';
  showMessage('#auth-message', '');
  showMessage('#auth-error', '');
  renderAuthForm();
  openDialog('#auth-modal', authModeAvailable() && ['login', 'register', 'forgot', 'resend'].includes(authMode) ? '#auth-email' : authModeAvailable() && authMode === 'reset' ? '#auth-password' : '#auth-title');
}

async function submitAuth(event) {
  event.preventDefault();
  if (authBusy || !authModeAvailable()) return;
  const mode = authMode;
  const password = $('#auth-password').value;
  if (['login', 'register', 'reset'].includes(mode) && (password.length < 15 || password.length > 128)) { showMessage('#auth-error', 'Use a password with 15–128 characters.'); $('#auth-password').focus(); return; }
  if (['register', 'reset'].includes(mode) && password !== $('#auth-confirm').value) { $('#auth-confirm').setCustomValidity('The passwords do not match.'); $('#auth-confirm').reportValidity(); return; }
  const email = $('#auth-email').value.trim();
  const bodies = { login: { email, password }, register: { email, password, name: $('#auth-name').value.trim() }, forgot: { email }, resend: { email }, verify: { token: emailAction?.token }, reset: { token: emailAction?.token, password } };
  const endpoints = { forgot: 'forgot-password', resend: 'resend-verification', verify: 'verify-email', reset: 'reset-password' };
  const revision = ++authRevision;
  sessionController?.abort();
  authController = new AbortController();
  authBusy = true;
  showMessage('#auth-error', '');
  showMessage('#auth-message', '');
  renderAuthForm();
  try {
    const data = await api(`/api/auth/${endpoints[mode] || mode}`, { method: 'POST', signal: authController.signal, body: JSON.stringify(bodies[mode]) });
    if (revision !== authRevision) return;
    $('#auth-password').value = '';
    $('#auth-confirm').value = '';
    if (mode === 'login') {
      if (!data.user?.id) throw new Error('Sign-in could not be confirmed. Please try again.');
      applySession({ ...auth, user: data.user }, true);
      closeDialog('#auth-modal');
    } else {
      authComplete = true;
      if (['verify', 'reset'].includes(mode)) emailAction = null;
      const fallback = mode === 'verify' ? 'Email verified. You can sign in and add your sources.' : mode === 'reset' ? 'Password updated. Sign in with your new password.' : 'Check your email. If the account is eligible, a link will arrive shortly.';
      showMessage('#auth-message', typeof data.message === 'string' ? data.message : fallback);
      if (mode === 'verify' || mode === 'reset') await checkSession();
    }
  } catch (error) {
    if (revision !== authRevision || error.name === 'AbortError') return;
    showMessage('#auth-error', errorText(error));
    $('#auth-password').value = '';
    $('#auth-confirm').value = '';
  } finally {
    authBusy = false;
    renderAccount();
    if (authComplete && $('#auth-modal').open) $('#auth-message').focus();
  }
}

async function signOut() {
  if (signingOut) return;
  const revision = ++authRevision;
  sessionController?.abort();
  authController?.abort();
  authBusy = false;
  signingOut = true;
  logoutFailed = false;
  emailAction = null;
  closeDialog('#auth-modal');
  $('#auth-form').reset();
  applySession({ ...auth, user: null }, true);
  try {
    await api('/api/auth/logout', { method: 'POST' });
    if (revision !== authRevision) return;
  } catch (error) {
    if (revision !== authRevision) return;
    logoutFailed = true;
  } finally {
    if (revision === authRevision) { signingOut = false; renderAccount(); updateResearchControls(); }
  }
}

function resetResult(mode) {
  const copy = modeCopy[mode];
  $('#result').classList.remove('hidden');
  $('#result-kicker').textContent = copy.kicker;
  $('#result-title').textContent = copy.title;
  $('#answer').innerHTML = '<p class="muted-answer">Your collection is being reviewed. This can take a moment for larger playlists.</p>';
  $('#citations').replaceChildren();
  $('#trajectory').replaceChildren();
  $('#researching').innerHTML = `<span class="spinner"></span> ${copy.loading}`;
  $('#researching').classList.remove('hidden');
}

function renderCitations(citations) {
  $('#citations').innerHTML = citations?.length ? citations.map((citation) => videoLink(citation, `${thumbnail(citation.video_id)}<span class="citation-content"><span class="citation-time">${esc(citation.formatted_time)}</span><span class="citation-title">${esc(citation.title)}</span><span class="citation-snippet">${esc((citation.text || '').replace(/\s+/g, ' ').slice(0, 160))}</span></span>`, 'citation-card', citation.timestamp)).join('') : '<div class="empty-citations">No transcript moments were returned for this response.</div>';
}

function addTrajectoryStep(record) {
  const item = document.createElement('li');
  item.textContent = ['tool', 'retrieve'].includes(record.type) ? 'Reviewed relevant transcript passages.' : record.type === 'synthesize' ? 'Prepared a cited response.' : record.type === 'verify' ? 'Checked the response against its evidence.' : 'Planned the next research step.';
  $('#trajectory').appendChild(item);
}

function createSSEParser(onEvent) {
  let buffer = '';
  let type = '';
  let data = [];
  const line = (value) => {
    if (!value) { if (data.length) onEvent({ type: type || 'message', data: data.join('\n') }); type = ''; data = []; return; }
    if (value.startsWith(':')) return;
    const colon = value.indexOf(':');
    const field = colon < 0 ? value : value.slice(0, colon);
    let content = colon < 0 ? '' : value.slice(colon + 1);
    if (content.startsWith(' ')) content = content.slice(1);
    if (field === 'event') type = content;
    if (field === 'data') data.push(content);
    // Ignore retry/id: never automatically replay a research request.
  };
  return (chunk, final = false) => {
    buffer += chunk;
    let match;
    while ((match = /[\r\n]/.exec(buffer))) {
      const index = match.index;
      if (buffer[index] === '\r' && index === buffer.length - 1 && !final) break;
      const width = buffer[index] === '\r' && buffer[index + 1] === '\n' ? 2 : 1;
      const value = buffer.slice(0, index);
      buffer = buffer.slice(index + width);
      line(value);
    }
    if (final) { if (buffer) line(buffer); buffer = ''; line(''); }
  };
}

async function runResearch(mode) {
  if (streaming || !['ready', 'partial'].includes(activeCorpus?.status) || signingOut) return;
  const question = $('#question').value.trim();
  if (mode === 'question' && !question) { $('#question').focus(); return; }
  const expected = generation;
  const controller = new AbortController();
  researchController = controller;
  streaming = true;
  updateResearchControls();
  resetResult(mode);
  let complete = false;
  let reader;
  const current = () => expected === generation && researchController === controller && !controller.signal.aborted;
  const done = (result) => {
    if (!current()) return;
    $('#answer').innerHTML = safeMarkdown(result.answer || 'No answer was returned.');
    renderCitations(result.citations);
    if (Array.isArray(result.trajectory)) { $('#trajectory').replaceChildren(); result.trajectory.forEach(addTrajectoryStep); }
    complete = true;
  };
  try {
    const response = await request('/api/research/stream', { method: 'POST', signal: controller.signal, headers: { Accept: 'text/event-stream' }, body: JSON.stringify({ corpus_id: activeCorpusId, mode, question: mode === 'question' ? question : '' }) });
    if (!current()) { await response.body?.cancel(); return; }
    if (response.headers.get('Content-Type')?.includes('application/json')) done(await response.json());
    else {
      if (!response.headers.get('Content-Type')?.includes('text/event-stream') || !response.body) throw new Error('The research response could not be read. Please try again.');
      const parse = createSSEParser((event) => {
        if (!current() || complete || !['step', 'answer', 'done', 'error'].includes(event.type)) return;
        let data;
        try { data = JSON.parse(event.data); } catch (_) { throw new Error('The research stream contained an unreadable response. Please try again.'); }
        if (event.type === 'step') addTrajectoryStep(data);
        else if (event.type === 'answer') $('#answer').innerHTML = safeMarkdown(data.answer || '');
        else if (event.type === 'done') done(data);
        else throw new Error(messageText(data, 'We could not complete this request. Please try again.'));
      });
      const decoder = new TextDecoder();
      reader = response.body.getReader();
      while (current() && !complete) {
        const chunk = await reader.read();
        if (!current()) return;
        parse(decoder.decode(chunk.value, { stream: !chunk.done }), chunk.done);
        if (chunk.done) break;
      }
      if (current() && !complete) throw new Error('The connection ended before the answer was complete. Please try again.');
    }
  } catch (error) {
    if (!current() || error.name === 'AbortError' || sessionExpired(error, expected)) return;
    $('#answer').innerHTML = `<p class="error-answer">${esc(errorText(error))}</p>`;
  } finally {
    if (reader) { try { await reader.cancel(); } catch (_) { /* already aborted */ } }
    if (current()) { researchController = null; streaming = false; $('#researching').classList.add('hidden'); updateResearchControls(); }
  }
}

$$('[data-view]').forEach((button) => button.addEventListener('click', () => setView(button.dataset.view)));
sourceButtons().forEach((button) => button.addEventListener('click', openSources));
$('#source-form').addEventListener('submit', createCollection);
$('#auth-form').addEventListener('submit', submitAuth);
$('#auth-confirm').addEventListener('input', () => $('#auth-confirm').setCustomValidity(''));
$('#auth-password').addEventListener('input', () => $('#auth-confirm').setCustomValidity(''));
['#top-account', '#sidebar-account'].forEach((selector) => $(selector).addEventListener('click', () => openAuth()));
$$('[data-auth-mode]').forEach((button) => button.addEventListener('click', () => openAuth(button.dataset.authMode)));
$('#refresh-account').addEventListener('click', () => { void checkSession(); });
$('#retry-session').addEventListener('click', () => { void checkSession(); });
['#sign-out', '#retry-logout'].forEach((selector) => $(selector).addEventListener('click', signOut));
$('#refresh-collections').addEventListener('click', refreshCollections);
$('#reload-collection').addEventListener('click', loadCorpus);
$('#retry-collection').addEventListener('click', retryCollection);
$('#collection-list').addEventListener('click', (event) => {
  const choice = event.target.closest('[data-corpus-id]');
  const deletion = event.target.closest('[data-delete-id]');
  if (choice) switchCorpus(choice.dataset.corpusId);
  if (deletion) openDelete(deletion.dataset.deleteId);
});
$('#confirm-delete').addEventListener('click', deleteCollection);
const dialogBusy = (id) => id === '#auth-modal' ? authBusy : id === '#delete-modal' ? deleteBusy : sourceBusy;
[
  ['#source-modal', ['#close-source-modal', '#cancel-source']],
  ['#auth-modal', ['#close-auth-modal']],
  ['#delete-modal', ['#close-delete-modal', '#cancel-delete']],
].forEach(([id, buttons]) => {
  buttons.forEach((selector) => $(selector).addEventListener('click', () => { if (!dialogBusy(id)) closeDialog(id); }));
  $(id).addEventListener('cancel', (event) => { if (dialogBusy(id)) event.preventDefault(); });
  $(id).addEventListener('click', (event) => { if (event.target === $(id) && !dialogBusy(id)) closeDialog(id); });
});
$('#auth-modal').addEventListener('close', () => { $('#auth-password').value = ''; $('#auth-confirm').value = ''; $('#auth-confirm').setCustomValidity(''); });
$('#open-sidebar').setAttribute('aria-controls', 'sidebar');
$('#open-sidebar').setAttribute('aria-expanded', 'false');
$('#open-sidebar').addEventListener('click', () => { $('#sidebar').classList.add('open'); $('#open-sidebar').setAttribute('aria-expanded', 'true'); $('#close-sidebar').focus(); });
$('#close-sidebar').addEventListener('click', () => { $('#sidebar').classList.remove('open'); $('#open-sidebar').setAttribute('aria-expanded', 'false'); $('#open-sidebar').focus(); });
$('#ask').addEventListener('click', () => { void runResearch('question'); });
$$('.action-button').forEach((button) => button.addEventListener('click', () => { void runResearch(button.dataset.mode); }));
$('#question').addEventListener('keydown', (event) => { if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) { event.preventDefault(); void runResearch('question'); } });

renderAccount();
renderCollections();
void loadCorpus();
void checkSession();
void loadLimits();
if (emailAction) openAuth(emailAction.mode);
