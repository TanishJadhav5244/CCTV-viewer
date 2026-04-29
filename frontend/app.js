const API = 'http://localhost:8000';
let analyticsChart = null;
let statusPoller = null;
let _seenRunning = false; // guard: don't stop poller before 'running' is ever seen

/**
 * Converts a stored crop_path (e.g. 'data\\crops\\hash.jpg' or 'crops/hash.jpg')
 * to a proper URL using the /crops static mount.
 */
function cropUrl(path) {
    if (!path) return '';
    // Normalise slashes
    const p = path.replace(/\\/g, '/');
    // Strip leading 'data/crops/' or 'crops/' prefix and use /crops mount
    const filename = p.split('/').pop();
    return `${API}/crops/${filename}`;
}

// ── Video Modal ──────────────────────────────────────────────────────────────
/**
 * Converts raw seconds to MM:SS format.
 * @param {number|null} seconds
 * @returns {string}
 */
function formatTime(seconds) {
    if (seconds == null || isNaN(seconds)) return null;
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return String(m).padStart(2, '0') + ':' + String(s).padStart(2, '0');
}

/**
 * Opens the video modal, seeks to the given time and plays.
 * @param {string} videoSrc  — path like "data/videos/test.mp4"
 * @param {number} seconds   — time offset in seconds
 * @param {string} label     — detection label (shown in modal title)
 */
function openVideoModal(videoSrc, seconds, label) {
    if (!videoSrc) return;
    const modal = document.getElementById('videoModal');
    const video = document.getElementById('mainVideo');
    const title = document.getElementById('modalTitle');

    // Build full URL served by FastAPI's StaticFiles mount at /data/videos
    const url = `${API}/${videoSrc.replace(/\\/g, '/')}`;
    video.src = url;
    title.textContent = `${label || 'Object'} — ${formatTime(seconds) || '00:00'}`;

    modal.classList.add('open');
    document.body.classList.add('modal-open');

    // Once metadata loads, seek then play
    video.onloadedmetadata = () => {
        if (seconds != null) video.currentTime = seconds;
        video.play().catch(() => {/* autoplay blocked – user can click play */});
    };
}

// ── Video Zoom / Pan Engine ─────────────────────────────────────────────────
const ZOOM_MIN = 1.0;
const ZOOM_MAX = 6.0;
const ZOOM_STEP_BTN  = 0.25;   // ± buttons step
const ZOOM_STEP_WHEEL = 0.15;  // scroll wheel step

let _zoom = 1.0;
let _panX = 0;
let _panY = 0;
let _dragging = false;
let _dragStartX = 0;
let _dragStartY = 0;
let _panStartX = 0;
let _panStartY = 0;

function _applyZoom() {
    const canvas = document.getElementById('videoZoomCanvas');
    const wrap   = document.getElementById('videoWrap');
    const label  = document.getElementById('zoomLevel');
    if (!canvas) return;

    // Clamp pan so video edges don't go inside the viewport when zoomed
    const ww = wrap ? wrap.clientWidth  : 0;
    const wh = wrap ? wrap.clientHeight : 0;
    const scaledW = ww * _zoom;
    const scaledH = wh * _zoom;
    const maxPanX = Math.max(0, scaledW - ww);
    const maxPanY = Math.max(0, scaledH - wh);
    _panX = Math.min(0, Math.max(-maxPanX, _panX));
    _panY = Math.min(0, Math.max(-maxPanY, _panY));

    canvas.style.transform = `scale(${_zoom}) translate(${_panX / _zoom}px, ${_panY / _zoom}px)`;
    if (label) label.textContent = _zoom.toFixed(1) + '×';
    if (wrap) wrap.classList.toggle('zoomed', _zoom > 1.0);
}

function _resetZoom() {
    _zoom = 1.0; _panX = 0; _panY = 0;
    _applyZoom();
}

function _changeZoom(delta, originX, originY) {
    const wrap = document.getElementById('videoWrap');
    const prevZoom = _zoom;
    _zoom = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, _zoom + delta));
    if (_zoom === prevZoom) return;

    // Adjust pan so the cursor/origin point stays fixed
    if (wrap && originX != null) {
        const ox = originX - wrap.getBoundingClientRect().left;
        const oy = originY - wrap.getBoundingClientRect().top;
        _panX = ox - (_zoom / prevZoom) * (ox - _panX);
        _panY = oy - (_zoom / prevZoom) * (oy - _panY);
    }
    _applyZoom();
}

// Scroll-to-zoom on the video wrap
document.getElementById('videoWrap')?.addEventListener('wheel', e => {
    e.preventDefault();
    const delta = e.deltaY < 0 ? ZOOM_STEP_WHEEL : -ZOOM_STEP_WHEEL;
    _changeZoom(delta, e.clientX, e.clientY);
}, { passive: false });

// Drag-to-pan
document.getElementById('videoWrap')?.addEventListener('mousedown', e => {
    if (_zoom <= 1.0 || e.button !== 0) return;
    // Don't hijack clicks on video controls bar (bottom ~44px of video)
    const wrap = document.getElementById('videoWrap');
    const rect = wrap.getBoundingClientRect();
    if (e.clientY > rect.bottom - 48) return;
    e.preventDefault();
    _dragging = true;
    _dragStartX = e.clientX;
    _dragStartY = e.clientY;
    _panStartX  = _panX;
    _panStartY  = _panY;
    document.getElementById('videoZoomCanvas')?.classList.add('dragging');
});

document.addEventListener('mousemove', e => {
    if (!_dragging) return;
    _panX = _panStartX + (e.clientX - _dragStartX);
    _panY = _panStartY + (e.clientY - _dragStartY);
    _applyZoom();
});

document.addEventListener('mouseup', () => {
    if (!_dragging) return;
    _dragging = false;
    document.getElementById('videoZoomCanvas')?.classList.remove('dragging');
});

// Touch pinch-to-zoom
let _lastTouchDist = null;
document.getElementById('videoWrap')?.addEventListener('touchstart', e => {
    if (e.touches.length === 2) {
        _lastTouchDist = Math.hypot(
            e.touches[1].clientX - e.touches[0].clientX,
            e.touches[1].clientY - e.touches[0].clientY
        );
    }
}, { passive: true });

document.getElementById('videoWrap')?.addEventListener('touchmove', e => {
    if (e.touches.length === 2 && _lastTouchDist) {
        e.preventDefault();
        const dist = Math.hypot(
            e.touches[1].clientX - e.touches[0].clientX,
            e.touches[1].clientY - e.touches[0].clientY
        );
        const midX = (e.touches[0].clientX + e.touches[1].clientX) / 2;
        const midY = (e.touches[0].clientY + e.touches[1].clientY) / 2;
        const delta = (dist - _lastTouchDist) * 0.015;
        _changeZoom(delta, midX, midY);
        _lastTouchDist = dist;
    }
}, { passive: false });

document.getElementById('videoWrap')?.addEventListener('touchend', () => {
    _lastTouchDist = null;
});

// Button controls
document.getElementById('zoomInBtn')?.addEventListener('click',  () => _changeZoom(+ZOOM_STEP_BTN));
document.getElementById('zoomOutBtn')?.addEventListener('click', () => _changeZoom(-ZOOM_STEP_BTN));
document.getElementById('zoomResetBtn')?.addEventListener('click', _resetZoom);

function closeVideoModal() {
    const modal = document.getElementById('videoModal');
    const video = document.getElementById('mainVideo');
    modal.classList.remove('open');
    document.body.classList.remove('modal-open');
    video.pause();
    video.src = '';
    _resetZoom();   // always reset zoom when modal closes
}

// Wire up close button and backdrop click
document.getElementById('modalClose').addEventListener('click', closeVideoModal);
document.getElementById('videoModal').addEventListener('click', e => {
    if (e.target === e.currentTarget) closeVideoModal();
});
// Note: Escape key is handled in the global keydown listener below


// ── Tab Navigation ──────────────────────────────────────────────────────────
document.querySelectorAll('.nav-item').forEach(btn => {
    btn.addEventListener('click', () => {
        const tab = btn.dataset.tab;
        document.querySelectorAll('.nav-item').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById(`tab-${tab}`).classList.add('active');
        document.getElementById('pageTitle').textContent = btn.querySelector('span').textContent;

        if (tab === 'analytics')  loadAnalytics();
        if (tab === 'detections') { loadSources(); loadDetections(); }
        if (tab === 'timeline')   loadSources();
        if (tab === 'persons')    loadPersons();
        if (tab === 'sessions')   loadSessions();
    });
});

// ── Keyboard shortcut: '/' focuses search ────────────────────────────────────
document.addEventListener('keydown', e => {
    if (e.key === '/' && document.activeElement.tagName !== 'INPUT'
                      && document.activeElement.tagName !== 'TEXTAREA') {
        e.preventDefault();
        // Switch to search tab first
        document.querySelector('[data-tab="search"]')?.click();
        document.getElementById('searchInput')?.focus();
    }
    if (e.key === 'Escape') closeVideoModal();
});

// ── Search ──────────────────────────────────────────────────────────────────
document.getElementById('searchBtn').addEventListener('click', doSearch);
document.getElementById('searchInput').addEventListener('keydown', e => {
    if (e.key === 'Enter') doSearch();
});

function triggerSearch(query) {
    document.getElementById('searchInput').value = query;
    doSearch();
}

// Returns the active colour pill value (empty string = Any)
function getSelectedColor() {
    const active = document.querySelector('#colorPills .color-pill.active');
    return active ? active.dataset.color : '';
}

// ── Color pill click handlers ────────────────────────────────────────────────
document.getElementById('colorPills').addEventListener('click', e => {
    const pill = e.target.closest('.color-pill');
    if (!pill) return;
    // Deactivate all, activate clicked one
    document.querySelectorAll('#colorPills .color-pill').forEach(p => p.classList.remove('active'));
    pill.classList.add('active');
    // Re-run search immediately if there's already a query in the box
    const q = document.getElementById('searchInput').value.trim();
    if (q) doSearch();
});

// Also re-search when the Class filter changes
document.getElementById('filterLabel')?.addEventListener('change', () => {
    const q = document.getElementById('searchInput').value.trim();
    if (q) doSearch();
});

// Builds a small shirt-color badge HTML for person cards
function shirtBadgeHtml(r) {
    if ((r.label || '').toLowerCase() !== 'person') return '';
    // Backend may return attributes as an object or a flat field
    const color = (r.attributes && r.attributes.shirt_color)
        || r.shirt_color
        || (r.attributes && r.attributes.clothing_color)
        || r.clothing_color
        || '';
    if (!color) return '';
    const swatch = cssColorForName(color);
    return `<div class="shirt-attr">
        <span class="shirt-swatch" style="background:${swatch}"></span>
        <span class="shirt-label">Shirt: ${color}</span>
      </div>`;
}

// Maps colour names → CSS colour values for the swatch dot
function cssColorForName(name) {
    const map = {
        red:'#ef4444', blue:'#3b82f6', green:'#10b981',
        black:'#1f2937', white:'#f9fafb', yellow:'#f59e0b',
        orange:'#f97316', purple:'#8b5cf6', grey:'#6b7280',
        gray:'#6b7280', navy:'#1e3a5f', pink:'#ec4899',
        brown:'#92400e', cyan:'#06b6d4', maroon:'#7f1d1d',
        beige:'#d4b896', khaki:'#a3956a',
    };
    return map[name.toLowerCase()] || '#8896b0';
}

async function doSearch() {
    const query = document.getElementById('searchInput').value.trim();
    if (!query) return;

    const grid = document.getElementById('resultsGrid');
    const status = document.getElementById('searchStatus');

    grid.innerHTML = `<div class="empty-state"><div class="spinner"></div> Searching for "${query}"…</div>`;
    status.classList.add('hidden');

    // Append clothing colour to query if a colour pill is selected
    const selectedColor = getSelectedColor();
    const fullQuery = selectedColor ? `${query} ${selectedColor} clothing` : query;

    // Class label hard-filter
    const labelFilter = document.getElementById('filterLabel')?.value || null;

    try {
        const res = await fetch(`${API}/search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: fullQuery,
                top_k: 24,
                label: labelFilter || null,
                color: selectedColor || null,
            }),
        });
        const data = await res.json();

        if (data.message) {
            grid.innerHTML = '';
            status.textContent = '⚠ ' + data.message;
            status.classList.remove('hidden');
            return;
        }

        const results = data.results || [];
        if (results.length === 0) {
            grid.innerHTML = `<div class="empty-state"><p>No matches found for "<strong>${query}</strong>". Try different keywords.</p></div>`;
            document.getElementById('searchExportBar')?.classList.add('hidden');
            return;
        }

        // Store for client-side CSV export
        window._lastSearchResults = results;
        window._lastSearchQuery = query;

        grid.innerHTML = results.map(r => {
            const imgSrc = cropUrl(r.crop_path);
            const ts = r.timestamp ? new Date(r.timestamp).toLocaleString() : '—';
            const score = (r.score * 100).toFixed(1);
            const timeFmt = formatTime(r.video_time);
            const hasVideo = r.video_src && timeFmt;
            const clickAttr = hasVideo
                ? `onclick="openVideoModal('${r.video_src.replace(/'/g, "\\'") }', ${r.video_time}, '${(r.label||'').replace(/'/g,"\\'")}')"`
                : '';
            const timeBadge = timeFmt
                ? `<span class="time-badge"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="11" height="11"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>${timeFmt}</span>`
                : '';
            return `
        <div class="result-card${hasVideo ? ' clickable' : ''}" ${clickAttr}>
          <div class="result-img-wrap">
            <img src="${imgSrc}" alt="${r.label}" loading="lazy"
                 onerror="this.style.display='none'"/>
            <span class="result-score-badge">${score}%</span>
            ${hasVideo ? '<span class="play-overlay"><svg viewBox="0 0 24 24" fill="currentColor" width="28" height="28"><polygon points="5 3 19 12 5 21 5 3"/></svg></span>' : ''}
          </div>
          <div class="result-info">
            <div class="result-label">${r.label} ${timeBadge}</div>
            ${shirtBadgeHtml(r)}
            <div class="result-meta">${ts}</div>
            <div class="result-meta">conf ${(r.confidence * 100).toFixed(1)}%</div>
          </div>
        </div>`;
        }).join('');
        // Reveal export bar
        const bar = document.getElementById('searchExportBar');
        if (bar) {
            bar.classList.remove('hidden');
            bar.querySelector('.export-label').textContent = `${results.length} result${results.length !== 1 ? 's' : ''}`;
        }
    } catch (err) {
        status.textContent = '✕ Could not connect to backend. Is the server running?';
        status.classList.remove('hidden');
        status.classList.add('error');
        grid.innerHTML = '';
    }
}

// ── Process Video ───────────────────────────────────────────────────────────
async function startProcessing() {
    const source = document.getElementById('videoSource').value.trim();
    if (!source) return;

    // Show progress immediately so the user sees feedback right away
    document.getElementById('progressSection').style.display = 'block';
    document.getElementById('pStatus').textContent = 'Starting…';
    document.getElementById('pFrames').textContent = '0';
    document.getElementById('pDetections').textContent = '0';
    document.getElementById('pFps').textContent = '0';
    setProgressBar(0, false);

    try {
        const res = await fetch(`${API}/process`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ source }),
        });
        const data = await res.json();
        if (res.ok) {
            document.getElementById('processBtn').style.display = 'none';
            document.getElementById('stopBtn').style.display = 'inline-flex';
            _seenRunning = false;
            startStatusPolling();
        } else {
            document.getElementById('progressSection').style.display = 'none';
            alert('Error: ' + (data.detail || JSON.stringify(data)));
        }
    } catch {
        document.getElementById('progressSection').style.display = 'none';
        alert('Could not connect to backend. Is the server running?');
    }
}

async function stopProcessing() {
    await fetch(`${API}/stop`, { method: 'POST' });
    stopStatusPolling();
    document.getElementById('processBtn').style.display = 'inline-flex';
    document.getElementById('stopBtn').style.display = 'none';
}

function startStatusPolling() {
    if (statusPoller) clearInterval(statusPoller);
    statusPoller = setInterval(updateStatus, 1500);
    updateStatus();
}

function stopStatusPolling() {
    if (statusPoller) clearInterval(statusPoller);
    statusPoller = null;
}

async function updateStatus() {
    try {
        const res = await fetch(`${API}/status`);
        const d = await res.json();

        const status = d.status || 'idle';
        const processed = d.processed_frames || 0;
        const total = d.total_frames || 0;
        const detections = d.total_detections || 0;
        const fps = d.current_fps || 0;

        document.getElementById('pFrames').textContent = processed.toLocaleString();
        document.getElementById('pDetections').textContent = detections.toLocaleString();
        document.getElementById('pFps').textContent = typeof fps === 'number' ? fps.toFixed(1) : fps;
        document.getElementById('pStatus').textContent = status.charAt(0).toUpperCase() + status.slice(1);

        // Sidebar status dot + label
        const dot = document.getElementById('statusDot');
        const label = document.getElementById('statusLabel');
        dot.className = 'status-dot ' + status;
        label.textContent = status.charAt(0).toUpperCase() + status.slice(1);

        // Progress bar
        if (total > 0) {
            const pct = Math.min(100, (processed / total) * 100);
            setProgressBar(pct, false);
        } else if (status === 'running') {
            // Indeterminate / streaming mode — animate
            setProgressBar(0, true);
        }

        // Update header index count
        refreshIndexCount();

        // Track whether we've ever seen 'running'
        if (status === 'running') _seenRunning = true;

        // Only stop polling once processing is actually done
        if (_seenRunning && (status === 'finished' || status === 'idle' || status === 'stopped')) {
            stopStatusPolling();
            document.getElementById('processBtn').style.display = 'inline-flex';
            document.getElementById('stopBtn').style.display = 'none';
            // Show final bar at 100% on finish
            if (status === 'finished') setProgressBar(100, false);
        }
    } catch { /* silent fail */ }
}

function setProgressBar(pct, indeterminate) {
    const bar = document.getElementById('progressBar');
    const wrap = bar.parentElement; // .progress-bar-wrap
    // Find or create the container row (parent of wrap)
    let row = document.getElementById('progressBarRow');
    if (!row) {
        row = document.createElement('div');
        row.id = 'progressBarRow';
        row.className = 'progress-bar-row';
        wrap.parentNode.insertBefore(row, wrap);
        row.appendChild(wrap);
    }
    // Find or create pct label inside the row (not inside wrap)
    let pctLabel = row.querySelector('.progress-pct');
    if (!pctLabel) {
        pctLabel = document.createElement('span');
        pctLabel.className = 'progress-pct';
        row.insertBefore(pctLabel, wrap);
    }
    if (indeterminate) {
        bar.style.width = '40%';
        bar.classList.add('indeterminate');
        pctLabel.textContent = 'Processing…';
    } else {
        bar.classList.remove('indeterminate');
        bar.style.width = pct + '%';
        pctLabel.textContent = pct > 0 ? Math.round(pct) + '%' : '';
    }
}

// ── Analytics ───────────────────────────────────────────────────────────────
async function loadAnalytics() {
    try {
        const res = await fetch(`${API}/stats`);
        const data = await res.json();

        document.getElementById('statTotal').textContent = (data.total_detections || 0).toLocaleString();
        document.getElementById('statIndexed').textContent = (data.index_size || 0).toLocaleString();
        const labels = data.label_counts ? Object.keys(data.label_counts) : [];
        document.getElementById('statLabels').textContent = labels.length;

        document.getElementById('headerIndexCount').textContent = (data.index_size || 0).toLocaleString();

        if (labels.length === 0) {
            document.getElementById('analyticsEmpty').classList.remove('hidden');
            return;
        }
        document.getElementById('analyticsEmpty').classList.add('hidden');

        const values = labels.map(l => data.label_counts[l]);
        const colors = generateColors(labels.length);

        const ctx = document.getElementById('analyticsChart').getContext('2d');
        if (analyticsChart) analyticsChart.destroy();
        analyticsChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels,
                datasets: [{
                    label: 'Detected Objects',
                    data: values,
                    backgroundColor: colors.map(c => c + '99'),
                    borderColor: colors,
                    borderWidth: 2,
                    borderRadius: 6,
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { display: false },
                    tooltip: { backgroundColor: '#0d1128', titleColor: '#f0f4ff', bodyColor: '#8896b0', borderColor: '#1e2d4a', borderWidth: 1 },
                },
                scales: {
                    x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#8896b0' } },
                    y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#8896b0' }, beginAtZero: true },
                }
            }
        });
        loadAnalyticsExtended();
    } catch { /* silent */ }
}

function generateColors(n) {
    const palette = ['#06b6d4', '#3b82f6', '#8b5cf6', '#10b981', '#f59e0b', '#ef4444', '#ec4899', '#14b8a6', '#f97316', '#84cc16'];
    return Array.from({ length: n }, (_, i) => palette[i % palette.length]);
}

// ── Extended Analytics ───────────────────────────────────────────────────
let _confChart = null;
let _hourChart = null;

async function loadAnalyticsExtended() {
    try {
        const data = await (await fetch(`${API}/detections?limit=2000`)).json();
        if (!data || data.length === 0) return;

        const chartOpts = (extra = {}) => ({
            responsive: true,
            plugins: {
                legend: { display: false },
                tooltip: { backgroundColor: '#0d1128', titleColor: '#f0f4ff', bodyColor: '#8896b0', borderColor: '#1e2d4a', borderWidth: 1 },
            },
            scales: {
                x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#8896b0' } },
                y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#8896b0' }, beginAtZero: true, ...extra },
            },
        });

        // ── Confidence histogram (10 buckets: 0–10 %, 10–20 %, … 90–100 %)
        const confBuckets = Array(10).fill(0);
        data.forEach(r => {
            const idx = Math.min(9, Math.floor((r.confidence || 0) * 10));
            confBuckets[idx]++;
        });
        const confLabels = confBuckets.map((_, i) => `${i * 10}–${i * 10 + 10}%`);

        const confCtx = document.getElementById('confChart')?.getContext('2d');
        if (confCtx) {
            if (_confChart) _confChart.destroy();
            _confChart = new Chart(confCtx, {
                type: 'bar',
                data: {
                    labels: confLabels,
                    datasets: [{ label: 'Detections', data: confBuckets,
                        backgroundColor: '#3b82f699', borderColor: '#3b82f6',
                        borderWidth: 2, borderRadius: 4 }]
                },
                options: chartOpts()
            });
            document.getElementById('confEmpty')?.classList.add('hidden');
        }

        // ── Detections by video-time (bucket into 1-minute intervals)
        const detsWithTime = data.filter(r => r.video_time != null);
        const maxMin = detsWithTime.length > 0
            ? Math.ceil(Math.max(...detsWithTime.map(r => r.video_time)) / 60) + 1
            : 10;
        const minuteBuckets = Array(maxMin).fill(0);
        detsWithTime.forEach(r => {
            const m = Math.min(maxMin - 1, Math.floor(r.video_time / 60));
            minuteBuckets[m]++;
        });
        const minuteLabels = minuteBuckets.map((_, m) => `${m}:00`);

        const hourCtx = document.getElementById('hourChart')?.getContext('2d');
        if (hourCtx) {
            if (_hourChart) _hourChart.destroy();
            _hourChart = new Chart(hourCtx, {
                type: 'bar',
                data: {
                    labels: minuteLabels,
                    datasets: [{ label: 'Detections', data: minuteBuckets,
                        backgroundColor: '#8b5cf699', borderColor: '#8b5cf6',
                        borderWidth: 2, borderRadius: 4 }]
                },
                options: {
                    ...chartOpts(),
                    plugins: {
                        ...chartOpts().plugins,
                        title: {
                            display: true,
                            text: 'Detections per minute of video',
                            color: '#8896b0',
                            font: { size: 11 },
                        },
                    },
                }
            });
            document.getElementById('hourEmpty')?.classList.add('hidden');
        }
    } catch { /* silent */ }
}

// ── Detections ──────────────────────────────────────────────────────────────
async function loadDetections() {
    const grid = document.getElementById('detectionsGrid');
    grid.innerHTML = `<div class="empty-state"><div class="spinner"></div> Loading…</div>`;

    const label  = document.getElementById('detFilterLabel')?.value  || '';
    const source = document.getElementById('detFilterSource')?.value || '';

    // Read attribute pills
    const genderActive = document.querySelector('#detGenderPills .attr-pill.active');
    const gender = genderActive ? genderActive.dataset.val : '';
    const accActive = document.querySelector('#detAccessoryPills .attr-pill.active');
    const acc = accActive ? accActive.dataset.val : '';

    let url = `${API}/detections?limit=80`;
    if (label)  url += `&label=${encodeURIComponent(label)}`;
    if (source) url += `&source=${encodeURIComponent(source)}`;
    if (gender) url += `&gender=${encodeURIComponent(gender)}`;
    if (acc === 'hat') url += `&has_hat=true`;
    if (acc === 'bag') url += `&has_bag=true`;

    try {
        const data = await (await fetch(url)).json();

        if (!data || data.length === 0) {
            grid.innerHTML = `<div class="empty-state" id="detectionsEmpty">
        <div class="empty-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M9 9h6M9 13h6M9 17h4"/></svg></div>
        <p>No detections yet. Process a video to begin.</p>
      </div>`;
            return;
        }

        grid.innerHTML = data.map(r => {
            const imgSrc = cropUrl(r.crop_path || '');
            const ts = r.timestamp ? new Date(r.timestamp).toLocaleString() : '—';
            const timeFmt = formatTime(r.video_time);
            const hasVideo = r.video_src && timeFmt;
            const clickAttr = hasVideo
                ? `onclick="openVideoModal('${r.video_src.replace(/'/g, "\\'") }', ${r.video_time}, '${(r.label||'').replace(/'/g,"\\'")}')"`
                : '';
            const timeBadge = timeFmt
                ? `<span class="time-badge"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="11" height="11"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>${timeFmt}</span>`
                : '';
            return `
        <div class="result-card${hasVideo ? ' clickable' : ''}" ${clickAttr}>
          <div class="result-img-wrap">
            <img src="${imgSrc}" alt="${r.label}" loading="lazy" onerror="this.style.display='none'"/>
            ${hasVideo ? '<span class="play-overlay"><svg viewBox="0 0 24 24" fill="currentColor" width="28" height="28"><polygon points="5 3 19 12 5 21 5 3"/></svg></span>' : ''}
          </div>
          <div class="result-info">
            <div class="result-label">${r.label || '—'} ${timeBadge}</div>
            ${shirtBadgeHtml(r)}
            <div class="result-meta">${ts}</div>
            <div class="result-meta">conf ${r.confidence ? (r.confidence * 100).toFixed(1) + '%' : '—'}</div>
          </div>
        </div>`;
        }).join('');
    } catch {
        grid.innerHTML = `<div class="empty-state"><p>Could not load detections. Is the backend running?</p></div>`;
    }
}

async function refreshIndexCount() {
    try {
        const res = await fetch(`${API}/stats`);
        const data = await res.json();
        document.getElementById('headerIndexCount').textContent = (data.index_size || 0).toLocaleString();
    } catch { /* silent */ }
}

// ── Sources (shared by Detections + Timeline) ─────────────────────────────
async function loadSources() {
    try {
        const sources = await (await fetch(`${API}/sources`)).json();
        // Populate Detections tab source filter
        const detSel = document.getElementById('detFilterSource');
        if (detSel) {
            detSel.innerHTML = '<option value="">All sources</option>' +
                sources.map(s => `<option value="${s.source}">${s.source.split('/').pop()} (${s.count})</option>`).join('');
        }
        // Populate Timeline source picker
        const tlSel = document.getElementById('tlSourceSelect');
        if (tlSel) {
            tlSel.innerHTML = '<option value="">— select a video —</option>' +
                sources.map(s => `<option value="${s.source}">${s.source.split('/').pop()} — ${s.count} detections</option>`).join('');
        }
    } catch { /* silent */ }
}

// ── Timeline ──────────────────────────────────────────────────────────────
const LABEL_COLORS = {
    person: '#06b6d4', car: '#3b82f6', truck: '#f97316', bicycle: '#10b981',
    motorcycle: '#8b5cf6', bus: '#f59e0b', dog: '#ec4899', cat: '#14b8a6',
    default: '#8896b0',
};
function labelColor(lbl) { return LABEL_COLORS[lbl] || LABEL_COLORS.default; }

let _tlSource = null;
let _tlDuration = 0;

async function loadTimeline() {
    const source = document.getElementById('tlSourceSelect')?.value;
    if (!source) return;
    _tlSource = source;

    const tlCard  = document.getElementById('tlCard');
    const tlEmpty = document.getElementById('tlEmpty');
    const tlTrack = document.getElementById('tlTrack');
    const tlSummary   = document.getElementById('tlSummary');
    const tlLegend    = document.getElementById('tlLegend');
    const tlTimestamps = document.getElementById('tlTimestamps');

    tlCard.style.display = 'none';
    tlEmpty.style.display = 'none';
    tlCard.style.display = 'block'; // show card with loading state
    tlTrack.innerHTML = '<div class="tl-cursor" id="tlCursor"></div>';
    tlSummary.innerHTML = '<div class="spinner"></div> Loading timeline…';

    try {
        const data = await (await fetch(`${API}/timeline?source=${encodeURIComponent(source)}`)).json();
        const dets = data.detections || [];
        _tlDuration = data.duration || 0;

        if (dets.length === 0) {
            tlCard.style.display = 'none';
            tlEmpty.style.display = 'flex';
            return;
        }

        // ── Summary row
        const labelCounts = {};
        dets.forEach(d => { labelCounts[d.label] = (labelCounts[d.label] || 0) + 1; });
        tlSummary.innerHTML = [
            `<span class="tl-stat">${dets.length} detections</span>`,
            `<span class="tl-stat">${formatTime(_tlDuration)} duration</span>`,
            ...Object.entries(labelCounts).map(([l, c]) =>
                `<span class="tl-stat" style="color:${labelColor(l)}">${c} ${l}</span>`)
        ].join('');

        // ── Legend
        const seenLabels = [...new Set(dets.map(d => d.label))];
        tlLegend.innerHTML = seenLabels.map(l =>
            `<span class="tl-leg-item"><span class="tl-leg-dot" style="background:${labelColor(l)}"></span>${l}</span>`
        ).join('');

        // ── Track: min 2400px wide, scale to duration
        const trackW = Math.max(2400, Math.ceil(_tlDuration * 8));
        tlTrack.style.width = trackW + 'px';
        tlTrack.innerHTML = '<div class="tl-cursor" id="tlCursor"></div>';

        // Timestamp labels every 10 s
        tlTimestamps.style.width = trackW + 'px';
        tlTimestamps.innerHTML = '';
        for (let t = 0; t <= _tlDuration; t += 10) {
            const pct = (t / _tlDuration) * 100;
            const span = document.createElement('span');
            span.className = 'tl-ts-label';
            span.style.left = pct + '%';
            span.textContent = formatTime(t);
            tlTimestamps.appendChild(span);
        }

        // Dots
        dets.forEach(d => {
            if (d.video_time == null) return;
            const pct = _tlDuration > 0 ? (d.video_time / _tlDuration) * 100 : 0;
            const dot = document.createElement('button');
            dot.className = 'tl-dot';
            dot.style.left = pct + '%';
            dot.style.background = labelColor(d.label);
            dot.title = `${d.label} — ${formatTime(d.video_time)} (conf ${(d.confidence * 100).toFixed(0)}%)`;
            dot.onclick = () => openVideoModal(source, d.video_time, d.label);
            tlTrack.appendChild(dot);
        });

    } catch (err) {
        tlSummary.innerHTML = 'Failed to load timeline.';
        console.error(err);
    }
}

// Sync the timeline cursor line to the video's current playback position
function syncTimelineCursor() {
    const video  = document.getElementById('mainVideo');
    const cursor = document.getElementById('tlCursor');
    if (!cursor || !video || _tlDuration === 0) return;
    const pct = (video.currentTime / _tlDuration) * 100;
    cursor.style.left = Math.min(100, pct) + '%';
}
document.getElementById('mainVideo').addEventListener('timeupdate', syncTimelineCursor);

// ── Export ────────────────────────────────────────────────────────────────
function exportDetectionsCSV() {
    const source = document.getElementById('detFilterSource')?.value || '';
    const url = source
        ? `${API}/export/detections?source=${encodeURIComponent(source)}`
        : `${API}/export/detections`;
    triggerDownload(url, source ? `detections_${source.split('/').pop()}.csv` : 'detections.csv');
}

function exportSearchResults() {
    const results = window._lastSearchResults;
    if (!results || results.length === 0) { alert('No search results to export.'); return; }
    const fields = ['id', 'timestamp', 'video_src', 'video_time', 'label', 'confidence', 'crop_path', 'score'];
    const rows = [fields.join(',')];
    results.forEach(r => {
        rows.push(fields.map(f => {
            const v = r[f] ?? '';
            return typeof v === 'string' && v.includes(',') ? `"${v}"` : v;
        }).join(','));
    });
    const blob = new Blob([rows.join('\n')], { type: 'text/csv' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `search_${window._lastSearchQuery || 'results'}.csv`;
    a.click();
    URL.revokeObjectURL(a.href);
}

function triggerDownload(url, filename) {
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}

// ── Upload Handlers ──────────────────────────────────────────────────────────
const fileInput = document.getElementById('fileInput');
const uploadZone = document.getElementById('uploadZone');

if (fileInput && uploadZone) {
    fileInput.addEventListener('change', e => {
        if (e.target.files.length > 0) uploadFile(e.target.files[0]);
    });

    uploadZone.addEventListener('dragover', e => {
        e.preventDefault();
        uploadZone.classList.add('dragover');
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('dragover');
    });

    uploadZone.addEventListener('drop', e => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) uploadFile(e.dataTransfer.files[0]);
    });
}

async function uploadFile(file) {
    const progWrap = document.getElementById('uploadProgress');
    const progBar = document.getElementById('uploadProgressBar');
    const progLabel = document.getElementById('uploadProgressLabel');

    progWrap.classList.remove('hidden');
    progBar.style.width = '0%';
    progLabel.textContent = `Uploading "${file.name}"… 0%`;

    const formData = new FormData();
    formData.append('file', file);

    try {
        const xhr = new XMLHttpRequest();
        xhr.open('POST', `${API}/upload`, true);

        xhr.upload.onprogress = e => {
            if (e.lengthComputable) {
                const pct = Math.round((e.loaded / e.total) * 100);
                progBar.style.width = pct + '%';
                progLabel.textContent = `Uploading "${file.name}"… ${pct}%`;
            }
        };

        xhr.onload = () => {
            if (xhr.status === 200) {
                const data = JSON.parse(xhr.responseText);
                progLabel.textContent = 'Upload complete! Starting pipeline…';
                setTimeout(() => {
                    progWrap.classList.add('hidden');
                    // Switch to progress view
                    document.getElementById('progressSection').style.display = 'block';
                    document.getElementById('processBtn').style.display = 'none';
                    document.getElementById('stopBtn').style.display = 'inline-flex';
                    _seenRunning = false;
                    startStatusPolling();
                }, 1000);
            } else {
                alert('Upload failed: ' + xhr.statusText);
                progWrap.classList.add('hidden');
            }
        };

        xhr.onerror = () => {
            alert('Network error during upload.');
            progWrap.classList.add('hidden');
        };

        xhr.send(formData);
    } catch (err) {
        console.error(err);
        progWrap.classList.add('hidden');
    }
}

// ── Init ─────────────────────────────────────────────────────────────────────
refreshIndexCount();
loadSources();

// ── Attribute pills for Detections tab ───────────────────────────────────────
document.getElementById('detGenderPills')?.addEventListener('click', e => {
    const pill = e.target.closest('.attr-pill');
    if (!pill) return;
    document.querySelectorAll('#detGenderPills .attr-pill').forEach(p => p.classList.remove('active'));
    pill.classList.add('active');
    loadDetections();
});

document.getElementById('detAccessoryPills')?.addEventListener('click', e => {
    const pill = e.target.closest('.attr-pill');
    if (!pill) return;
    document.querySelectorAll('#detAccessoryPills .attr-pill').forEach(p => p.classList.remove('active'));
    pill.classList.add('active');
    loadDetections();
});

// Re-load detections when dropdowns change
document.getElementById('detFilterLabel')?.addEventListener('change', loadDetections);
document.getElementById('detFilterSource')?.addEventListener('change', loadDetections);

// ── Person Re-ID Gallery ──────────────────────────────────────────────────────
async function loadPersons() {
    const grid = document.getElementById('personsGrid');
    const countEl = document.getElementById('personCount');
    grid.innerHTML = `<div class="empty-state"><div class="spinner"></div> Building gallery…</div>`;

    try {
        const groups = await (await fetch(`${API}/persons`)).json();

        if (!groups || groups.length === 0) {
            grid.innerHTML = `<div class="empty-state" id="personsEmpty">
              <div class="empty-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/></svg></div>
              <h3>No person detections yet</h3>
              <p>Process a video that contains people to build the gallery.</p>
            </div>`;
            if (countEl) countEl.textContent = '—';
            return;
        }

        if (countEl) countEl.textContent = `${groups.length} unique appearance${groups.length !== 1 ? 's' : ''}`;

        grid.innerHTML = groups.map(g => {
            const imgSrc = cropUrl(g.crop_path);
            const hasVideo = g.video_src && g.video_time != null;
            const clickAttr = hasVideo
                ? `onclick="openVideoModal('${g.video_src.replace(/'/g,"\\'")  }', ${g.video_time}, 'person')"`
                : '';
            const timeFmt = formatTime(g.video_time);
            const upperSwatch = `<span class="attr-swatch" style="background:${cssColorForName(g.upper_color)}" title="Upper: ${g.upper_color}"></span>`;
            const lowerSwatch = `<span class="attr-swatch" style="background:${cssColorForName(g.lower_color)}" title="Lower: ${g.lower_color}"></span>`;

            const badges = [
                `<span class="person-badge gender-badge">${g.gender === 'male' ? '♂' : g.gender === 'female' ? '♀' : '?'} ${g.gender}</span>`,
                g.has_hat ? `<span class="person-badge acc-badge">🎩 Hat</span>` : '',
                g.has_bag ? `<span class="person-badge acc-badge">👜 Bag</span>` : '',
            ].filter(Boolean).join('');

            return `
            <div class="person-group-card${hasVideo ? ' clickable' : ''}" ${clickAttr}>
              <div class="pgc-img-wrap">
                <img src="${imgSrc}" alt="person" loading="lazy" onerror="this.style.display='none'"/>
                ${hasVideo ? '<span class="play-overlay"><svg viewBox="0 0 24 24" fill="currentColor" width="22" height="22"><polygon points="5 3 19 12 5 21 5 3"/></svg></span>' : ''}
                <span class="pgc-count">${g.count} × seen</span>
              </div>
              <div class="pgc-info">
                <div class="pgc-swatches">${upperSwatch}${lowerSwatch}
                  <span class="pgc-color-text">${g.upper_color} / ${g.lower_color}</span>
                </div>
                <div class="pgc-badges">${badges}</div>
                ${timeFmt ? `<div class="result-meta">First seen @ ${timeFmt}</div>` : ''}
              </div>
            </div>`;
        }).join('');
    } catch (err) {
        grid.innerHTML = `<div class="empty-state"><p>Could not load person gallery. Is the backend running?</p></div>`;
        console.error(err);
    }
}

// ── Session History ───────────────────────────────────────────────────────────
async function loadSessions() {
    const wrap = document.getElementById('sessionsWrap');
    wrap.innerHTML = `<div class="empty-state"><div class="spinner"></div> Loading history…</div>`;

    try {
        const sessions = await (await fetch(`${API}/sessions`)).json();

        if (!sessions || sessions.length === 0) {
            wrap.innerHTML = `<div class="empty-state" id="sessionsEmpty">
              <div class="empty-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg></div>
              <h3>No sessions yet</h3>
              <p>Processing history will appear here once you run the pipeline.</p>
            </div>`;
            return;
        }

        const rows = sessions.map(s => {
            const started = s.started_at ? new Date(s.started_at).toLocaleString() : '—';
            const finished = s.finished_at ? new Date(s.finished_at).toLocaleString() : '—';
            // Duration in seconds
            let duration = '—';
            if (s.started_at && s.finished_at) {
                const secs = Math.round((new Date(s.finished_at) - new Date(s.started_at)) / 1000);
                duration = formatTime(secs) || `${secs}s`;
            }
            const srcName = s.source ? s.source.split('/').pop().split('\\').pop() : '—';
            const statusClass = s.status === 'finished' ? 'status-finished'
                              : s.status === 'running'  ? 'status-running'
                              : s.status === 'error'    ? 'status-error'
                              : 'status-idle';
            return `<tr>
              <td><span class="session-src" title="${s.source || ''}">📹 ${srcName}</span></td>
              <td>${started}</td>
              <td>${finished}</td>
              <td>${duration}</td>
              <td>${(s.total_detections || 0).toLocaleString()}</td>
              <td>${(s.processed_frames || 0).toLocaleString()}</td>
              <td><span class="session-status ${statusClass}">${s.status || 'unknown'}</span></td>
            </tr>`;
        }).join('');

        wrap.innerHTML = `
          <div class="sessions-table-wrap">
            <table class="sessions-table">
              <thead>
                <tr>
                  <th>Source</th><th>Started</th><th>Finished</th>
                  <th>Duration</th><th>Detections</th><th>Frames</th><th>Status</th>
                </tr>
              </thead>
              <tbody>${rows}</tbody>
            </table>
          </div>`;
    } catch (err) {
        wrap.innerHTML = `<div class="empty-state"><p>Could not load sessions. Is the backend running?</p></div>`;
        console.error(err);
    }
}
