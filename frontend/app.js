const API = 'http://localhost:8000';
let analyticsChart = null;
let statusPoller = null;
let _seenRunning = false; // guard: don't stop poller before 'running' is ever seen

// ── Tab Navigation ──────────────────────────────────────────────────────────
document.querySelectorAll('.nav-item').forEach(btn => {
    btn.addEventListener('click', () => {
        const tab = btn.dataset.tab;
        document.querySelectorAll('.nav-item').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById(`tab-${tab}`).classList.add('active');
        document.getElementById('pageTitle').textContent = btn.querySelector('span').textContent;

        if (tab === 'analytics') loadAnalytics();
        if (tab === 'detections') loadDetections();
    });
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

async function doSearch() {
    const query = document.getElementById('searchInput').value.trim();
    if (!query) return;

    const grid = document.getElementById('resultsGrid');
    const status = document.getElementById('searchStatus');
    const empty = document.getElementById('searchEmpty');

    grid.innerHTML = `<div class="empty-state"><div class="spinner"></div> Searching for "${query}"…</div>`;
    status.classList.add('hidden');

    try {
        const res = await fetch(`${API}/search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, top_k: 24 }),
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
            return;
        }

        grid.innerHTML = results.map(r => {
            const imgSrc = `${API}/${r.crop_path.replace(/\\/g, '/')}`;
            const ts = r.timestamp ? new Date(r.timestamp).toLocaleString() : '—';
            const score = (r.score * 100).toFixed(1);
            return `
        <div class="result-card">
          <div class="result-img-wrap">
            <img src="${imgSrc}" alt="${r.label}" loading="lazy"
                 onerror="this.style.display='none'"/>
            <span class="result-score-badge">${score}%</span>
          </div>
          <div class="result-info">
            <div class="result-label">${r.label}</div>
            <div class="result-meta">${ts}</div>
            <div class="result-meta">conf ${(r.confidence * 100).toFixed(1)}%</div>
          </div>
        </div>`;
        }).join('');
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
    } catch { /* silent */ }
}

function generateColors(n) {
    const palette = ['#06b6d4', '#3b82f6', '#8b5cf6', '#10b981', '#f59e0b', '#ef4444', '#ec4899', '#14b8a6', '#f97316', '#84cc16'];
    return Array.from({ length: n }, (_, i) => palette[i % palette.length]);
}

// ── Detections ──────────────────────────────────────────────────────────────
async function loadDetections() {
    const grid = document.getElementById('detectionsGrid');
    grid.innerHTML = `<div class="empty-state"><div class="spinner"></div> Loading…</div>`;

    try {
        const res = await fetch(`${API}/detections?limit=60`);
        const data = await res.json();

        if (!data || data.length === 0) {
            grid.innerHTML = `<div class="empty-state" id="detectionsEmpty">
        <div class="empty-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M9 9h6M9 13h6M9 17h4"/></svg></div>
        <p>No detections yet. Process a video to begin.</p>
      </div>`;
            return;
        }

        grid.innerHTML = data.map(r => {
            const imgSrc = `${API}/${(r.crop_path || '').replace(/\\/g, '/')}`;
            const ts = r.timestamp ? new Date(r.timestamp).toLocaleString() : '—';
            return `
        <div class="result-card">
          <div class="result-img-wrap">
            <img src="${imgSrc}" alt="${r.label}" loading="lazy" onerror="this.style.display='none'"/>
          </div>
          <div class="result-info">
            <div class="result-label">${r.label || '—'}</div>
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

// ── Init ─────────────────────────────────────────────────────────────────────
refreshIndexCount();
