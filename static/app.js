/* ==========================================
   Factor Analysis Dashboard — App Logic
   ========================================== */

// ==========================================
// State
// ==========================================

let factorInfo = {};
let fullChart = null;
let rollingChart = null;

// Chart color palette
const COLORS = {
    strategy: { line: '#58a6ff', fill: 'rgba(88,166,255,0.08)' },
    spy: { line: '#d29922', fill: 'rgba(210,153,34,0.05)' },
    qqq: { line: '#bc8cff', fill: 'rgba(188,140,255,0.05)' },
    rolling: [
        '#58a6ff', '#3fb950', '#f85149', '#d29922', '#bc8cff',
        '#f778ba', '#79c0ff', '#56d364', '#ffa657', '#ff7b72',
        '#a5d6ff', '#7ee787', '#ffdcd7', '#e3b341', '#d2a8ff',
        '#ffdfb6', '#a371f7', '#9ecbff', '#b1bac4', '#8b949e'
    ]
};

const CHART_DEFAULTS = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    plugins: {
        legend: {
            labels: {
                color: '#8b949e',
                font: { family: 'Inter', size: 11 },
                padding: 16,
                usePointStyle: true,
                pointStyleWidth: 10
            }
        },
        tooltip: {
            backgroundColor: '#21262d',
            titleColor: '#e6edf3',
            bodyColor: '#8b949e',
            borderColor: '#30363d',
            borderWidth: 1,
            cornerRadius: 8,
            padding: 10,
            bodyFont: { family: 'JetBrains Mono', size: 11 },
            titleFont: { family: 'Inter', size: 12, weight: 600 },
            callbacks: {
                label: ctx => `${ctx.dataset.label}: ${ctx.parsed.y?.toFixed(4) || '—'}`
            }
        }
    },
    scales: {
        x: {
            ticks: { color: '#6e7681', font: { size: 10 }, maxTicksLimit: 12 },
            grid: { color: 'rgba(48,54,61,0.5)', drawBorder: false }
        },
        y: {
            ticks: { color: '#6e7681', font: { family: 'JetBrains Mono', size: 10 } },
            grid: { color: 'rgba(48,54,61,0.5)', drawBorder: false }
        }
    }
};


// ==========================================
// Init
// ==========================================

document.addEventListener('DOMContentLoaded', () => {
    loadFactors();
    loadSaved();
    setupUniverseMode();
    createLoadingOverlay();
    createToastContainer();
});

async function loadFactors() {
    try {
        const res = await fetch('/api/factors');
        factorInfo = await res.json();
        renderFactorControls(factorInfo);
    } catch (e) {
        showToast('Failed to load factors: ' + e.message, 'error');
    }
}

function setupUniverseMode() {
    const modeSelect = document.getElementById('universeMode');
    const customGroup = document.getElementById('customTickersGroup');
    modeSelect.addEventListener('change', () => {
        customGroup.style.display = modeSelect.value === 'custom' ? 'block' : 'none';
    });
}


// ==========================================
// Factor Controls
// ==========================================

function renderFactorControls(info) {
    const container = document.getElementById('factorControls');
    container.innerHTML = '';

    for (const [key, factor] of Object.entries(info)) {
        const enabled = factor.default_weight > 0;
        const card = document.createElement('div');
        card.className = `factor-card ${enabled ? '' : 'disabled'}`;
        card.id = `factor-card-${key}`;

        let paramsHTML = '';
        for (const [pKey, pInfo] of Object.entries(factor.params)) {
            paramsHTML += `
                <div class="param-row">
                    <label title="${pInfo.label}">${pInfo.label}</label>
                    <input type="range" id="param-${key}-${pKey}"
                        min="${pInfo.min}" max="${pInfo.max}" step="${pInfo.step}" value="${pInfo.default}"
                        oninput="syncParamInput('${key}','${pKey}', this.value)"/>
                    <input type="number" class="param-value-input" id="paramVal-${key}-${pKey}"
                        value="${pInfo.default}" min="${pInfo.min}" max="${pInfo.max}" step="${pInfo.step}"
                        onchange="syncParamSlider('${key}','${pKey}', this.value)"/>
                </div>`;
        }

        card.innerHTML = `
            <div class="factor-header">
                <span class="factor-name">${factor.name}</span>
                <label class="toggle">
                    <input type="checkbox" id="enable-${key}" ${enabled ? 'checked' : ''}
                        onchange="toggleFactor('${key}', this.checked)" />
                    <span class="toggle-slider"></span>
                </label>
            </div>
            <div class="factor-desc">${factor.description}</div>
            <div class="factor-params">
                ${paramsHTML}
                <div class="weight-row">
                    <label>Weight</label>
                    <input type="range" id="weight-${key}"
                        min="0" max="1" step="0.01" value="${factor.default_weight}"
                        oninput="onWeightSliderChange('${key}', this.value)" />
                    <input type="number" class="weight-display" id="weightVal-${key}"
                        value="${factor.default_weight.toFixed(2)}" min="0" max="1" step="0.01"
                        onchange="onWeightInputChange('${key}', this.value)"/>
                </div>
            </div>`;

        container.appendChild(card);
    }
}

/* Sync param slider → input box */
function syncParamInput(factorKey, paramKey, val) {
    document.getElementById(`paramVal-${factorKey}-${paramKey}`).value = val;
}

/* Sync param input box → slider */
function syncParamSlider(factorKey, paramKey, val) {
    document.getElementById(`param-${factorKey}-${paramKey}`).value = val;
}

function toggleFactor(key, enabled) {
    const card = document.getElementById(`factor-card-${key}`);
    card.classList.toggle('disabled', !enabled);
    normalizeWeights();
}

/* When weight SLIDER changes: update input, then auto-normalize others */
function onWeightSliderChange(changedKey, value) {
    const v = parseFloat(value);
    document.getElementById(`weightVal-${changedKey}`).value = v.toFixed(2);
    autoNormalizeOthers(changedKey, v);
}

/* When weight INPUT BOX changes: update slider, then auto-normalize others */
function onWeightInputChange(changedKey, value) {
    let v = Math.max(0, Math.min(1, parseFloat(value) || 0));
    document.getElementById(`weight-${changedKey}`).value = v;
    document.getElementById(`weightVal-${changedKey}`).value = v.toFixed(2);
    autoNormalizeOthers(changedKey, v);
}

/**
 * Auto-normalize: keep changedKey's weight fixed,
 * proportionally adjust all OTHER enabled factors so total = 1.0
 */
function autoNormalizeOthers(changedKey, fixedValue) {
    const others = [];
    for (const key of Object.keys(factorInfo)) {
        if (key === changedKey) continue;
        const enableEl = document.getElementById(`enable-${key}`);
        if (enableEl && enableEl.checked) {
            const w = parseFloat(document.getElementById(`weight-${key}`).value) || 0;
            others.push({ key, weight: w });
        }
    }

    const remaining = Math.max(0, 1.0 - fixedValue);

    if (others.length === 0) return;

    const otherSum = others.reduce((s, o) => s + o.weight, 0);

    for (const o of others) {
        let newW;
        if (otherSum > 1e-8) {
            // Proportional redistribution
            newW = (o.weight / otherSum) * remaining;
        } else {
            // All others were zero — equal split
            newW = remaining / others.length;
        }
        newW = Math.round(newW * 100) / 100; // round to 0.01

        document.getElementById(`weight-${o.key}`).value = newW;
        document.getElementById(`weightVal-${o.key}`).value = newW.toFixed(2);
    }
}

/**
 * Auto-normalize weights of all enabled factors to sum to 1.0
 * Called when toggling a factor on/off — redistributes equally among enabled only.
 * Disabled factors keep their weight untouched.
 */
function normalizeWeights() {
    const enabledFactors = [];

    for (const key of Object.keys(factorInfo)) {
        const enableEl = document.getElementById(`enable-${key}`);
        if (enableEl && enableEl.checked) {
            enabledFactors.push(key);
        }
    }

    if (enabledFactors.length === 0) return;

    const equalWeight = Math.round((1.0 / enabledFactors.length) * 100) / 100;

    for (const key of enabledFactors) {
        const weightEl = document.getElementById(`weight-${key}`);
        const dispEl = document.getElementById(`weightVal-${key}`);
        if (!weightEl || !dispEl) continue;

        weightEl.value = equalWeight;
        dispEl.value = equalWeight.toFixed(2);
    }
}


// ==========================================
// Gather Config
// ==========================================

function gatherFactorConfig() {
    const config = {};
    for (const key of Object.keys(factorInfo)) {
        const enableEl = document.getElementById(`enable-${key}`);
        const weightEl = document.getElementById(`weight-${key}`);
        const enabled = enableEl?.checked || false;
        const weight = parseFloat(weightEl?.value || 0);

        const params = { enabled, weight };
        for (const pKey of Object.keys(factorInfo[key].params)) {
            const el = document.getElementById(`param-${key}-${pKey}`);
            params[pKey] = parseInt(el?.value || factorInfo[key].params[pKey].default);
        }
        config[key] = params;
    }
    return config;
}

function gatherFullConfig() {
    const universeMode = document.getElementById('universeMode').value;
    let tickers = null;
    if (universeMode === 'custom') {
        tickers = document.getElementById('customTickers').value;
    }

    return {
        factors: gatherFactorConfig(),
        rebalance: document.getElementById('rebalanceFreq').value,
        years: parseInt(document.getElementById('backtestYears').value),
        top_n: parseInt(document.getElementById('topN').value),
        tickers: tickers,
        universe_mode: universeMode,
        rolling_window: parseInt(document.getElementById('rollingWindow').value)
    };
}


// ==========================================
// Run Backtest
// ==========================================

async function runBacktest() {
    const config = gatherFullConfig();
    const btn = document.getElementById('btnRunBacktest');

    // Validate at least one factor enabled
    const anyEnabled = Object.values(config.factors).some(f => f.enabled && f.weight > 0);
    if (!anyEnabled) {
        showToast('Enable at least one factor with weight > 0', 'error');
        return;
    }

    setStatus('running', 'Running backtest...');
    showLoading('Running backtest — downloading data & computing factors...');
    btn.disabled = true;

    try {
        // Run main backtest
        const res = await fetch('/api/backtest', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });
        const data = await res.json();

        if (data.status === 'error') {
            throw new Error(data.message);
        }

        renderFullChart(data.data);
        renderStats(data.data.stats);
        renderTradeSummary(data.data.trade_summary);
        renderHoldingsHistory(data.data.holdings_history);
        showToast('Backtest completed successfully!', 'success');

        // Run rolling if enabled
        if (config.rolling_window > 0) {
            updateLoading('Running rolling window analysis...');
            const rollingRes = await fetch('/api/rolling', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    ...config,
                    window_years: config.rolling_window
                })
            });
            const rollingData = await rollingRes.json();

            if (rollingData.status === 'ok') {
                renderRollingChart(rollingData.windows);
                renderRollingStats(rollingData.windows);
            }
        } else {
            hideRolling();
        }

        setStatus('ready', 'Complete');

    } catch (e) {
        console.error(e);
        showToast('Backtest failed: ' + e.message, 'error');
        setStatus('error', 'Failed');
    } finally {
        hideLoading();
        btn.disabled = false;
    }
}


// ==========================================
// Charts
// ==========================================

function renderFullChart(data) {
    const wrapper = document.getElementById('fullChart').parentElement;
    const placeholder = document.getElementById('fullChartPlaceholder');

    wrapper.classList.add('visible');
    placeholder.classList.add('hidden');

    const strategyData = parseEquityData(data.strategy);
    const spyData = parseEquityData(data.spy);
    const qqqData = parseEquityData(data.qqq);

    if (fullChart) fullChart.destroy();

    const ctx = document.getElementById('fullChart').getContext('2d');
    fullChart = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: [
                {
                    label: 'Strategy',
                    data: strategyData,
                    borderColor: COLORS.strategy.line,
                    backgroundColor: COLORS.strategy.fill,
                    borderWidth: 2.5,
                    fill: true,
                    pointRadius: 0,
                    tension: 0.1
                },
                {
                    label: 'SPY',
                    data: spyData,
                    borderColor: COLORS.spy.line,
                    backgroundColor: 'transparent',
                    borderWidth: 1.5,
                    borderDash: [5, 3],
                    pointRadius: 0,
                    tension: 0.1
                },
                {
                    label: 'QQQ',
                    data: qqqData,
                    borderColor: COLORS.qqq.line,
                    backgroundColor: 'transparent',
                    borderWidth: 1.5,
                    borderDash: [5, 3],
                    pointRadius: 0,
                    tension: 0.1
                }
            ]
        },
        options: {
            ...CHART_DEFAULTS,
            scales: {
                x: {
                    type: 'time',
                    time: { unit: 'month', tooltipFormat: 'yyyy-MM-dd' },
                    ticks: { color: '#6e7681', font: { size: 10 }, maxTicksLimit: 14 },
                    grid: { color: 'rgba(48,54,61,0.5)', drawBorder: false }
                },
                y: {
                    ...CHART_DEFAULTS.scales.y,
                    title: { display: true, text: 'Growth of $1', color: '#6e7681', font: { size: 11 } }
                }
            }
        }
    });
}

function renderRollingChart(windows) {
    const wrapper = document.getElementById('rollingChart').parentElement;
    const placeholder = document.getElementById('rollingChartPlaceholder');

    wrapper.classList.add('visible');
    placeholder.classList.add('hidden');

    if (rollingChart) rollingChart.destroy();

    const datasets = windows.map((w, i) => {
        const data = Object.entries(w.equity).map(([_, val], idx) => ({
            x: idx,
            y: val
        }));

        return {
            label: w.label,
            data: data,
            borderColor: COLORS.rolling[i % COLORS.rolling.length],
            backgroundColor: 'transparent',
            borderWidth: 1.5,
            pointRadius: 0,
            tension: 0.1
        };
    });

    const ctx = document.getElementById('rollingChart').getContext('2d');
    rollingChart = new Chart(ctx, {
        type: 'line',
        data: { datasets },
        options: {
            ...CHART_DEFAULTS,
            scales: {
                x: {
                    type: 'linear',
                    title: { display: true, text: 'Trading Days', color: '#6e7681', font: { size: 11 } },
                    ticks: { color: '#6e7681', font: { size: 10 } },
                    grid: { color: 'rgba(48,54,61,0.5)', drawBorder: false }
                },
                y: {
                    ...CHART_DEFAULTS.scales.y,
                    title: { display: true, text: 'Growth (normalized)', color: '#6e7681', font: { size: 11 } }
                }
            }
        }
    });
}

function renderRollingStats(windows) {
    const container = document.getElementById('rollingStatsContainer');
    const tbody = document.getElementById('rollingStatsBody');
    container.style.display = 'block';
    tbody.innerHTML = '';

    for (const w of windows) {
        const s = w.stats || {};
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td style="color: var(--text-primary); font-weight: 500;">${w.label}</td>
            <td class="${(s.calmar || 0) >= 1 ? 'positive' : ''}" style="color: ${(s.calmar || 0) >= 1 ? 'var(--accent-green)' : 'var(--text-secondary)'}">
                ${s.calmar?.toFixed(2) || '—'}
            </td>
            <td class="${s.cagr >= 0 ? 'positive' : 'negative'}" style="color: ${s.cagr >= 0 ? 'var(--accent-green)' : 'var(--accent-red)'}">
                ${formatPct(s.cagr)}
            </td>
            <td style="color: var(--accent-red)">${formatPct(s.max_dd)}</td>
            <td>${s.sharpe?.toFixed(2) || '—'}</td>
            <td>${formatPct(s.win_rate)}</td>`;
        tbody.appendChild(tr);
    }
}

function hideRolling() {
    document.getElementById('rollingChart').parentElement.classList.remove('visible');
    document.getElementById('rollingChartPlaceholder').classList.remove('hidden');
    document.getElementById('rollingStatsContainer').style.display = 'none';
}


// ==========================================
// Stats
// ==========================================

function renderStats(stats) {
    const grid = document.getElementById('statsGrid');
    grid.style.display = 'grid';

    const s = stats.strategy || {};
    setStatValue('statCalmar', s.calmar?.toFixed(2) || '—', s.calmar);
    setStatValue('statCAGR', formatPct(s.cagr), s.cagr);
    setStatValue('statMaxDD', formatPct(s.max_dd), s.max_dd);
    setStatValue('statSharpe', s.sharpe?.toFixed(2) || '—', s.sharpe);
    setStatValue('statWinRate', formatPct(s.win_rate), s.win_rate - 0.5);

    const spy = stats.spy || {};
    setStatValue('statSPY_CAGR', formatPct(spy.cagr), spy.cagr);
    setStatValue('statSPY_MaxDD', formatPct(spy.max_dd), spy.max_dd);
    setStatValue('statSPY_Sharpe', spy.sharpe?.toFixed(2) || '—', spy.sharpe);

    const qqq = stats.qqq || {};
    setStatValue('statQQQ_CAGR', formatPct(qqq.cagr), qqq.cagr);
    setStatValue('statQQQ_MaxDD', formatPct(qqq.max_dd), qqq.max_dd);
    setStatValue('statQQQ_Sharpe', qqq.sharpe?.toFixed(2) || '—', qqq.sharpe);
}

function setStatValue(id, text, numValue) {
    const el = document.getElementById(id);
    if (!el) return;
    el.textContent = text;
    el.className = 'stat-val';
    if (numValue > 0) el.classList.add('positive');
    else if (numValue < 0) el.classList.add('negative');
}


// ==========================================
// Save / Load
// ==========================================

async function saveFactorConfig() {
    const name = document.getElementById('saveName').value.trim();
    if (!name) { showToast('Enter a name for the config', 'error'); return; }

    try {
        const res = await fetch('/api/save-factor', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, config: gatherFactorConfig() })
        });
        const data = await res.json();
        showToast(data.message, data.status === 'ok' ? 'success' : 'error');
        if (data.status === 'ok') {
            document.getElementById('saveName').value = '';
            loadSaved();
        }
    } catch (e) {
        showToast('Save failed: ' + e.message, 'error');
    }
}

async function saveStrategy() {
    const name = document.getElementById('saveName').value.trim();
    if (!name) { showToast('Enter a name for the strategy', 'error'); return; }

    try {
        const res = await fetch('/api/save-strategy', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, strategy: gatherFullConfig() })
        });
        const data = await res.json();
        showToast(data.message, data.status === 'ok' ? 'success' : 'error');
        if (data.status === 'ok') {
            document.getElementById('saveName').value = '';
            loadSaved();
        }
    } catch (e) {
        showToast('Save failed: ' + e.message, 'error');
    }
}

async function loadSaved() {
    try {
        const res = await fetch('/api/saved');
        const data = await res.json();
        renderSavedFactors(data.factors || {});
        renderSavedStrategies(data.strategies || {});
    } catch (e) {
        console.error('Failed to load saved:', e);
    }
}

function renderSavedFactors(factors) {
    const container = document.getElementById('savedFactorsList');
    if (Object.keys(factors).length === 0) {
        container.innerHTML = '<p class="empty-state">No saved factor configs</p>';
        return;
    }

    container.innerHTML = '';
    for (const [name, info] of Object.entries(factors)) {
        const date = info.saved_at ? new Date(info.saved_at).toLocaleDateString() : '';
        const item = document.createElement('div');
        item.className = 'saved-item';
        item.innerHTML = `
            <div class="saved-item-name" onclick="loadFactorConfig('${escapeHtml(name)}')">
                ${escapeHtml(name)}
                <span class="saved-item-date">${date}</span>
            </div>
            <div class="saved-item-actions">
                <button class="btn btn-danger" onclick="deleteSaved('${escapeHtml(name)}', 'factor')">✕</button>
            </div>`;
        container.appendChild(item);
    }
}

function renderSavedStrategies(strategies) {
    const container = document.getElementById('savedStrategiesList');
    if (Object.keys(strategies).length === 0) {
        container.innerHTML = '<p class="empty-state">No saved strategies</p>';
        return;
    }

    container.innerHTML = '';
    for (const [name, info] of Object.entries(strategies)) {
        const date = info.saved_at ? new Date(info.saved_at).toLocaleDateString() : '';
        const item = document.createElement('div');
        item.className = 'saved-item';
        item.innerHTML = `
            <div class="saved-item-name" onclick="loadStrategy('${escapeHtml(name)}')">
                ${escapeHtml(name)}
                <span class="saved-item-date">${date}</span>
            </div>
            <div class="saved-item-actions">
                <button class="btn btn-danger" onclick="deleteSaved('${escapeHtml(name)}', 'strategy')">✕</button>
            </div>`;
        container.appendChild(item);
    }
}

async function loadFactorConfig(name) {
    try {
        const res = await fetch('/api/saved');
        const data = await res.json();
        const config = data.factors?.[name]?.config;
        if (!config) { showToast('Config not found', 'error'); return; }
        applyFactorConfig(config);
        showToast(`Loaded factor config "${name}"`, 'info');
    } catch (e) {
        showToast('Load failed: ' + e.message, 'error');
    }
}

async function loadStrategy(name) {
    try {
        const res = await fetch('/api/saved');
        const data = await res.json();
        const strategy = data.strategies?.[name]?.strategy;
        if (!strategy) { showToast('Strategy not found', 'error'); return; }
        applyFullConfig(strategy);
        showToast(`Loaded strategy "${name}"`, 'info');
    } catch (e) {
        showToast('Load failed: ' + e.message, 'error');
    }
}

function applyFactorConfig(config) {
    for (const [key, params] of Object.entries(config)) {
        const enableEl = document.getElementById(`enable-${key}`);
        if (enableEl) {
            enableEl.checked = params.enabled;
            toggleFactor(key, params.enabled);
        }

        const weightEl = document.getElementById(`weight-${key}`);
        if (weightEl) {
            weightEl.value = params.weight;
            const disp = document.getElementById(`weightVal-${key}`);
            if (disp) disp.textContent = parseFloat(params.weight).toFixed(2);
        }

        for (const [pKey, pVal] of Object.entries(params)) {
            if (pKey === 'enabled' || pKey === 'weight') continue;
            const el = document.getElementById(`param-${key}-${pKey}`);
            if (el) {
                el.value = pVal;
                const disp = document.getElementById(`paramVal-${key}-${pKey}`);
                if (disp) disp.textContent = pVal;
            }
        }
    }
}

function applyFullConfig(strategy) {
    if (strategy.factors) applyFactorConfig(strategy.factors);

    if (strategy.rebalance) document.getElementById('rebalanceFreq').value = strategy.rebalance;
    if (strategy.years) document.getElementById('backtestYears').value = strategy.years;
    if (strategy.top_n) document.getElementById('topN').value = strategy.top_n;
    if (strategy.rolling_window !== undefined) document.getElementById('rollingWindow').value = strategy.rolling_window;

    if (strategy.universe_mode) {
        document.getElementById('universeMode').value = strategy.universe_mode;
        document.getElementById('customTickersGroup').style.display =
            strategy.universe_mode === 'custom' ? 'block' : 'none';
    }
    if (strategy.tickers) {
        document.getElementById('customTickers').value = strategy.tickers;
    }
}

async function deleteSaved(name, type) {
    try {
        await fetch('/api/delete-saved', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, type })
        });
        loadSaved();
        showToast(`Deleted "${name}"`, 'info');
    } catch (e) {
        showToast('Delete failed: ' + e.message, 'error');
    }
}

// ==========================================
// Trade Summary & Holdings History
// ==========================================

function renderTradeSummary(summary) {
    const container = document.getElementById('tradeSummaryContainer');
    const grid = document.getElementById('tradeSummaryGrid');
    if (!summary || summary.total_ops === 0) {
        container.style.display = 'none';
        return;
    }
    container.style.display = '';

    const fmtRet = (v) => {
        const sign = v >= 0 ? '+' : '';
        const cls = v >= 0 ? 'positive' : 'negative';
        return `<span class="return ${cls}">${sign}${v.toFixed(2)}%</span>`;
    };

    const tradeList = (trades) => trades.map(t =>
        `<div class="trade-item">
            <span class="ticker">${t.ticker}</span>
            ${fmtRet(t.return_pct)}
            <span class="period-label">${t.period.split(' ~ ')[0]}</span>
        </div>`
    ).join('');

    grid.innerHTML = `
        <div class="summary-card">
            <h4>Overview</h4>
            <div class="summary-stat-row">
                <span class="label">Total Trades</span>
                <span class="value">${summary.total_ops}</span>
            </div>
            <div class="summary-stat-row">
                <span class="label">Periods</span>
                <span class="value">${summary.total_periods}</span>
            </div>
            <div class="summary-stat-row">
                <span class="label">Win Rate</span>
                <span class="value ${summary.win_rate >= 50 ? 'positive' : 'negative'}">${summary.win_rate}%</span>
            </div>
            <div class="summary-stat-row">
                <span class="label">Wins / Losses</span>
                <span class="value"><span class="positive">${summary.win_count}</span> / <span class="negative">${summary.loss_count}</span></span>
            </div>
        </div>
        <div class="summary-card">
            <h4>🏆 Top Gainers</h4>
            ${tradeList(summary.top_gainers)}
        </div>
        <div class="summary-card">
            <h4>📉 Top Losers</h4>
            ${tradeList(summary.top_losers)}
        </div>
    `;
}

function renderHoldingsHistory(periods) {
    const container = document.getElementById('holdingsHistoryContainer');
    const scroll = document.getElementById('holdingsScroll');
    const countEl = document.getElementById('periodCount');
    if (!periods || periods.length === 0) {
        container.style.display = 'none';
        return;
    }
    container.style.display = '';
    countEl.textContent = `(${periods.length} periods)`;

    scroll.innerHTML = periods.map((period, idx) => {
        const isRecent = idx === 0;
        const rows = period.holdings.map(h => {
            const sign = h.return_pct >= 0 ? '+' : '';
            const cls = h.return_pct >= 0 ? 'positive' : 'negative';
            return `<div class="holding-row">
                <span class="ticker">${h.ticker}</span>
                <span class="return ${cls}">${sign}${h.return_pct.toFixed(2)}%</span>
            </div>`;
        }).join('');

        return `<div class="period-card ${isRecent ? 'most-recent' : ''}">
            <div class="period-card-header">${isRecent ? '★ ' : ''}${period.label}</div>
            <div class="period-card-body">${rows}</div>
        </div>`;
    }).join('');
}


// ==========================================
// Utilities
// ==========================================

function parseEquityData(equityObj) {
    if (!equityObj) return [];
    return Object.entries(equityObj).map(([date, val]) => ({
        x: new Date(date),
        y: val
    }));
}

function formatPct(val) {
    if (val === null || val === undefined) return '—';
    return (val * 100).toFixed(2) + '%';
}

function escapeHtml(str) {
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;').replace(/'/g, '&#039;');
}

function setStatus(type, text) {
    const pill = document.getElementById('statusPill');
    const textEl = document.getElementById('statusText');
    pill.className = 'status-pill';
    if (type === 'running') pill.classList.add('running');
    if (type === 'error') pill.classList.add('error');
    textEl.textContent = text;
}


// ==========================================
// Loading Overlay
// ==========================================

function createLoadingOverlay() {
    const overlay = document.createElement('div');
    overlay.className = 'loading-overlay';
    overlay.id = 'loadingOverlay';
    overlay.innerHTML = `
        <div class="spinner"></div>
        <div class="loading-text" id="loadingText">Loading...</div>`;
    document.body.appendChild(overlay);
}

function showLoading(text) {
    const overlay = document.getElementById('loadingOverlay');
    document.getElementById('loadingText').textContent = text || 'Loading...';
    overlay.classList.add('visible');
}

function updateLoading(text) {
    document.getElementById('loadingText').textContent = text;
}

function hideLoading() {
    document.getElementById('loadingOverlay').classList.remove('visible');
}


// ==========================================
// Toast Notifications
// ==========================================

function createToastContainer() {
    const container = document.createElement('div');
    container.className = 'toast-container';
    container.id = 'toastContainer';
    document.body.appendChild(container);
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(100%)';
        toast.style.transition = 'all 0.3s';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}
