/* ═══════════════════════════════════════════════════════════════════
   CityMind — Frontend Controller & Canvas Renderer
   ═══════════════════════════════════════════════════════════════════ */

// ── State ─────────────────────────────────────────────────────────────
let currentState = null;
let isLoading = false;

// ── Color Map ─────────────────────────────────────────────────────────
const TYPE_COLORS = {
    'Residential':    '#4A9EFF',
    'Hospital':       '#FF6B6B',
    'School':         '#FBBF24',
    'Industrial':     '#6B7280',
    'PowerPlant':     '#FF8C42',
    'AmbulanceDepot': '#34D399',
    'Empty':          '#1E293B'
};

const RISK_COLORS = {
    0: 'rgba(52, 211, 153, 0.45)',   // Low  — green
    1: 'rgba(251, 191, 36, 0.45)',   // Medium — yellow
    2: 'rgba(244, 63, 94, 0.50)'     // High — red
};

// ── Icons (emoji-style for canvas) ───────────────────────────────────
const TYPE_ICONS = {
    'Hospital':       '🏥',
    'School':         '🏫',
    'Industrial':     '🏭',
    'PowerPlant':     '⚡',
    'AmbulanceDepot': '🚑',
};

// ── API Calls ─────────────────────────────────────────────────────────

function showLoading(msg) {
    isLoading = true;
    const overlay = document.createElement('div');
    overlay.className = 'loading-overlay';
    overlay.id = 'loading-overlay';
    overlay.innerHTML = `<div class="loading-spinner"></div><div class="loading-text">${msg}</div>`;
    document.body.appendChild(overlay);
}

function hideLoading() {
    isLoading = false;
    const el = document.getElementById('loading-overlay');
    if (el) el.remove();
}

async function apiCall(endpoint, method = 'POST') {
    const res = await fetch(`/api/${endpoint}`, { method });
    return res.json();
}

async function initializeSimulation() {
    if (isLoading) return;
    showLoading('Initializing CityMind — Running CSP, GA, ML pipelines...');
    try {
        const data = await apiCall('initialize');
        currentState = data;
        updateUI(data);
        document.getElementById('btn-step').disabled = false;
        document.getElementById('btn-run-all').disabled = false;
    } catch (e) {
        console.error(e);
        addLogEntry('Error initializing: ' + e.message, 'error');
    }
    hideLoading();
}

async function simulationStep() {
    if (isLoading) return;
    try {
        const data = await apiCall('step');
        currentState = data;
        updateUI(data);
    } catch (e) {
        console.error(e);
    }
}

async function runAll() {
    if (isLoading) return;
    showLoading('Running full 20-step simulation...');
    try {
        const data = await apiCall('run_all');
        currentState = data;
        updateUI(data);
    } catch (e) {
        console.error(e);
    }
    hideLoading();
}

async function resetSimulation() {
    try {
        const data = await apiCall('reset');
        currentState = data;
        updateUI(data);
        document.getElementById('btn-step').disabled = true;
        document.getElementById('btn-run-all').disabled = true;
        document.getElementById('event-log').innerHTML = '';
    } catch (e) {
        console.error(e);
    }
}

async function modifyConstraint(constraint, value) {
    if (isLoading) return;
    showLoading(`Applying modification: ${constraint} = ${value}...`);
    try {
        const res = await fetch('/api/update_constraint', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ constraint, value })
        });
        const data = await res.json();
        currentState = data;
        updateUI(data);
        document.getElementById('btn-step').disabled = false;
        document.getElementById('btn-run-all').disabled = false;
    } catch (e) {
        console.error(e);
    }
    hideLoading();
}

// ── UI Update ─────────────────────────────────────────────────────────

function updateUI(data) {
    // Status chip
    const chip = document.getElementById('status-chip');
    const statusText = document.getElementById('status-text');
    chip.className = 'status-chip';
    if (data.state === 'initialized' || data.state === 'running') {
        chip.classList.add('running');
        statusText.textContent = data.state === 'initialized' ? 'Ready' : 'Running';
    } else if (data.state === 'complete') {
        chip.classList.add('complete');
        statusText.textContent = 'Complete';
    } else {
        statusText.textContent = 'Idle';
    }

    // Step counter
    document.getElementById('step-current').textContent = data.current_step;
    document.getElementById('step-total').textContent = data.total_steps;

    // Stats
    if (data.results) {
        const rn = data.results.road_network;
        const amb = data.results.ambulance_placement;
        const rs = data.router_status;
        document.getElementById('stat-roads').textContent = rn ? rn.edge_count : '—';
        document.getElementById('stat-cost').textContent = rn ? rn.total_cost : '—';
        document.getElementById('stat-ambulances').textContent = amb ? amb.num_ambulances : '—';
        document.getElementById('stat-rescued').textContent = rs ? rs.civilians_rescued.length : '0';
        document.getElementById('stat-blocked').textContent = data.blocked_roads ? data.blocked_roads.length : '0';
    }

    // Challenge Results
    updateChallengeResults(data);

    // Event Log
    updateEventLog(data);

    // Render the grid
    renderGrid();

    // Disable buttons if complete
    if (data.state === 'complete') {
        document.getElementById('btn-step').disabled = true;
        document.getElementById('btn-run-all').disabled = true;
    }
}

function updateChallengeResults(data) {
    const r = data.results || {};

    // C1: Layout
    const c1 = document.getElementById('result-c1');
    if (r.layout) {
        const layout = r.layout;
        if (layout.success) {
            c1.className = 'result-item success';
            c1.querySelector('.result-text').textContent =
                `Layout ✓ Hops: H=${layout.hospital_hops}, P=${layout.powerplant_hops}`;
        } else {
            c1.className = 'result-item running';
            const conflict = layout.conflict || {};
            let text = `Layout ⚠ ${conflict.rule || 'Conflict'}`;

            // Show proposed solution if available
            if (conflict.proposed_solution) {
                const sol = conflict.proposed_solution;
                text += ` | Fix: ${sol.suggestion}`;
                console.log('CSP Conflict Proposal:', sol);
            }

            c1.querySelector('.result-text').textContent = text;
        }
    }

    // C2: Roads
    const c2 = document.getElementById('result-c2');
    if (r.road_network) {
        c2.className = 'result-item success';
        c2.querySelector('.result-text').textContent =
            `Roads ✓ ${r.road_network.edge_count} edges, cost=${r.road_network.total_cost}, ` +
            `${r.road_network.redundancy_satisfied ? '2 redundant paths ✓' : 'redundancy ✗'}`;
    }

    // C3: Ambulances
    const c3 = document.getElementById('result-c3');
    if (r.ambulance_placement) {
        c3.className = 'result-item success';
        c3.querySelector('.result-text').textContent =
            `Ambulances ✓ Worst-case: ${r.ambulance_placement.max_response_distance}, ` +
            `Avg: ${r.ambulance_placement.avg_response_distance}`;
    }

    // C4: Routing
    const c4 = document.getElementById('result-c4');
    if (data.router_status) {
        const rs = data.router_status;
        c4.className = 'result-item ' + (rs.mission_complete ? 'success' : 'running');
        c4.querySelector('.result-text').textContent = rs.mission_complete
            ? `Routing ✓ All rescued! Cost: ${rs.total_cost}`
            : `Routing: ${rs.civilians_rescued.length}/${rs.civilians_rescued.length + rs.civilians_remaining.length} rescued`;
    }

    // C5: Crime
    const c5 = document.getElementById('result-c5');
    if (data.crime_risk_distribution) {
        c5.className = 'result-item success';
        const d = data.crime_risk_distribution;
        c5.querySelector('.result-text').textContent =
            `Crime ML ✓ H:${d.High} M:${d.Medium} L:${d.Low}`;
    }

    // Feature Importances (Decision Tree explainability)
    if (data.feature_importances) {
        const fi = data.feature_importances;
        const topFeature = Object.entries(fi).sort((a, b) => b[1] - a[1])[0];
        if (topFeature) {
            console.log(`Top risk factor: ${topFeature[0]} (${(topFeature[1]*100).toFixed(1)}%)`);
        }
    }

    // C5b: Police Deployment
    const c5b = document.getElementById('result-c5b');
    if (r.police_deployment) {
        c5b.className = 'result-item success';
        const pd = r.police_deployment;
        c5b.querySelector('.result-text').textContent =
            `Police ✓ ${pd.num_officers} officers (High:${pd.high_risk_covered})`;
    }
}

function updateEventLog(data) {
    const logEl = document.getElementById('event-log');
    const logs = data.step_log || [];

    logEl.innerHTML = '';
    logs.forEach(entry => {
        const msg = entry.message || '';
        let cls = 'log-entry';
        if (msg.includes('✓')) cls += ' success';
        else if (msg.includes('⚠') || msg.includes('WARNING')) cls += ' warning';
        else if (msg.includes('ERROR')) cls += ' error';
        else if (msg.includes('FLOODING') || msg.includes('🌊')) cls += ' flood';
        else if (msg.includes('═══')) cls += ' info';

        const div = document.createElement('div');
        div.className = cls;
        div.innerHTML = `<span class="log-step">[${entry.step}]</span>${escapeHtml(msg)}`;
        logEl.appendChild(div);
    });

    logEl.scrollTop = logEl.scrollHeight;
}

function addLogEntry(msg, type = 'info') {
    const logEl = document.getElementById('event-log');
    const div = document.createElement('div');
    div.className = `log-entry ${type}`;
    div.textContent = msg;
    logEl.appendChild(div);
    logEl.scrollTop = logEl.scrollHeight;
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

// ── Canvas Rendering ──────────────────────────────────────────────────

function renderGrid() {
    const canvas = document.getElementById('city-canvas');
    const ctx = canvas.getContext('2d');

    if (!currentState || !currentState.graph) {
        // Draw empty grid
        ctx.fillStyle = '#111827';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#64748b';
        ctx.font = '500 16px Inter';
        ctx.textAlign = 'center';
        ctx.fillText('Click "Initialize City" to begin', canvas.width / 2, canvas.height / 2);
        return;
    }

    const g = currentState.graph;
    const W = g.width;
    const H = g.height;
    const padding = 30;
    const cellW = (canvas.width - padding * 2) / W;
    const cellH = (canvas.height - padding * 2) / H;

    // Clear
    ctx.fillStyle = '#0d1117';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Grid coordinates label
    ctx.fillStyle = '#475569';
    ctx.font = '500 10px JetBrains Mono';
    ctx.textAlign = 'center';
    for (let x = 0; x < W; x++) {
        ctx.fillText(x, padding + x * cellW + cellW / 2, padding - 10);
    }
    ctx.textAlign = 'right';
    for (let y = 0; y < H; y++) {
        ctx.fillText(y, padding - 8, padding + y * cellH + cellH / 2 + 4);
    }

    // Build lookup maps
    const nodeMap = {};
    g.nodes.forEach(n => { nodeMap[`${n.x},${n.y}`] = n; });

    const showRoads = document.getElementById('toggle-roads').checked;
    const showAmbulances = document.getElementById('toggle-ambulances').checked;
    const showHeatmap = document.getElementById('toggle-heatmap').checked;
    const showCivilians = document.getElementById('toggle-civilians').checked;

    // ── Draw cells ─────────────────────────────────────────────────
    g.nodes.forEach(n => {
        const cx = padding + n.x * cellW;
        const cy = padding + n.y * cellH;

        // Base color
        const baseColor = TYPE_COLORS[n.type] || TYPE_COLORS['Empty'];

        // Cell background
        ctx.fillStyle = baseColor;
        ctx.globalAlpha = 0.85;
        ctx.beginPath();
        ctx.roundRect(cx + 1.5, cy + 1.5, cellW - 3, cellH - 3, 4);
        ctx.fill();
        ctx.globalAlpha = 1.0;

        // Cell border
        ctx.strokeStyle = 'rgba(255,255,255,0.08)';
        ctx.lineWidth = 0.5;
        ctx.stroke();

        // Crime heatmap overlay - continuous gradient based on risk_index
        if (showHeatmap && n.risk_index > 0) {
            // Calculate alpha based on risk index (0.1 to 0.9)
            const alpha = Math.min(0.7, n.risk_index * 0.8);

            // Color based on risk level with smooth transition
            let riskColor;
            if (n.risk_index > 0.7) {
                riskColor = 'rgba(239, 68, 68, ';  // High - Red
            } else if (n.risk_index > 0.3) {
                riskColor = 'rgba(251, 191, 36, ';  // Medium - Yellow/Orange
            } else {
                riskColor = 'rgba(34, 197, 94, ';   // Low - Green
            }

            ctx.fillStyle = riskColor + alpha + ')';
            ctx.beginPath();
            ctx.roundRect(cx + 1.5, cy + 1.5, cellW - 3, cellH - 3, 4);
            ctx.fill();

            // Add risk value text for high-risk areas
            if (n.risk_index > 0.5 && cellW > 25) {
                ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
                ctx.font = 'bold 9px Inter';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText((n.risk_index * 100).toFixed(0) + '%', cx + cellW/2, cy + cellH/2);
            }
        }

        // Type icon
        const icon = TYPE_ICONS[n.type];
        if (icon) {
            ctx.font = `${Math.max(14, cellW * 0.4)}px serif`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(icon, cx + cellW / 2, cy + cellH / 2);
        }
    });

    // ── Draw roads ─────────────────────────────────────────────────
    if (showRoads && g.edges) {
        g.edges.forEach(edge => {
            const x1 = padding + edge.from[0] * cellW + cellW / 2;
            const y1 = padding + edge.from[1] * cellH + cellH / 2;
            const x2 = padding + edge.to[0] * cellW + cellW / 2;
            const y2 = padding + edge.to[1] * cellH + cellH / 2;

            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);

            if (edge.blocked) {
                ctx.strokeStyle = '#F43F5E';
                ctx.lineWidth = 2.5;
                ctx.setLineDash([4, 4]);
            } else {
                ctx.strokeStyle = 'rgba(255, 255, 255, 0.35)';
                ctx.lineWidth = 1.5;
                ctx.setLineDash([]);
            }
            ctx.stroke();
            ctx.setLineDash([]);
        });
    }

    // ── Draw blocked road X markers ────────────────────────────────
    if (currentState.blocked_roads) {
        currentState.blocked_roads.forEach(([p1, p2]) => {
            const mx = padding + (p1[0] + p2[0]) / 2 * cellW + cellW / 2;
            const my = padding + (p1[1] + p2[1]) / 2 * cellH + cellH / 2;

            ctx.fillStyle = '#F43F5E';
            ctx.font = 'bold 14px Inter';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText('✕', mx, my);
        });
    }

    // ── Draw ambulance coverage ────────────────────────────────────
    if (showAmbulances && currentState.ambulance_positions) {
        currentState.ambulance_positions.forEach(pos => {
            const cx = padding + pos[0] * cellW + cellW / 2;
            const cy = padding + pos[1] * cellH + cellH / 2;
            const radius = cellW * 2.5;

            // Coverage radius glow
            const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, radius);
            grad.addColorStop(0, 'rgba(232, 121, 249, 0.25)');
            grad.addColorStop(0.7, 'rgba(232, 121, 249, 0.08)');
            grad.addColorStop(1, 'rgba(232, 121, 249, 0)');
            ctx.fillStyle = grad;
            ctx.beginPath();
            ctx.arc(cx, cy, radius, 0, Math.PI * 2);
            ctx.fill();

            // Ambulance marker
            ctx.fillStyle = '#E879F9';
            ctx.beginPath();
            ctx.arc(cx, cy, 8, 0, Math.PI * 2);
            ctx.fill();
            ctx.fillStyle = '#0a0e1a';
            ctx.font = 'bold 9px Inter';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText('A', cx, cy);

            // Outer ring pulse
            ctx.strokeStyle = 'rgba(232, 121, 249, 0.5)';
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.arc(cx, cy, 12, 0, Math.PI * 2);
            ctx.stroke();
        });
    }

    // ── Draw police coverage ─────────────────────────────────────────
    const showPolice = document.getElementById('toggle-police').checked;
    if (showPolice && currentState.police_positions) {
        currentState.police_positions.forEach(pos => {
            const cx = padding + pos[0] * cellW + cellW / 2;
            const cy = padding + pos[1] * cellH + cellH / 2;

            // Police badge/marker
            ctx.fillStyle = '#3B82F6';
            ctx.beginPath();
            ctx.arc(cx, cy, 6, 0, Math.PI * 2);
            ctx.fill();

            // Inner badge
            ctx.fillStyle = '#0a0e1a';
            ctx.font = 'bold 8px Inter';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText('P', cx, cy);

            // Small glow for visibility
            ctx.strokeStyle = 'rgba(59, 130, 246, 0.5)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.arc(cx, cy, 9, 0, Math.PI * 2);
            ctx.stroke();
        });
    }

    // ── Draw civilians & route ─────────────────────────────────────
    if (showCivilians) {
        // Civilians remaining
        if (currentState.router_status) {
            const rs = currentState.router_status;

            // Rescued civilians (green check)
            rs.civilians_rescued.forEach(pos => {
                const cx = padding + pos[0] * cellW + cellW / 2;
                const cy = padding + pos[1] * cellH + cellH / 2;
                ctx.fillStyle = '#34D399';
                ctx.font = 'bold 16px Inter';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText('✓', cx, cy);
            });

            // Remaining civilians (red pulse)
            rs.civilians_remaining.forEach(pos => {
                const cx = padding + pos[0] * cellW + cellW / 2;
                const cy = padding + pos[1] * cellH + cellH / 2;

                // Red glow
                const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, cellW * 0.8);
                grad.addColorStop(0, 'rgba(244, 63, 94, 0.5)');
                grad.addColorStop(1, 'rgba(244, 63, 94, 0)');
                ctx.fillStyle = grad;
                ctx.beginPath();
                ctx.arc(cx, cy, cellW * 0.8, 0, Math.PI * 2);
                ctx.fill();

                ctx.fillStyle = '#F43F5E';
                ctx.font = 'bold 14px Inter';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText('⚠', cx, cy);
            });

            // Current position of team
            if (rs.current_position) {
                const cx = padding + rs.current_position[0] * cellW + cellW / 2;
                const cy = padding + rs.current_position[1] * cellH + cellH / 2;

                ctx.strokeStyle = '#6BDFFF';
                ctx.lineWidth = 2.5;
                ctx.beginPath();
                ctx.arc(cx, cy, 10, 0, Math.PI * 2);
                ctx.stroke();

                ctx.fillStyle = '#6BDFFF';
                ctx.font = 'bold 10px Inter';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText('🚐', cx, cy - 1);
            }
        } else if (currentState.civilian_positions) {
            currentState.civilian_positions.forEach(pos => {
                const cx = padding + pos[0] * cellW + cellW / 2;
                const cy = padding + pos[1] * cellH + cellH / 2;
                ctx.fillStyle = '#F43F5E';
                ctx.font = 'bold 14px Inter';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText('⚠', cx, cy);
            });
        }
    }
}

// ── Canvas Tooltip ────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('city-canvas');
    const tooltip = document.getElementById('canvas-tooltip');

    canvas.addEventListener('mousemove', (e) => {
        if (!currentState || !currentState.graph) return;

        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        const mx = (e.clientX - rect.left) * scaleX;
        const my = (e.clientY - rect.top) * scaleY;

        const g = currentState.graph;
        const padding = 30;
        const cellW = (canvas.width - padding * 2) / g.width;
        const cellH = (canvas.height - padding * 2) / g.height;

        const gx = Math.floor((mx - padding) / cellW);
        const gy = Math.floor((my - padding) / cellH);

        if (gx >= 0 && gx < g.width && gy >= 0 && gy < g.height) {
            const node = g.nodes.find(n => n.x === gx && n.y === gy);
            if (node) {
                let riskLabel = 'Low';
                if (node.risk_index > 0.7) riskLabel = 'High';
                else if (node.risk_index > 0.3) riskLabel = 'Medium';

                tooltip.style.display = 'block';
                tooltip.style.left = (e.clientX - canvas.parentElement.getBoundingClientRect().left + 15) + 'px';
                tooltip.style.top = (e.clientY - canvas.parentElement.getBoundingClientRect().top - 10) + 'px';
                tooltip.innerHTML =
                    `<strong>(${gx}, ${gy})</strong> ${node.type}<br>` +
                    `Pop: ${node.population_density}<br>` +
                    `Risk: ${riskLabel} (${node.risk_index.toFixed(2)})`;
            }
        } else {
            tooltip.style.display = 'none';
        }
    });

    canvas.addEventListener('mouseleave', () => {
        tooltip.style.display = 'none';
    });

    // Initial render
    renderGrid();
});
