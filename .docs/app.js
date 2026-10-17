/**
 * app.js — PyMAPF 2D Multi-Agent Pathfinding Research & Simulation Platform.
 *
 * Drives:
 *  1. Three-page research workflow:
 *     - Page 1 (setup): Interactive 2D Grid & Multi-Agent Editor + Batch Experiment Config
 *     - Page 2 (simulation): Synchronized 3-Algorithm Playback & Algorithm-Specific Execution Logs
 *     - Page 3 (results): Comparative Metrics, 6-Point Formal Validation Audit, Batch Charts & CSV Export
 *  2. Side-by-side execution and synchronized timeline playback for
 *     Prioritized Space-Time A*, Conflict-Based Search (CBS), and MAPPO under one-shot stay-at-goal semantics;
 *  3. Native Python server API integration (/api/*) with automatic fallback to
 *     the built-in browser engine + pre-trained MAPPO checkpoint when served statically.
 */

import {
  SCENARIOS,
  agentName,
  buildScenario,
  mappoRollout,
  problemFromState,
  solve,
  validateProblemSpec,
  validateSolution,
} from './mapf.js';

/* ------------------------------------------------------------- helpers --- */

const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const SERIES = () => [1, 2, 3, 4, 5, 6, 7, 8].map((i) => css(`--s${i}`));
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const fmtTime = (seconds) => {
  if (seconds == null || Number.isNaN(seconds)) return '—';
  if (seconds >= 1) return `${seconds.toFixed(2)} s`;
  if (seconds >= 0.001) return `${(seconds * 1000).toFixed(1)} ms`;
  return `${(seconds * 1e6).toFixed(0)} µs`;
};

const ALGORITHMS = ['prioritized', 'cbs', 'mappo'];

const ALGORITHM_LABELS = {
  prioritized: 'Prioritized A*',
  cbs: 'CBS (Optimal)',
  mappo: 'MAPPO (RL)',
};

const ALGORITHM_BLURBS = {
  prioritized: 'Sequential Space-Time A* with reservation table. Fast baseline, incomplete on tight bottlenecks.',
  cbs: 'Bi-level Conflict-Based Search over constraint tree. Complete and provably sum-of-costs optimal.',
  mappo: 'Centralized-critic Multi-Agent PPO policy using 9×9 local observations and one-shot goal locking.',
};

/* --------------------------------------------------------------- state --- */

const state = {
  activePage: 'setup',
  scenario: 'corner_swap',
  seed: 0,
  mappoEvalSeed: 0,
  agents: 4,
  algorithm: 'cbs',
  heuristic: 'manhattan',
  timeLimit: 5.0,
  mappoMode: 'sampled',
  tool: 'wall',
  activeAgentIndex: 0,
  matrix: [],
  agentList: [],
  // Results per algorithm: { prioritized, cbs, mappo }
  comparison: {
    prioritized: null,
    cbs: null,
    mappo: null,
  },
  // Pre-trained or instance-trained MAPPO policy JSON
  policyData: null,
  backendAvailable: false,
  running: false,
  // Synchronized playback state
  t: 0,
  horizon: 0,
  playing: false,
  speed: 3.2,
  lastFrame: 0,
  // Batch experiment state
  batchData: null,
};

/* --------------------------------------------------- 3-Page Navigation --- */

function normalizePageName(raw) {
  const clean = String(raw || '').replace(/^#/, '').trim().toLowerCase();
  if (clean === 'simulation' || clean === 'comparison') return 'simulation';
  if (clean === 'results' || clean === 'validation' || clean === 'experiments' || clean === 'methodology') {
    return 'results';
  }
  return 'setup';
}

function navigateToPage(pageName, { updateHash = true } = {}) {
  const target = normalizePageName(pageName);
  state.activePage = target;

  document.querySelectorAll('.page-view').forEach((sec) => {
    const isMatch = sec.dataset.page === target;
    sec.hidden = !isMatch;
    sec.classList.toggle('active-page', isMatch);
  });

  document.querySelectorAll('.nav-tab').forEach((btn) => {
    const isMatch = btn.dataset.navPage === target;
    btn.classList.toggle('active', isMatch);
    btn.setAttribute('aria-current', isMatch ? 'page' : 'false');
  });

  if (updateHash) {
    const nextHash = `#${target}`;
    if (window.location.hash !== nextHash) {
      history.replaceState(null, '', nextHash);
    }
  }

  window.scrollTo({ top: 0, behavior: 'instant' });

  // Re-measure and redraw canvases/charts once the target page is visible
  requestAnimationFrame(() => {
    drawAllCanvases();
    if (target === 'results' && state.batchData) {
      renderBatchResults(state.batchData);
    }
  });
}

/* ------------------------------------------------------------- logging --- */

function clearLog() {
  const el = $('log');
  if (el) el.innerHTML = '';
}

function log(text, className = '') {
  const el = $('log');
  if (!el) return;
  const line = document.createElement('div');
  if (className) line.className = className;
  line.textContent = text;
  el.appendChild(line);
  el.scrollTop = el.scrollHeight;
}

function renderAlgorithmExecutionLog(algo, entries) {
  const box = $(`log-${algo}`);
  const countEl = $(`log-count-${algo}`);
  if (!box) return;
  box.innerHTML = '';
  const list = Array.isArray(entries) ? entries : [];
  if (countEl) {
    countEl.textContent = `${list.length} ${list.length === 1 ? 'event' : 'events'}`;
  }
  if (!list.length) {
    const empty = document.createElement('div');
    empty.className = 'branch';
    empty.textContent = 'No execution events recorded yet. Run the algorithm to inspect its step-by-step trace.';
    box.appendChild(empty);
    return;
  }
  for (const item of list) {
    const row = document.createElement('div');
    if (typeof item === 'string') {
      row.textContent = item;
    } else if (item && typeof item === 'object') {
      row.className = item.kind || 'step';
      row.textContent = item.text || JSON.stringify(item);
    }
    box.appendChild(row);
  }
  box.scrollTop = 0;
}

function setFeedback(text, kind = 'neutral') {
  const box = $('editor-feedback');
  if (!box) return;
  box.textContent = text;
  box.classList.remove('error', 'ok');
  if (kind === 'error') box.classList.add('error');
  if (kind === 'ok') box.classList.add('ok');
}

/* ----------------------------------------------------- canvas rendering --- */

const ease = (f) => f * f * (3 - 2 * f);

function positionAt(path, t) {
  if (!path || !path.length) return [0, 0];
  const last = path.length - 1;
  const clamped = Number.isFinite(t) ? Math.max(0, Math.min(t, last)) : 0;
  const whole = Math.floor(clamped);
  const a = path[whole];
  const b = path[Math.min(whole + 1, last)];
  const k = ease(clamped - whole);
  return [a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k];
}

function fitCanvas(canvas, rows, cols, maxHeight = 460) {
  const dpr = window.devicePixelRatio || 1;
  const rawWidth = canvas.parentElement?.clientWidth || 360;
  const parentWidth = Math.max(220, rawWidth - 20);
  const cell = clamp(Math.floor(Math.min(parentWidth / cols, maxHeight / rows)), 10, 44);
  const width = cell * cols;
  const height = cell * rows;

  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, cell, rows, cols, width, height };
}

function drawGridAndAgents(canvas, paths, currentT, options = {}) {
  if (!canvas || !state.matrix.length) return null;
  const rows = state.matrix.length;
  const cols = state.matrix[0]?.length || 1;
  const { ctx, cell, width, height } = fitCanvas(canvas, rows, cols, options.maxHeight || 460);
  const colors = SERIES();
  const center = (v) => (v + 0.5) * cell;

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = css('--surface-1');
  ctx.fillRect(0, 0, width, height);

  // 1. Obstacles
  ctx.fillStyle = css('--obstacle');
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      if (state.matrix[r][c]) {
        ctx.fillRect(c * cell, r * cell, cell, cell);
      }
    }
  }

  // 2. Grid lines
  ctx.strokeStyle = css('--grid');
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let c = 0; c <= cols; c += 1) {
    ctx.moveTo(c * cell + 0.5, 0);
    ctx.lineTo(c * cell + 0.5, height);
  }
  for (let r = 0; r <= rows; r += 1) {
    ctx.moveTo(0, r * cell + 0.5);
    ctx.lineTo(width, r * cell + 0.5);
  }
  ctx.stroke();

  const agentNames = state.agentList.map((a) => a.name);
  const colorOf = (name) => {
    const idx = agentNames.indexOf(name);
    return colors[(idx >= 0 ? idx : 0) % colors.length];
  };

  // 3. Goal markers (open ring + subtle center label)
  state.agentList.forEach((agent, index) => {
    const color = colors[index % colors.length];
    const gx = center(agent.goal[1]);
    const gy = center(agent.goal[0]);
    ctx.strokeStyle = color;
    ctx.lineWidth = Math.max(1.8, cell * 0.1);
    ctx.globalAlpha = 0.85;
    ctx.beginPath();
    ctx.arc(gx, gy, cell * 0.32, 0, Math.PI * 2);
    ctx.stroke();

    if (cell >= 18) {
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.85;
      ctx.font = `600 ${Math.max(8, Math.round(cell * 0.25))}px ${css('--mono') || 'monospace'}`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(agent.name, gx, gy);
    }
    ctx.globalAlpha = 1;
  });

  // 4. Planned trajectories / rails
  const hasPaths = paths && Object.keys(paths).length > 0;
  if (hasPaths) {
    const railSpan = cell * 0.24;
    agentNames.forEach((name, index) => {
      const path = paths[name];
      if (!path || !path.length) return;
      const offset = agentNames.length > 1
        ? (index / (agentNames.length - 1) - 0.5) * railSpan
        : 0;
      const color = colorOf(name);

      for (const [stroke, lw, alpha] of [
        [css('--surface-1'), Math.max(2.6, cell * 0.2), 0.9],
        [color, Math.max(1.6, cell * 0.1), 0.9],
      ]) {
        ctx.strokeStyle = stroke;
        ctx.lineWidth = lw;
        ctx.globalAlpha = alpha;
        ctx.lineJoin = 'round';
        ctx.lineCap = 'round';
        ctx.beginPath();
        path.forEach(([r, c], i) => {
          const x = center(c) + offset;
          const y = center(r) + offset;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
    });
  }

  // 5. Agent discs (at currentT if paths exist, otherwise at start position)
  state.agentList.forEach((agent, index) => {
    const color = colors[index % colors.length];
    const path = hasPaths ? paths[agent.name] : null;
    const [r, c] = path && path.length ? positionAt(path, currentT) : agent.start;

    // Determine whether agent has arrived and locked at its goal by currentT
    let arrived = false;
    if (path && path.length) {
      const stepIdx = Math.min(Math.floor(Math.max(0, currentT)), path.length - 1);
      const posAtStep = path[stepIdx];
      arrived = posAtStep[0] === agent.goal[0] && posAtStep[1] === agent.goal[1];
    } else {
      arrived = agent.start[0] === agent.goal[0] && agent.start[1] === agent.goal[1];
    }

    const x = center(c);
    const y = center(r);

    // Highlight active agent in editor
    if (options.highlightActive && index === state.activeAgentIndex) {
      ctx.beginPath();
      ctx.arc(x, y, cell * 0.44, 0, Math.PI * 2);
      ctx.strokeStyle = css('--ink');
      ctx.lineWidth = 1.5;
      ctx.setLineDash([3, 2]);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    ctx.beginPath();
    ctx.arc(x, y, cell * 0.31, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.lineWidth = Math.max(1.4, cell * 0.07);
    ctx.strokeStyle = css('--surface-1');
    ctx.stroke();

    if (arrived) {
      ctx.beginPath();
      ctx.arc(x, y, cell * 0.42, 0, Math.PI * 2);
      ctx.strokeStyle = color;
      ctx.lineWidth = Math.max(1.6, cell * 0.08);
      ctx.globalAlpha = 0.55;
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    if (cell >= 14) {
      ctx.fillStyle = '#ffffff';
      ctx.font = `700 ${Math.max(9, Math.round(cell * 0.34))}px ${css('--sans') || 'sans-serif'}`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(agent.name, x, y + 0.5);
    }
  });

  // 6. Conflict marker if validation found a conflict
  if (options.conflict && options.conflict.cell_a) {
    const [cr, cc] = options.conflict.cell_a;
    const x = center(cc);
    const y = center(cr);
    const s = cell * 0.28;
    ctx.strokeStyle = css('--critical');
    ctx.lineWidth = Math.max(2.2, cell * 0.12);
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(x - s, y - s);
    ctx.lineTo(x + s, y + s);
    ctx.moveTo(x + s, y - s);
    ctx.lineTo(x - s, y + s);
    ctx.stroke();
  }

  return { cell, rows, cols, width, height };
}

function drawAllCanvases() {
  // Master editor board shows the clean problem setup on Page 1
  drawGridAndAgents(
    $('board'),
    null,
    0,
    {
      maxHeight: 460,
      highlightActive: true,
      conflict: null,
    },
  );

  // 3 Side-by-Side Comparison Canvases on Page 2
  for (const algo of ALGORITHMS) {
    const res = state.comparison[algo];
    drawGridAndAgents(
      $(`canvas-${algo}`),
      res?.paths || null,
      state.t,
      {
        maxHeight: 290,
        highlightActive: false,
        conflict: res?.validation?.first_conflict || null,
      },
    );
  }
}

/* ------------------------------------------------ UI & Roster Sync --- */

function syncRosterAndControls() {
  const rows = state.matrix.length;
  const cols = state.matrix[0]?.length || 0;

  $('grid-rows').value = String(rows);
  $('grid-rows-out').textContent = String(rows);
  $('grid-cols').value = String(cols);
  $('grid-cols-out').textContent = String(cols);
  $('agents').value = String(state.agentList.length);
  $('agents-out').textContent = String(state.agentList.length);
  $('seed').value = String(state.seed);
  $('seed-out').textContent = String(state.seed);
  if ($('mappo-eval-seed')) {
    $('mappo-eval-seed').value = String(state.mappoEvalSeed);
  }
  $('grid-summary').textContent = `${rows}×${cols} · ${state.agentList.length} agents`;

  const specLabel = SCENARIOS[state.scenario]?.label || state.scenario;
  const simSummary = $('sim-problem-summary');
  if (simSummary) {
    simSummary.textContent = `Active instance: ${specLabel} (${rows}×${cols} grid, ${state.agentList.length} agents, scenario seed=${state.seed}, MAPPO mode=${state.mappoMode}, MAPPO eval seed=${state.mappoEvalSeed}).`;
  }

  if (state.activeAgentIndex >= state.agentList.length) {
    state.activeAgentIndex = Math.max(0, state.agentList.length - 1);
  }

  // Active agent dropdown
  const select = $('active-agent-select');
  select.innerHTML = '';
  state.agentList.forEach((agent, idx) => {
    const opt = document.createElement('option');
    opt.value = String(idx);
    opt.textContent = `Agent ${agent.name} (${agent.start[0]},${agent.start[1]} → ${agent.goal[0]},${agent.goal[1]})`;
    if (idx === state.activeAgentIndex) opt.selected = true;
    select.appendChild(opt);
  });

  // Roster table
  const colors = SERIES();
  const tbody = $('agent-roster-body');
  tbody.innerHTML = '';
  state.agentList.forEach((agent, idx) => {
    const tr = document.createElement('tr');
    if (idx === state.activeAgentIndex) tr.classList.add('active-row');
    const dist = Math.abs(agent.start[0] - agent.goal[0]) + Math.abs(agent.start[1] - agent.goal[1]);
    const color = colors[idx % colors.length];
    tr.innerHTML = `
      <td><span class="agent-dot" style="background:${color}"></span><strong>${agent.name}</strong></td>
      <td><code>(${agent.start[0]}, ${agent.start[1]})</code></td>
      <td><code>(${agent.goal[0]}, ${agent.goal[1]})</code></td>
      <td>${dist}</td>
    `;
    tr.addEventListener('click', () => {
      state.activeAgentIndex = idx;
      syncRosterAndControls();
      drawAllCanvases();
    });
    tbody.appendChild(tr);
  });

  updateEquivalentCode();
}

function resetComparisonResults() {
  state.playing = false;
  state.t = 0;
  state.horizon = 0;
  state.comparison = { prioritized: null, cbs: null, mappo: null };

  $('play').textContent = '⏵ Play';
  $('play').disabled = true;
  $('step-prev').disabled = true;
  $('step-next').disabled = true;
  $('step-reset').disabled = true;
  $('timeline').disabled = true;
  $('timeline').max = '0';
  $('timeline').value = '0';
  $('clock').textContent = 't = 0 / 0';

  for (const algo of ALGORITHMS) {
    const badge = $(`badge-${algo}`);
    badge.textContent = 'Ready';
    badge.className = 'status-label';
    $(`mini-cost-${algo}`).textContent = '—';
    $(`mini-ms-${algo}`).textContent = '—';
    $(`mini-time-${algo}`).textContent = '—';
    $(`mini-valid-${algo}`).textContent = '—';
    renderAlgorithmExecutionLog(algo, []);
  }
  drawAllCanvases();
}

/* ------------------------------------------- Synchronized Playback --- */

function syncTransportToResults() {
  const pathHorizons = ALGORITHMS.map((a) => {
    const res = state.comparison[a];
    if (!res) return 0;
    if (Number.isFinite(res.makespan) && res.makespan > 0) return res.makespan;
    const paths = res.paths || {};
    const lens = Object.values(paths).map((p) => Math.max(0, (p?.length || 1) - 1));
    return lens.length ? Math.max(...lens) : 0;
  });
  const horizon = pathHorizons.length ? Math.max(...pathHorizons) : 0;
  state.horizon = horizon;
  state.t = 0;

  const hasAnyPath = ALGORITHMS.some((a) => {
    const paths = state.comparison[a]?.paths;
    return paths && Object.keys(paths).length > 0;
  });

  $('timeline').max = String(horizon);
  $('timeline').value = '0';
  $('timeline').disabled = !hasAnyPath || horizon === 0;
  $('play').disabled = !hasAnyPath || horizon === 0;
  $('step-prev').disabled = !hasAnyPath || horizon === 0;
  $('step-next').disabled = !hasAnyPath || horizon === 0;
  $('step-reset').disabled = !hasAnyPath || horizon === 0;
  $('clock').textContent = `t = 0 / ${horizon}`;
}

function setPlayhead(newT) {
  state.t = clamp(newT, 0, state.horizon);
  $('timeline').value = String(state.t);
  $('clock').textContent = `t = ${Math.floor(state.t)} / ${state.horizon}`;
  drawAllCanvases();
}

function tickPlayback(now) {
  if (!state.playing) return;
  const dt = Math.min(0.1, Math.max(0, (now - state.lastFrame) / 1000));
  state.lastFrame = now;
  const nextT = state.t + dt * state.speed;
  if (nextT >= state.horizon) {
    setPlayhead(state.horizon);
    state.playing = false;
    $('play').textContent = '⏵ Play';
    return;
  }
  setPlayhead(nextT);
  requestAnimationFrame(tickPlayback);
}

function togglePlay() {
  if (state.horizon <= 0) return;
  if (state.playing) {
    state.playing = false;
    $('play').textContent = '⏵ Play';
    return;
  }
  if (state.t >= state.horizon) state.t = 0;
  state.playing = true;
  state.lastFrame = performance.now();
  $('play').textContent = '⏸ Pause';
  requestAnimationFrame(tickPlayback);
}

/* ----------------------------------- Metrics & Validation Table Updates --- */

function formatBoolCell(el, flag, passText = 'PASS', failText = 'FAIL') {
  if (!el) return;
  if (flag == null) {
    el.textContent = '—';
    el.className = '';
    return;
  }
  el.textContent = flag ? `✓ ${passText}` : `✕ ${failText}`;
  el.className = flag ? 'cell-pass' : 'cell-fail';
}

function updateComparisonUI() {
  const cbsCost = state.comparison.cbs?.solved ? (state.comparison.cbs.sum_of_costs ?? state.comparison.cbs.sumOfCosts) : null;
  const violationsBox = $('validation-violations-box');
  if (violationsBox) violationsBox.innerHTML = '';

  for (const algo of ALGORITHMS) {
    const res = state.comparison[algo];
    const badge = $(`badge-${algo}`);
    if (!res) {
      badge.textContent = 'Not run';
      badge.className = 'status-label';
      renderAlgorithmExecutionLog(algo, []);
      continue;
    }

    const cost = res.sum_of_costs ?? res.sumOfCosts;
    const val = res.validation || {};
    const timeSec = algo === 'mappo' ? (res.inference_time ?? res.runtime) : res.runtime;

    if (res.solved && res.valid) {
      badge.textContent = '✓ Solved & Valid';
      badge.className = 'status-label ok';
    } else if (!res.available && res.available !== undefined) {
      badge.textContent = 'Unavailable';
      badge.className = 'status-label warn';
    } else if (res.truncated) {
      badge.textContent = '✕ Truncated (Horizon)';
      badge.className = 'status-label fail';
    } else {
      badge.textContent = '✕ Unsolved / Incomplete';
      badge.className = 'status-label fail';
    }

    // Mini panel summary
    $(`mini-cost-${algo}`).textContent = cost != null ? String(cost) : '—';
    $(`mini-ms-${algo}`).textContent = res.makespan != null ? String(res.makespan) : '—';
    $(`mini-time-${algo}`).textContent = fmtTime(timeSec);
    $(`mini-valid-${algo}`).textContent = res.valid ? '✓ Valid' : '✕ Invalid';
    $(`mini-valid-${algo}`).className = `v ${res.valid ? 'cell-pass' : 'cell-fail'}`;
    $(`note-${algo}`).textContent = res.status_reason || (res.solved ? 'All agents reached assigned goals without conflict.' : 'Failed to find a valid joint plan.');

    // Render dedicated algorithm-specific execution log on Page 2
    renderAlgorithmExecutionLog(algo, res.execution_log || []);

    // Main comparative performance table on Page 3
    formatBoolCell($(`tbl-solved-${algo}`), Boolean(res.solved), 'Solved', 'Unsolved');
    formatBoolCell($(`tbl-valid-${algo}`), Boolean(res.valid), 'Valid', 'Invalid');
    $(`tbl-cost-${algo}`).textContent = cost != null ? String(cost) : '—';

    if (cost != null && cbsCost != null && cbsCost > 0) {
      const ratio = cost / cbsCost;
      $(`tbl-subopt-${algo}`).textContent = `${ratio.toFixed(3)}× (${cost - cbsCost >= 0 ? '+' : ''}${cost - cbsCost})`;
    } else if (algo === 'cbs' && cost != null) {
      $(`tbl-subopt-${algo}`).textContent = '1.000× (Optimal)';
    } else {
      $(`tbl-subopt-${algo}`).textContent = '—';
    }

    $(`tbl-ms-${algo}`).textContent = res.makespan != null ? `${res.makespan} steps` : '—';
    $(`tbl-exp-${algo}`).textContent = res.expansions != null
      ? res.expansions.toLocaleString()
      : (algo === 'mappo' ? `Rollout (${res.steps_taken ?? res.makespan ?? 0} steps)` : '—');
    $(`tbl-time-${algo}`).textContent = algo === 'mappo'
      ? `${fmtTime(timeSec)} (policy inference)`
      : `${fmtTime(timeSec)} (online search)`;

    if (algo === 'mappo') {
      const trainSec = res.training_time ?? state.policyData?.training_meta?.training_time_sec ?? 0;
      const trainSteps = res.training_steps ?? state.policyData?.training_meta?.training_steps ?? 0;
      $(`tbl-train-mappo`).textContent = `${fmtTime(trainSec)} (${Number(trainSteps).toLocaleString()} env steps)`;
    }

    $(`tbl-reason-${algo}`).textContent = res.status_reason || '—';

    // 6 Formal Validation Criteria
    formatBoolCell($(`val-obs-${algo}`), val.in_bounds_and_obstacle_free);
    formatBoolCell($(`val-move-${algo}`), val.legal_moves);
    formatBoolCell($(`val-vert-${algo}`), val.vertex_conflict_free);
    formatBoolCell($(`val-edge-${algo}`), val.edge_swap_free);
    formatBoolCell($(`val-goal-${algo}`), val.all_reached_goal);
    formatBoolCell($(`val-occ-${algo}`), val.goal_occupancy_valid);

    if (violationsBox) {
      const line = document.createElement('div');
      if (val.valid) {
        line.className = 'solved';
        line.textContent = `[${ALGORITHM_LABELS[algo]}] ✓ All 6 formal validation checks passed (0 violations).`;
      } else {
        line.className = 'conflict';
        const vList = (val.violations && val.violations.length)
          ? val.violations.join(' | ')
          : (res.status_reason || 'No valid complete plan.');
        line.textContent = `[${ALGORITHM_LABELS[algo]}] ✕ ${vList}`;
      }
      violationsBox.appendChild(line);
    }
  }
}

/* ------------------------------------------- Algorithm Execution --- */

function currentProblemPayload() {
  state.mappoEvalSeed = Number($('mappo-eval-seed')?.value || 0);
  return {
    matrix: state.matrix,
    agents: state.agentList.map((a) => ({
      name: a.name,
      start: [...a.start],
      goal: [...a.goal],
    })),
    heuristic: state.heuristic,
    time_limit: Number($('time-limit').value || 5.0),
    mappo_mode: state.mappoMode,
    seed: state.seed,
    mappo_eval_seed: state.mappoEvalSeed,
  };
}

function buildLocalSearchExecutionLog(algo, problem, sol, runtime, options) {
  const entries = [];
  const rows = problem.grid.height;
  const cols = problem.grid.width;
  const agentOrder = problem.agents.map((a) => a.name).join(' → ');

  if (algo === 'prioritized') {
    entries.push({
      kind: 'init',
      text: `[INIT] Prioritized Space-Time A* | grid=${rows}x${cols} | priority=[${agentOrder}] | heuristic=${options.heuristic}`,
    });
    entries.push({
      kind: 'step',
      text: '[RULE] Sequential reservation table: higher-priority trajectories & occupied goals are reserved for all future t.',
    });
  } else {
    entries.push({
      kind: 'init',
      text: `[INIT] Conflict-Based Search (CBS) | grid=${rows}x${cols} | agents=${problem.agents.length} | heuristic=${options.heuristic} | budget=${options.timeLimit}s`,
    });
  }

  for (const ev of (sol?.traceEvents || [])) {
    const kind = ev.kind || ev.type;
    if (kind === 'root') {
      entries.push({
        kind: 'step',
        text: `[ROOT] Initialized search root | agents=[${(ev.agents || []).join(', ')}] | initial sum_of_costs=${ev.cost ?? 0}`,
      });
    } else if (kind === 'agent_planned') {
      const pathLen = ev.length ?? (ev.path ? ev.path.length : 0);
      const cost = ev.cost ?? Math.max(0, pathLen - 1);
      entries.push({
        kind: 'step',
        text: `[PLAN] Agent ${ev.agent}: space-time A* path found | cost=${cost} (${pathLen} states)`,
      });
    } else if (kind === 'expand') {
      entries.push({
        kind: 'step',
        text: `[CT#${ev.node}] Expanded node | cost=${ev.cost} | open=${ev.open ?? 0}`,
      });
    } else if (kind === 'conflict') {
      const cType = ev.type || ev.conflict?.kind || 'vertex';
      const agentA = ev.a || ev.conflict?.agent_a || '?';
      const agentB = ev.b || ev.conflict?.agent_b || '?';
      const tStep = ev.t ?? ev.conflict?.time ?? 0;
      const cell = ev.cell || ev.conflict?.cell_a || [0, 0];
      entries.push({
        kind: 'conflict',
        text: `[CONFLICT] ${cType} conflict between Agent ${agentA} and Agent ${agentB} at (${cell[0]},${cell[1]}) at t=${tStep}`,
      });
    } else if (kind === 'branch') {
      entries.push({
        kind: 'branch',
        text: `[BRANCH] Constrained Agent ${ev.agent} (${ev.constraint || 'conflict'}) → replanned child CT cost=${ev.cost}`,
      });
    }
  }

  if (sol) {
    entries.push({
      kind: 'solved',
      text: `[SOLVED] ✓ Valid joint plan found | sum_of_costs=${sol.sumOfCosts} | makespan=${sol.makespan} | expansions=${sol.expansions} | time=${(runtime * 1000).toFixed(1)}ms`,
    });
  } else {
    entries.push({
      kind: 'conflict',
      text: `[FAILED] ✕ No conflict-free plan found within ${options.timeLimit}s budget.`,
    });
  }
  return entries;
}

function runAlgorithmLocally(algo, payload) {
  const problem = problemFromState({
    matrix: payload.matrix,
    agents: payload.agents,
    allowDiagonals: false,
  });

  if (algo === 'mappo') {
    return mappoRollout(problem, state.policyData, {
      mappo_mode: payload.mappo_mode,
      seed: payload.seed,
      mappo_eval_seed: payload.mappo_eval_seed ?? payload.seed,
    });
  }

  const traceEvents = [];
  const started = performance.now();
  const sol = solve(problem, algo, {
    heuristic: payload.heuristic,
    timeLimit: payload.time_limit,
    maxExpansions: 25000,
    trace: (ev) => traceEvents.push(ev),
  });
  const runtime = (performance.now() - started) / 1000;
  if (sol) sol.traceEvents = traceEvents;

  const execution_log = buildLocalSearchExecutionLog(algo, problem, sol, runtime, {
    heuristic: payload.heuristic,
    timeLimit: payload.time_limit,
  });

  if (!sol) {
    const val = validateSolution(problem, {});
    return {
      algorithm: algo,
      available: true,
      solved: false,
      valid: false,
      sum_of_costs: null,
      sumOfCosts: null,
      makespan: null,
      expansions: null,
      runtime,
      status_reason: algo === 'cbs'
        ? `No conflict-free solution found within ${payload.time_limit}s budget`
        : 'Prioritized reservation table blocked a lower-priority agent',
      paths: {},
      validation: val,
      execution_log,
    };
  }

  const val = sol.validation || validateSolution(problem, sol.paths);
  return {
    algorithm: algo,
    available: true,
    solved: val.valid,
    valid: val.valid,
    sum_of_costs: sol.sumOfCosts,
    sumOfCosts: sol.sumOfCosts,
    makespan: sol.makespan,
    expansions: sol.expansions,
    runtime: sol.runtime ?? runtime,
    status_reason: val.valid
      ? (algo === 'cbs' ? 'Optimal conflict-free plan verified' : 'Valid prioritized plan verified')
      : (val.violations?.[0] || 'Validation failed'),
    paths: sol.paths,
    validation: val,
    execution_log,
  };
}

async function runComparison(onlyAlgorithm = null, { switchPage = false } = {}) {
  if (state.running) return;
  const payload = currentProblemPayload();
  const check = validateProblemSpec(payload);
  if (!check.ok) {
    setFeedback(`Invalid problem configuration: ${check.errors[0]}`, 'error');
    log(`Problem validation failed: ${check.errors.join(' | ')}`, 'conflict');
    return;
  }

  if (switchPage) {
    navigateToPage('simulation');
  }

  state.running = true;
  $('run-compare').disabled = true;
  $('run-single').disabled = true;
  $('hero-run-compare').disabled = true;
  if ($('sim-rerun-compare')) $('sim-rerun-compare').disabled = true;

  const targets = onlyAlgorithm ? [onlyAlgorithm] : ALGORITHMS;
  clearLog();
  log(
    `Running ${onlyAlgorithm ? ALGORITHM_LABELS[onlyAlgorithm] : '3-way comparison'} on ${state.matrix.length}×${state.matrix[0].length} grid (${state.agentList.length} agents, scenario_seed=${state.seed}, mappo_mode=${state.mappoMode}, mappo_eval_seed=${state.mappoEvalSeed})…`,
  );

  try {
    if (state.backendAvailable) {
      if (!onlyAlgorithm) {
        const resp = await fetch('/api/compare', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        const data = await resp.json();
        if (!resp.ok || !data.ok) {
          throw new Error((data.errors && data.errors.join('; ')) || 'Comparison API error');
        }
        state.comparison = data.algorithms;
      } else {
        const resp = await fetch('/api/solve', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ ...payload, algorithm: onlyAlgorithm }),
        });
        const data = await resp.json();
        if (!resp.ok || !data.ok) {
          throw new Error((data.errors && data.errors.join('; ')) || 'Solve API error');
        }
        state.comparison[onlyAlgorithm] = data.result;
      }
    } else {
      for (const algo of targets) {
        state.comparison[algo] = runAlgorithmLocally(algo, payload);
      }
    }

    for (const algo of targets) {
      const r = state.comparison[algo];
      if (!r) continue;
      const cost = r.sum_of_costs ?? r.sumOfCosts;
      const timeSec = algo === 'mappo' ? (r.inference_time ?? r.runtime) : r.runtime;
      if (r.solved && r.valid) {
        log(
          `[${ALGORITHM_LABELS[algo]}] ✓ SOLVED & VALID — sum_of_costs=${cost}, makespan=${r.makespan}, time=${fmtTime(timeSec)} (${r.status_reason})`,
          'solved',
        );
      } else {
        const vMsg = r.validation?.violations?.[0] ? ` | First violation: ${r.validation.violations[0]}` : '';
        log(
          `[${ALGORITHM_LABELS[algo]}] ✕ UNSOLVED — ${r.status_reason}${vMsg}`,
          'conflict',
        );
      }
    }

    syncRosterAndControls();
    updateComparisonUI();
    syncTransportToResults();
    drawAllCanvases();
    if (state.horizon > 0 && state.activePage === 'simulation') {
      togglePlay();
    }
  } catch (err) {
    log(`Execution error: ${err.message || err}`, 'conflict');
    setFeedback(`Execution error: ${err.message || err}`, 'error');
  } finally {
    state.running = false;
    $('run-compare').disabled = false;
    $('run-single').disabled = false;
    $('hero-run-compare').disabled = false;
    if ($('sim-rerun-compare')) $('sim-rerun-compare').disabled = false;
  }
}

async function trainMappoOnActiveGrid() {
  if (state.running) return;
  const payload = currentProblemPayload();
  const check = validateProblemSpec(payload);
  if (!check.ok) {
    setFeedback(`Fix problem configuration before training: ${check.errors[0]}`, 'error');
    return;
  }

  const steps = Number($('mappo-train-steps').value || 16384);
  const btn = $('train-mappo-btn');
  const statusEl = $('mappo-policy-status');

  if (!state.backendAvailable) {
    statusEl.textContent = 'Using pre-trained MAPPO policy checkpoint (start Python server for live on-instance training).';
    return;
  }

  state.running = true;
  btn.disabled = true;
  btn.textContent = 'Training MAPPO…';
  statusEl.textContent = `Training MAPPO on active ${state.matrix.length}×${state.matrix[0].length} grid for ${steps.toLocaleString()} steps…`;
  log(`Starting on-instance MAPPO training (${steps.toLocaleString()} environment steps)…`);

  try {
    const resp = await fetch('/api/train_mappo', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        ...payload,
        steps,
        mappo_mode: state.mappoMode,
      }),
    });
    const data = await resp.json();
    if (!resp.ok || !data.ok) {
      throw new Error((data.errors && data.errors.join('; ')) || 'MAPPO training error');
    }
    state.comparison.mappo = data.result;
    const tSec = data.result.training_time ?? 0;
    statusEl.textContent = `Active policy: Instance-trained MAPPO (${steps.toLocaleString()} steps in ${fmtTime(tSec)}).`;
    log(
      `[MAPPO Training] Completed ${steps.toLocaleString()} steps in ${fmtTime(tSec)} → rollout ${data.result.solved ? 'SOLVED' : 'UNSOLVED'} (${data.result.status_reason})`,
      data.result.solved ? 'solved' : 'conflict',
    );
    updateComparisonUI();
    syncTransportToResults();
    drawAllCanvases();
  } catch (err) {
    statusEl.textContent = `Training failed: ${err.message || err}`;
    log(`MAPPO training failed: ${err.message || err}`, 'conflict');
  } finally {
    state.running = false;
    btn.disabled = false;
    btn.textContent = 'Train on Active Grid';
  }
}

/* ------------------------------------------- Interactive 2D Grid Editor --- */

function loadScenarioPreset(name = state.scenario) {
  state.scenario = name;
  const spec = SCENARIOS[name];
  $('scenario-blurb').textContent = spec?.blurb || '';

  const maxAgents = name === 'corner_swap' ? 8 : 10;
  $('agents').max = String(maxAgents);
  if (state.agents > maxAgents) state.agents = maxAgents;

  let built;
  try {
    built = buildScenario(name, { agents: state.agents, seed: state.seed });
  } catch (e) {
    state.agents = Math.max(2, state.agents - 1);
    built = buildScenario(name, { agents: state.agents, seed: state.seed });
  }

  state.matrix = built.matrix.map((row) => [...row]);
  state.agentList = built.agents.map((a) => ({
    name: a.name,
    start: [...a.start],
    goal: [...a.goal],
  }));
  state.activeAgentIndex = 0;

  syncRosterAndControls();
  resetComparisonResults();
  setFeedback(`Loaded '${spec.label}' (${state.matrix.length}×${state.matrix[0].length}, ${state.agentList.length} agents, scenario seed ${state.seed}).`, 'ok');
}

function resizeEmptyGrid() {
  const rows = clamp(Number($('grid-rows').value) || 12, 5, 22);
  const cols = clamp(Number($('grid-cols').value) || 12, 5, 22);
  const matrix = Array.from({ length: rows }, (_, r) =>
    Array.from({ length: cols }, (_, c) => (r === 0 || r === rows - 1 || c === 0 || c === cols - 1 ? 1 : 0)));

  const freeCells = [];
  for (let r = 1; r < rows - 1; r += 1) {
    for (let c = 1; c < cols - 1; c += 1) freeCells.push([r, c]);
  }
  const count = Math.min(state.agentList.length || 4, Math.floor(freeCells.length / 2));
  const agents = [];
  for (let i = 0; i < count; i += 1) {
    agents.push({
      name: agentName(i),
      start: [...freeCells[i]],
      goal: [...freeCells[freeCells.length - 1 - i]],
    });
  }

  state.matrix = matrix;
  state.agentList = agents;
  state.activeAgentIndex = 0;
  syncRosterAndControls();
  resetComparisonResults();
  setFeedback(`Resized to ${rows}×${cols} empty bordered grid with ${agents.length} agents.`, 'ok');
}

function clearInteriorObstacles() {
  const rows = state.matrix.length;
  const cols = state.matrix[0]?.length || 0;
  for (let r = 1; r < rows - 1; r += 1) {
    for (let c = 1; c < cols - 1; c += 1) {
      state.matrix[r][c] = 0;
    }
  }
  resetComparisonResults();
  setFeedback('Cleared all interior obstacles.', 'ok');
}

function addNewAgentAt(targetCell = null) {
  const rows = state.matrix.length;
  const cols = state.matrix[0]?.length || 0;
  const occupiedStarts = new Set(state.agentList.map((a) => `${a.start[0]},${a.start[1]}`));
  const occupiedGoals = new Set(state.agentList.map((a) => `${a.goal[0]},${a.goal[1]}`));

  let start = null;
  if (targetCell) {
    const [tr, tc] = targetCell;
    if (state.matrix[tr]?.[tc] === 0 && !occupiedStarts.has(`${tr},${tc}`)) {
      start = [tr, tc];
    } else {
      setFeedback(`Cannot add agent start at (${tr}, ${tc}): cell is blocked or already has a start.`, 'error');
      return;
    }
  }

  const freeStarts = [];
  const freeGoals = [];
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      if (state.matrix[r][c] !== 0) continue;
      if (!occupiedStarts.has(`${r},${c}`)) freeStarts.push([r, c]);
      if (!occupiedGoals.has(`${r},${c}`)) freeGoals.push([r, c]);
    }
  }

  if (!start) start = freeStarts[0];
  if (!start || !freeGoals.length) {
    setFeedback('No free unoccupied cells remaining to add another agent.', 'error');
    return;
  }

  // Pick a free goal furthest from start
  let bestGoal = freeGoals[0];
  let bestDist = -1;
  for (const cand of freeGoals) {
    if (cand[0] === start[0] && cand[1] === start[1] && freeGoals.length > 1) continue;
    const d = Math.abs(cand[0] - start[0]) + Math.abs(cand[1] - start[1]);
    if (d > bestDist) {
      bestDist = d;
      bestGoal = cand;
    }
  }

  const name = agentName(state.agentList.length);
  state.agentList.push({ name, start: [...start], goal: [...bestGoal] });
  state.activeAgentIndex = state.agentList.length - 1;
  state.agents = state.agentList.length;

  syncRosterAndControls();
  resetComparisonResults();
  setFeedback(`Added Agent ${name} with start (${start[0]}, ${start[1]}) and goal (${bestGoal[0]}, ${bestGoal[1]}).`, 'ok');
}

function removeActiveAgent() {
  if (state.agentList.length <= 1) {
    setFeedback('At least 1 agent is required in the problem.', 'error');
    return;
  }
  const removed = state.agentList.splice(state.activeAgentIndex, 1)[0];
  state.agentList.forEach((a, i) => { a.name = agentName(i); });
  state.activeAgentIndex = Math.max(0, Math.min(state.activeAgentIndex, state.agentList.length - 1));
  state.agents = state.agentList.length;

  syncRosterAndControls();
  resetComparisonResults();
  setFeedback(`Removed agent ${removed.name}; ${state.agentList.length} agents active.`, 'ok');
}

function handleBoardCellAction(r, c, dragValue = null) {
  const rows = state.matrix.length;
  const cols = state.matrix[0]?.length || 0;
  if (r < 0 || r >= rows || c < 0 || c >= cols) return null;

  const hasStart = state.agentList.find((a) => a.start[0] === r && a.start[1] === c);
  const hasGoal = state.agentList.find((a) => a.goal[0] === r && a.goal[1] === c);

  if (state.tool === 'wall') {
    if (hasStart || hasGoal) {
      setFeedback(
        `Cannot place obstacle on (${r}, ${c}) — occupied by Agent ${(hasStart || hasGoal).name}'s ${hasStart ? 'start' : 'goal'}.`,
        'error',
      );
      return null;
    }
    const nextVal = dragValue != null ? dragValue : (state.matrix[r][c] ? 0 : 1);
    state.matrix[r][c] = nextVal;
    resetComparisonResults();
    updateEquivalentCode();
    return nextVal;
  }

  if (state.tool === 'start') {
    const active = state.agentList[state.activeAgentIndex];
    if (!active) return null;
    if (state.matrix[r][c]) {
      setFeedback(`Cannot move Agent ${active.name} start to (${r}, ${c}): cell is an obstacle.`, 'error');
      return null;
    }
    if (hasStart && hasStart.name !== active.name) {
      setFeedback(`Cannot move Agent ${active.name} start to (${r}, ${c}): Agent ${hasStart.name} already starts there.`, 'error');
      return null;
    }
    active.start = [r, c];
    syncRosterAndControls();
    resetComparisonResults();
    setFeedback(`Moved Agent ${active.name} start to (${r}, ${c}).`, 'ok');
    return null;
  }

  if (state.tool === 'goal') {
    const active = state.agentList[state.activeAgentIndex];
    if (!active) return null;
    if (state.matrix[r][c]) {
      setFeedback(`Cannot move Agent ${active.name} goal to (${r}, ${c}): cell is an obstacle.`, 'error');
      return null;
    }
    if (hasGoal && hasGoal.name !== active.name) {
      setFeedback(`Cannot move Agent ${active.name} goal to (${r}, ${c}): Agent ${hasGoal.name} already has that goal.`, 'error');
      return null;
    }
    active.goal = [r, c];
    syncRosterAndControls();
    resetComparisonResults();
    setFeedback(`Moved Agent ${active.name} goal to (${r}, ${c}).`, 'ok');
    return null;
  }

  if (state.tool === 'add') {
    addNewAgentAt([r, c]);
    return null;
  }

  return null;
}

function wireBoardInteractions() {
  const board = $('board');
  let paintingWall = null;

  const cellFromEvent = (ev) => {
    const rect = board.getBoundingClientRect();
    const rows = state.matrix.length;
    const cols = state.matrix[0]?.length || 1;
    const c = Math.floor(((ev.clientX - rect.left) / rect.width) * cols);
    const r = Math.floor(((ev.clientY - rect.top) / rect.height) * rows);
    return [r, c];
  };

  board.addEventListener('pointerdown', (ev) => {
    const [r, c] = cellFromEvent(ev);
    const res = handleBoardCellAction(r, c, null);
    if (state.tool === 'wall' && res !== null) {
      paintingWall = res;
      board.setPointerCapture(ev.pointerId);
    }
  });

  board.addEventListener('pointermove', (ev) => {
    const [r, c] = cellFromEvent(ev);
    const rows = state.matrix.length;
    const cols = state.matrix[0]?.length || 1;
    if (r >= 0 && r < rows && c >= 0 && c < cols) {
      $('cursor-coords').textContent = `Cell: (${r}, ${c})`;
      if (paintingWall !== null && state.tool === 'wall') {
        handleBoardCellAction(r, c, paintingWall);
      }
    }
  });

  const stopPaint = () => { paintingWall = null; };
  board.addEventListener('pointerup', stopPaint);
  board.addEventListener('pointercancel', stopPaint);
}

/* ------------------------------------------- Batch Experiments & Charts --- */

const svgEl = (tag, attrs = {}, text = null) => {
  const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
  if (text != null) node.textContent = text;
  return node;
};

const ALGO_COLORS = () => ({
  prioritized: css('--s2'),
  cbs: css('--s1'),
  mappo: css('--s3'),
});

function renderLineChart(svg, rows, field, { logScale = false, yLabel = '' } = {}) {
  if (!svg) return;
  svg.innerHTML = '';
  const W = 420;
  const H = 260;
  const pad = { l: 54, r: 20, t: 28, b: 38 };
  const plotW = W - pad.l - pad.r;
  const plotH = H - pad.t - pad.b;

  const usable = rows.filter((r) => r[field] != null && (!logScale || r[field] > 0));
  if (!usable.length) {
    svg.appendChild(svgEl('text', {
      x: W / 2, y: H / 2, 'text-anchor': 'middle', fill: css('--muted'), 'font-size': 12,
    }, 'Run batch experiment to plot results'));
    return;
  }

  const xs = [...new Set(rows.map((r) => r.n_agents))].sort((a, b) => a - b);
  const tr = (v) => (logScale ? Math.log10(Math.max(1e-5, v)) : v);
  const vals = usable.map((r) => tr(r[field]));
  let yMin = Math.min(...vals);
  let yMax = Math.max(...vals);
  if (yMin === yMax) { yMin -= 0.5; yMax += 0.5; }
  const span = yMax - yMin;
  yMin -= span * 0.08;
  yMax += span * 0.12;

  const xOf = (x) => pad.l + (xs.length === 1 ? plotW / 2 : ((x - xs[0]) / (xs[xs.length - 1] - xs[0])) * plotW);
  const yOf = (v) => pad.t + plotH - ((tr(v) - yMin) / (yMax - yMin)) * plotH;

  // Grid & Y ticks
  for (let i = 0; i <= 4; i += 1) {
    const gy = pad.t + (i / 4) * plotH;
    const raw = yMax - (i / 4) * (yMax - yMin);
    const val = logScale ? 10 ** raw : raw;
    svg.appendChild(svgEl('line', {
      x1: pad.l, x2: W - pad.r, y1: gy, y2: gy, stroke: css('--grid'),
    }));
    const label = logScale
      ? (val >= 0.1 ? `${val.toFixed(2)}s` : `${(val * 1000).toFixed(1)}ms`)
      : val.toFixed(0);
    svg.appendChild(svgEl('text', {
      x: pad.l - 6, y: gy + 4, 'text-anchor': 'end', fill: css('--muted'), 'font-size': 10,
    }, label));
  }

  // X ticks
  for (const x of xs) {
    const gx = xOf(x);
    svg.appendChild(svgEl('text', {
      x: gx, y: H - 12, 'text-anchor': 'middle', fill: css('--muted'), 'font-size': 11,
    }, `${x} agents`));
  }

  if (yLabel) {
    svg.appendChild(svgEl('text', {
      x: pad.l, y: 15, fill: css('--muted'), 'font-size': 10,
    }, yLabel));
  }

  const palette = ALGO_COLORS();
  let lx = pad.l + 80;
  for (const algo of ALGORITHMS) {
    const color = palette[algo];
    const pts = [];
    for (const x of xs) {
      const sub = usable.filter((r) => r.algorithm === algo && r.n_agents === x);
      if (!sub.length) continue;
      const mean = sub.reduce((acc, r) => acc + r[field], 0) / sub.length;
      pts.push([xOf(x), yOf(mean)]);
    }
    if (pts.length > 1) {
      const d = pts.map(([px, py], i) => `${i ? 'L' : 'M'}${px.toFixed(1)},${py.toFixed(1)}`).join(' ');
      svg.appendChild(svgEl('path', { d, fill: 'none', stroke: color, 'stroke-width': 2.2 }));
    }
    for (const [px, py] of pts) {
      svg.appendChild(svgEl('circle', {
        cx: px, cy: py, r: 3.8, fill: color, stroke: css('--surface-1'), 'stroke-width': 1.4,
      }));
    }
    svg.appendChild(svgEl('circle', { cx: lx, cy: 12, r: 4, fill: color }));
    svg.appendChild(svgEl('text', { x: lx + 7, y: 15, fill: css('--ink'), 'font-size': 10.5 }, ALGORITHM_LABELS[algo]));
    lx += 110;
  }
}

function renderSuccessChart(svg, summary) {
  if (!svg) return;
  svg.innerHTML = '';
  const W = 420;
  const paddingL = 130;
  const barW = W - paddingL - 58;
  const palette = ALGO_COLORS();

  ALGORITHMS.forEach((algo, idx) => {
    const item = summary[algo] || { success_rate: 0 };
    const rate = item.success_rate || 0;
    const y = 26 + idx * 52;
    const color = palette[algo];

    svg.appendChild(svgEl('text', {
      x: paddingL - 10, y: y + 17, 'text-anchor': 'end', fill: css('--ink'), 'font-size': 12, 'font-weight': 600,
    }, ALGORITHM_LABELS[algo]));
    svg.appendChild(svgEl('rect', {
      x: paddingL, y, width: barW, height: 24, rx: 4, fill: css('--surface-2'),
    }));
    svg.appendChild(svgEl('rect', {
      x: paddingL, y, width: Math.max(2, barW * rate), height: 24, rx: 4, fill: color,
    }));
    svg.appendChild(svgEl('text', {
      x: paddingL + barW + 8, y: y + 16, fill: css('--ink'), 'font-size': 11.5,
    }, `${Math.round(rate * 100)}%`));
  });
}

function summarizeLocalBatch(rows) {
  const cbsCosts = {};
  for (const r of rows) {
    if (r.algorithm === 'cbs' && r.solved && r.sum_of_costs != null) {
      cbsCosts[`${r.scenario}:${r.n_agents}:${r.seed}`] = r.sum_of_costs;
    }
  }
  const summary = {};
  for (const algo of ALGORITHMS) {
    const subset = rows.filter((r) => r.algorithm === algo);
    const solved = subset.filter((r) => r.solved && r.valid);
    const total = subset.length;
    const ratios = [];
    for (const r of solved) {
      const ref = cbsCosts[`${r.scenario}:${r.n_agents}:${r.seed}`];
      if (ref && ref > 0 && r.sum_of_costs != null) ratios.push(r.sum_of_costs / ref);
    }
    summary[algo] = {
      instances: total,
      solved: solved.length,
      success_rate: total ? solved.length / total : 0,
      valid_rate: total ? subset.filter((r) => r.valid).length / total : 0,
      mean_cost: solved.length ? solved.reduce((a, b) => a + b.sum_of_costs, 0) / solved.length : null,
      mean_makespan: solved.length ? solved.reduce((a, b) => a + b.makespan, 0) / solved.length : null,
      mean_runtime: total ? subset.reduce((a, b) => a + b.runtime, 0) / total : 0,
      mean_suboptimality_vs_cbs: ratios.length ? ratios.reduce((a, b) => a + b, 0) / ratios.length : null,
      timing_label: algo === 'mappo' ? 'Rollout inference time (policy pre-trained)' : 'Online search planning time',
    };
  }
  return summary;
}

function renderBatchResults(batch) {
  state.batchData = batch;
  const hasRows = Boolean(batch && batch.rows?.length);
  $('export-csv').disabled = !hasRows;
  if ($('results-export-csv')) $('results-export-csv').disabled = !hasRows;

  const subEl = $('results-batch-subtitle');
  if (subEl && batch) {
    const scList = (batch.scenarios || [batch.scenario]).join(', ');
    subEl.textContent = `Evaluated ${batch.rows?.length || 0} trials across scenarios [${scList}], agent counts [${(batch.agent_counts || []).join(', ')}], scenario seeds [${(batch.seeds || []).join(', ')}].`;
  }

  // 1. Aggregate summary table
  const sumBody = $('bench-summary-body');
  sumBody.innerHTML = '';
  for (const algo of ALGORITHMS) {
    const s = batch.summary?.[algo];
    if (!s) continue;
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><strong>${ALGORITHM_LABELS[algo]}</strong></td>
      <td>${s.instances}</td>
      <td class="${s.success_rate >= 0.8 ? 'cell-pass' : (s.success_rate > 0 ? '' : 'cell-fail')}">${(s.success_rate * 100).toFixed(1)}% (${s.solved}/${s.instances})</td>
      <td>${(s.valid_rate * 100).toFixed(1)}%</td>
      <td>${s.mean_cost != null ? s.mean_cost.toFixed(1) : '—'}</td>
      <td>${s.mean_makespan != null ? s.mean_makespan.toFixed(1) : '—'}</td>
      <td>${s.mean_suboptimality_vs_cbs != null ? `${s.mean_suboptimality_vs_cbs.toFixed(3)}×` : '—'}</td>
      <td>${fmtTime(s.mean_runtime)}</td>
      <td><span class="hint">${s.timing_label}</span></td>
    `;
    sumBody.appendChild(tr);
  }

  // 2. Charts
  renderLineChart($('chart-runtime'), batch.rows, 'runtime', {
    logScale: true,
    yLabel: 'Runtime (log scale)',
  });
  renderLineChart(
    $('chart-cost'),
    batch.rows.filter((r) => r.solved && r.sum_of_costs != null),
    'sum_of_costs',
    { logScale: false, yLabel: 'Sum of Path Costs' },
  );
  renderSuccessChart($('chart-success'), batch.summary);

  // 3. Per-instance table
  const table = $('bench-table');
  table.innerHTML = `
    <thead>
      <tr>
        <th>Scenario</th>
        <th>Agents</th>
        <th>Scen Seed</th>
        <th>Eval Seed</th>
        <th>Algorithm</th>
        <th>Solved</th>
        <th>Valid</th>
        <th>Cost</th>
        <th>Makespan</th>
        <th>Time</th>
      </tr>
    </thead>
    <tbody>
      ${batch.rows.map((r) => `
        <tr>
          <td>${SCENARIOS[r.scenario]?.label || r.scenario}</td>
          <td>${r.n_agents}</td>
          <td>${r.scenario_seed ?? r.seed}</td>
          <td>${r.mappo_eval_seed != null ? r.mappo_eval_seed : '—'}</td>
          <td>${ALGORITHM_LABELS[r.algorithm] || r.algorithm}</td>
          <td class="${r.solved ? 'cell-pass' : 'cell-fail'}">${r.solved ? '✓' : '✕'}</td>
          <td class="${r.valid ? 'cell-pass' : 'cell-fail'}">${r.valid ? '✓' : '✕'}</td>
          <td>${r.sum_of_costs ?? '—'}</td>
          <td>${r.makespan ?? '—'}</td>
          <td>${fmtTime(r.runtime)}</td>
        </tr>
      `).join('')}
    </tbody>
  `;
}

function getSelectedBatchConfig() {
  const scenarios = [...document.querySelectorAll('#bench-scenarios-group input[type="checkbox"]:checked')]
    .map((el) => el.value);
  const agentCounts = [...document.querySelectorAll('#bench-agents-checks input[type="checkbox"]:checked')]
    .map((el) => Number(el.value))
    .filter((n) => Number.isFinite(n) && n > 0);
  const seedStart = Number($('bench-seed-start')?.value || 0);
  const numSeeds = Number($('bench-seeds')?.value || 3);
  const seeds = Array.from({ length: Math.max(1, numSeeds) }, (_, i) => seedStart + i);
  const mappoMode = $('bench-mappo-mode')?.value || 'sampled';
  const mappoSeedOffset = Number($('bench-mappo-seed-offset')?.value || 1000);

  return {
    scenarios: scenarios.length ? scenarios : ['random_obstacles'],
    agentCounts: agentCounts.length ? agentCounts : [2, 4, 6],
    seeds,
    mappoMode,
    mappoSeedOffset,
  };
}

async function runBatchExperiment({ switchPage = false } = {}) {
  if (state.running) return;
  const {
    scenarios,
    agentCounts,
    seeds,
    mappoMode,
    mappoSeedOffset,
  } = getSelectedBatchConfig();

  if (switchPage) {
    navigateToPage('results');
  }

  state.running = true;
  $('run-bench').disabled = true;
  if ($('results-rerun-batch')) $('results-rerun-batch').disabled = true;
  const totalTrials = scenarios.length * agentCounts.length * seeds.length * 3;
  $('bench-status').textContent = `Running ${totalTrials} trials across ${scenarios.length} scenario family(s)…`;

  try {
    if (state.backendAvailable) {
      const resp = await fetch('/api/batch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          scenario: scenarios[0],
          scenarios,
          agent_counts: agentCounts,
          seeds,
          time_limit: 2.5,
          mappo_mode: mappoMode,
          mappo_seed_offset: mappoSeedOffset,
        }),
      });
      const data = await resp.json();
      if (!resp.ok || !data.ok) throw new Error('Batch API error');
      renderBatchResults(data);
    } else {
      const rows = [];
      for (const scName of scenarios) {
        for (const n of agentCounts) {
          for (const seed of seeds) {
            let built;
            try {
              const cappedN = scName === 'corner_swap' ? Math.min(n, 8) : n;
              built = buildScenario(scName, { agents: cappedN, seed });
            } catch (e) {
              continue;
            }
            const evalSeed = seed + mappoSeedOffset;
            const payload = {
              matrix: built.matrix,
              agents: built.agents,
              heuristic: 'manhattan',
              time_limit: 2.5,
              mappo_mode: mappoMode,
              seed,
              mappo_eval_seed: evalSeed,
            };
            for (const algo of ALGORITHMS) {
              const res = runAlgorithmLocally(algo, payload);
              rows.push({
                scenario: scName,
                n_agents: built.agents.length,
                seed,
                scenario_seed: seed,
                mappo_eval_seed: algo === 'mappo' ? evalSeed : null,
                algorithm: algo,
                solved: res.solved,
                valid: res.valid,
                sum_of_costs: res.sum_of_costs ?? res.sumOfCosts,
                makespan: res.makespan,
                expansions: res.expansions,
                runtime: algo === 'mappo' ? (res.inference_time ?? res.runtime) : res.runtime,
                timing_type: algo === 'mappo' ? 'policy_inference' : 'online_search',
                status_reason: res.status_reason,
              });
            }
            await new Promise((r) => setTimeout(r, 0));
          }
        }
      }
      const summary = summarizeLocalBatch(rows);
      const csvHeader = 'scenario,n_agents,seed,scenario_seed,mappo_eval_seed,algorithm,solved,valid,sum_of_costs,makespan,expansions,runtime,timing_type,status_reason';
      const csvLines = rows.map((r) => [
        r.scenario, r.n_agents, r.seed, r.scenario_seed, r.mappo_eval_seed ?? '',
        r.algorithm, r.solved, r.valid,
        r.sum_of_costs ?? '', r.makespan ?? '', r.expansions ?? '',
        (r.runtime ?? 0).toFixed(6), r.timing_type, `"${String(r.status_reason || '').replace(/"/g, '""')}"`,
      ].join(','));
      renderBatchResults({
        scenario: scenarios[0],
        scenarios,
        agent_counts: agentCounts,
        seeds,
        rows,
        summary,
        csv: [csvHeader, ...csvLines].join('\n'),
      });
    }
    $('bench-status').textContent = `Completed ${state.batchData.rows.length} algorithm evaluations across ${scenarios.length} scenario family(s).`;
  } catch (err) {
    $('bench-status').textContent = `Batch experiment error: ${err.message || err}`;
  } finally {
    state.running = false;
    $('run-bench').disabled = false;
    if ($('results-rerun-batch')) $('results-rerun-batch').disabled = false;
  }
}

function downloadBatchCsv() {
  if (!state.batchData?.csv) return;
  const blob = new Blob([state.batchData.csv], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  const tag = (state.batchData.scenarios || [state.batchData.scenario || 'experiment']).join('-');
  a.download = `pymapf-batch-${tag}.csv`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

/* ------------------------------------------- Reproducible Code Snippet --- */

function updateEquivalentCode() {
  const agentLines = state.agentList
    .map((a) => `    Agent("${a.name}", start=(${a.start[0]}, ${a.start[1]}), goal=(${a.goal[0]}, ${a.goal[1]})),`)
    .join('\n');

  $('equivalent-code').textContent = `from pymapf import Agent, GridMap, MAPFProblem, solve, validate_solution
from pymapf.research import compare_all_algorithms

grid = GridMap(${JSON.stringify(state.matrix)})
agents = [
${agentLines}
]
problem = MAPFProblem(grid, agents, allow_diagonals=False)

# Run Prioritized Space-Time A*, CBS, and MAPPO on the identical problem:
comparison = compare_all_algorithms({
    "matrix": grid.as_matrix(),
    "agents": [{"name": a.name, "start": a.start, "goal": a.goal} for a in agents],
    "heuristic": "${state.heuristic}",
    "mappo_mode": "${state.mappoMode}",
    "seed": ${state.seed},
    "mappo_eval_seed": ${state.mappoEvalSeed},
})
for algo, metrics in comparison["algorithms"].items():
    print(algo, "solved=", metrics["solved"], "valid=", metrics["valid"], "cost=", metrics["sum_of_costs"])`;
}

/* ----------------------------------------------------- Initialization --- */

async function detectBackendAndPolicy() {
  try {
    const policyResp = await fetch('assets/mappo-policy.json');
    if (policyResp.ok) {
      state.policyData = await policyResp.json();
    }
  } catch (e) {
    // Ignore if unavailable
  }

  try {
    const resp = await fetch('/api/health');
    if (resp.ok) {
      const info = await resp.json();
      if (info && info.ok) {
        state.backendAvailable = true;
        $('engine-glyph').className = 'status-glyph live';
        $('engine-label').textContent = `Python ${info.python || ''} · pymapf ${info.version}`;
        return;
      }
    }
  } catch (e) {
    // Fallback to browser execution
  }

  state.backendAvailable = false;
  $('engine-glyph').className = 'status-glyph live';
  $('engine-label').textContent = 'Browser Engine + MAPPO Checkpoint';
}

function initControls() {
  // Wire 3-page navigation buttons & links
  document.querySelectorAll('[data-nav-page]').forEach((el) => {
    el.addEventListener('click', (ev) => {
      ev.preventDefault();
      navigateToPage(el.dataset.navPage);
    });
  });

  if ($('goto-sim-btn')) {
    $('goto-sim-btn').addEventListener('click', () => navigateToPage('simulation'));
  }
  if ($('goto-results-btn')) {
    $('goto-results-btn').addEventListener('click', () => navigateToPage('results'));
  }

  window.addEventListener('hashchange', () => {
    if (window.location.hash) {
      navigateToPage(window.location.hash, { updateHash: false });
    }
  });

  // Populate scenario select & batch scenario multi-select checkboxes
  const scenarioSelect = $('scenario');
  const benchScenariosGroup = $('bench-scenarios-group');
  for (const [keyName, spec] of Object.entries(SCENARIOS)) {
    const opt1 = document.createElement('option');
    opt1.value = keyName;
    opt1.textContent = spec.label;
    if (keyName === state.scenario) opt1.selected = true;
    scenarioSelect.appendChild(opt1);

    if (benchScenariosGroup) {
      const lbl = document.createElement('label');
      lbl.className = 'check-pill';
      const checked = keyName === 'empty_room' || keyName === 'random_obstacles';
      lbl.innerHTML = `<input type="checkbox" value="${keyName}" ${checked ? 'checked' : ''}> ${spec.label}`;
      benchScenariosGroup.appendChild(lbl);
    }
  }

  if ($('bench-select-all-scenarios')) {
    $('bench-select-all-scenarios').addEventListener('click', () => {
      document.querySelectorAll('#bench-scenarios-group input[type="checkbox"]').forEach((cb) => {
        cb.checked = true;
      });
    });
  }
  if ($('bench-select-default-scenarios')) {
    $('bench-select-default-scenarios').addEventListener('click', () => {
      document.querySelectorAll('#bench-scenarios-group input[type="checkbox"]').forEach((cb) => {
        cb.checked = cb.value === 'empty_room' || cb.value === 'random_obstacles';
      });
    });
  }

  $('algorithm-blurb').textContent = ALGORITHM_BLURBS[state.algorithm];

  scenarioSelect.addEventListener('change', () => {
    state.seed = 0;
    loadScenarioPreset(scenarioSelect.value);
    runComparison(null, { switchPage: false });
  });

  $('agents').addEventListener('input', () => {
    state.agents = Number($('agents').value);
    $('agents-out').textContent = String(state.agents);
  });
  $('agents').addEventListener('change', () => {
    loadScenarioPreset(state.scenario);
    runComparison(null, { switchPage: false });
  });

  $('seed').addEventListener('input', () => {
    state.seed = Number($('seed').value);
    $('seed-out').textContent = String(state.seed);
  });
  $('seed').addEventListener('change', () => {
    loadScenarioPreset(state.scenario);
    runComparison(null, { switchPage: false });
  });

  if ($('mappo-eval-seed')) {
    $('mappo-eval-seed').addEventListener('change', () => {
      state.mappoEvalSeed = Number($('mappo-eval-seed').value || 0);
      syncRosterAndControls();
    });
  }

  $('grid-rows').addEventListener('input', () => {
    $('grid-rows-out').textContent = $('grid-rows').value;
  });
  $('grid-cols').addEventListener('input', () => {
    $('grid-cols-out').textContent = $('grid-cols').value;
  });

  $('regenerate').addEventListener('click', () => {
    state.seed = (state.seed + 1) % 41;
    loadScenarioPreset(state.scenario);
    runComparison(null, { switchPage: false });
  });
  $('resize-apply').addEventListener('click', resizeEmptyGrid);
  $('clear-walls').addEventListener('click', clearInteriorObstacles);

  $('algorithm').addEventListener('change', () => {
    state.algorithm = $('algorithm').value;
    $('algorithm-blurb').textContent = ALGORITHM_BLURBS[state.algorithm] || '';
    drawAllCanvases();
  });

  $('heuristic').addEventListener('change', () => {
    state.heuristic = $('heuristic').value;
    updateEquivalentCode();
  });

  $('time-limit').addEventListener('change', () => {
    state.timeLimit = Number($('time-limit').value);
  });

  // MAPPO mode toggle
  document.querySelectorAll('#mappo-mode-group button').forEach((btn) => {
    btn.addEventListener('click', () => {
      state.mappoMode = btn.dataset.mode;
      document.querySelectorAll('#mappo-mode-group button').forEach((b) => {
        b.setAttribute('aria-pressed', String(b === btn));
      });
      syncRosterAndControls();
    });
  });

  $('train-mappo-btn').addEventListener('click', trainMappoOnActiveGrid);
  $('run-compare').addEventListener('click', () => runComparison(null, { switchPage: true }));
  $('hero-run-compare').addEventListener('click', () => runComparison(null, { switchPage: true }));
  $('run-single').addEventListener('click', () => runComparison(state.algorithm, { switchPage: true }));
  if ($('sim-rerun-compare')) {
    $('sim-rerun-compare').addEventListener('click', () => runComparison(null, { switchPage: false }));
  }

  // Editor tool buttons
  document.querySelectorAll('.tool-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      state.tool = btn.dataset.tool;
      document.querySelectorAll('.tool-btn').forEach((b) => {
        b.setAttribute('aria-pressed', String(b === btn));
      });
      const hints = {
        wall: 'Click or drag on any unoccupied cell to add or remove obstacles.',
        start: 'Click any free cell to move the selected Active Agent’s start position.',
        goal: 'Click any free cell to move the selected Active Agent’s goal position.',
        add: 'Click any free cell to spawn a new agent starting on that cell.',
      };
      setFeedback(hints[state.tool] || '');
    });
  });

  $('active-agent-select').addEventListener('change', () => {
    state.activeAgentIndex = Number($('active-agent-select').value);
    syncRosterAndControls();
    drawAllCanvases();
  });

  $('remove-agent-btn').addEventListener('click', removeActiveAgent);

  // Playback transport controls
  $('play').addEventListener('click', togglePlay);
  $('step-prev').addEventListener('click', () => {
    state.playing = false;
    $('play').textContent = '⏵ Play';
    setPlayhead(Math.max(0, Math.floor(state.t) - 1));
  });
  $('step-next').addEventListener('click', () => {
    state.playing = false;
    $('play').textContent = '⏵ Play';
    setPlayhead(Math.min(state.horizon, Math.floor(state.t) + 1));
  });
  $('step-reset').addEventListener('click', () => {
    state.playing = false;
    $('play').textContent = '⏵ Play';
    setPlayhead(0);
  });
  $('timeline').addEventListener('input', () => {
    state.playing = false;
    $('play').textContent = '⏵ Play';
    setPlayhead(Number($('timeline').value));
  });
  $('playback-speed').addEventListener('change', () => {
    state.speed = Number($('playback-speed').value || 3.2);
  });

  // Batch experiments
  $('run-bench').addEventListener('click', () => runBatchExperiment({ switchPage: true }));
  if ($('results-rerun-batch')) {
    $('results-rerun-batch').addEventListener('click', () => runBatchExperiment({ switchPage: false }));
  }
  $('export-csv').addEventListener('click', downloadBatchCsv);
  if ($('results-export-csv')) {
    $('results-export-csv').addEventListener('click', downloadBatchCsv);
  }

  // Theme toggle
  $('theme-toggle').addEventListener('click', () => {
    const html = document.documentElement;
    html.dataset.theme = html.dataset.theme === 'light' ? 'dark' : 'light';
    drawAllCanvases();
    if (state.batchData) renderBatchResults(state.batchData);
  });

  window.addEventListener('resize', () => {
    drawAllCanvases();
  });
}

async function boot() {
  initControls();
  wireBoardInteractions();
  loadScenarioPreset('corner_swap');
  if (window.location.hash) {
    navigateToPage(window.location.hash, { updateHash: false });
  } else {
    navigateToPage('setup', { updateHash: false });
  }
  await detectBackendAndPolicy();
  await runComparison(null, { switchPage: false });
  runBatchExperiment({ switchPage: false });
}

boot();
