/**
 * mapf.js — the playground's instant-response engine.
 *
 * This is a faithful port of pymapf's core search (space-time A*, prioritized
 * planning, CBS and weighted CBS) so the page is fully interactive from the
 * first paint, before the ~10 MB Pyodide runtime has finished downloading —
 * and so it still works if that download is blocked.
 *
 * The solvers are written as generators that yield SearchEvents, exactly the
 * vocabulary pymapf.core.trace defines. The driver pumps them with a per-frame
 * time budget, which is what makes the search animate live instead of freezing
 * the tab. When Pyodide is ready the real Python library takes over and emits
 * the same events, so the UI never has to care which engine ran.
 *
 * Cells are [row, col]; a solution's paths[name][t] is the cell at time t.
 */

const ORTHOGONAL = [[-1, 0], [1, 0], [0, -1], [0, 1]];
const DIAGONAL = [[-1, -1], [-1, 1], [1, -1], [1, 1]];

export const key = (r, c) => r + ',' + c;

/* ---------------------------------------------------------------- grid --- */

export function makeGrid(height, width, blocked = []) {
  return { height, width, blocked: new Set(blocked.map(([r, c]) => key(r, c))) };
}

export function gridFromMatrix(matrix) {
  const blocked = [];
  matrix.forEach((row, r) => row.forEach((v, c) => { if (v) blocked.push([r, c]); }));
  return makeGrid(matrix.length, matrix[0].length, blocked);
}

export function toMatrix(grid) {
  return Array.from({ length: grid.height }, (_, r) =>
    Array.from({ length: grid.width }, (_, c) => (isFree(grid, r, c) ? 0 : 1)));
}

export const inBounds = (grid, r, c) => r >= 0 && r < grid.height && c >= 0 && c < grid.width;
export const isFree = (grid, r, c) => inBounds(grid, r, c) && !grid.blocked.has(key(r, c));

export function freeCellCount(grid) {
  return grid.height * grid.width - grid.blocked.size;
}

export function neighbors(grid, [r, c], allowDiagonals = false) {
  const out = [];
  for (const [dr, dc] of ORTHOGONAL) {
    if (isFree(grid, r + dr, c + dc)) out.push([r + dr, c + dc]);
  }
  if (allowDiagonals) {
    for (const [dr, dc] of DIAGONAL) {
      // No corner cutting: both orthogonal cells must be free too.
      if (isFree(grid, r + dr, c + dc) && isFree(grid, r + dr, c) && isFree(grid, r, c + dc)) {
        out.push([r + dr, c + dc]);
      }
    }
  }
  return out;
}

/* ---------------------------------------------------------- heuristics --- */

export const HEURISTICS = {
  manhattan: (a, b) => Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]),
  euclidean: (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]),
  chebyshev: (a, b) => Math.max(Math.abs(a[0] - b[0]), Math.abs(a[1] - b[1])),
  octile: (a, b) => {
    const dr = Math.abs(a[0] - b[0]); const dc = Math.abs(a[1] - b[1]);
    return (dr + dc) + (Math.SQRT2 - 2) * Math.min(dr, dc);
  },
};

/* --------------------------------------------------------- constraints --- */

export function newConstraints() {
  return { vertex: new Set(), edge: new Set(), lastVertex: new Map() };
}

export function addVertex(constraints, [r, c], t) {
  constraints.vertex.add(r + ',' + c + ',' + t);
  const k = key(r, c);
  constraints.lastVertex.set(k, Math.max(constraints.lastVertex.get(k) ?? -1, t));
}

export function addEdge(constraints, [ur, uc], [vr, vc], t) {
  constraints.edge.add(ur + ',' + uc + '>' + vr + ',' + vc + ',' + t);
}

export function copyConstraints(constraints) {
  return {
    vertex: new Set(constraints.vertex),
    edge: new Set(constraints.edge),
    lastVertex: new Map(constraints.lastVertex),
  };
}

const blocksVertex = (constraints, [r, c], t) => constraints.vertex.has(r + ',' + c + ',' + t);
const blocksEdge = (constraints, [ur, uc], [vr, vc], t) =>
  constraints.edge.has(ur + ',' + uc + '>' + vr + ',' + vc + ',' + t);

/* -------------------------------------------------------- binary heap --- */

class Heap {
  constructor(compare) { this.items = []; this.compare = compare; }
  get size() { return this.items.length; }
  push(item) {
    const items = this.items;
    items.push(item);
    let i = items.length - 1;
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (this.compare(items[i], items[parent]) >= 0) break;
      [items[i], items[parent]] = [items[parent], items[i]];
      i = parent;
    }
  }
  pop() {
    const items = this.items;
    const top = items[0];
    const last = items.pop();
    if (items.length) {
      items[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1; const r = l + 1;
        let best = i;
        if (l < items.length && this.compare(items[l], items[best]) < 0) best = l;
        if (r < items.length && this.compare(items[r], items[best]) < 0) best = r;
        if (best === i) break;
        [items[i], items[best]] = [items[best], items[i]];
        i = best;
      }
    }
    return top;
  }
  peek() { return this.items[0]; }
}

const byFirst = (a, b) => (a[0] - b[0]) || (a[1] - b[1]) || (a[2] - b[2]);

/* ------------------------------------------------- space-time A* (low) --- */

export function spaceTimeAStar(grid, start, goal, constraints = null, options = {}) {
  const { heuristic = 'manhattan', allowDiagonals = false } = options;
  const h = typeof heuristic === 'function' ? heuristic : HEURISTICS[heuristic];
  const cons = constraints || newConstraints();

  const settleTime = cons.lastVertex.get(key(goal[0], goal[1])) ?? -1;
  let lastConstraintT = 0;
  for (const entry of cons.vertex) lastConstraintT = Math.max(lastConstraintT, +entry.split(',')[2]);
  for (const entry of cons.edge) lastConstraintT = Math.max(lastConstraintT, +entry.split(',').pop());
  const maxTimestep = options.maxTimestep ?? (freeCellCount(grid) + lastConstraintT + 1);

  const open = new Heap(byFirst);
  let tie = 0;
  open.push([h(start, goal), tie++, 0, start]);
  const visited = new Set();
  const parent = new Map([[key(start[0], start[1]) + ',0', null]]);

  while (open.size) {
    const [, , t, cell] = open.pop();
    const state = key(cell[0], cell[1]) + ',' + t;
    if (visited.has(state)) continue;
    visited.add(state);

    if (cell[0] === goal[0] && cell[1] === goal[1] && t > settleTime) {
      const path = [];
      let cursor = state;
      while (cursor != null) {
        const [r, c] = cursor.split(',');
        path.push([+r, +c]);
        cursor = parent.get(cursor);
      }
      return path.reverse();
    }
    if (t >= maxTimestep) continue;

    for (const next of [...neighbors(grid, cell, allowDiagonals), cell]) {
      const nt = t + 1;
      const nstate = key(next[0], next[1]) + ',' + nt;
      if (visited.has(nstate)) continue;
      if (blocksVertex(cons, next, nt)) continue;
      if (blocksEdge(cons, cell, next, nt)) continue;
      if (!parent.has(nstate)) parent.set(nstate, state);
      open.push([nt + h(next, goal), tie++, nt, next]);
    }
  }
  return null;
}

/* ---------------------------------------------------------- conflicts --- */

const cellAt = (path, t) => (t < path.length ? path[t] : path[path.length - 1]);
const same = (a, b) => a[0] === b[0] && a[1] === b[1];

export function findFirstConflict(paths) {
  const names = Object.keys(paths);
  const horizon = Math.max(0, ...names.map((n) => paths[n].length));
  for (let t = 0; t < horizon; t += 1) {
    for (let i = 0; i < names.length; i += 1) {
      for (let j = i + 1; j < names.length; j += 1) {
        const pa = paths[names[i]]; const pb = paths[names[j]];
        const aNow = cellAt(pa, t); const bNow = cellAt(pb, t);
        if (same(aNow, bNow)) {
          return { kind: 'vertex', a: names[i], b: names[j], t, cellA: aNow, cellB: bNow };
        }
      }
    }
    if (t > 0) {
      for (let i = 0; i < names.length; i += 1) {
        for (let j = i + 1; j < names.length; j += 1) {
          const pa = paths[names[i]]; const pb = paths[names[j]];
          const aNow = cellAt(pa, t); const bNow = cellAt(pb, t);
          const aPrev = cellAt(pa, t - 1); const bPrev = cellAt(pb, t - 1);
          if (same(aNow, bPrev) && same(bNow, aPrev) && !same(aPrev, bPrev)) {
            return { kind: 'edge', a: names[i], b: names[j], t, cellA: aNow, cellB: bNow };
          }
        }
      }
    }
  }
  return null;
}

export function countConflicts(paths) {
  const names = Object.keys(paths);
  const horizon = Math.max(0, ...names.map((n) => paths[n].length));
  let total = 0;
  for (let t = 0; t < horizon; t += 1) {
    for (let i = 0; i < names.length; i += 1) {
      for (let j = i + 1; j < names.length; j += 1) {
        const pa = paths[names[i]]; const pb = paths[names[j]];
        const aNow = cellAt(pa, t); const bNow = cellAt(pb, t);
        if (same(aNow, bNow)) {
          total += 1;
        } else if (t > 0) {
          const aPrev = cellAt(pa, t - 1); const bPrev = cellAt(pb, t - 1);
          if (same(aNow, bPrev) && same(bNow, aPrev) && !same(aPrev, bPrev)) {
            total += 1;
          }
        }
      }
    }
  }
  return total;
}

export function solutionMetrics(paths, algorithm, expansions, runtime, problem = null) {
  const lengths = Object.values(paths).map((p) => p.length - 1);
  const validation = problem ? validateSolution({ ...problem, algorithm }, paths) : null;
  return {
    paths,
    algorithm,
    expansions,
    runtime,
    makespan: Math.max(0, ...lengths),
    sumOfCosts: lengths.reduce((a, b) => a + b, 0),
    valid: validation ? validation.valid : findFirstConflict(paths) === null,
    validation,
  };
}

export function validateProblemSpec(spec) {
  const errors = [];
  const { matrix, agents } = spec || {};
  if (!Array.isArray(matrix) || !matrix.length || !Array.isArray(matrix[0])) {
    return { ok: false, errors: ['Grid matrix must be a non-empty 2D array.'] };
  }
  const height = matrix.length;
  const width = matrix[0].length;
  if (!Array.isArray(agents) || !agents.length) {
    return { ok: false, errors: ['At least one agent must be defined.'] };
  }
  const names = new Set();
  const starts = new Map();
  const goals = new Map();

  for (const agent of agents) {
    const name = String(agent.name);
    if (names.has(name)) errors.push(`Duplicate agent name '${name}'.`);
    names.add(name);
    const [sr, sc] = agent.start || [];
    const [gr, gc] = agent.goal || [];
    if (sr < 0 || sr >= height || sc < 0 || sc >= width) {
      errors.push(`Agent '${name}' start (${sr}, ${sc}) is outside the grid.`);
    } else if (matrix[sr][sc]) {
      errors.push(`Agent '${name}' start (${sr}, ${sc}) is on an obstacle.`);
    }
    if (gr < 0 || gr >= height || gc < 0 || gc >= width) {
      errors.push(`Agent '${name}' goal (${gr}, ${gc}) is outside the grid.`);
    } else if (matrix[gr][gc]) {
      errors.push(`Agent '${name}' goal (${gr}, ${gc}) is on an obstacle.`);
    }
    const sk = key(sr, sc);
    if (starts.has(sk)) errors.push(`Agents '${starts.get(sk)}' and '${name}' share start (${sr}, ${sc}).`);
    else starts.set(sk, name);

    const gk = key(gr, gc);
    if (goals.has(gk)) errors.push(`Agents '${goals.get(gk)}' and '${name}' share goal (${gr}, ${gc}).`);
    else goals.set(gk, name);
  }
  return { ok: errors.length === 0, errors };
}

export function validateSolution(problem, paths) {
  const { grid, agents, allowDiagonals = false } = problem;
  const algorithm = paths?.algorithm || problem?.algorithm || '';
  const starts = new Set(agents.map((a) => key(a.start[0], a.start[1])));
  const goals = new Set(agents.map((a) => key(a.goal[0], a.goal[1])));
  const distinctStarts = starts.size === agents.length;
  const distinctGoals = goals.size === agents.length;
  const violations = [];

  if (!distinctStarts) violations.push('Multiple agents share the same start cell.');
  if (!distinctGoals) violations.push('Multiple agents share the same goal cell.');

  if (!paths || !Object.keys(paths).length) {
    violations.push('No solution paths returned.');
    return {
      valid: false,
      in_bounds_and_obstacle_free: false,
      legal_moves: false,
      vertex_conflict_free: false,
      edge_swap_free: false,
      all_reached_goal: false,
      goal_occupancy_valid: false,
      distinct_starts: distinctStarts,
      distinct_goals: distinctGoals,
      violations,
      first_conflict: null,
    };
  }

  let inBoundsAndObstacleFree = true;
  let legalMoves = true;
  let vertexConflictFree = true;
  let edgeSwapFree = true;
  let allReachedGoal = true;
  let goalOccupancyValid = distinctGoals;

  for (const agent of agents) {
    const path = paths[agent.name];
    if (!path || !path.length) {
      allReachedGoal = false;
      legalMoves = false;
      goalOccupancyValid = false;
      violations.push(`Agent ${agent.name} has no path in solution.`);
      continue;
    }
    if (!same(path[0], agent.start)) {
      legalMoves = false;
      violations.push(`Agent ${agent.name} starts at (${path[0]}) instead of (${agent.start}).`);
    }
    for (let t = 0; t < path.length; t += 1) {
      const [r, c] = path[t];
      if (!inBounds(grid, r, c)) {
        inBoundsAndObstacleFree = false;
        violations.push(`Agent ${agent.name} moves outside the grid to (${r}, ${c}) at t=${t}.`);
      } else if (!isFree(grid, r, c)) {
        inBoundsAndObstacleFree = false;
        violations.push(`Agent ${agent.name} moves through obstacle at (${r}, ${c}) at t=${t}.`);
      }
    }
    for (let t = 0; t < path.length - 1; t += 1) {
      const curr = path[t];
      const nxt = path[t + 1];
      if (same(curr, nxt)) continue;
      const validNext = isFree(grid, curr[0], curr[1])
        ? neighbors(grid, curr, allowDiagonals)
        : [];
      if (!validNext.some((n) => same(n, nxt))) {
        legalMoves = false;
        violations.push(`Agent ${agent.name} makes illegal move (${curr}) -> (${nxt}) at t=${t} -> ${t + 1}.`);
      }
    }
    if (!same(path[path.length - 1], agent.goal)) {
      allReachedGoal = false;
      goalOccupancyValid = false;
      violations.push(`Agent ${agent.name} ends at (${path[path.length - 1]}) instead of goal (${agent.goal}).`);
    }
    if (!['prioritized', 'cbs', 'wcbs', 'space_time_astar'].includes(algorithm)) {
      for (let t = 0; t < path.length - 1; t += 1) {
        if (same(path[t], agent.goal) && !same(path[t + 1], agent.goal)) {
          goalOccupancyValid = false;
          violations.push(`Agent ${agent.name} moved away from assigned goal (${agent.goal}) between t=${t} and t=${t + 1}.`);
        }
      }
    }
  }

  const activeNames = agents.map((a) => a.name).filter((n) => paths[n]?.length);
  const horizon = Math.max(0, ...activeNames.map((n) => paths[n].length));
  let firstConflict = null;

  for (let t = 0; t < horizon; t += 1) {
    for (let i = 0; i < activeNames.length; i += 1) {
      for (let j = i + 1; j < activeNames.length; j += 1) {
        const a = activeNames[i]; const b = activeNames[j];
        const pa = paths[a]; const pb = paths[b];
        const aNow = cellAt(pa, t); const bNow = cellAt(pb, t);
        if (same(aNow, bNow)) {
          vertexConflictFree = false;
          if (t >= pa.length - 1 || t >= pb.length - 1) goalOccupancyValid = false;
          if (!firstConflict) {
            firstConflict = { kind: 'vertex', a, b, t, cell_a: aNow, cell_b: bNow };
          }
          violations.push(`Vertex conflict between ${a} and ${b} at (${aNow}) (t=${t}).`);
        }
        const aNext = cellAt(pa, t + 1); const bNext = cellAt(pb, t + 1);
        if (same(aNow, bNext) && same(bNow, aNext) && !same(aNow, bNow)) {
          edgeSwapFree = false;
          if (!firstConflict) {
            firstConflict = { kind: 'edge', a, b, t: t + 1, cell_a: aNext, cell_b: bNext };
          }
          violations.push(`Edge-swap conflict between ${a} and ${b} across (${aNow}) <-> (${bNow}) (t=${t + 1}).`);
        }
      }
    }
  }

  const valid = (
    distinctStarts
    && distinctGoals
    && inBoundsAndObstacleFree
    && legalMoves
    && vertexConflictFree
    && edgeSwapFree
    && allReachedGoal
    && goalOccupancyValid
  );
  return {
    valid,
    in_bounds_and_obstacle_free: inBoundsAndObstacleFree,
    legal_moves: legalMoves,
    vertex_conflict_free: vertexConflictFree,
    edge_swap_free: edgeSwapFree,
    all_reached_goal: allReachedGoal,
    goal_occupancy_valid: goalOccupancyValid,
    distinct_starts: distinctStarts,
    distinct_goals: distinctGoals,
    violations,
    first_conflict: firstConflict,
  };
}

/* -------------------------------------------- solvers (as generators) --- */

function reservationsToConstraints(reservations, horizon) {
  const cons = newConstraints();
  for (const path of Object.values(reservations)) {
    path.forEach((cell, t) => addVertex(cons, cell, t));
    for (let t = path.length; t <= horizon; t += 1) addVertex(cons, path[path.length - 1], t);
    for (let t = 0; t < path.length - 1; t += 1) addEdge(cons, path[t + 1], path[t], t + 1);
  }
  return cons;
}

/** Prioritized planning: agents in order, each avoiding the ones before it. */
export function* prioritizedPlanning(problem, options = {}) {
  const started = performance.now();
  const { grid, agents, allowDiagonals = false } = problem;
  const order = options.priority || agents.map((a) => a.name);
  const lookup = Object.fromEntries(agents.map((a) => [a.name, a]));

  yield { kind: 'root', agents: order, cost: 0 };
  const reservations = {};
  let expansions = 0;

  for (const name of order) {
    const agent = lookup[name];
    const reserved = Object.values(reservations).reduce((n, p) => n + p.length, 0);
    const horizon = freeCellCount(grid) + reserved + order.length + 1;
    const path = spaceTimeAStar(grid, agent.start, agent.goal,
      reservationsToConstraints(reservations, horizon),
      { heuristic: options.heuristic, allowDiagonals, maxTimestep: horizon });
    if (!path) {
      yield {
        kind: 'failed',
        reason: `agent ${name} has no path under the reservations of the ` +
                `${Object.keys(reservations).length} higher-priority agents`,
      };
      return null;
    }
    reservations[name] = path;
    expansions += 1;
    yield { kind: 'agent_planned', agent: name, path, cost: path.length - 1 };
    yield {
      kind: 'expand',
      node: expansions,
      cost: Object.values(reservations).reduce((n, p) => n + p.length - 1, 0),
      open: order.length - expansions,
      paths: { ...reservations },
    };
  }

  const solution = solutionMetrics(reservations, 'prioritized', expansions, (performance.now() - started) / 1000, problem);
  yield { kind: 'solved', ...solution };
  return solution;
}

/**
 * Conflict-based search. `weight === 1` is optimal CBS; a larger weight runs
 * the focal (ECBS) variant, whose cost is bounded by weight × optimal.
 */
export function* conflictBasedSearch(problem, options = {}) {
  const started = performance.now();
  const { grid, agents, allowDiagonals = false } = problem;
  const weight = options.weight ?? 1;
  const maxExpansions = options.maxExpansions ?? 10000;
  const timeLimit = options.timeLimit ?? null;
  const searchOptions = { heuristic: options.heuristic, allowDiagonals };

  const lowLevel = (agent, cons) => spaceTimeAStar(grid, agent.start, agent.goal, cons, searchOptions);

  const rootConstraints = {};
  const rootPaths = {};
  for (const agent of agents) {
    rootConstraints[agent.name] = newConstraints();
    const path = lowLevel(agent, rootConstraints[agent.name]);
    if (!path) {
      yield { kind: 'failed', reason: `agent ${agent.name} has no individual path` };
      return null;
    }
    rootPaths[agent.name] = path;
    yield { kind: 'agent_planned', agent: agent.name, path, cost: path.length - 1 };
  }

  const makeNode = (constraints, paths) => ({
    constraints,
    paths,
    cost: Object.values(paths).reduce((n, p) => n + p.length - 1, 0),
    conflicts: countConflicts(paths),
    expanded: false,
  });

  const root = makeNode(rootConstraints, rootPaths);
  yield { kind: 'root', agents: agents.map((a) => a.name), cost: root.cost };

  let tie = 0;
  const open = new Heap((a, b) => (a[0] - b[0]) || (a[1] - b[1]));
  const pending = new Heap((a, b) => (a[0] - b[0]) || (a[1] - b[1]));
  const focal = new Heap((a, b) => (a[0] - b[0]) || (a[1] - b[1]) || (a[2] - b[2]));
  const tag = tie++;
  open.push([root.cost, tag, root]);
  if (weight > 1) pending.push([root.cost, tag, root]);
  else focal.push([root.conflicts, root.cost, tag, root]);

  let expansions = 0;
  let timedOut = false;

  while (open.size && expansions < maxExpansions) {
    if (timeLimit != null && (performance.now() - started) / 1000 > timeLimit) { timedOut = true; break; }
    while (open.size && open.peek()[2].expanded) open.pop();
    if (!open.size) break;

    const bound = weight * open.peek()[0];
    while (pending.size && pending.peek()[0] <= bound) {
      const [cost, nodeTag, candidate] = pending.pop();
      focal.push([candidate.conflicts, cost, nodeTag, candidate]);
    }

    let node = null;
    while (focal.size) {
      const candidate = focal.pop()[3];
      if (candidate.expanded) continue;
      node = candidate;
      break;
    }
    if (!node) node = open.pop()[2];
    node.expanded = true;
    expansions += 1;

    yield {
      kind: 'expand',
      node: expansions,
      cost: node.cost,
      open: open.size,
      bound,
      paths: node.paths,
    };

    const conflict = findFirstConflict(node.paths);
    if (!conflict) {
      const solution = solutionMetrics(node.paths, weight > 1 ? 'wcbs' : 'cbs',
        expansions, (performance.now() - started) / 1000, problem);
      yield { kind: 'solved', ...solution };
      return solution;
    }

    yield {
      kind: 'conflict',
      type: conflict.kind,
      a: conflict.a,
      b: conflict.b,
      t: conflict.t,
      cell: conflict.cellA,
    };

    for (const name of [conflict.a, conflict.b]) {
      const constraints = {};
      for (const [other, cons] of Object.entries(node.constraints)) {
        constraints[other] = other === name ? copyConstraints(cons) : cons;
      }
      const mine = constraints[name];
      if (conflict.kind === 'vertex') {
        addVertex(mine, conflict.cellA, conflict.t);
      } else if (name === conflict.a) {
        addEdge(mine, conflict.cellB, conflict.cellA, conflict.t);
      } else {
        addEdge(mine, conflict.cellA, conflict.cellB, conflict.t);
      }

      const agent = agents.find((a) => a.name === name);
      const path = lowLevel(agent, mine);
      if (!path) continue;
      const child = makeNode(constraints, { ...node.paths, [name]: path });
      yield { kind: 'branch', agent: name, constraint: conflict.kind, cost: child.cost };

      const childTag = tie++;
      open.push([child.cost, childTag, child]);
      if (child.cost <= bound) focal.push([child.conflicts, child.cost, childTag, child]);
      else pending.push([child.cost, childTag, child]);
    }
  }

  const reason = timedOut
    ? `time limit (${timeLimit}s) reached after ${expansions} nodes`
    : (expansions >= maxExpansions
      ? `expansion limit (${maxExpansions}) reached`
      : 'constraint tree exhausted');
  yield { kind: 'failed', reason };
  return null;
}

/* ------------------------------------------- MAPPO rollout (exact port) --- */

const MAPF_ENV_ACTIONS = [[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]];

function encodeLocalObservation(grid, positions, goals, agentName, radius = 4) {
  const [row, col] = positions[agentName];
  const goal = goals[agentName];
  const size = 2 * radius + 1;
  const planeArea = size * size;
  const flat = new Float32Array(4 * planeArea + 3);

  const others = new Set();
  const otherGoals = new Set();
  for (const [name, pos] of Object.entries(positions)) {
    if (name !== agentName) others.add(key(pos[0], pos[1]));
  }
  for (const [name, g] of Object.entries(goals)) {
    if (name !== agentName) otherGoals.add(key(g[0], g[1]));
  }

  for (let dr = -radius; dr <= radius; dr += 1) {
    for (let dc = -radius; dc <= radius; dc += 1) {
      const cr = row + dr; const cc = col + dc;
      const r = dr + radius; const c = dc + radius;
      const idx = r * size + c;
      if (!inBounds(grid, cr, cc) || !isFree(grid, cr, cc)) {
        flat[idx] = 1.0;
        continue;
      }
      const ck = key(cr, cc);
      if (others.has(ck)) flat[planeArea + idx] = 1.0;
      if (otherGoals.has(ck)) flat[2 * planeArea + idx] = 1.0;
      if (cr === goal[0] && cc === goal[1]) flat[3 * planeArea + idx] = 1.0;
    }
  }

  const deltaRow = goal[0] - row;
  const deltaCol = goal[1] - col;
  const dist = Math.hypot(deltaRow, deltaCol);
  const diag = Math.hypot(grid.height, grid.width) || 1.0;
  const base = 4 * planeArea;
  if (dist > 1e-9) {
    flat[base] = deltaRow / dist;
    flat[base + 1] = deltaCol / dist;
  }
  flat[base + 2] = Math.min(1.0, dist / diag);
  return flat;
}

function mlpForward(input, actorParams) {
  let current = input;
  const layers = actorParams.length / 2;
  for (let l = 0; l < layers; l += 1) {
    const W = actorParams[2 * l];
    const b = actorParams[2 * l + 1];
    const outDim = b.length;
    const next = new Float64Array(outDim);
    for (let j = 0; j < outDim; j += 1) {
      let sum = b[j];
      for (let i = 0; i < current.length; i += 1) {
        sum += current[i] * W[i][j];
      }
      next[j] = l < layers - 1 ? Math.tanh(sum) : sum;
    }
    current = next;
  }
  return current;
}

function selectAction(logits, deterministic, rng) {
  let maxLogit = -Infinity;
  for (let i = 0; i < logits.length; i += 1) if (logits[i] > maxLogit) maxLogit = logits[i];
  const probs = new Float64Array(logits.length);
  let sumExp = 0;
  for (let i = 0; i < logits.length; i += 1) {
    probs[i] = Math.exp(logits[i] - maxLogit);
    sumExp += probs[i];
  }
  if (deterministic) {
    let best = 0;
    for (let i = 1; i < probs.length; i += 1) if (probs[i] > probs[best]) best = i;
    return best;
  }
  const draw = rng() * sumExp;
  let cum = 0;
  for (let i = 0; i < probs.length; i += 1) {
    cum += probs[i];
    if (draw <= cum) return i;
  }
  return probs.length - 1;
}

const MAPF_ACTION_LABELS = ['STAY', 'UP', 'DOWN', 'LEFT', 'RIGHT'];

function resolveEnvStep(grid, agentNames, positions, actions, goals = null, finished = null) {
  const intended = {};
  const blocked = {};
  const collided = {};
  for (const name of agentNames) {
    blocked[name] = false;
    collided[name] = false;
    const [r, c] = positions[name];
    const isFinished = (finished && finished[name]) || (goals && goals[name] && same([r, c], goals[name]));
    if (isFinished) {
      intended[name] = [r, c];
      continue;
    }
    const [dr, dc] = MAPF_ENV_ACTIONS[actions[name] ?? 0] || [0, 0];
    const target = [r + dr, c + dc];
    if (dr === 0 && dc === 0) {
      intended[name] = [r, c];
    } else if (!isFree(grid, target[0], target[1])) {
      intended[name] = [r, c];
      blocked[name] = true;
    } else {
      intended[name] = target;
    }
  }

  for (;;) {
    let changed = false;
    const claims = new Map();
    for (const name of agentNames) {
      const ck = key(intended[name][0], intended[name][1]);
      if (!claims.has(ck)) claims.set(ck, []);
      claims.get(ck).push(name);
    }
    for (const claimants of claims.values()) {
      if (claimants.length < 2) continue;
      for (const name of claimants) {
        if (!same(intended[name], positions[name])) {
          intended[name] = [...positions[name]];
          collided[name] = true;
          changed = true;
        }
      }
    }
    for (let i = 0; i < agentNames.length; i += 1) {
      for (let j = i + 1; j < agentNames.length; j += 1) {
        const a = agentNames[i]; const b = agentNames[j];
        if (same(intended[a], positions[b]) && same(intended[b], positions[a]) && !same(positions[a], positions[b])) {
          for (const name of [a, b]) {
            if (!same(intended[name], positions[name])) {
              intended[name] = [...positions[name]];
              collided[name] = true;
              changed = true;
            }
          }
        }
      }
    }
    if (!changed) break;
  }
  const collisionCount = Object.values(collided).filter(Boolean).length;
  return { resolved: intended, blocked, collided, collisionCount };
}

export function mappoRollout(problem, policyData, options = {}) {
  const started = performance.now();
  const { grid, agents } = problem;
  if (!policyData || !policyData.actor) {
    return {
      algorithm: 'mappo',
      available: false,
      solved: false,
      valid: false,
      sumOfCosts: null,
      sum_of_costs: null,
      makespan: null,
      expansions: null,
      runtime: 0,
      inference_time: 0,
      training_time: 0,
      status_reason: 'MAPPO policy checkpoint not loaded',
      paths: {},
      execution_log: [{ step: 0, kind: 'failed', text: 'MAPPO policy checkpoint not loaded' }],
      validation: validateSolution(problem, {}),
    };
  }

  const actionMode = String(options.mappo_mode || options.mappoMode || 'sampled').toLowerCase();
  const deterministic = actionMode === 'greedy';
  const scenarioSeed = Number(options.seed ?? 0);
  const evalSeed = Number(options.mappo_eval_seed ?? options.mappoSeed ?? scenarioSeed);
  const maxSteps = Number(options.maxSteps ?? options.max_steps ?? (4 * (grid.height + grid.width)));
  const radius = policyData.observation_kwargs?.radius ?? 4;
  const attempts = deterministic ? 1 : Number(options.rolloutAttempts ?? 1);
  const trainingMeta = policyData.training_meta || {};
  const trainingModeLabel = trainingMeta.mode || 'Pre-trained general MAPPO policy';
  const trainingSteps = trainingMeta.training_steps ?? 0;
  const trainingTimeSec = trainingMeta.training_time_sec ?? 0;

  const agentNames = agents.map((a) => a.name);
  const goals = Object.fromEntries(agents.map((a) => [a.name, a.goal]));

  let bestPaths = null;
  let bestSolved = false;
  let bestCost = Infinity;
  let bestCollisions = 0;
  let bestSteps = 0;
  let bestAtGoal = 0;
  let bestLogs = [];
  let bestFinishedList = [];

  for (let attempt = 0; attempt < attempts; attempt += 1) {
    const attemptSeed = evalSeed + attempt * 997;
    const rng = mulberry32(attemptSeed + 17);
    let positions = Object.fromEntries(agents.map((a) => [a.name, [...a.start]]));
    const finished = Object.fromEntries(agents.map((a) => [a.name, same(a.start, a.goal)]));
    const trajectories = Object.fromEntries(agents.map((a) => [a.name, [[...a.start]]]));
    let collisions = 0;
    let solved = false;
    let steps = 0;

    const logs = [
      {
        step: 0,
        kind: 'init',
        text: `[INIT] MAPPO rollout on ${grid.height}×${grid.width} grid (${agentNames.length} agents, mode=${actionMode}, scenario_seed=${scenarioSeed}, mappo_eval_seed=${attemptSeed}, horizon=${maxSteps} steps)`,
      },
      {
        step: 0,
        kind: 'policy',
        text: `[POLICY] ${trainingModeLabel} | ${Number(trainingSteps).toLocaleString()} training steps (offline training time=${Number(trainingTimeSec).toFixed(2)}s, 9×9 local window + goal bearing)`,
      },
    ];

    for (const name of agentNames) {
      if (finished[name]) {
        logs.push({
          step: 0,
          kind: 'goal',
          text: `[t=0] Agent ${name} starts on assigned goal (${goals[name]}) → marked FINISHED (stationary at goal)`,
        });
      }
    }

    for (let step = 0; step < maxSteps; step += 1) {
      if (agentNames.every((n) => finished[n])) {
        solved = true;
        break;
      }
      const prevPositions = Object.fromEntries(agentNames.map((n) => [n, [...positions[n]]]));
      const prevFinished = { ...finished };
      const actions = {};
      for (const name of agentNames) {
        if (finished[name]) {
          actions[name] = 0;
          continue;
        }
        const obs = encodeLocalObservation(grid, positions, goals, name, radius);
        const logits = mlpForward(obs, policyData.actor);
        actions[name] = selectAction(logits, deterministic, rng);
      }
      const { resolved, blocked, collided, collisionCount } = resolveEnvStep(
        grid,
        agentNames,
        positions,
        actions,
        goals,
        finished,
      );
      positions = resolved;
      collisions += collisionCount;
      steps += 1;

      const arrivedNow = [];
      const blockedEvents = [];
      const collisionEvents = [];
      const moveSummaries = [];

      for (const name of agentNames) {
        trajectories[name].push([...positions[name]]);
        if (same(positions[name], goals[name])) {
          finished[name] = true;
        }
        const prevPos = prevPositions[name];
        const currPos = positions[name];
        const actIdx = actions[name] ?? 0;
        const actLabel = prevFinished[name] ? 'LOCKED@GOAL' : (MAPF_ACTION_LABELS[actIdx] || String(actIdx));
        moveSummaries.push(`${name}:${actLabel}(${prevPos})→(${currPos})`);

        if (!prevFinished[name] && finished[name]) {
          arrivedNow.push(`Agent ${name} reached goal (${currPos}) (FINISHED & locked)`);
        }
        if (blocked[name]) {
          blockedEvents.push(`Agent ${name} ${actLabel} from (${prevPos}) blocked by obstacle/wall`);
        }
        if (collided[name]) {
          const [dr, dc] = MAPF_ENV_ACTIONS[actIdx] || [0, 0];
          const attempted = [prevPos[0] + dr, prevPos[1] + dc];
          const parkedOwner = agentNames.find(
            (other) => other !== name && prevFinished[other] && same(prevPositions[other], attempted),
          );
          if (parkedOwner) {
            collisionEvents.push(
              `Agent ${name} move (${prevPos})→(${attempted}) refused (occupied goal of finished Agent ${parkedOwner})`,
            );
          } else {
            collisionEvents.push(`Agent ${name} move (${prevPos})→(${attempted}) refused (vertex/edge conflict)`);
          }
        }
      }

      const allDone = agentNames.every((n) => finished[n]);
      const notable = arrivedNow.length > 0 || blockedEvents.length > 0 || collisionEvents.length > 0;
      if (steps <= 20 || notable || steps === maxSteps || allDone) {
        const finCnt = agentNames.filter((n) => finished[n]).length;
        logs.push({
          step: steps,
          kind: 'step',
          text: `[t=${steps}] ${moveSummaries.join(', ')} | finished=${finCnt}/${agentNames.length}`,
        });
      }
      for (const msg of arrivedNow) logs.push({ step: steps, kind: 'goal', text: `[t=${steps} GOAL] ${msg}` });
      for (const msg of collisionEvents) logs.push({ step: steps, kind: 'conflict', text: `[t=${steps} REFUSED] ${msg}` });
      for (const msg of blockedEvents) logs.push({ step: steps, kind: 'warn', text: `[t=${steps} BLOCKED] ${msg}` });

      if (allDone) {
        solved = true;
        break;
      }
    }

    const trimmed = {};
    for (const name of agentNames) {
      const path = trajectories[name];
      const g = goals[name];
      let end = path.length;
      while (end > 1 && same(path[end - 1], g) && same(path[end - 2], g)) end -= 1;
      trimmed[name] = path.slice(0, end);
    }

    const cost = Object.values(trimmed).reduce((acc, p) => acc + (p.length - 1), 0);
    const ms = Math.max(0, ...Object.values(trimmed).map((p) => p.length - 1));
    const atGoal = agentNames.filter((n) => finished[n]).length;

    if (solved) {
      logs.push({
        step: steps,
        kind: 'solved',
        text: `[TERMINATED: SOLVED] All ${agentNames.length}/${agentNames.length} agents reached and remained stationary at their goals at t=${steps} (sum_of_costs=${cost}, makespan=${ms}, refused_collisions=${collisions})`,
      });
    } else {
      const unfinished = agentNames
        .filter((n) => !finished[n])
        .map((n) => `${n}@(${positions[n]})→goal(${goals[n]})`);
      logs.push({
        step: steps,
        kind: 'failed',
        text: `[TRUNCATED: UNSOLVED] Horizon expired at t=${steps}/${maxSteps}: ${atGoal}/${agentNames.length} agents finished; unfinished=[${unfinished.join(', ')}]; refused_collisions=${collisions}`,
      });
    }

    if (!bestPaths || (solved && (!bestSolved || cost < bestCost))) {
      bestPaths = trimmed;
      bestSolved = solved;
      bestCost = cost;
      bestCollisions = collisions;
      bestSteps = steps;
      bestAtGoal = atGoal;
      bestLogs = logs;
      bestFinishedList = agentNames.filter((n) => finished[n]);
    }
    if (solved) break;
  }

  const inferenceTime = (performance.now() - started) / 1000;
  const validation = validateSolution(problem, bestPaths);
  const solvedAndValid = bestSolved && validation.valid;
  const rolloutHorizon = Math.max(0, ...Object.values(bestPaths).map((p) => p.length - 1));

  return {
    algorithm: 'mappo',
    available: true,
    solved: solvedAndValid,
    truncated: !solvedAndValid,
    valid: validation.valid,
    sumOfCosts: solvedAndValid ? bestCost : null,
    sum_of_costs: solvedAndValid ? bestCost : null,
    makespan: solvedAndValid ? rolloutHorizon : null,
    rollout_horizon: rolloutHorizon,
    expansions: null,
    runtime: inferenceTime,
    inference_time: inferenceTime,
    training_time: trainingTimeSec,
    training_steps: trainingSteps,
    training_mode: trainingModeLabel,
    mappo_mode: actionMode,
    scenario_seed: scenarioSeed,
    mappo_eval_seed: evalSeed,
    collisions_refused: bestCollisions,
    agents_at_goal: bestAtGoal,
    finished_agents: bestFinishedList,
    steps_taken: bestSteps,
    max_steps: maxSteps,
    status_reason: solvedAndValid
      ? `Solved via ${actionMode} rollout (${bestCollisions} refused collision attempts)`
      : `Truncated at ${bestSteps}/${maxSteps} steps (${bestAtGoal}/${agentNames.length} agents at goal, ${bestCollisions} refused collision attempts)`,
    paths: bestPaths,
    execution_log: bestLogs,
    validation,
  };
}

export function* mappoGenerator(problem, options = {}) {
  yield { kind: 'root', agents: problem.agents.map((a) => a.name), cost: 0 };
  const result = mappoRollout(problem, options.policyData, options);
  if (result.solved) {
    yield { kind: 'solved', ...result };
    return result;
  }
  yield { kind: 'failed', reason: result.status_reason, result };
  return result;
}

export function solveGenerator(problem, algorithm, options = {}) {
  if (algorithm === 'prioritized') return prioritizedPlanning(problem, options);
  if (algorithm === 'mappo') return mappoGenerator(problem, options);
  if (algorithm === 'wcbs') return conflictBasedSearch(problem, { ...options, weight: options.weight ?? 1.5 });
  return conflictBasedSearch(problem, { ...options, weight: 1 });
}

/** Drain a solver generator synchronously and attach recorded trace events. */
export function solve(problem, algorithm, options = {}) {
  const generator = solveGenerator(problem, algorithm, options);
  const events = [];
  let step = generator.next();
  while (!step.done) {
    if (step.value) events.push(step.value);
    step = generator.next();
  }
  if (step.value && typeof step.value === 'object') {
    step.value.traceEvents = events;
    return step.value;
  }
  return null;
}

/* ---------------------------------------------------------- scenarios --- */

/** Deterministic 32-bit PRNG, so a seed reproduces an instance exactly. */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
export const agentName = (i) => (i < 26 ? ALPHABET[i] : ALPHABET[i % 26] + (Math.floor(i / 26) + 1));

function largestComponent(matrix) {
  const height = matrix.length; const width = matrix[0].length;
  const seen = new Set(); let best = [];
  for (let r = 0; r < height; r += 1) {
    for (let c = 0; c < width; c += 1) {
      if (matrix[r][c] || seen.has(key(r, c))) continue;
      const stack = [[r, c]]; const component = [];
      seen.add(key(r, c));
      while (stack.length) {
        const [cr, cc] = stack.pop();
        component.push([cr, cc]);
        for (const [dr, dc] of ORTHOGONAL) {
          const nr = cr + dr; const nc = cc + dc;
          if (nr >= 0 && nr < height && nc >= 0 && nc < width && !matrix[nr][nc] && !seen.has(key(nr, nc))) {
            seen.add(key(nr, nc));
            stack.push([nr, nc]);
          }
        }
      }
      if (component.length > best.length) best = component;
    }
  }
  return best;
}

function sampleAgents(matrix, count, rng, minSeparation = 3) {
  const cells = largestComponent(matrix);
  if (cells.length < 2 * count) {
    throw new Error(`map has ${cells.length} reachable cells, need ${2 * count} for ${count} agents`);
  }
  const pool = [...cells];
  for (let i = pool.length - 1; i > 0; i -= 1) {
    const j = Math.floor(rng() * (i + 1));
    [pool[i], pool[j]] = [pool[j], pool[i]];
  }
  const starts = pool.slice(0, count);
  const goals = pool.slice(count);
  const distance = (a, b) => Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]);

  return starts.map((start, i) => {
    let index = 0;
    for (let k = 0; k < goals.length; k += 1) {
      if (distance(goals[k], start) >= minSeparation) { index = k; break; }
      if (distance(goals[k], start) > distance(goals[index], start)) index = k;
    }
    const [goal] = goals.splice(index, 1);
    return { name: agentName(i), start, goal };
  });
}

const bordered = (height, width, fill = 0) =>
  Array.from({ length: height }, (_, r) => Array.from({ length: width },
    (_, c) => ((r === 0 || r === height - 1 || c === 0 || c === width - 1) ? 1 : fill)));

export const SCENARIOS = {
  empty_room: {
    label: 'Empty room',
    blurb: 'Open floor — every interaction comes from the agents themselves.',
    defaults: { height: 12, width: 12, agents: 4 },
    build({ height, width, agents, seed }) {
      const matrix = bordered(height, width);
      return { matrix, agents: sampleAgents(matrix, agents, mulberry32(seed)) };
    },
  },
  random_obstacles: {
    label: 'Random obstacles',
    blurb: 'Uniform clutter: the standard MAPF stress map.',
    defaults: { height: 16, width: 16, agents: 6, density: 0.18 },
    build({ height, width, agents, seed, density = 0.18 }) {
      const rng = mulberry32(seed);
      const matrix = bordered(height, width);
      for (let r = 1; r < height - 1; r += 1) {
        for (let c = 1; c < width - 1; c += 1) matrix[r][c] = rng() < density ? 1 : 0;
      }
      return { matrix, agents: sampleAgents(matrix, agents, rng) };
    },
  },
  warehouse: {
    label: 'Warehouse',
    blurb: 'Shelf blocks and one-cell aisles — agents meet head-on constantly.',
    defaults: { height: 12, width: 19, agents: 8 },
    build({ agents, seed }) {
      const shelfRows = 3; const shelfCols = 4; const shelfH = 2; const shelfW = 3;
      const height = 2 + shelfRows * (shelfH + 1) + 1;
      const width = 2 + shelfCols * (shelfW + 1) + 1;
      const matrix = bordered(height, width);
      for (let br = 0; br < shelfRows; br += 1) {
        for (let bc = 0; bc < shelfCols; bc += 1) {
          const top = 2 + br * (shelfH + 1); const left = 2 + bc * (shelfW + 1);
          for (let r = top; r < Math.min(top + shelfH, height - 1); r += 1) {
            for (let c = left; c < Math.min(left + shelfW, width - 1); c += 1) matrix[r][c] = 1;
          }
        }
      }
      return { matrix, agents: sampleAgents(matrix, agents, mulberry32(seed), 6) };
    },
  },
  maze: {
    label: 'Maze',
    blurb: 'A perfect maze: exactly one route between any two cells.',
    defaults: { height: 15, width: 15, agents: 4 },
    build({ height, width, agents, seed }) {
      const h = height % 2 ? height : height + 1;
      const w = width % 2 ? width : width + 1;
      const matrix = Array.from({ length: h }, () => Array(w).fill(1));
      const rng = mulberry32(seed);
      matrix[1][1] = 0;
      const stack = [[1, 1]];
      while (stack.length) {
        const [r, c] = stack[stack.length - 1];
        const candidates = [];
        for (const [dr, dc] of [[-2, 0], [2, 0], [0, -2], [0, 2]]) {
          const nr = r + dr; const nc = c + dc;
          if (nr >= 1 && nr < h - 1 && nc >= 1 && nc < w - 1 && matrix[nr][nc]) candidates.push([nr, nc, dr, dc]);
        }
        if (!candidates.length) { stack.pop(); continue; }
        const [nr, nc, dr, dc] = candidates[Math.floor(rng() * candidates.length)];
        matrix[r + dr / 2][c + dc / 2] = 0;
        matrix[nr][nc] = 0;
        stack.push([nr, nc]);
      }
      return { matrix, agents: sampleAgents(matrix, agents, rng, 8) };
    },
  },
  bottleneck: {
    label: 'Bottleneck',
    blurb: 'Two rooms, one corridor. Somebody has to yield.',
    defaults: { agents: 4 },
    build({ agents, seed }) {
      const room = 5; const corridor = 5;
      const height = 2 * room + 1; const width = 2 * room + corridor + 2;
      const matrix = Array.from({ length: height }, () => Array(width).fill(1));
      const mid = Math.floor(height / 2);
      const left = []; const right = [];
      for (let r = 1; r < height - 1; r += 1) {
        for (let c = 1; c <= room; c += 1) { matrix[r][c] = 0; left.push([r, c]); }
        for (let c = room + corridor + 1; c < width - 1; c += 1) { matrix[r][c] = 0; right.push([r, c]); }
      }
      for (let c = room + 1; c <= room + corridor; c += 1) matrix[mid][c] = 0;

      const rng = mulberry32(seed);
      const shuffle = (list) => {
        for (let i = list.length - 1; i > 0; i -= 1) {
          const j = Math.floor(rng() * (i + 1));
          [list[i], list[j]] = [list[j], list[i]];
        }
      };
      shuffle(left); shuffle(right);
      const perSide = Math.max(1, Math.floor(agents / 2));
      const out = [];
      for (let i = 0; i < perSide; i += 1) out.push({ name: agentName(out.length), start: left[i], goal: right[i] });
      for (let i = 0; i < agents - perSide; i += 1) {
        out.push({ name: agentName(out.length), start: right[perSide + i], goal: left[perSide + i] });
      }
      return { matrix, agents: out };
    },
  },
  corner_swap: {
    label: 'Corner swap',
    blurb: 'Symmetric crossings — every route runs through the middle.',
    defaults: { agents: 4 },
    build({ agents }) {
      const size = 9;
      const matrix = bordered(size, size);
      const mid = Math.floor(size / 2); const last = size - 2;
      const ring = [
        [[1, 1], [last, last]], [[last, last], [1, 1]],
        [[1, last], [last, 1]], [[last, 1], [1, last]],
        [[mid, 1], [mid, last]], [[mid, last], [mid, 1]],
        [[1, mid], [last, mid]], [[last, mid], [1, mid]],
      ];
      const count = Math.min(agents, ring.length);
      return {
        matrix,
        agents: ring.slice(0, count).map(([start, goal], i) => ({ name: agentName(i), start, goal })),
      };
    },
  },
};

export function buildScenario(name, overrides = {}) {
  const spec = SCENARIOS[name];
  if (!spec) throw new Error(`unknown scenario ${name}`);
  const params = { seed: 0, ...spec.defaults, ...overrides };
  const { matrix, agents } = spec.build(params);
  return { name, label: spec.label, blurb: spec.blurb, matrix, agents, params };
}

export function problemFromState(state) {
  return {
    grid: gridFromMatrix(state.matrix),
    agents: state.agents.map((a) => ({ name: a.name, start: a.start, goal: a.goal })),
    allowDiagonals: !!state.allowDiagonals,
  };
}
