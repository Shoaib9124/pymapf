# PyMAPF — 2D Multi-Agent Pathfinding Research & Simulation Platform

A focused, reproducible 2D Multi-Agent Pathfinding (MAPF) research and simulation platform built on top of `pymapf` for controlled algorithmic evaluation, side-by-side visualization, and experimental benchmarking.

---

## 1. Overview & Core Capabilities

- **Interactive 2D Grid & Multi-Agent Editor**:
  - Configure 2D grid dimensions (`5×5` to `22×22`) or load 6 deterministic 2D benchmark scenario families (`empty_room`, `random_obstacles`, `warehouse`, `maze`, `bottleneck`, and `corner_swap`).
  - Click or drag cells to add/remove obstacles (`Walls`), place new agents (`Add Agent`), or move any agent's start (`Set Start`) and goal (`Set Goal`) positions.
  - Enforces strict validity invariants in real time: agents cannot be placed on obstacles, no two agents may share a start cell, and no two agents may share a goal cell.

- **Three Target Algorithmic Approaches**:
  1. **Prioritized Planning (`prioritized`)**: Sequential Space-Time A* with a dynamic vertex/edge reservation table (`pymapf/algorithms/prioritized_planning.py` & `pymapf/algorithms/space_time_astar.py`). Fast classical baseline; incomplete under adversarial priority orderings.
  2. **Conflict-Based Search (`cbs`)**: Canonical two-level solver (`pymapf/algorithms/cbs.py`) combining high-level best-first constraint-tree search over sum-of-costs with low-level Space-Time A* (`space_time_astar.py`). Complete and provably optimal for sum-of-costs.
  3. **Multi-Agent Proximal Policy Optimization (`mappo`)**: Centralized-training, decentralized-execution (CTDE) reinforcement learning (`pymapf/rl/algorithms.py`, `pymapf/rl/networks.py`, `pymapf/rl/env.py`). Uses a centralized value critic over the normalized global state (`5 * N + 1` features) during training and a shared actor over `9×9` egocentric 4-channel local windows + normalized goal vectors (`327` features) at execution time. Supports both a **pre-trained general policy** (`.docs/assets/mappo-policy.json`) and **live on-instance training**, with user-selectable **Greedy (argmax)** or **Sampled (stochastic)** rollout modes.

- **Synchronized 3-Panel Comparison & Playback**:
  - Configure a problem once and click **Run Comparison** to solve the exact same instance across Prioritized A*, CBS, and MAPPO, then scrub or play back all three trajectories simultaneously on a unified discrete timeline (`t = 0 … H_max`).
  - Run any single algorithm independently via **Run Single Algorithm**.

- **Formal Solution Validator (`pymapf.core.validate_solution`)**:
  - Every solution path or RL rollout is audited against 6 formal MAPF validity rules:
    1. **In-bounds & Obstacle-Free**: Every visited cell `(r, c)` is within grid bounds and not an obstacle.
    2. **Legal 4-Connected Moves**: Every transition from `t` to `t+1` is either a cardinal step (`|dr| + |dc| == 1`) or a wait action (`(0, 0)`), starting at `agent.start`.
    3. **Vertex Conflict-Free**: No two agents occupy the same cell at the same timestep `t`.
    4. **Edge-Swap Conflict-Free**: No two agents traverse the same undirected edge in opposite directions between `t-1` and `t`.
    5. **All Agents Reach Goal**: Every agent finishes its path on its assigned `agent.goal`.
    6. **Stay-at-Goal Occupancy**: Agents remain on their goal cells after arrival (`t >= T_i`) and block collisions with still-moving agents at all subsequent timesteps up to `makespan`.

- **Comparative Metrics & Batch Experiment Runner**:
  - Explicitly separates **online search planning time** (Prioritized A*, CBS) from **offline/instance RL training time** and **rollout inference time** (MAPPO).
  - Includes a multi-seed batch experiment runner across scenario families and agent counts with aggregated success rates, validity rates, suboptimality ratios vs. CBS, and **CSV export**.

---

## 2. Repository Structure

```text
├── main.py                              # Web server (0.0.0.0:3000) & CLI entry point
├── requirements.txt                     # Python dependencies (numpy)
├── pyproject.toml                       # Package metadata & pytest configuration
├── pymapf/
│   ├── __init__.py                      # Public 2D package exports & solve() helper
│   ├── scenarios.py                     # 6 deterministic 2D scenario builders & ASCII parser
│   ├── benchmark.py                     # Solver benchmarking & scaling study utilities
│   ├── research.py                      # Unified 3-algorithm comparison, MAPPO runner & batch harness
│   ├── core/
│   │   ├── grid.py                      # 2D GridMap representation & neighbor queries
│   │   ├── heuristics.py                # Manhattan, Euclidean, Chebyshev, Octile & true_distance
│   │   ├── solver.py                    # Agent, MAPFProblem, Solution, Constraints, Conflict & validate_solution
│   │   └── trace.py                     # SearchTrace & SearchEvent observer instrumentation
│   ├── algorithms/
│   │   ├── search.py                    # Single-agent A*, Dijkstra, and distance_table primitives
│   │   ├── space_time_astar.py          # Constraint-aware Space-Time A* with wait actions & goal settling
│   │   ├── prioritized_planning.py      # Sequential Prioritized Planning with reservation table
│   │   └── cbs.py                       # Optimal Conflict-Based Search (CBS)
│   └── rl/
│       ├── env.py                       # 2D MAPFEnv with simultaneous conflict resolution & cascade
│       ├── observation.py               # LocalWindow (4-channel 9x9 + goal vector) & GlobalGrid encoders
│       ├── reward.py                    # Potential-based ShapedReward (exact BFS distance) & SparseReward
│       ├── networks.py                  # Pure-NumPy (and optional Torch) Actor-Critic MLPs & Adam
│       ├── algorithms.py                # MAPPO & IPPO trainers, RolloutBuffer & GAE
│       ├── wrappers.py                  # VectorMAPFEnv & PettingZoo/Gymnasium adapters
│       ├── evaluate.py                  # Rollout & planner comparison utilities
│       └── spaces.py                    # Lightweight Box & Discrete space descriptors
├── .docs/
│   ├── index.html                       # Interactive 2D Research Platform UI
│   ├── styles.css                       # Research platform stylesheet
│   ├── app.js                           # Grid editor, synchronized 3-panel playback & batch runner UI
│   ├── mapf.js                          # In-browser fallback solvers & MAPPO inference engine
│   ├── worker.js                        # Web Worker for Pyodide / JS execution
│   ├── pymapf-bundle.json               # Generated bundle of pure-stdlib pymapf modules
│   └── assets/
│       ├── favicon.svg                  # Application icon
│       └── mappo-policy.json            # Pre-trained MAPPO actor-critic checkpoint
├── scripts/
│   ├── build_web_bundle.py              # Regenerates .docs/pymapf-bundle.json
│   └── train_rl.py                      # CLI training & evaluation script for MAPPO/IPPO
└── tests/                               # Unit & integration test suite
```

---

## 3. Quickstart & Commands

### Start the Interactive Web Application (Port 3000)

```bash
python3 main.py
```

Then open `http://localhost:3000` in your browser.

### Run a Headless CLI Comparison

```bash
python3 main.py --cli-demo
# Or customize scenario, agent count, and seed:
python3 main.py --cli --scenario bottleneck --agents 4 --seed 0
```

### Run the Test Suite

```bash
python3 -m unittest discover -s tests -p "test_research_platform.py" -v
```

### Rebuild the Browser Python Bundle

After modifying `pymapf/core`, `pymapf/algorithms`, `pymapf/scenarios.py`, or `pymapf/benchmark.py`:

```bash
python3 scripts/build_web_bundle.py
```

---

## 4. License & Attribution

Adapted from the open-source **PyMAPF** library by Erwin Lejeune (`MIT` / `Apache-2.0`, see [LICENSE](LICENSE) and [CITATION.cff](CITATION.cff)).
