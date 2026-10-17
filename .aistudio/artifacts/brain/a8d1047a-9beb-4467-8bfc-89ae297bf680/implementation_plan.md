# PyMAPF 2D Multi-Agent Pathfinding Research & Simulation Platform

A focused, academic-grade 2D Multi-Agent Pathfinding (MAPF) research workbench built on the existing PyMAPF library. Users can design or select 2D grid instances, configure multi-agent start and goal positions with strict invariant validation, run and compare **Prioritized Planning (Space-Time A\*)**, **Conflict-Based Search (CBS)**, and **Multi-Agent Proximal Policy Optimization (MAPPO)** side by side on a synchronized playback timeline, audit formal solution validity, and run reproducible batch experiments with CSV export.

### User Review & Critical Decisions

> [!IMPORTANT]
> The following architectural and product choices incorporate your confirmed answers from Phase 1 and define how the three target algorithms and research workflows will operate together.

- **Confirmed Decision 1 — MAPPO Execution Strategy**: Use a pre-trained general MAPPO policy (using the fixed-size `LocalWindow` observation encoder) for immediate evaluation rollouts on any 2D grid and team size, paired with an optional **On-Instance Fine-Tuning / Training** trigger that trains the centralized-critic MAPPO trainer on the active problem and separately records **Training Time** vs. **Evaluation/Inference Time**.
- **Confirmed Decision 2 — MAPPO Action Selection**: Provide an explicit toggle between **Greedy (`argmax`)** and **Sampled (`stochastic`)** rollout modes, allowing researchers to observe and document how stochastic sampling breaks deterministic period-2 orbits and livelocks at the expense of higher path cost.
- **Confirmed Decision 3 — Batch Experiment Suite**: Provide full controls over **2D Scenario Family**, **Agent Counts**, **Random Seeds**, and **MAPPO Rollout Mode**, computing aggregate success rate, validity rate, mean path cost, makespan, suboptimality vs. CBS, and timing metrics, with one-click **CSV Export** for research reports.
- **Scope Simplification & Preservation**: All 3D voxel views, quadrotor docking, continuous trajectory optimization, swarm/flocking/ORCA promotional sections, and extra solvers (LaCAM, PIBT, LNS, WCBS) are removed from the main application interface while preserving all underlying Python library modules, test suites, and license attributions intact.

---

### 1. Overview & Core Concept

- **What It Does**: Transforms the general-purpose PyMAPF showcase into a streamlined, rigorous 2D MAPF experimental platform tailored for university demonstrations and empirical evaluation. Users configure a 2D grid problem once and either inspect a single algorithm or run a 3-way side-by-side comparison across classical prioritized search, optimal conflict-resolution search, and decentralized reinforcement learning.
- **Target Audience / Persona**: Final-year university researchers, academic supervisors, and students evaluating the algorithmic trade-offs between completeness, optimality, planning time, and learned decentralized execution in discrete 2D multi-agent pathfinding.
- **Key Value**: Guarantees a controlled apples-to-apples comparison—every algorithm receives the exact same 2D grid, obstacles, agent starts/goals, and movement/collision rules, and every resulting plan or rollout is audited by the same formal validator and replayed on a single synchronized clock.

---

### 2. User Experience & Visual Design

- **Key User Flows**:
  1. **Configure the 2D Grid & Agents**: Select from the 6 planar scenario families (`empty_room`, `random_obstacles`, `warehouse`, `maze`, `bottleneck`, `corner_swap`) or set custom grid dimensions (rows × cols). Use the interactive canvas tool mode selector (**Toggle Wall**, **Add Agent**, **Move Start**, **Move Goal**, **Remove Agent**) to edit the instance. The editor enforces placement invariants in real time (preventing walls on starts/goals, overlapping starts, overlapping goals, or out-of-bounds placement).
  2. **Select Algorithm or Run 3-Way Comparison**: Choose a single algorithm (**Prioritized A\***, **CBS**, or **MAPPO**) or click **Run Comparison** to execute all three on the active problem instance. Configure shared solver parameters (heuristic, time budget) and MAPPO parameters (Greedy vs. Sampled rollout, pre-trained weights vs. on-instance training budget).
  3. **Synchronized Side-by-Side Playback**: Inspect the three synchronized canvas panels (**Prioritized Space-Time A\***, **Conflict-Based Search**, **MAPPO**). Scrub the shared timestep slider or use Play, Pause, Step Backward, Step Forward, and Speed controls (`0.5×`, `1×`, `2×`) to observe how agents navigate bottlenecks, wait for priority reservations, resolve conflicts, or experience decentralized deadlocks at the exact same timestep $t$.
  4. **Audit Validation & Comparative Metrics**: Review the side-by-side comparative metrics table and per-algorithm validation breakdown (checking obstacle clearance, cardinal adjacency, vertex collisions, edge swaps, goal arrival, and permanent goal occupancy semantics).
  5. **Run Batch Experiments & Export Data**: Switch to the **Batch Experiments** workspace, select scenario families, agent count sweeps, and seed ranges, run the batch evaluation across all three approaches, inspect the comparative SVG charts and aggregate summary table, and download the raw and aggregated results as CSV for inclusion in a thesis or report.

- **Visual Identity & Theme**:
  - *Aesthetic Direction*: Calibrated Scientific Instrumentation & Empirical Workbench. High-contrast, distraction-free layout with crisp 1px structural borders, clear spatial legends, and zero promotional fluff.
  - *Color Palette & Mood* (60-30-10 budget, supporting both Dark and Light lab modes):
    - **60% Canvas & Stage (`--plane` / `--surface-1`)**: Deep lab obsidian (`#0d0d0d` / `#1a1a19`) in dark mode and clinical paper (`#f9f9f7` / `#fcfcfb`) in light mode.
    - **30% Structural Surfaces & Grids (`--surface-2` / `--grid` / `--border`)**: Subdued slate panels (`#201f1e` / `#f2f1ee`) and hairline cell lattices (`#2c2c2a` / `#e1e0d9`).
    - **10% Calibrated Agent & Semantic Accents**: Distinct, colorblind-tested agent trajectories (`#3987e5`, `#d95926`, `#199e70`, `#c98500`, `#d55181`, `#008300`, `#9085e9`, `#e66767`) and semantic validation states (`#0ca30c` Nominal/Valid, `#fab219` Warning/Truncated, `#d03b3b` Conflict/Invalid) always paired with explicit text glyphs (`✓ Valid`, `✕ Conflict`, `▲ Timeout`).
  - *Typography & Hierarchy*:
    - **Headings & UI Labels**: Clean technical sans-serif (`Plus Jakarta Sans` / system geometric sans) with balanced headline wrapping and clear section hierarchy.
    - **Metrics, Timesteps, Coordinates & Tables**: Tabular monospace (`IBM Plex Mono` / `ui-monospace` with `font-variant-numeric: tabular-nums`) so timestep counters, path costs, and runtime columns never jitter during playback or live sweeps.
  - *Component Styling & Layout*:
    - **Top Bar Contract**: Single-row header with 3 zones separated by clean spacing: Brand Title (`PyMAPF Research Workbench`), concise navigation links (`Workspace`, `Comparison`, `Validation Rules`, `Batch Experiments`), and theme/reset action.
    - **Two-Zone Sandbox & Three-Panel Comparison Grid**: Left parameter & editor control column (320px) paired with the main stage (single interactive editor canvas + 3-column synchronized comparison viewport below or beside it).

- **Interactive Feedback & Motion**:
  - Smooth interpolated agent movement between discrete timesteps $t$ and $t+1$ with clear start markers (solid labeled discs), goal markers (concentric target rings), path rails (offset slightly per agent to keep overlapping corridors legible), and explicit conflict markers (`✕` at collision cells/timesteps).
  - Immediate inline toast/banner feedback when preventing invalid grid edits (e.g., attempting to place an obstacle on an agent's start or placing two agents on the same cell).

---

### 3. Key Product Decisions & Trade-Offs

- **Decision 1 — Unified Python + Browser Execution Architecture**:
  - *Chosen Approach*: Serve the web application on port `3000` via a lightweight Python standard-library HTTP server that both serves the static frontend and exposes clean JSON endpoints for native Python execution (including fast `numpy`-backed MAPPO rollouts, on-instance MAPPO training, and formal validation), while also bundling the pure-Python MAPPO/CBS/Prioritized inference engine into the browser bundle and JS fallback so the playground works standalone too.
  - *Why*: Running MAPPO training and evaluation natively in Python with NumPy is orders of magnitude faster and completely deterministic, while preserving the zero-build vanilla HTML/CSS/JS frontend architecture.
  - *Alternatives Considered*: Introducing React/Vite or FastAPI was rejected per project constraints to keep the stack simple, dependency-light, and faithful to the existing `.docs/` architecture.

- **Decision 2 — Pre-Trained General MAPPO Policy + Optional On-Instance Training**:
  - *Chosen Approach*: Ship a serialised, pre-trained MAPPO actor-critic checkpoint trained across the 2D scenario families (using the 327-dimensional `LocalWindow(radius=4)` egocentric observation), and provide an in-app **Train on Current Problem** option (configurable step budget, e.g. 10k–50k steps) that trains a fresh or fine-tuned `MAPPO` centralized-critic policy on the exact user grid.
  - *Why*: A pre-trained `LocalWindow` policy works immediately on any grid dimension and agent count when the user clicks **Run Comparison**, while on-instance training lets the user study how centralized-critic training improves coordination on hard bottlenecks—with **Training Time** and **Inference/Rollout Time** strictly separated in the UI.
  - *Alternatives Considered*: Training from scratch on every click without a pre-trained policy would either block the UI for minutes or produce near-random paths if under-trained.

- **Decision 3 — Strengthened Formal Solution Validator**:
  - *Chosen Approach*: Extend the validation pipeline to return a structured `ValidationReport` checking all six required criteria against a `MAPFProblem` and `Solution`: (1) in-bounds & obstacle-free cells, (2) legal single-step cardinal transitions ($\Delta r + \Delta c \le 1$), (3) vertex conflict freedom at all $t$, (4) edge-swap conflict freedom between $t$ and $t+1$, (5) goal reachability for every agent at its path end, and (6) goal-occupancy semantics (agents remaining on their goals block other agents at later timesteps).
  - *Why*: `Solution.is_valid()` previously only checked pairwise vertex/edge conflicts on the paths dict without verifying against the `MAPFProblem` grid obstacles, start cells, goal completion, or legal step adjacency. Strengthening this gives verifiable guarantees across Prioritized A\*, CBS, and MAPPO rollouts.

---

### 4. Technical Architecture & Data Strategy *(Technical Reference)*

- **Architecture & Component Diagram**:

```
┌────────────────────────────────────────────────────────────────────────────┐
│                   PyMAPF 2D Research & Simulation Platform                 │
├──────────────────────┬─────────────────────────────────────────────────────┤
│  2D Problem Builder  │         Synchronized Comparison & Analysis          │
│                      │                                                     │
│  ┌────────────────┐  │  ┌───────────────┐ ┌───────────────┐ ┌───────────┐  │
│  │ Scenario &     │  │  │ Panel 1:      │ │ Panel 2:      │ │ Panel 3:  │  │
│  │ Grid Dimension │  │  │ Prioritized   │ │ Conflict-Based│ │ MAPPO     │  │
│  │ Controls       │  │  │ Space-Time A* │ │ Search (CBS)  │ │ Policy    │  │
│  └───────┬────────┘  │  └───────┬───────┘ └───────┬───────┘ └─────┬─────┘  │
│          ▼           │          └─────────────────┼───────────────┘        │
│  ┌────────────────┐  │                            ▼                        │
│  │ Interactive 2D │  │  ┌───────────────────────────────────────────────┐  │
│  │ Grid & Agent   │  │  │ Shared Synchronized Playback Timeline (t=0…H) │  │
│  │ Canvas Editor  │  │  └─────────────────────────┬─────────────────────┘  │
│  └───────┬────────┘  │                            ▼                        │
│          │           │  ┌───────────────────────────────────────────────┐  │
│          │           │  │ Comparative Metrics & Formal Validation Audit │  │
│          │           │  └─────────────────────────┬─────────────────────┘  │
│          │           │                            ▼                        │
│          │           │  ┌───────────────────────────────────────────────┐  │
│          │           │  │ Batch Experiment Runner, SVG Charts & CSV Out │  │
│          │           │  └───────────────────────────────────────────────┘  │
└──────────┼───────────┴────────────────────────────▲────────────────────────┘
           │         Shared 2D Problem Spec         │
           ▼                                        │
┌───────────────────────────────────────────────────┴────────────────────────┐
│               Core Execution & Validation Layer (Python / WASM)            │
│                                                                            │
│  • MAPFProblem & GridMap (2D Planar Invariants)                            │
│  • Solvers: PrioritizedPlanning ("prioritized"), ConflictBasedSearch("cbs")│
│  • RL: MAPFEnv + MAPPO Trainer (Centralized Critic) + Pre-trained Weights  │
│  • Formal Validator: validate_solution(problem, solution) -> Report        │
│  • Batch Harness: compare_algorithms + RL evaluate -> JSON / CSV           │
└────────────────────────────────────────────────────────────────────────────┘
```

- **Data Model & State**:
  - **Problem Specification (`ProblemSpec`)**:
    - `matrix`: 2D binary array (`height × width`, `0` = free, `1` = obstacle).
    - `agents`: Ordered list of `{ name, start: [r, c], goal: [r, c] }` with uniqueness and free-cell invariants enforced.
    - `allowDiagonals`: Fixed to `false` (4-connected cardinal movement + wait action, consistent across planners and `MAPFEnv`).
  - **Algorithm Run Result (`AlgorithmOutcome`)**:
    - `algorithm`: `"prioritized"` | `"cbs"` | `"mappo"`.
    - `solved`: Boolean indicating whether all agents reached their goals and passed formal validation.
    - `valid`: Boolean from the comprehensive validator, paired with `validationDetails` (`obstacleFree`, `legalMoves`, `vertexConflictFree`, `edgeSwapFree`, `allReachedGoal`, `goalOccupancyValid`, and `firstViolation` description).
    - `paths`: Dict mapping agent name to `[[r, c], ...]` trajectory.
    - `sumOfCosts`: Integer sum of path lengths before settling at goal (or `null` if unsolved).
    - `makespan`: Integer maximum path length across agents.
    - `expansions`: Search node expansions (`prioritized`: agents planned / low-level calls; `cbs`: high-level CT nodes expanded; `mappo`: `N/A — policy rollout`).
    - `planningTime` / `inferenceTime` / `trainingTime`: Measured wall-clock seconds, explicitly separated for MAPPO (`trainingTime` vs. `inferenceTime`).
    - `statusReason`: Human-readable status or failure explanation (e.g., `"Optimal solution found"`, `"Agent C blocked by higher-priority reservations"`, `"Time limit (5.0s) reached"`, `"Truncated at horizon (2 agents did not reach goal)"`).

- **Interactive Component & State Mapping**:
  - **Grid & Agent Editor Controller**:
    - Resize grid (`height`, `width` inputs between `5` and `25`) → rebuilds matrix border, preserves in-bounds obstacles and agents, and relocates any out-of-bounds agents to valid free cells.
    - Tool Mode selector (`wall`, `add_agent`, `move_start`, `move_goal`, `delete_agent`) → handles pointer click/drag on the master editor canvas; validates that walls cannot overwrite starts/goals and starts/goals remain pairwise distinct per category.
  - **Execution & Comparison Controller**:
    - **Run Single Algorithm**: Executes the selected algorithm on the active `ProblemSpec`, updates its panel and the main view, and populates the validation audit.
    - **Run Comparison (All 3)**: Sequentially executes `prioritized`, `cbs`, and `mappo` on the identical `ProblemSpec`, normalizes the shared playback horizon $H = \max(\text{makespan}_{\text{prioritized}}, \text{makespan}_{\text{cbs}}, \text{makespan}_{\text{mappo}})$, renders all three comparison canvases, and populates the side-by-side comparison table.
    - **Train MAPPO on Instance**: Runs `MAPPO` training (`centralized_critic=True`) on the active problem for the user-selected step budget, records the wall-clock `trainingTime` and training solve-rate curve, and immediately evaluates the trained policy.
  - **Synchronized Playback Controller**:
    - Single `requestAnimationFrame` loop advancing shared timestep $t \in [0, H]$ at configurable playback speed, simultaneously redrawing the master canvas and all three comparison canvases (`Prioritized A*`, `CBS`, `MAPPO`) with per-panel status overlays and conflict highlights.
  - **Batch Experiment Controller**:
    - Executes multi-seed, multi-agent-count sweeps across `prioritized`, `cbs`, and `mappo`, updates live SVG charts (Runtime vs. Agents, Sum of Costs vs. Agents, Success Rate by Algorithm) and the aggregate results table, and provides a `Download CSV` handler that exports all experimental rows.
