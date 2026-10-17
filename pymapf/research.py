"""Unified 2D Multi-Agent Pathfinding research and comparison harness.

This module connects the three target approaches required by the research
platform under a single problem definition, validation pipeline, and metrics
schema:

1. **Prioritized Planning (Space-Time A*)** -- ``"prioritized"``
2. **Conflict-Based Search (CBS)** -- ``"cbs"``
3. **Multi-Agent Proximal Policy Optimization (MAPPO)** -- ``"mappo"``

All three approaches receive the exact same 2D :class:`~pymapf.core.MAPFProblem`
and are audited by :func:`~pymapf.core.validate_solution`. For MAPPO, model
training time and policy evaluation/inference time are tracked and reported
separately so that online search planning time is never conflated with offline
reinforcement learning training time.
"""

from __future__ import annotations

import csv
import io
import json
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

from .algorithms.cbs import ConflictBasedSearch
from .algorithms.prioritized_planning import PrioritizedPlanning
from .core.grid import GridMap
from .core.solver import (
    Agent,
    MAPFProblem,
    Solution,
    validate_solution,
)
from . core.trace import SearchTrace
from .scenarios import build_scenario

__all__ = [
    "TARGET_ALGORITHMS",
    "PLANAR_SCENARIOS",
    "DEFAULT_MAPPO_POLICY_PATH",
    "problem_from_spec",
    "validate_problem_spec",
    "load_pretrained_mappo_state",
    "train_pretrained_mappo_checkpoint",
    "run_single_algorithm",
    "run_comparison",
    "run_batch_experiment",
    "batch_rows_to_csv",
]

TARGET_ALGORITHMS: Tuple[str, ...] = ("prioritized", "cbs", "mappo")
PLANAR_SCENARIOS: Tuple[str, ...] = (
    "empty_room",
    "random_obstacles",
    "warehouse",
    "maze",
    "bottleneck",
    "corner_swap",
)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MAPPO_POLICY_PATH = os.path.join(
    ROOT_DIR, ".docs", "assets", "mappo-policy.json"
)

_CACHED_MAPPO_STATE: Optional[dict] = None


def validate_problem_spec(spec: dict) -> Tuple[bool, List[str]]:
    """Validate a raw 2D grid & agent problem specification before solving."""
    errors: List[str] = []
    matrix = spec.get("matrix")
    agents = spec.get("agents")

    if not isinstance(matrix, list) or not matrix or not isinstance(matrix[0], list):
        errors.append("Grid matrix must be a non-empty 2D array.")
        return False, errors

    height = len(matrix)
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        errors.append("All grid rows must have the same width.")
        return False, errors

    if not isinstance(agents, list) or len(agents) < 1:
        errors.append("At least one agent must be defined.")
        return False, errors

    names = set()
    starts = {}
    goals = {}

    for idx, entry in enumerate(agents):
        name = str(entry.get("name") or f"A{idx + 1}")
        if name in names:
            errors.append(f"Duplicate agent name '{name}'.")
        names.add(name)

        start = tuple(entry.get("start", ()))
        goal = tuple(entry.get("goal", ()))
        if len(start) != 2 or len(goal) != 2:
            errors.append(f"Agent '{name}' start and goal must be 2D [row, col] coordinates.")
            continue

        sr, sc = int(start[0]), int(start[1])
        gr, gc = int(goal[0]), int(goal[1])
        if not (0 <= sr < height and 0 <= sc < width):
            errors.append(f"Agent '{name}' start ({sr}, {sc}) is outside the {height}x{width} grid.")
        elif matrix[sr][sc]:
            errors.append(f"Agent '{name}' start ({sr}, {sc}) is placed on an obstacle.")

        if not (0 <= gr < height and 0 <= gc < width):
            errors.append(f"Agent '{name}' goal ({gr}, {gc}) is outside the {height}x{width} grid.")
        elif matrix[gr][gc]:
            errors.append(f"Agent '{name}' goal ({gr}, {gc}) is placed on an obstacle.")

        if (sr, sc) in starts:
            errors.append(
                f"Agents '{starts[(sr, sc)]}' and '{name}' share the same start cell ({sr}, {sc})."
            )
        else:
            starts[(sr, sc)] = name

        if (gr, gc) in goals:
            errors.append(
                f"Agents '{goals[(gr, gc)]}' and '{name}' share the same goal cell ({gr}, {gc})."
            )
        else:
            goals[(gr, gc)] = name

    return len(errors) == 0, errors


def problem_from_spec(spec: dict) -> MAPFProblem:
    """Construct a validated 2D :class:`MAPFProblem` from a JSON-compatible spec."""
    ok, errors = validate_problem_spec(spec)
    if not ok:
        raise ValueError("; ".join(errors))
    grid = GridMap(spec["matrix"])
    agents = [
        Agent(
            str(a["name"]),
            (int(a["start"][0]), int(a["start"][1])),
            (int(a["goal"][0]), int(a["goal"][1])),
        )
        for a in spec["agents"]
    ]
    return MAPFProblem(grid=grid, agents=agents, allow_diagonals=False)


def train_pretrained_mappo_checkpoint(
    output_path: str = DEFAULT_MAPPO_POLICY_PATH,
    steps: int = 80_000,
    seed: int = 0,
) -> dict:
    """Train a general 2D MAPPO policy with centralized critic and export JSON weights."""
    import numpy as np
    from .rl import MAPFEnv, make_trainer

    started = time.perf_counter()
    stage1_steps = max(16_384, int(steps * 0.65))
    stage2_steps = max(0, steps - stage1_steps)

    # Stage 1: Fast goal-acquisition and agent-avoidance on 8x8 empty room
    env1 = MAPFEnv(
        "empty_room",
        observation="local",
        reward="shaped",
        observation_kwargs={"radius": 4, "include_goal_vector": True},
        height=8,
        width=8,
        n_agents=2,
        seed=seed,
    )
    trainer1 = make_trainer(
        "mappo",
        env1,
        backend="numpy",
        n_envs=16,
        rollout_steps=128,
        hidden=(64, 64),
        lr=3e-4,
        seed=seed,
    )
    trainer1.learn(total_steps=stage1_steps, verbose=False)
    best_stage1_state = trainer1.policy.state_dict()
    best_solve_rate = float(max(0.0, trainer1.best_score))
    total_steps = int(trainer1.total_steps)

    # Stage 2: Obstacle & multi-agent fine-tuning on random_obstacles
    final_state = best_stage1_state
    if stage2_steps > 0:
        env2 = MAPFEnv(
            "random_obstacles",
            observation="local",
            reward="shaped",
            observation_kwargs={"radius": 4, "include_goal_vector": True},
            height=9,
            width=9,
            n_agents=3,
            density=0.12,
            seed=seed + 1,
        )
        trainer2 = make_trainer(
            "mappo",
            env2,
            backend="numpy",
            n_envs=16,
            rollout_steps=128,
            hidden=(64, 64),
            lr=2e-4,
            seed=seed + 1,
        )
        trainer2.policy.load_state_dict(best_stage1_state)
        trainer2.learn(total_steps=stage2_steps, verbose=False)
        total_steps += int(trainer2.total_steps)
        if trainer2.best_score > 0.15:
            final_state = trainer2.policy.state_dict()
            best_solve_rate = max(best_solve_rate, float(trainer2.best_score))

    training_time = time.perf_counter() - started
    payload = {
        "algorithm": "mappo",
        "centralized_critic": True,
        "observation": "local",
        "observation_kwargs": {"radius": 4, "include_goal_vector": True},
        "hidden": [64, 64],
        "obs_dim": int(final_state["obs_dim"]),
        "n_actions": int(final_state["n_actions"]),
        "critic_dim": int(final_state["critic_dim"]),
        "training_meta": {
            "mode": "Pre-trained general MAPPO (centralized critic, local 9x9 observation)",
            "training_steps": total_steps,
            "training_time_sec": round(float(training_time), 3),
            "best_solve_rate": round(best_solve_rate, 4),
            "training_distribution": "2-stage curriculum: empty_room (8x8) -> random_obstacles (9x9)",
            "seed": seed,
        },
        "actor": [
            np.round(param, 6).tolist() for param in final_state["actor"]
        ],
        "critic": [
            np.round(param, 6).tolist() for param in final_state["critic"]
        ],
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)

    global _CACHED_MAPPO_STATE
    _CACHED_MAPPO_STATE = payload
    return payload


def load_pretrained_mappo_state(path: str = DEFAULT_MAPPO_POLICY_PATH) -> dict:
    """Load the pre-trained MAPPO JSON checkpoint, generating it if missing."""
    global _CACHED_MAPPO_STATE
    if _CACHED_MAPPO_STATE is not None:
        return _CACHED_MAPPO_STATE
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            _CACHED_MAPPO_STATE = json.load(handle)
            return _CACHED_MAPPO_STATE
    return train_pretrained_mappo_checkpoint(output_path=path, steps=40_000, seed=42)


_ACTION_NAMES: Tuple[str, ...] = ("STAY", "UP", "DOWN", "LEFT", "RIGHT")


def _build_mappo_rollout_with_log(
    env,
    trainer,
    deterministic: bool,
    eval_seed: int,
    scenario_seed: int,
    action_mode: str,
    training_mode_label: str,
    training_steps: int,
    training_time: float,
):
    """Execute one MAPPO rollout episode and record a detailed step-by-step execution log."""
    import numpy as np

    if hasattr(trainer, "_random"):
        trainer._random = np.random.default_rng(eval_seed)
    if hasattr(trainer, "policy") and hasattr(trainer.policy, "_random"):
        trainer.policy._random = np.random.default_rng(eval_seed)

    observations, _ = env.reset(seed=eval_seed)
    logs: List[dict] = [
        {
            "step": 0,
            "kind": "init",
            "text": (
                f"[INIT] MAPPO rollout on {env.grid.height}×{env.grid.width} grid "
                f"({len(env.possible_agents)} agents, mode={action_mode}, "
                f"scenario_seed={scenario_seed}, mappo_eval_seed={eval_seed}, horizon={env.max_steps} steps)"
            ),
        },
        {
            "step": 0,
            "kind": "policy",
            "text": (
                f"[POLICY] {training_mode_label} | {training_steps:,} training steps "
                f"(offline training time={training_time:.2f}s, 9×9 local window + goal bearing)"
            ),
        },
    ]

    for agent in env.possible_agents:
        if env.finished.get(agent, False):
            logs.append(
                {
                    "step": 0,
                    "kind": "goal",
                    "text": (
                        f"[t=0] Agent {agent} starts on assigned goal {env.goals[agent]} "
                        f"→ marked FINISHED (stationary at goal)"
                    ),
                }
            )

    while env.agents:
        prev_positions = dict(env.positions)
        prev_finished = dict(env.finished)
        actions = trainer.act(observations, deterministic=deterministic)
        for agent in env.agents:
            if prev_finished.get(agent, False):
                actions[agent] = 0

        observations, _, terminations, truncations, infos = env.step(actions)
        t = env.step_count

        arrived_now: List[str] = []
        blocked_events: List[str] = []
        collision_events: List[str] = []
        move_summaries: List[str] = []

        for agent in env.possible_agents:
            prev_pos = prev_positions[agent]
            curr_pos = env.positions[agent]
            info = infos.get(agent, {})
            act_idx = int(actions.get(agent, 0))
            act_label = (
                "LOCKED@GOAL"
                if prev_finished.get(agent, False)
                else _ACTION_NAMES[act_idx] if 0 <= act_idx < len(_ACTION_NAMES) else str(act_idx)
            )
            move_summaries.append(f"{agent}:{act_label}{prev_pos}→{curr_pos}")

            if not prev_finished.get(agent, False) and env.finished.get(agent, False):
                arrived_now.append(f"Agent {agent} reached goal {curr_pos} (FINISHED & locked)")

            if info.get("blocked"):
                blocked_events.append(f"Agent {agent} {act_label} from {prev_pos} blocked by obstacle/wall")
            if info.get("collided"):
                delta = env.actions[act_idx] if 0 <= act_idx < len(env.actions) else (0, 0)
                attempted = (prev_pos[0] + delta[0], prev_pos[1] + delta[1])
                parked_owner = next(
                    (
                        other
                        for other in env.possible_agents
                        if other != agent
                        and prev_finished.get(other, False)
                        and prev_positions[other] == attempted
                    ),
                    None,
                )
                if parked_owner is not None:
                    collision_events.append(
                        f"Agent {agent} move {prev_pos}→{attempted} refused (occupied goal of finished Agent {parked_owner})"
                    )
                else:
                    collision_events.append(
                        f"Agent {agent} move {prev_pos}→{attempted} refused (vertex/edge conflict)"
                    )

        has_notable_event = bool(arrived_now or blocked_events or collision_events)
        if t <= 20 or has_notable_event or t == env.max_steps or any(terminations.values()):
            finished_cnt = sum(1 for a in env.possible_agents if env.finished.get(a, False))
            step_line = f"[t={t}] {', '.join(move_summaries)} | finished={finished_cnt}/{len(env.possible_agents)}"
            logs.append({"step": t, "kind": "step", "text": step_line})

        for msg in arrived_now:
            logs.append({"step": t, "kind": "goal", "text": f"[t={t} GOAL] {msg}"})
        for msg in collision_events:
            logs.append({"step": t, "kind": "conflict", "text": f"[t={t} REFUSED] {msg}"})
        for msg in blocked_events:
            logs.append({"step": t, "kind": "warn", "text": f"[t={t} BLOCKED] {msg}"})

        if any(terminations.values()) or any(truncations.values()):
            break

    solution = env.solution(algorithm="mappo")
    summary = env.episode_summary()
    if summary.get("solved"):
        logs.append(
            {
                "step": env.step_count,
                "kind": "solved",
                "text": (
                    f"[TERMINATED: SOLVED] All {len(env.possible_agents)}/{len(env.possible_agents)} "
                    f"agents reached and remained stationary at their goals at t={env.step_count} "
                    f"(sum_of_costs={solution.sum_of_costs}, makespan={solution.makespan}, "
                    f"refused_collisions={summary['collisions']})"
                ),
            }
        )
    else:
        unfinished = [
            f"{a}@{env.positions[a]}→goal{env.goals[a]}"
            for a in env.possible_agents
            if not env.finished.get(a, False)
        ]
        logs.append(
            {
                "step": env.step_count,
                "kind": "failed",
                "text": (
                    f"[TRUNCATED: UNSOLVED] Horizon expired at t={env.step_count}/{env.max_steps}: "
                    f"{summary['agents_at_goal']}/{len(env.possible_agents)} agents finished; "
                    f"unfinished=[{', '.join(unfinished)}]; refused_collisions={summary['collisions']}"
                ),
            }
        )
    return solution, summary, logs


def _build_search_execution_log(
    problem: MAPFProblem,
    algo: str,
    trace: SearchTrace,
    solution: Optional[Solution],
    report,
    elapsed: float,
    heuristic: str,
    time_limit: float,
    max_expansions: int,
) -> List[dict]:
    """Format a structured algorithm-specific execution log from a SearchTrace."""
    logs: List[dict] = []
    agent_lookup = {a.name: a for a in problem.agents}

    if algo == "prioritized":
        order = [a.name for a in problem.agents]
        root_events = trace.of_kind("root")
        if root_events and root_events[0].get("agents"):
            order = list(root_events[0].get("agents"))
        logs.append(
            {
                "step": 0,
                "kind": "init",
                "text": (
                    f"[INIT] Prioritized Space-Time A* on {problem.grid.height}×{problem.grid.width} grid "
                    f"({len(problem.agents)} agents, heuristic={heuristic}, priority order: {' → '.join(order)})"
                ),
            }
        )
        planned_count = 0
        reserved_vertices = 0
        for ev in trace.events:
            if ev.kind == "agent_planned":
                name = ev.get("agent", "?")
                path = ev.get("path") or []
                cost = ev.get("cost", max(0, len(path) - 1))
                ag = agent_lookup.get(name)
                l1 = (
                    abs(ag.start[0] - ag.goal[0]) + abs(ag.start[1] - ag.goal[1])
                    if ag
                    else 0
                )
                preview = " → ".join(
                    f"({r},{c})" for r, c in (path[:6] if len(path) <= 7 else path[:4] + path[-2:])
                )
                if len(path) > 7:
                    preview = (
                        " → ".join(f"({r},{c})" for r, c in path[:4])
                        + f" → … ({len(path) - 6} steps) → "
                        + " → ".join(f"({r},{c})" for r, c in path[-2:])
                    )
                logs.append(
                    {
                        "step": planned_count + 1,
                        "kind": "plan",
                        "text": (
                            f"[PLAN #{planned_count + 1}] Agent {name}: start {ag.start if ag else '?'} → "
                            f"goal {ag.goal if ag else '?'} (L1={l1}, detours={cost - l1}) | "
                            f"avoiding {planned_count} higher-priority trajectories ({reserved_vertices} reserved states) | "
                            f"arrived at t={cost}: {preview}"
                        ),
                    }
                )
                planned_count += 1
                reserved_vertices += len(path)
            elif ev.kind == "failed":
                logs.append(
                    {
                        "step": planned_count + 1,
                        "kind": "failed",
                        "text": f"[FAILED] {ev.get('reason', 'No path found under higher-priority reservations')}",
                    }
                )
        if solution is not None and report.valid:
            logs.append(
                {
                    "step": planned_count,
                    "kind": "solved",
                    "text": (
                        f"[SOLVED] Prioritized Space-Time A* succeeded in {elapsed * 1000:.2f} ms: "
                        f"sum_of_costs={solution.sum_of_costs}, makespan={solution.makespan}, "
                        f"low_level_expansions={solution.expansions}, validation=PASS"
                    ),
                }
            )
        elif solution is not None:
            logs.append(
                {
                    "step": planned_count,
                    "kind": "failed",
                    "text": f"[INVALID] Plan failed formal validation: {'; '.join(report.violations)}",
                }
            )
        return logs

    # Conflict-Based Search (CBS)
    logs.append(
        {
            "step": 0,
            "kind": "init",
            "text": (
                f"[INIT] Conflict-Based Search (CBS) on {problem.grid.height}×{problem.grid.width} grid "
                f"({len(problem.agents)} agents, low-level heuristic={heuristic}, "
                f"time_limit={time_limit:.1f}s, max_ct_nodes={max_expansions})"
            ),
        }
    )
    current_node = 0
    for ev in trace.events:
        if ev.kind == "agent_planned":
            name = ev.get("agent", "?")
            cost = ev.get("cost", 0)
            ag = agent_lookup.get(name)
            logs.append(
                {
                    "step": 0,
                    "kind": "plan",
                    "text": (
                        f"[ROOT-LOW] Agent {name} individual unconstrained shortest path "
                        f"{ag.start if ag else '?'} → {ag.goal if ag else '?'}: cost={cost}"
                    ),
                }
            )
        elif ev.kind == "root":
            logs.append(
                {
                    "step": 0,
                    "kind": "init",
                    "text": (
                        f"[ROOT-CT] Built Root CT Node #0: unconstrained sum_of_costs={ev.get('cost', 0)}"
                    ),
                }
            )
        elif ev.kind == "expand":
            current_node = int(ev.get("node", current_node + 1))
            if current_node <= 25 or current_node % 25 == 0:
                logs.append(
                    {
                        "step": current_node,
                        "kind": "expand",
                        "text": (
                            f"[CT EXPAND #{current_node}] Popped CT node with g-cost={ev.get('cost', 0)} "
                            f"(open_list={ev.get('open', 0)})"
                        ),
                    }
                )
        elif ev.kind == "conflict":
            if current_node <= 25 or current_node % 25 == 0:
                logs.append(
                    {
                        "step": current_node,
                        "kind": "conflict",
                        "text": (
                            f"[CONFLICT @ CT #{current_node}] {str(ev.get('type', 'vertex')).upper()} conflict "
                            f"between Agent {ev.get('a')} and Agent {ev.get('b')} at cell {ev.get('cell')} (t={ev.get('t')})"
                        ),
                    }
                )
        elif ev.kind == "branch":
            if current_node <= 25 or current_node % 25 == 0:
                logs.append(
                    {
                        "step": current_node,
                        "kind": "branch",
                        "text": (
                            f"[BRANCH @ CT #{current_node}] Constrained Agent {ev.get('agent')} "
                            f"({ev.get('constraint')} constraint) → replanned child CT node g-cost={ev.get('cost')}"
                        ),
                    }
                )
        elif ev.kind == "failed":
            logs.append(
                {
                    "step": current_node,
                    "kind": "failed",
                    "text": f"[FAILED] CBS terminated: {ev.get('reason', 'No conflict-free solution found')}",
                }
            )

    if solution is not None and report.valid:
        logs.append(
            {
                "step": solution.expansions,
                "kind": "solved",
                "text": (
                    f"[SOLVED] Conflict-free CT node verified in {elapsed * 1000:.2f} ms! "
                    f"Optimal sum_of_costs={solution.sum_of_costs}, makespan={solution.makespan}, "
                    f"CT nodes expanded={solution.expansions}, conflicts resolved={len(trace.of_kind('conflict'))}"
                ),
            }
        )
    elif solution is not None:
        logs.append(
            {
                "step": solution.expansions,
                "kind": "failed",
                "text": f"[INVALID] Solution failed validation: {'; '.join(report.violations)}",
            }
        )
    return logs


def _run_mappo(
    problem: MAPFProblem,
    options: dict,
) -> dict:
    """Execute MAPPO on ``problem`` and return a unified outcome dictionary."""
    try:
        from .rl import MAPFEnv, make_trainer
    except ImportError as exc:
        return {
            "algorithm": "mappo",
            "available": False,
            "solved": False,
            "valid": False,
            "sum_of_costs": None,
            "makespan": None,
            "expansions": None,
            "runtime": 0.0,
            "inference_time": 0.0,
            "training_time": 0.0,
            "status_reason": f"MAPPO unavailable: numpy is required ({exc})",
            "paths": {},
            "execution_log": [
                {
                    "step": 0,
                    "kind": "failed",
                    "text": f"MAPPO unavailable: numpy is required ({exc})",
                }
            ],
            "validation": validate_solution(problem, None).as_dict(),
        }

    action_mode = str(options.get("mappo_mode", "sampled")).lower()
    deterministic = action_mode == "greedy"
    scenario_seed = int(options.get("seed", 0))
    mappo_eval_seed = int(
        options.get("mappo_eval_seed", options.get("mappo_seed", scenario_seed))
    )
    train_on_instance = bool(options.get("train_on_instance", False))
    train_steps = int(options.get("train_steps", 16_384))
    max_steps = options.get("max_steps")

    pretrained = load_pretrained_mappo_state()
    obs_kwargs = pretrained.get(
        "observation_kwargs", {"radius": 4, "include_goal_vector": True}
    )
    hidden = tuple(pretrained.get("hidden", [64, 64]))

    env = MAPFEnv(
        problem,
        observation="local",
        reward="shaped",
        observation_kwargs=obs_kwargs,
        max_steps=max_steps,
        randomise=False,
        seed=mappo_eval_seed,
    )

    trainer = make_trainer(
        "mappo",
        env,
        backend="numpy",
        n_envs=4 if train_on_instance else 1,
        rollout_steps=64,
        hidden=hidden,
        lr=3e-4,
        seed=mappo_eval_seed,
    )
    trainer.policy.load_state_dict(pretrained)

    training_time = float(
        pretrained.get("training_meta", {}).get("training_time_sec", 0.0)
    )
    training_steps = int(
        pretrained.get("training_meta", {}).get("training_steps", 0)
    )
    training_mode_label = (
        "Pre-trained general MAPPO policy"
        if not train_on_instance
        else f"Pre-trained + On-instance fine-tuned ({train_steps:,} steps)"
    )

    if train_on_instance and train_steps > 0:
        train_started = time.perf_counter()
        trainer.learn(total_steps=train_steps, verbose=False)
        training_time = time.perf_counter() - train_started
        training_steps = int(trainer.total_steps)

    # Measure inference/rollout time strictly separately from training time.
    rollout_attempts = 1 if deterministic else int(options.get("rollout_attempts", 1))
    infer_started = time.perf_counter()
    best_solution: Optional[Solution] = None
    best_summary: Optional[dict] = None
    best_logs: List[dict] = []

    for attempt in range(max(1, rollout_attempts)):
        sol, summary, logs = _build_mappo_rollout_with_log(
            env,
            trainer,
            deterministic=deterministic,
            eval_seed=mappo_eval_seed + attempt * 997,
            scenario_seed=scenario_seed,
            action_mode=action_mode,
            training_mode_label=training_mode_label,
            training_steps=training_steps,
            training_time=training_time,
        )
        if best_solution is None:
            best_solution, best_summary, best_logs = sol, summary, logs
        if summary.get("solved") and sol.is_valid(problem):
            if (
                best_summary is None
                or not best_summary.get("solved")
                or sol.sum_of_costs < best_solution.sum_of_costs
            ):
                best_solution, best_summary, best_logs = sol, summary, logs
            break

    inference_time = time.perf_counter() - infer_started
    assert best_solution is not None and best_summary is not None
    best_solution.algorithm = "mappo"
    best_solution.runtime = inference_time

    report = validate_solution(problem, best_solution)
    solved = bool(best_summary["solved"]) and report.valid
    truncated = bool(best_summary.get("truncated", not solved))

    if solved:
        reason = (
            f"Solved via {action_mode} rollout "
            f"({best_summary['collisions']} blocked/refused collision attempts during rollout)"
        )
    else:
        at_goal = best_summary.get("agents_at_goal", 0)
        total_agents = len(problem.agents)
        reason = (
            f"Truncated at {best_summary['steps']}/{env.max_steps} steps "
            f"({at_goal}/{total_agents} agents at goal, "
            f"{best_summary['collisions']} refused collision attempts)"
        )

    return {
        "algorithm": "mappo",
        "available": True,
        "solved": solved,
        "truncated": truncated,
        "valid": report.valid,
        "sum_of_costs": best_solution.sum_of_costs if solved else None,
        "makespan": best_solution.makespan if solved else None,
        "rollout_horizon": best_solution.makespan,
        "expansions": None,
        "runtime": inference_time,
        "inference_time": inference_time,
        "training_time": training_time,
        "training_steps": training_steps,
        "training_mode": training_mode_label,
        "mappo_mode": action_mode,
        "scenario_seed": scenario_seed,
        "mappo_eval_seed": mappo_eval_seed,
        "collisions_refused": int(best_summary.get("collisions", 0)),
        "agents_at_goal": int(best_summary.get("agents_at_goal", 0)),
        "finished_agents": list(best_summary.get("finished_agents", [])),
        "steps_taken": int(best_summary.get("steps", 0)),
        "max_steps": int(env.max_steps),
        "status_reason": reason,
        "paths": {
            name: [list(cell) for cell in path]
            for name, path in best_solution.paths.items()
        },
        "execution_log": best_logs,
        "validation": report.as_dict(),
    }


def run_single_algorithm(
    problem: MAPFProblem,
    algorithm: str,
    options: Optional[dict] = None,
) -> dict:
    """Run one of the three target algorithms on ``problem`` and validate its output."""
    opts = dict(options or {})
    algo = algorithm.lower().strip()
    if algo not in TARGET_ALGORITHMS:
        raise ValueError(
            f"Unsupported algorithm {algorithm!r}. Target algorithms: {TARGET_ALGORITHMS}"
        )

    if algo == "mappo":
        return _run_mappo(problem, opts)

    heuristic = opts.get("heuristic", "manhattan")
    time_limit = float(opts.get("time_limit", 5.0))
    max_expansions = int(opts.get("max_expansions", 10000))

    trace = SearchTrace()
    started = time.perf_counter()
    if algo == "prioritized":
        priority = opts.get("priority")
        solver = PrioritizedPlanning(heuristic=heuristic, priority=priority)
        solution = solver.solve(problem, observer=trace)
    else:
        solver = ConflictBasedSearch(
            heuristic=heuristic,
            max_expansions=max_expansions,
            time_limit=time_limit,
        )
        solution = solver.solve(problem, observer=trace)
    elapsed = time.perf_counter() - started

    conflicts_detected = len(trace.of_kind("conflict"))
    failed_events = trace.of_kind("failed")
    failure_reason = (
        failed_events[-1].payload.get("reason", "No solution found")
        if failed_events
        else "No solution found"
    )

    if solution is None:
        report = validate_solution(problem, None)
        execution_log = _build_search_execution_log(
            problem,
            algo,
            trace,
            None,
            report,
            elapsed,
            heuristic,
            time_limit,
            max_expansions,
        )
        return {
            "algorithm": algo,
            "available": True,
            "solved": False,
            "valid": False,
            "sum_of_costs": None,
            "makespan": None,
            "expansions": len(trace.of_kind("expand")),
            "conflicts_detected": conflicts_detected,
            "runtime": elapsed,
            "planning_time": elapsed,
            "status_reason": failure_reason,
            "paths": {},
            "execution_log": execution_log,
            "validation": report.as_dict(),
        }

    report = validate_solution(problem, solution)
    solved = report.valid
    status_reason = (
        "Optimal conflict-free plan found"
        if algo == "cbs" and solved
        else (
            "Sequential priority plan found"
            if algo == "prioritized" and solved
            else ("; ".join(report.violations) if report.violations else "Invalid plan")
        )
    )
    execution_log = _build_search_execution_log(
        problem,
        algo,
        trace,
        solution,
        report,
        elapsed,
        heuristic,
        time_limit,
        max_expansions,
    )

    return {
        "algorithm": algo,
        "available": True,
        "solved": solved,
        "valid": report.valid,
        "sum_of_costs": solution.sum_of_costs if solved else None,
        "makespan": solution.makespan if solved else None,
        "expansions": solution.expansions,
        "conflicts_detected": conflicts_detected,
        "runtime": elapsed,
        "planning_time": elapsed,
        "status_reason": status_reason,
        "paths": {
            name: [list(cell) for cell in path]
            for name, path in solution.paths.items()
        },
        "execution_log": execution_log,
        "validation": report.as_dict(),
    }


def run_comparison(
    problem: MAPFProblem,
    options: Optional[dict] = None,
) -> dict:
    """Run Prioritized A*, CBS, and MAPPO on the identical ``problem``."""
    opts = dict(options or {})
    outcomes = {}
    for algo in TARGET_ALGORITHMS:
        outcomes[algo] = run_single_algorithm(problem, algo, opts)

    cbs_cost = outcomes["cbs"].get("sum_of_costs")
    for algo, res in outcomes.items():
        cost = res.get("sum_of_costs")
        if cost is not None and cbs_cost and cbs_cost > 0:
            res["suboptimality"] = round(cost / cbs_cost, 4)
        elif cost == 0 and cbs_cost == 0:
            res["suboptimality"] = 1.0
        else:
            res["suboptimality"] = None

    shared_horizon = max(
        (
            res.get("makespan")
            or res.get("rollout_horizon")
            or res.get("steps_taken")
            or 0
            for res in outcomes.values()
        ),
        default=0,
    )
    return {
        "outcomes": outcomes,
        "shared_horizon": shared_horizon,
        "n_agents": len(problem.agents),
        "grid_shape": [problem.grid.height, problem.grid.width],
    }


def run_batch_experiment(
    scenario_name: str | Sequence[str] = "random_obstacles",
    agent_counts: Sequence[int] = (2, 4, 6),
    seeds: Sequence[int] = (0, 1, 2),
    options: Optional[dict] = None,
    scenarios: Optional[Sequence[str]] = None,
) -> dict:
    """Run a reproducible batch experiment across one or more scenarios, agent counts, and seeds."""
    opts = dict(options or {})
    if scenarios is None and opts.get("scenarios"):
        scenarios = opts.get("scenarios")
    if scenarios is not None:
        scenario_list = [str(s) for s in scenarios]
    elif isinstance(scenario_name, (list, tuple)):
        scenario_list = [str(s) for s in scenario_name]
    else:
        scenario_list = [str(scenario_name)]

    if not scenario_list:
        scenario_list = ["random_obstacles"]

    for scen in scenario_list:
        if scen not in PLANAR_SCENARIOS:
            raise ValueError(
                f"Scenario {scen!r} is not a 2D planar scenario. "
                f"Choose from {PLANAR_SCENARIOS}."
            )

    mappo_seed_offset = int(opts.get("mappo_seed_offset", 0))
    explicit_mappo_seed = opts.get("mappo_eval_seed", opts.get("mappo_seed"))
    rows: List[dict] = []

    for scen in scenario_list:
        for n_agents in agent_counts:
            max_allowed = 8 if scen == "corner_swap" else 10
            effective_agents = min(int(n_agents), max_allowed)
            for seed in seeds:
                scen_seed = int(seed)
                eval_seed = (
                    int(explicit_mappo_seed) + scen_seed
                    if explicit_mappo_seed is not None
                    else scen_seed + mappo_seed_offset
                )
                scenario = build_scenario(
                    scen, n_agents=effective_agents, seed=scen_seed
                )
                problem = scenario.to_problem()
                comp = run_comparison(
                    problem,
                    {
                        **opts,
                        "seed": scen_seed,
                        "mappo_eval_seed": eval_seed,
                    },
                )
                for algo in TARGET_ALGORITHMS:
                    res = comp["outcomes"][algo]
                    rows.append(
                        {
                            "scenario": scen,
                            "n_agents": len(problem.agents),
                            "seed": scen_seed,
                            "scenario_seed": scen_seed,
                            "mappo_eval_seed": eval_seed if algo == "mappo" else None,
                            "mappo_mode": res.get("mappo_mode") if algo == "mappo" else None,
                            "algorithm": algo,
                            "solved": bool(res["solved"]),
                            "valid": bool(res["valid"]),
                            "sum_of_costs": res["sum_of_costs"],
                            "makespan": res["makespan"] if res["solved"] else None,
                            "expansions": res["expansions"],
                            "runtime": round(float(res["runtime"]), 6),
                            "training_time": (
                                round(float(res.get("training_time", 0.0)), 3)
                                if algo == "mappo"
                                else None
                            ),
                            "suboptimality": res.get("suboptimality"),
                            "status_reason": res.get("status_reason", ""),
                        }
                    )

    summary_by_algo: List[dict] = []
    for algo in TARGET_ALGORITHMS:
        algo_rows = [r for r in rows if r["algorithm"] == algo]
        total = max(1, len(algo_rows))
        solved_rows = [r for r in algo_rows if r["solved"]]
        valid_rows = [r for r in algo_rows if r["valid"]]
        costs = [r["sum_of_costs"] for r in solved_rows if r["sum_of_costs"] is not None]
        makespans = [r["makespan"] for r in solved_rows if r["makespan"] is not None]
        ratios = [r["suboptimality"] for r in solved_rows if r["suboptimality"] is not None]
        runtimes = [r["runtime"] for r in algo_rows]

        summary_by_algo.append(
            {
                "algorithm": algo,
                "instances": len(algo_rows),
                "solved_count": len(solved_rows),
                "success_rate": round(len(solved_rows) / total, 4),
                "validity_rate": round(len(valid_rows) / total, 4),
                "mean_cost": round(sum(costs) / len(costs), 2) if costs else None,
                "mean_makespan": (
                    round(sum(makespans) / len(makespans), 2) if makespans else None
                ),
                "mean_suboptimality": (
                    round(sum(ratios) / len(ratios), 3) if ratios else None
                ),
                "mean_runtime": round(sum(runtimes) / len(runtimes), 6) if runtimes else 0.0,
                "timing_kind": (
                    "Rollout inference time (excludes offline training)"
                    if algo == "mappo"
                    else "Online search planning time"
                ),
            }
        )

    return {
        "scenario": scenario_list[0],
        "scenarios": scenario_list,
        "agent_counts": [int(c) for c in agent_counts],
        "seeds": [int(s) for s in seeds],
        "rows": rows,
        "summary": summary_by_algo,
    }


def batch_rows_to_csv(rows: Sequence[dict]) -> str:
    """Serialize batch experiment rows to CSV text."""
    fieldnames = [
        "scenario",
        "n_agents",
        "seed",
        "scenario_seed",
        "mappo_eval_seed",
        "mappo_mode",
        "algorithm",
        "solved",
        "valid",
        "sum_of_costs",
        "makespan",
        "expansions",
        "runtime",
        "training_time",
        "suboptimality",
        "status_reason",
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k, "") for k in fieldnames})
    return buffer.getvalue()
