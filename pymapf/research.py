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


def _run_mappo(
    problem: MAPFProblem,
    options: dict,
) -> dict:
    """Execute MAPPO on ``problem`` and return a unified outcome dictionary."""
    try:
        from .rl import MAPFEnv, make_trainer, rollout
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
            "validation": validate_solution(problem, None).as_dict(),
        }

    action_mode = str(options.get("mappo_mode", "sampled")).lower()
    deterministic = action_mode == "greedy"
    rollout_seed = int(options.get("seed", 0))
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
        seed=rollout_seed,
    )

    trainer = make_trainer(
        "mappo",
        env,
        backend="numpy",
        n_envs=4 if train_on_instance else 1,
        rollout_steps=64,
        hidden=hidden,
        lr=3e-4,
        seed=rollout_seed,
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
    # When in sampled mode, if the first rollout truncates, we allow up to
    # `rollout_attempts` seeded rollouts inside the evaluation budget and keep
    # the best valid rollout (or the final trajectory for visualization).
    rollout_attempts = 1 if deterministic else int(options.get("rollout_attempts", 5))
    infer_started = time.perf_counter()
    best_solution: Optional[Solution] = None
    best_summary: Optional[dict] = None

    for attempt in range(max(1, rollout_attempts)):
        sol, summary = rollout(
            env,
            trainer,
            deterministic=deterministic,
            seed=rollout_seed + attempt * 997,
        )
        if best_solution is None:
            best_solution, best_summary = sol, summary
        if summary.get("solved") and sol.is_valid(problem):
            if (
                best_summary is None
                or not best_summary.get("solved")
                or sol.sum_of_costs < best_solution.sum_of_costs
            ):
                best_solution, best_summary = sol, summary
            break

    inference_time = time.perf_counter() - infer_started
    assert best_solution is not None and best_summary is not None
    best_solution.algorithm = "mappo"
    best_solution.runtime = inference_time

    report = validate_solution(problem, best_solution)
    solved = bool(best_summary["solved"]) and report.valid

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
        "valid": report.valid,
        "sum_of_costs": best_solution.sum_of_costs if solved else None,
        "makespan": best_solution.makespan,
        "expansions": None,
        "runtime": inference_time,
        "inference_time": inference_time,
        "training_time": training_time,
        "training_steps": training_steps,
        "training_mode": training_mode_label,
        "mappo_mode": action_mode,
        "collisions_refused": int(best_summary.get("collisions", 0)),
        "agents_at_goal": int(best_summary.get("agents_at_goal", 0)),
        "steps_taken": int(best_summary.get("steps", 0)),
        "max_steps": int(env.max_steps),
        "status_reason": reason,
        "paths": {
            name: [list(cell) for cell in path]
            for name, path in best_solution.paths.items()
        },
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

    return {
        "algorithm": algo,
        "available": True,
        "solved": solved,
        "valid": report.valid,
        "sum_of_costs": solution.sum_of_costs if solved else None,
        "makespan": solution.makespan,
        "expansions": solution.expansions,
        "conflicts_detected": conflicts_detected,
        "runtime": elapsed,
        "planning_time": elapsed,
        "status_reason": status_reason,
        "paths": {
            name: [list(cell) for cell in path]
            for name, path in solution.paths.items()
        },
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
        (res.get("makespan") or 0 for res in outcomes.values()),
        default=0,
    )
    return {
        "outcomes": outcomes,
        "shared_horizon": shared_horizon,
        "n_agents": len(problem.agents),
        "grid_shape": [problem.grid.height, problem.grid.width],
    }


def run_batch_experiment(
    scenario_name: str = "random_obstacles",
    agent_counts: Sequence[int] = (2, 4, 6),
    seeds: Sequence[int] = (0, 1, 2),
    options: Optional[dict] = None,
) -> dict:
    """Run a reproducible batch experiment across scenarios, agent counts, and seeds."""
    if scenario_name not in PLANAR_SCENARIOS:
        raise ValueError(
            f"Scenario {scenario_name!r} is not a 2D planar scenario. "
            f"Choose from {PLANAR_SCENARIOS}."
        )
    opts = dict(options or {})
    rows: List[dict] = []

    for n_agents in agent_counts:
        for seed in seeds:
            scenario = build_scenario(
                scenario_name, n_agents=int(n_agents), seed=int(seed)
            )
            problem = scenario.to_problem()
            comp = run_comparison(problem, {**opts, "seed": int(seed)})
            for algo in TARGET_ALGORITHMS:
                res = comp["outcomes"][algo]
                rows.append(
                    {
                        "scenario": scenario_name,
                        "n_agents": int(n_agents),
                        "seed": int(seed),
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
        "scenario": scenario_name,
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
