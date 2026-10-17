#!/usr/bin/env python3
"""Entry point for the PyMAPF 2D Multi-Agent Pathfinding Research Platform.

Serves the interactive browser application from ``.docs/`` and exposes JSON API
endpoints backed directly by ``pymapf`` (Prioritized Space-Time A*, Conflict-Based
Search, Multi-Agent PPO, formal solution validation, and batch experiments).

Usage:
    python main.py                  # starts server on http://0.0.0.0:3000
    python main.py --port 8000      # starts server on custom port
    python main.py --cli-demo       # runs a headless CLI comparison & exits
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import traceback
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pymapf  # noqa: E402
from pymapf.research import (  # noqa: E402
    PLANAR_SCENARIOS,
    TARGET_ALGORITHMS,
    batch_rows_to_csv,
    load_pretrained_mappo_state,
    problem_from_spec,
    run_batch_experiment,
    run_comparison,
    run_single_algorithm,
    validate_problem_spec,
)

DOCS_DIR = os.path.join(ROOT_DIR, ".docs")


def _extract_options(data: dict) -> dict:
    opts = dict(data.get("options") or {})
    for key in (
        "heuristic",
        "time_limit",
        "mappo_mode",
        "seed",
        "mappo_eval_seed",
        "mappo_seed",
        "mappo_seed_offset",
        "train_on_instance",
        "train_steps",
        "max_steps",
    ):
        if key in data and key not in opts:
            opts[key] = data[key]
    if "steps" in data and "train_steps" not in opts:
        opts["train_steps"] = data["steps"]
    return opts


class ResearchPlatformHandler(SimpleHTTPRequestHandler):
    """Serves static files from ``.docs/`` and handles ``/api/*`` requests."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DOCS_DIR, **kwargs)

    def log_message(self, format: str, *args) -> None:
        # Keep console output clean during normal operation.
        pass

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw) if raw else {}

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            pretrained = load_pretrained_mappo_state()
            self._send_json(
                200,
                {
                    "ok": True,
                    "version": pymapf.__version__,
                    "python": sys.version.split()[0],
                    "algorithms": list(TARGET_ALGORITHMS),
                    "scenarios": list(PLANAR_SCENARIOS),
                    "mappo_meta": pretrained.get("training_meta", {}),
                },
            )
            return
        super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            data = self._read_json()
            if parsed.path == "/api/validate":
                ok, errors = validate_problem_spec(data)
                self._send_json(200, {"ok": ok, "errors": errors})
                return

            if parsed.path == "/api/solve":
                problem = problem_from_spec(data)
                algorithm = data.get("algorithm", "cbs")
                options = _extract_options(data)
                outcome = run_single_algorithm(problem, algorithm, options)
                self._send_json(
                    200, {"ok": True, "outcome": outcome, "result": outcome}
                )
                return

            if parsed.path == "/api/compare":
                problem = problem_from_spec(data)
                options = _extract_options(data)
                comparison = run_comparison(problem, options)
                self._send_json(
                    200,
                    {
                        "ok": True,
                        "algorithms": comparison["outcomes"],
                        **comparison,
                    },
                )
                return

            if parsed.path == "/api/train_mappo":
                problem = problem_from_spec(data)
                options = _extract_options(data)
                options["train_on_instance"] = True
                options.setdefault("train_steps", 16384)
                outcome = run_single_algorithm(problem, "mappo", options)
                self._send_json(
                    200, {"ok": True, "outcome": outcome, "result": outcome}
                )
                return

            if parsed.path in ("/api/benchmark", "/api/batch"):
                scenarios = data.get("scenarios")
                scenario = data.get("scenario", "random_obstacles")
                agent_counts = data.get("agent_counts", [2, 4, 6])
                seeds = data.get("seeds", [0, 1, 2])
                options = _extract_options(data)
                report = run_batch_experiment(
                    scenario_name=scenario,
                    scenarios=scenarios,
                    agent_counts=agent_counts,
                    seeds=seeds,
                    options=options,
                )
                report["csv"] = batch_rows_to_csv(report["rows"])
                # Provide both dict-keyed and list summary representations
                if isinstance(report.get("summary"), list):
                    summary_dict = {}
                    for item in report["summary"]:
                        summary_dict[item["algorithm"]] = {
                            **item,
                            "solved": item.get("solved_count", 0),
                            "valid_rate": item.get("validity_rate", 0.0),
                            "mean_suboptimality_vs_cbs": item.get("mean_suboptimality"),
                            "timing_label": item.get("timing_kind", ""),
                        }
                    report["summary_list"] = report["summary"]
                    report["summary"] = summary_dict
                self._send_json(200, {"ok": True, **report})
                return

            self._send_json(
                404, {"ok": False, "error": f"Unknown endpoint {parsed.path}"}
            )
        except ValueError as exc:
            self._send_json(
                400, {"ok": False, "error": str(exc), "errors": [str(exc)]}
            )
        except Exception as exc:  # pragma: no cover
            self._send_json(
                500,
                {
                    "ok": False,
                    "error": str(exc),
                    "errors": [str(exc)],
                    "trace": traceback.format_exc(limit=3),
                },
            )


# Aliases for backward compatibility in tests
ResearchHTTPServer = ThreadingHTTPServer
ResearchRequestHandler = ResearchPlatformHandler


def run_cli_demo() -> int:
    """Run a quick CLI demonstration comparing Prioritized A*, CBS, and MAPPO."""
    scenario = pymapf.build_scenario("random_obstacles", n_agents=4, seed=0)
    problem = scenario.to_problem()
    result = run_comparison(problem, {"time_limit": 5.0, "mappo_mode": "sampled"})
    print(
        f"PyMAPF {pymapf.__version__} — 2D Research Comparison on "
        f"{scenario.name} ({problem.grid.height}x{problem.grid.width}, {len(problem.agents)} agents)"
    )
    print("-" * 78)
    print(
        f"{'Algorithm':<16} {'Solved':<8} {'Valid':<8} {'Cost':<8} {'Makespan':<10} {'Time (ms)':<12} {'Status'}"
    )
    print("-" * 78)
    for algo in TARGET_ALGORITHMS:
        out = result["outcomes"][algo]
        cost_str = str(out["sum_of_costs"]) if out["sum_of_costs"] is not None else "—"
        ms_str = str(out["makespan"]) if out["makespan"] is not None else "—"
        time_ms = f"{out['runtime'] * 1000:.1f}"
        print(
            f"{algo:<16} {str(out['solved']):<8} {str(out['valid']):<8} "
            f"{cost_str:<8} {ms_str:<10} {time_ms:<12} {out['status_reason']}"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        default=os.environ.get("HOST", "0.0.0.0"),
        help="Host interface to bind (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", "3000")),
        help="Port to listen on (default: 3000)",
    )
    parser.add_argument(
        "--cli-demo",
        "--cli",
        action="store_true",
        dest="cli_demo",
        help="Run a one-shot CLI comparison instead of starting the HTTP server",
    )
    args, _ = parser.parse_known_args()

    if args.cli_demo:
        return run_cli_demo()

    mimetypes.add_type("application/javascript", ".js")
    mimetypes.add_type("application/json", ".json")

    server = ThreadingHTTPServer((args.host, args.port), ResearchPlatformHandler)
    print(
        f"PyMAPF Research & Simulation Platform running at http://{args.host}:{args.port}"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
