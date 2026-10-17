"""Tests for the 2D Multi-Agent Pathfinding Research & Simulation Platform."""

import json
import os
import threading
import unittest
import urllib.request

from main import ResearchHTTPServer, ResearchRequestHandler
from pymapf.core import Agent, GridMap, MAPFProblem, Solution, validate_solution
from pymapf.research import (
    problem_from_spec,
    run_batch_experiment,
    run_comparison,
    run_single_algorithm,
    validate_problem_spec,
)
from pymapf.scenarios import bottleneck, empty_room


class TestResearchPlatform(unittest.TestCase):
    def test_validate_solution_accepts_valid_cbs_plan(self):
        scenario = bottleneck(n_agents=2, seed=0)
        problem = scenario.to_problem()
        res = run_comparison(problem)
        cbs = res["outcomes"]["cbs"]
        self.assertTrue(cbs["solved"])
        self.assertTrue(cbs["valid"])
        val = cbs["validation"]
        self.assertTrue(val["in_bounds_and_obstacle_free"])
        self.assertTrue(val["legal_moves"])
        self.assertTrue(val["vertex_conflict_free"])
        self.assertTrue(val["edge_swap_free"])
        self.assertTrue(val["all_reached_goal"])
        self.assertTrue(val["goal_occupancy_valid"])
        self.assertEqual(val["violations"], [])

    def test_validate_solution_detects_obstacle_and_illegal_move(self):
        grid = GridMap(
            [
                [1, 1, 1, 1],
                [1, 0, 0, 1],
                [1, 0, 0, 1],
                [1, 1, 1, 1],
            ]
        )
        agents = [Agent("A", (1, 1), (2, 2))]
        problem = MAPFProblem(grid, agents, allow_diagonals=False)

        # Moves through obstacle (0, 1) and diagonal move (0, 1) -> (2, 2)
        bad_paths = {"A": [(1, 1), (0, 1), (2, 2)]}
        report = validate_solution(problem, bad_paths)
        self.assertFalse(report.valid)
        self.assertFalse(report.in_bounds_and_obstacle_free)
        self.assertFalse(report.legal_moves)
        self.assertGreaterEqual(len(report.violations), 2)

    def test_validate_solution_detects_vertex_edge_and_goal_occupancy_conflicts(self):
        grid = GridMap(
            [
                [1, 1, 1, 1, 1],
                [1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1],
            ]
        )
        agents = [
            Agent("A", (1, 1), (1, 3)),
            Agent("B", (1, 2), (1, 1)),
        ]
        problem = MAPFProblem(grid, agents, allow_diagonals=False)

        # Edge-swap conflict between (1, 1) and (1, 2) at t=0->1
        swap_paths = {
            "A": [(1, 1), (1, 2), (1, 3)],
            "B": [(1, 2), (1, 1)],
        }
        report_swap = validate_solution(problem, swap_paths)
        self.assertFalse(report_swap.valid)
        self.assertFalse(report_swap.edge_swap_free)

        # Vertex conflict at (1, 2) at t=1
        agents_v = [
            Agent("A", (1, 1), (1, 3)),
            Agent("B", (1, 3), (1, 1)),
        ]
        problem_v = MAPFProblem(grid, agents_v, allow_diagonals=False)
        vertex_paths = {
            "A": [(1, 1), (1, 2), (1, 3)],
            "B": [(1, 3), (1, 2), (1, 1)],
        }
        report_vertex = validate_solution(problem_v, vertex_paths)
        self.assertFalse(report_vertex.valid)
        self.assertFalse(report_vertex.vertex_conflict_free)

        # Goal occupancy violation: A arrives at (1, 2) at t=1 and stays; B passes through (1, 2) at t=2
        agents_occ = [
            Agent("A", (1, 1), (1, 2)),
            Agent("B", (2, 2), (1, 3)),
        ]
        prob_occ = MAPFProblem(grid, agents_occ, allow_diagonals=False)
        occ_paths = {
            "A": [(1, 1), (1, 2)],
            "B": [(2, 2), (2, 2), (1, 2), (1, 3)],
        }
        report_occ = validate_solution(
            prob_occ, Solution(paths=occ_paths, algorithm="test")
        )
        self.assertFalse(report_occ.valid)
        self.assertFalse(report_occ.vertex_conflict_free)
        self.assertFalse(report_occ.goal_occupancy_valid)

    def test_validate_problem_spec_rejects_invalid_placements(self):
        matrix = [
            [1, 1, 1, 1],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 1],
        ]
        ok, errors = validate_problem_spec(
            {
                "matrix": matrix,
                "agents": [
                    {"name": "A", "start": [0, 0], "goal": [1, 2]},
                    {"name": "B", "start": [2, 1], "goal": [1, 2]},
                ],
            }
        )
        self.assertFalse(ok)
        self.assertTrue(any("obstacle" in e for e in errors))
        self.assertTrue(any("share the same goal" in e for e in errors))

    def test_all_three_algorithms_and_instance_mappo_training(self):
        scenario = empty_room(height=8, width=8, n_agents=2, seed=0)
        problem = problem_from_spec(
            {
                "matrix": scenario.grid.as_matrix(),
                "agents": [
                    {"name": a.name, "start": a.start, "goal": a.goal}
                    for a in scenario.agents
                ],
            }
        )

        comp = run_comparison(problem, {"heuristic": "manhattan", "mappo_mode": "sampled"})
        for algo in ("prioritized", "cbs", "mappo"):
            self.assertIn(algo, comp["outcomes"])
            res = comp["outcomes"][algo]
            self.assertTrue(res["available"])
            self.assertIn("validation", res)
            self.assertTrue(res["solved"])
            self.assertTrue(res["valid"])

        # On-instance MAPPO training
        trained = run_single_algorithm(
            problem,
            "mappo",
            {"train_on_instance": True, "train_steps": 2048, "mappo_mode": "sampled"},
        )
        self.assertGreaterEqual(trained["training_steps"], 2048)
        self.assertGreater(trained["training_time"], 0.0)

    def test_batch_experiments_and_http_api(self):
        batch = run_batch_experiment(
            scenario_name="empty_room",
            agent_counts=[2, 3],
            seeds=[0],
            options={"mappo_mode": "sampled"},
        )
        self.assertEqual(len(batch["rows"]), 6)  # 2 agent counts * 1 seed * 3 algorithms
        self.assertEqual(len(batch["summary"]), 3)

        # Spin up ResearchHTTPServer on an ephemeral port and test /api/health, /api/compare, and /api/batch
        server = ResearchHTTPServer(("127.0.0.1", 0), ResearchRequestHandler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health") as resp:
                health = json.loads(resp.read().decode("utf-8"))
                self.assertTrue(health["ok"])
                self.assertIn("mappo", health["algorithms"])

            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/api/compare",
                data=json.dumps(
                    {
                        "matrix": [
                            [1, 1, 1, 1, 1],
                            [1, 0, 0, 0, 1],
                            [1, 0, 0, 0, 1],
                            [1, 1, 1, 1, 1],
                        ],
                        "agents": [
                            {"name": "A", "start": [1, 1], "goal": [1, 3]},
                            {"name": "B", "start": [2, 3], "goal": [2, 1]},
                        ],
                    }
                ).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req) as resp:
                comp = json.loads(resp.read().decode("utf-8"))
                self.assertTrue(comp["ok"])
                self.assertTrue(comp["algorithms"]["cbs"]["solved"])
                self.assertTrue(comp["algorithms"]["prioritized"]["solved"])
        finally:
            server.shutdown()
            server.server_close()

    def test_published_docs_json_and_asset_links(self):
        docs_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".docs"
        )
        # Verify all .json files under .docs/ contain no NaN/Infinity tokens
        for root, _, files in os.walk(docs_dir):
            for name in files:
                if name.endswith(".json"):
                    path = os.path.join(root, name)
                    with open(path, encoding="utf-8") as handle:
                        json.load(
                            handle,
                            parse_constant=lambda c: (_ for _ in ()).throw(
                                ValueError(f"Invalid JSON constant {c} in {path}")
                            ),
                        )


if __name__ == "__main__":
    unittest.main()
