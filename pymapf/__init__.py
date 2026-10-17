"""PyMAPF: a 2D Multi-Agent Pathfinding research and simulation library.

The :mod:`pymapf.core` package defines the shared 2D problem and validation
framework (``GridMap``, ``MAPFProblem``, ``Solution``, ``ValidationReport``,
``validate_solution``, and the solver registry). Concrete algorithms live in
:mod:`pymapf.algorithms` (Prioritized Planning using Space-Time A* and
Conflict-Based Search) and :mod:`pymapf.rl` (Multi-Agent PPO).
"""

from .core import (
    Agent,
    Cell,
    Conflict,
    Constraints,
    GridMap,
    MAPFProblem,
    MAPFSolver,
    Observer,
    SearchEvent,
    SearchTrace,
    Solution,
    ValidationReport,
    available_solvers,
    find_first_conflict,
    get_solver,
    register_solver,
    validate_solution,
)

# Importing the algorithms package registers the built-in solvers by name.
from . import algorithms  # noqa: F401  (side-effect: populates the registry)
from . import scenarios  # noqa: F401
from .algorithms import (
    ConflictBasedSearch,
    PrioritizedPlanning,
    astar,
    dijkstra,
    space_time_astar,
)
from .scenarios import Scenario, available_scenarios, build_scenario

__version__ = "0.9.0"


def solve(
    problem: MAPFProblem,
    algorithm: str = "cbs",
    observer: Observer = None,
    **kwargs,
):
    """Solve ``problem`` with a registered algorithm (default: CBS).

    ``observer`` receives :class:`SearchEvent` objects while the search runs
    (see :class:`SearchTrace`); ``kwargs`` are forwarded to the solver
    constructor (e.g. ``heuristic``, ``time_limit``, ``max_expansions``).
    """
    return get_solver(algorithm, **kwargs).solve(problem, observer=observer)


__all__ = [
    "Agent",
    "Cell",
    "Conflict",
    "Constraints",
    "GridMap",
    "MAPFProblem",
    "MAPFSolver",
    "Observer",
    "Scenario",
    "SearchEvent",
    "SearchTrace",
    "Solution",
    "ValidationReport",
    "available_scenarios",
    "build_scenario",
    "ConflictBasedSearch",
    "PrioritizedPlanning",
    "astar",
    "dijkstra",
    "space_time_astar",
    "available_solvers",
    "find_first_conflict",
    "get_solver",
    "register_solver",
    "validate_solution",
    "solve",
    "__version__",
]
