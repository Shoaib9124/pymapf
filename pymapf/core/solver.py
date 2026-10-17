"""Core framework types shared by every centralized MAPF algorithm.

This module defines the small, stable vocabulary that makes PyMAPF usable as a
framework: a :class:`MAPFProblem` (the input), a :class:`Solution` (the output),
:class:`Constraints` (used by constraint-based low-level search), conflict
detection utilities, an abstract :class:`MAPFSolver`, and a name-based solver
registry so new algorithms can be discovered and swapped without touching call
sites.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Type

from .grid import Cell, GridMap
from .trace import Observer


@dataclass(frozen=True)
class Agent:
    """A planning agent: a unique ``name`` with ``start`` and ``goal`` cells."""

    name: str
    start: Cell
    goal: Cell


@dataclass
class MAPFProblem:
    """A multi-agent path finding instance."""

    grid: GridMap
    agents: List[Agent]
    allow_diagonals: bool = False

    def __post_init__(self):
        names = [a.name for a in self.agents]
        if len(names) != len(set(names)):
            raise ValueError("agent names must be unique")
        for a in self.agents:
            if not self.grid.is_free(a.start):
                raise ValueError(
                    "agent %r start %s is blocked/out of bounds" % (a.name, a.start)
                )
            if not self.grid.is_free(a.goal):
                raise ValueError(
                    "agent %r goal %s is blocked/out of bounds" % (a.name, a.goal)
                )


class Constraints:
    """Vertex and edge constraints for a single agent's low-level search.

    * A *vertex* constraint ``(cell, t)`` forbids occupying ``cell`` at time
      ``t``.
    * An *edge* constraint ``(u, v, t)`` forbids traversing from ``u`` to ``v``
      and arriving at ``v`` at time ``t`` (used to prevent agents swapping).
    """

    __slots__ = ("vertex", "edge")

    def __init__(self):
        self.vertex = set()  # {(cell, t)}
        self.edge = set()  # {(u, v, t)}

    def copy(self) -> "Constraints":
        c = Constraints()
        c.vertex = set(self.vertex)
        c.edge = set(self.edge)
        return c

    def add_vertex(self, cell: Cell, t: int) -> None:
        self.vertex.add((cell, t))

    def add_edge(self, u: Cell, v: Cell, t: int) -> None:
        self.edge.add((u, v, t))

    def blocks_vertex(self, cell: Cell, t: int) -> bool:
        return (cell, t) in self.vertex

    def blocks_edge(self, u: Cell, v: Cell, t: int) -> bool:
        return (u, v, t) in self.edge

    def last_vertex_time(self, cell: Cell) -> int:
        """Latest time a vertex constraint applies to ``cell`` (``-1`` if none).

        The low-level search uses this to know it may only "settle" on the goal
        once no further constraint can push it off.
        """
        times = [t for (c, t) in self.vertex if c == cell]
        return max(times) if times else -1


# A conflict between two agents in a candidate solution.
@dataclass(frozen=True)
class Conflict:
    kind: str  # "vertex" or "edge"
    a: str
    b: str
    t: int
    cell_a: Cell
    cell_b: Cell  # equals cell_a for vertex conflicts


@dataclass
class Solution:
    """Result of a solve: a path per agent plus cost metrics.

    ``paths[name]`` is a list of cells where ``paths[name][t]`` is the agent's
    location at timestep ``t`` (index 0 is the start).
    """

    paths: Dict[str, List[Cell]]
    algorithm: str = ""
    expansions: int = 0
    runtime: float = 0.0  # wall-clock seconds spent in ``solve``

    @property
    def makespan(self) -> int:
        """Time at which the last agent reaches its goal."""
        return max((len(p) - 1 for p in self.paths.values()), default=0)

    @property
    def sum_of_costs(self) -> int:
        """Sum over agents of the time each spends before settling at its goal."""
        return sum(len(p) - 1 for p in self.paths.values())

    def first_conflict(self) -> Optional[Conflict]:
        return find_first_conflict(self.paths)

    def is_valid(self, problem: Optional[MAPFProblem] = None) -> bool:
        if problem is not None:
            return validate_solution(problem, self).valid
        return self.first_conflict() is None

    def validate(self, problem: MAPFProblem) -> "ValidationReport":
        """Validate this solution against ``problem``'s grid, starts, goals, and rules."""
        return validate_solution(problem, self)

    def position_at(self, agent: str, t: int) -> Cell:
        """Where ``agent`` is at time ``t`` (it parks on its goal afterwards)."""
        return _cell_at(self.paths[agent], t)

    def congestion(self) -> Dict[Cell, int]:
        """How many agent-timesteps each cell absorbs -- the traffic hot spots."""
        counts: Dict[Cell, int] = {}
        for path in self.paths.values():
            for cell in path:
                counts[cell] = counts.get(cell, 0) + 1
        return counts

    def as_dict(self) -> Dict:
        """JSON-serialisable view (paths as lists), used by the web playground."""
        return {
            "algorithm": self.algorithm,
            "expansions": self.expansions,
            "runtime": self.runtime,
            "makespan": self.makespan,
            "sum_of_costs": self.sum_of_costs,
            "paths": {
                name: [list(cell) for cell in path] for name, path in self.paths.items()
            },
        }


@dataclass(frozen=True)
class ValidationReport:
    """Detailed audit of a candidate :class:`Solution` against a :class:`MAPFProblem`."""

    valid: bool
    in_bounds_and_obstacle_free: bool
    legal_moves: bool
    vertex_conflict_free: bool
    edge_swap_free: bool
    all_reached_goal: bool
    goal_occupancy_valid: bool
    distinct_starts: bool
    distinct_goals: bool
    violations: List[str]
    first_conflict: Optional[Conflict] = None

    def as_dict(self) -> Dict:
        return {
            "valid": self.valid,
            "in_bounds_and_obstacle_free": self.in_bounds_and_obstacle_free,
            "legal_moves": self.legal_moves,
            "vertex_conflict_free": self.vertex_conflict_free,
            "edge_swap_free": self.edge_swap_free,
            "all_reached_goal": self.all_reached_goal,
            "goal_occupancy_valid": self.goal_occupancy_valid,
            "distinct_starts": self.distinct_starts,
            "distinct_goals": self.distinct_goals,
            "violations": list(self.violations),
            "first_conflict": (
                {
                    "kind": self.first_conflict.kind,
                    "a": self.first_conflict.a,
                    "b": self.first_conflict.b,
                    "t": self.first_conflict.t,
                    "cell_a": list(self.first_conflict.cell_a),
                    "cell_b": list(self.first_conflict.cell_b),
                }
                if self.first_conflict is not None
                else None
            ),
        }


def _cell_at(path: List[Cell], t: int) -> Cell:
    """Position of an agent at time ``t`` (it waits on its goal after arrival)."""
    return path[t] if t < len(path) else path[-1]


def validate_solution(
    problem: MAPFProblem, solution: Optional[Solution]
) -> ValidationReport:
    """Audit ``solution`` against ``problem`` for movement, obstacle, conflict, and goal rules.

    Checks:
    1. Distinct starts and distinct goals across all agents.
    2. Every trajectory cell is inside the grid and not an obstacle.
    3. Every step ``t -> t+1`` is either a wait or a legal graph/grid neighbour step,
       and ``path[0] == agent.start``.
    4. No two agents occupy the same vertex at the same timestep ``t``.
    5. No two agents swap vertices along the same edge between ``t`` and ``t+1``.
    6. Every agent reaches its assigned goal at the end of its path.
    7. Stay-at-goal occupancy semantics hold (agents remain on their goals after arrival
       without colliding with still-moving agents).
    """
    starts = [a.start for a in problem.agents]
    goals = [a.goal for a in problem.agents]
    distinct_starts = len(set(starts)) == len(starts)
    distinct_goals = len(set(goals)) == len(goals)
    violations: List[str] = []

    if not distinct_starts:
        violations.append("Multiple agents share the same start cell.")
    if not distinct_goals:
        violations.append("Multiple agents share the same goal cell.")

    if isinstance(solution, dict):
        solution = Solution(paths=solution, algorithm="custom")

    if solution is None or not solution.paths:
        violations.append("No solution paths returned.")
        return ValidationReport(
            valid=False,
            in_bounds_and_obstacle_free=False,
            legal_moves=False,
            vertex_conflict_free=False,
            edge_swap_free=False,
            all_reached_goal=False,
            goal_occupancy_valid=False,
            distinct_starts=distinct_starts,
            distinct_goals=distinct_goals,
            violations=violations,
            first_conflict=None,
        )

    in_bounds_and_obstacle_free = True
    legal_moves = True
    vertex_conflict_free = True
    edge_swap_free = True
    all_reached_goal = True
    goal_occupancy_valid = distinct_goals

    grid = problem.grid
    has_in_bounds = hasattr(grid, "in_bounds")

    for agent in problem.agents:
        path = solution.paths.get(agent.name)
        if not path:
            all_reached_goal = False
            legal_moves = False
            goal_occupancy_valid = False
            violations.append("Agent %s has no path in solution." % agent.name)
            continue

        if tuple(path[0]) != tuple(agent.start):
            legal_moves = False
            violations.append(
                "Agent %s starts at %s instead of assigned start %s."
                % (agent.name, tuple(path[0]), tuple(agent.start))
            )

        for t, raw_cell in enumerate(path):
            cell = tuple(raw_cell)
            if has_in_bounds and not grid.in_bounds(cell):
                in_bounds_and_obstacle_free = False
                violations.append(
                    "Agent %s moves outside the grid to %s at t=%d."
                    % (agent.name, cell, t)
                )
            elif not grid.is_free(cell):
                in_bounds_and_obstacle_free = False
                violations.append(
                    "Agent %s moves through obstacle at %s at t=%d."
                    % (agent.name, cell, t)
                )

        for t in range(len(path) - 1):
            curr = tuple(path[t])
            nxt = tuple(path[t + 1])
            if nxt == curr:
                continue
            valid_neighbors = (
                grid.neighbors(curr, problem.allow_diagonals)
                if (not has_in_bounds or grid.in_bounds(curr)) and grid.is_free(curr)
                else []
            )
            if nxt not in valid_neighbors:
                legal_moves = False
                violations.append(
                    "Agent %s makes illegal move %s -> %s between t=%d and t=%d."
                    % (agent.name, curr, nxt, t, t + 1)
                )

        if tuple(path[-1]) != tuple(agent.goal):
            all_reached_goal = False
            goal_occupancy_valid = False
            violations.append(
                "Agent %s ends at %s instead of assigned goal %s."
                % (agent.name, tuple(path[-1]), tuple(agent.goal))
            )

        if getattr(solution, "algorithm", "") not in (
            "prioritized",
            "cbs",
            "wcbs",
            "space_time_astar",
        ):
            for t in range(len(path) - 1):
                if tuple(path[t]) == tuple(agent.goal) and tuple(
                    path[t + 1]
                ) != tuple(agent.goal):
                    goal_occupancy_valid = False
                    violations.append(
                        "Agent %s moved away from assigned goal %s between t=%d and t=%d."
                        % (agent.name, tuple(agent.goal), t, t + 1)
                    )

    names = [a.name for a in problem.agents if a.name in solution.paths]
    normalized_paths: Dict[str, List[Cell]] = {
        name: [tuple(c) for c in solution.paths[name]]
        for name in names
        if solution.paths[name]
    }
    active_names = list(normalized_paths)
    horizon = max((len(p) for p in normalized_paths.values()), default=0)
    first_conflict: Optional[Conflict] = None

    for t in range(horizon):
        for i in range(len(active_names)):
            for j in range(i + 1, len(active_names)):
                a, b = active_names[i], active_names[j]
                pa, pb = normalized_paths[a], normalized_paths[b]
                a_now, b_now = _cell_at(pa, t), _cell_at(pb, t)
                if a_now == b_now:
                    vertex_conflict_free = False
                    if t >= len(pa) - 1 or t >= len(pb) - 1:
                        goal_occupancy_valid = False
                    if first_conflict is None:
                        first_conflict = Conflict("vertex", a, b, t, a_now, b_now)
                    violations.append(
                        "Vertex conflict between %s and %s at cell %s (t=%d)."
                        % (a, b, a_now, t)
                    )
                a_next, b_next = _cell_at(pa, t + 1), _cell_at(pb, t + 1)
                if a_now == b_next and b_now == a_next and a_now != b_now:
                    edge_swap_free = False
                    if first_conflict is None:
                        first_conflict = Conflict("edge", a, b, t + 1, a_next, b_next)
                    violations.append(
                        "Edge-swap conflict between %s and %s across %s <-> %s (t=%d)."
                        % (a, b, a_now, b_now, t + 1)
                    )

    valid = (
        distinct_starts
        and distinct_goals
        and in_bounds_and_obstacle_free
        and legal_moves
        and vertex_conflict_free
        and edge_swap_free
        and all_reached_goal
        and goal_occupancy_valid
    )
    return ValidationReport(
        valid=valid,
        in_bounds_and_obstacle_free=in_bounds_and_obstacle_free,
        legal_moves=legal_moves,
        vertex_conflict_free=vertex_conflict_free,
        edge_swap_free=edge_swap_free,
        all_reached_goal=all_reached_goal,
        goal_occupancy_valid=goal_occupancy_valid,
        distinct_starts=distinct_starts,
        distinct_goals=distinct_goals,
        violations=violations,
        first_conflict=first_conflict,
    )


def find_first_conflict(paths: Dict[str, List[Cell]]) -> Optional[Conflict]:
    """Return the earliest vertex or edge conflict between any pair of agents.

    Paths are implicitly padded: once an agent reaches its goal it is assumed to
    stay there, so a shorter path still occupies its goal at later timesteps.
    """
    names = list(paths)
    horizon = max((len(p) for p in paths.values()), default=0)
    for t in range(horizon):
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                pa, pb = paths[a], paths[b]
                a_now, b_now = _cell_at(pa, t), _cell_at(pb, t)
                # Vertex conflict: both agents occupy the same cell at t.
                if a_now == b_now:
                    return Conflict("vertex", a, b, t, a_now, b_now)
                # Edge conflict: agents swap cells between t-1 and t.
                if t > 0:
                    a_prev, b_prev = _cell_at(pa, t - 1), _cell_at(pb, t - 1)
                    if a_now == b_prev and b_now == a_prev and a_prev != b_prev:
                        return Conflict("edge", a, b, t, a_now, b_now)
    return None


def count_conflicts(paths: Dict[str, List[Cell]]) -> int:
    """Total number of pairwise conflicts in a joint plan.

    Where :func:`find_first_conflict` answers "is this plan valid?",
    this answers "how far from valid is it?" -- the quantity CBS uses to
    tie-break between nodes of equal cost.
    """
    names = list(paths)
    horizon = max((len(p) for p in paths.values()), default=0)
    total = 0
    for t in range(horizon):
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                pa, pb = paths[names[i]], paths[names[j]]
                a_now, b_now = _cell_at(pa, t), _cell_at(pb, t)
                if a_now == b_now:
                    total += 1
                if t > 0:
                    a_prev, b_prev = _cell_at(pa, t - 1), _cell_at(pb, t - 1)
                    if a_now == b_prev and b_now == a_prev and a_prev != b_prev:
                        total += 1
    return total


class MAPFSolver(ABC):
    """Abstract base class for centralized MAPF algorithms.

    Subclasses implement :meth:`solve`; registering them with
    :func:`register_solver` makes them retrievable by name via
    :func:`get_solver`.
    """

    name = "abstract"

    @abstractmethod
    def solve(
        self, problem: MAPFProblem, observer: Optional[Observer] = None
    ) -> Optional[Solution]:
        """Return a :class:`Solution` or ``None`` if the instance is unsolved.

        ``observer`` is an optional callable receiving
        :class:`~pymapf.core.trace.SearchEvent` objects as the search runs; it
        is what the live views in :mod:`pymapf.viz` subscribe to.
        """
        raise NotImplementedError


_REGISTRY: Dict[str, Type[MAPFSolver]] = {}


def register_solver(name: str):
    """Class decorator that registers a solver under ``name``."""

    def decorator(cls: Type[MAPFSolver]) -> Type[MAPFSolver]:
        key = name.lower()
        if key in _REGISTRY:
            raise ValueError("solver %r already registered" % name)
        cls.name = key
        _REGISTRY[key] = cls
        return cls

    return decorator


def available_solvers() -> List[str]:
    return sorted(_REGISTRY)


def get_solver(name: str, **kwargs) -> MAPFSolver:
    """Instantiate a registered solver by name, forwarding ``kwargs``."""
    key = name.lower()
    try:
        cls = _REGISTRY[key]
    except KeyError as error:
        raise ValueError(
            "Unknown solver %r. Available: %s" % (name, ", ".join(available_solvers()))
        ) from error
    return cls(**kwargs)
