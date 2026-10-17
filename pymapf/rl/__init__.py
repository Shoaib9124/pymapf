"""Multi-agent reinforcement learning for 2D grid MAPF.

Provides the PettingZoo-compatible :class:`MAPFEnv`, centralized-critic
:class:`MAPPO` (and :class:`IPPO`), local observation encoders, shaped/sparse
reward functions, actor-critic networks, and rollout evaluation.
"""

from __future__ import annotations

from .algorithms import (
    IPPO,
    MAPPO,
    PPOTrainer,
    RolloutBuffer,
    available_algorithms,
    compute_gae,
    make_trainer,
    register_algorithm,
)
from .env import (
    DIAGONAL_ACTIONS,
    ORTHOGONAL_ACTIONS,
    MAPFEnv,
)
from .evaluate import EvaluationResult, compare, evaluate, plan, rollout
from .networks import (
    MLP,
    ActorCritic,
    Adam,
    NumpyActorCritic,
    TorchActorCritic,
    available_backends,
    make_actor_critic,
)
from .observation import (
    GlobalGrid,
    LocalWindow,
    ObservationEncoder,
    available_observations,
    get_observation,
    register_observation,
)
from .reward import (
    RewardFunction,
    ShapedReward,
    SparseReward,
    available_rewards,
    get_reward,
    register_reward,
)
from .spaces import Box, Discrete, Space, to_gymnasium
from .wrappers import PettingZooParallel, SingleAgentGym, VectorMAPFEnv

__all__ = [
    # environment
    "MAPFEnv",
    "ORTHOGONAL_ACTIONS",
    "DIAGONAL_ACTIONS",
    # spaces
    "Space",
    "Discrete",
    "Box",
    "to_gymnasium",
    # observations
    "ObservationEncoder",
    "LocalWindow",
    "GlobalGrid",
    "register_observation",
    "get_observation",
    "available_observations",
    # rewards
    "RewardFunction",
    "SparseReward",
    "ShapedReward",
    "register_reward",
    "get_reward",
    "available_rewards",
    # wrappers
    "PettingZooParallel",
    "SingleAgentGym",
    "VectorMAPFEnv",
    # networks & trainers
    "ActorCritic",
    "NumpyActorCritic",
    "TorchActorCritic",
    "MLP",
    "Adam",
    "make_actor_critic",
    "available_backends",
    "PPOTrainer",
    "IPPO",
    "MAPPO",
    "RolloutBuffer",
    "compute_gae",
    "make_trainer",
    "available_algorithms",
    "register_algorithm",
    # evaluation
    "EvaluationResult",
    "rollout",
    "plan",
    "evaluate",
    "compare",
]
