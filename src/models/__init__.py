"""Neural network architectures for DRL agents."""

from .dqn_network import DuelingDQN
from .ppo_network import ActorCritic

__all__ = ['DuelingDQN', 'ActorCritic']