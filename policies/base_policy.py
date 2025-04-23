from typing import Tuple, Dict
from abc import abstractmethod
import gym
import numpy as np


ActionDict = Dict[str, np.ndarray]


class Policy:
    """Abstract interface for policies"""
    @abstractmethod
    def __init__(self, env: gym.Env, batch_size: int, fixed_cost: float = 0., gamma: float = 1.0):
        pass

    @abstractmethod
    def get_action(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, ActionDict]:
        pass

    @abstractmethod
    def store_transition(
            self,
            obs: np.ndarray,
            action:np.ndarray,
            next_obs: np.ndarray,
            reward: float,
            done: bool,
            last_done: bool,
            info: Dict,
            action_dict: ActionDict):
        pass

    @abstractmethod
    def learn(self, obs: np.ndarray, done: bool, current_progress_remaining: float):
        pass
