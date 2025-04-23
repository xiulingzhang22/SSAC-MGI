from typing import Tuple, Dict

import gym
import torch
import numpy as np

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import get_device

from .algos import ppo_train
from .buffers import RolloutBuffer
from .base_policy import Policy, ActionDict

class SinglePlayerPolicy(Policy):
    '''
        Standard single agent RL policy using PPO.

        To account for safety violations, 'cost' of safety violation is
        combined with the reward.    
    '''
    def __init__(self, 
        env:gym.Env, 
        network:list,

        batch_size:int, 
        pi_1_lr:float, 
        gamma:float=0.99):
        self.device = get_device("cuda")
        self.lr_schedule_fn = lambda progress: pi_1_lr

        self.buffer = \
            RolloutBuffer(buffer_size=batch_size, 
                          observation_space=env.observation_space,
                          action_space=env.action_space, 
                          gamma=gamma,
                          device=self.device

            )

        self.policy = ActorCriticPolicy(
                          observation_space=env.observation_space,
                          action_space=env.action_space,
                          net_arch=network,

                          lr_schedule=self.lr_schedule_fn,
                        #   lr_schedule=lr_schedule_fn_1,
                          #net_arch=[{"pi": [64, 64], "vf": [64, 64]}],
                          activation_fn=torch.nn.ReLU,
            ).to(self.device)

    def get_action(self, 
                   obs: np.ndarray, 
                   deterministic: bool = False) -> Tuple[np.ndarray, ActionDict]:
        with torch.no_grad():
            obs_tensor = torch.as_tensor([obs]).to(self.device)
            actions_1, values_1, log_prob_1 = self.policy.forward(
                obs_tensor, deterministic=deterministic)
            actions = actions_1.cpu().numpy()[0, :]
            action_dict = {'value': values_1, 'log_prob': log_prob_1, 'intervene': False}
            return actions, action_dict

    def store_transition(
        self,
        observation:np.ndarray,
        action:np.ndarray,
        next_observation: np.ndarray,
        reward:float,
        done:bool,
        last_done:bool,
        info:dict,
        action_dict:ActionDict):
        '''
            Store interaction in buffer
        '''
        #cost = 0. * info['cost'] original
        cost = 50. * info['cost']
        self.buffer.add(
            observation, action, reward - cost, last_done, action_dict['value'], action_dict['log_prob'], True)

    def learn(self, 
        observation:np.ndarray, 
        done:bool, 
        progress_remaining:float):
        '''
            Update parameters of policy using PPO the reset rollout buffer

            Arguments:
                observation - last observation in current policy rollout
                done - whether environment has terminated (used to learn
                    value function)
                progress_remaining - used in more complex learning rate
                    scheduler functions
        '''
        with torch.no_grad():
            observation = torch.as_tensor([observation]).to(self.device)
            _, value, _ = self.policy.forward(observation)
            self.buffer.compute_returns_and_advantage(last_values=value, dones=done)

        ppo_train(self.policy, self.buffer, progress_remaining, self.lr_schedule_fn)

        self.buffer.reset()