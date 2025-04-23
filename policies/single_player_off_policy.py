from typing import Tuple, Dict

import numpy as np
import torch
import gym

from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.utils import get_device

from .algos import sac_train
from .base_policy import Policy, ActionDict

from torch.utils.tensorboard import SummaryWriter

class SinglePlayerOffPolicy(Policy):
    '''
        Standard single agent RL policy using SAC.

        To account for safety violations, 'cost' of safety violation is
        combined with the reward.    
    '''    
    def __init__(
        self, 
        env:gym.Env,
        network:list,
        buffer_size:int,
        pi_1_lr=float,
        gamma:float=0.99,
        #gradient_steps:int=20,
        gradient_steps: int = 10,

        #batch_size:int = 1000,
        batch_size: int = 1024,
        tau:float = 0.005,
        writer:SummaryWriter = None
    ):
        self.device = get_device("cuda")
        self.lr_schedule_fn = lambda progress: pi_1_lr
        self.gradient_steps = gradient_steps
        
        
        self.batch_size = batch_size
        self.tau = tau
        self.writer = writer
        self.gamma = gamma
        

        self.buffer = \
            ReplayBuffer(
                buffer_size=buffer_size,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=self.device
            )

        self.policy = \
            SACPolicy(
                observation_space=env.observation_space,
                action_space=env.action_space,
                net_arch=network,
                lr_schedule=self.lr_schedule_fn,
                activation_fn=torch.nn.ReLU,
            ).to(self.device)

        self.log_ent_coef = torch.log(torch.ones(1, device=self.device)).requires_grad_(True)
        self.ent_coef_optimizer = torch.optim.Adam(
            [self.log_ent_coef], lr=self.lr_schedule_fn(1))
        self.target_entropy = -np.prod(env.action_space.shape).astype(np.float32)

    @torch.no_grad()
    def get_action(self, 
                   observation:np.ndarray,
                   deterministic:bool=False) -> Tuple[np.ndarray, ActionDict]:
        '''
            Return action for current observation.

            Arguments:
                observation - np.ndarray | current environment observation
            
            Returns:
                action - np.ndarray | policy's action for observation
                action_dict - IGNORE, just there so code works nicely
        '''
        with torch.no_grad():
            observation = torch.as_tensor([observation]).to(self.device)
            action = self.policy.forward(
                        observation, deterministic=deterministic)
            action = action.cpu().numpy()[0, :]
            return action, {'intervene': False}

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
        '''
            Store interaction in buffer
        '''
        cost = 50. * info['cost']
        #cost = 5. * info['cost']
        #cost=info['cost']
        info = [info]
        for i in info:
            i['Timelimit.truncated'] = False
        self.buffer.add(obs, next_obs, action, reward - cost, done, info)

    def learn(
        self,
        obs: np.ndarray,
        done: bool,
        current_progress_remaining: float,
        episode:int = None
    ):
        sac_train(
            self.policy,
            self.buffer,
            self.log_ent_coef,
            self.ent_coef_optimizer,
            self.lr_schedule_fn,
            current_progress_remaining,
            gradient_steps=self.gradient_steps,
            target_entropy=self.target_entropy,
            batch_size=self.batch_size,
            gamma=self.gamma,
            tau=self.tau,
            writer=self.writer,
            episode=episode,
            policy_name='single_player'
        )
