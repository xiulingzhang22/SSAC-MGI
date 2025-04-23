import imp
from typing import Dict, Tuple, Union

import gym
import torch
import numpy as np

from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.utils import get_device

from .algos import sac_train, sac_train_joint_action
from .base_policy import Policy, ActionDict

import pdb

from torch.utils.tensorboard import SummaryWriter

class ThreeModelFreeOffPolicy_JointAction(Policy):
    '''
        DESTA with:
            - SAC for standard policy
            - SAC for safe policy
            - SAC for intervention policy
    '''
    def __init__(
        self, 
        environment:gym.Env,
        
        buffer_size:int,
        
        action_distance_threshold:float,
        intervention_cost:float,
        
        learning_rate_standard_policy:float,
        learning_rate_safe_policy:float,
        learning_rate_intervention_policy:float,
        gradient_steps:int=20,

        network:list = [256, 256],
        batch_size:int = 1000,
        gamma:float = 1.,
        tau:float = 0.005,
        writer:SummaryWriter = None
    ):
        self.device = get_device("cuda") if torch.cuda.is_available() \
                        else get_device('cpu')
        
        self.batch_size = batch_size
        
        self.action_distance_threshold = action_distance_threshold
        self.gamma = gamma
        self.intervention_cost = intervention_cost
        self.tau = tau

        self.gradient_steps = gradient_steps
        
        # Learning rate schedules                        
        self.standard_policy_learning_rate_schedule = \
            lambda progress: learning_rate_standard_policy
        self.safe_policy_learning_rate_schedule = \
            lambda progress: learning_rate_safe_policy
        self.intervene_policy_learning_rate_schedule = \
            lambda progress: learning_rate_intervention_policy

        # Buffers
        # pdb.set_trace()
        self.standard_policy_buffer = \
            ReplayBuffer(
                buffer_size=buffer_size, 
                observation_space=environment.observation_space,
                action_space=environment.action_space,
                device=self.device,
                handle_timeout_termination=False
        )

        self.safe_policy_buffer = \
            ReplayBuffer(
                buffer_size=buffer_size,
                observation_space=environment.observation_space,
                action_space=environment.action_space,
                device=self.device,
                handle_timeout_termination=False
        )

        intervene_action_space = gym.spaces.Box(0, 1, shape=(1,))
        self.intervene_policy_buffer = \
            ReplayBuffer(
                buffer_size=buffer_size,
                observation_space=environment.observation_space,
                action_space=intervene_action_space,
                device=self.device,
                handle_timeout_termination=False
        )

        # Policies
        self.standard_policy = \
            SACPolicy(
                observation_space=environment.observation_space,
                action_space=environment.action_space,
                lr_schedule=self.standard_policy_learning_rate_schedule,
                net_arch=network,
                activation_fn=torch.nn.ReLU,
                use_other_player_actions=True,
        ).to(self.device)

        self.safe_policy = \
            SACPolicy(
                observation_space=environment.observation_space,
                action_space=environment.action_space,
                lr_schedule=self.safe_policy_learning_rate_schedule,
                net_arch=network,
                activation_fn=torch.nn.ReLU,
                use_other_player_actions=True,
        ).to(self.device)

        self.intervene_policy = \
            SACPolicy(
                observation_space=environment.observation_space,
                action_space=intervene_action_space,
                lr_schedule=self.intervene_policy_learning_rate_schedule,
                net_arch=network,
                activation_fn=torch.nn.ReLU,
                # use_other_player_actions=False,
        ).to(self.device)

        # Entropy
        # np.prod()计算所有元素乘积
        self.standard_policy_target_entropy = \
            -np.prod(environment.action_space.shape).astype(np.float32)
        self.standard_policy_log_ent_coef = \
            torch.log(torch.ones(1, device=self.device)).requires_grad_(True)
        self.standard_policy_ent_coef_optimizer = torch.optim.Adam(
            [self.standard_policy_log_ent_coef], lr=self.standard_policy_learning_rate_schedule(1))

        self.safe_policy_target_entropy = \
            -np.prod(environment.action_space.shape).astype(np.float32)
        self.safe_policy_log_ent_coef = \
            torch.log(torch.ones(1, device=self.device)).requires_grad_(True)
        self.safe_policy_ent_coef_optimizer = torch.optim.Adam(
            [self.safe_policy_log_ent_coef], lr=self.safe_policy_learning_rate_schedule(1))

        self.intervene_policy_target_entropy = \
            -np.prod(intervene_action_space.shape).astype(np.float32)
        self.intervene_policy_log_ent_coef = \
            torch.log(torch.ones(1, device=self.device)).requires_grad_(True)
        self.intervene_policy_ent_coef_optimizer = torch.optim.Adam(
            [self.intervene_policy_log_ent_coef], lr=self.intervene_policy_learning_rate_schedule(1))
        
        self.writer = writer

    @torch.no_grad()
    def get_action(self, 
        observation:np.ndarray,
        deterministic:bool=False
    ) -> Tuple[np.ndarray, ActionDict]:
        '''
            Compute actions of standard, safe, and intervene for 
            current observation.

            Arguments:
                observation - current environment observation
            
            Returns:
                action - policy's action for observation
                action_dict - other useful info
        '''
        observation = torch.as_tensor([observation]).to(self.device)
        standard_action = \
            self.standard_policy.forward(observation, deterministic=deterministic)
        safe_action = \
            self.safe_policy.forward(observation, deterministic=deterministic)
        intervene_action = \
            self.intervene_policy.forward(observation, deterministic=deterministic)

        # NOTE: The implementation of SAC used here doesn't allow for
        # discrete action spaces. Hences to 'simulate' a binary option
        # for intervening or not intervening, we simply intervene if SAC's
        # picked action is > 0 else we don't intervene. 
        intervene = True if intervene_action >= 0 else False
        action = safe_action if intervene else standard_action
        action = action.cpu().numpy()[0, :]

        action_dict = \
            {
                'intervene': intervene,
                'intervene_action': intervene_action.cpu().numpy()[0, :],
                'safe_action':safe_action.cpu().numpy()[0,:]
            }
        
        return action, action_dict

    def store_transition(
            self,
            observation:np.ndarray,
            action:np.ndarray,
            next_observation:np.ndarray,
            reward:float,
            done:bool,
            last_done:bool,
            info:dict,
            action_dict: ActionDict):
        '''
            Store appropriate interaction data in each policy's respective
            buffer.
        '''
        self.standard_policy_buffer.add( \
            observation,
            next_observation,
            action,
            reward,
            done,
            info
        )

        # Safe  
        safe_action = action_dict['safe_action']
        action_difference = np.linalg.norm(action - safe_action)
        if action_difference <= self.action_distance_threshold:
            safe_buffer_cost = info['cost'] + self.intervention_cost
            self.safe_policy_buffer.add(\
                observation,
                next_observation,
                action,
                -safe_buffer_cost,
                done,
                info
            )
        
        # Intervene
        intervene = action_dict['intervene']
        intervene_action = action_dict['intervene_action']
        intervene_buffer_cost = \
            info['cost'] + self.intervention_cost if intervene else info['cost']
        self.intervene_policy_buffer.add( \
            observation, 
            next_observation, 
            intervene_action, 
            -intervene_buffer_cost, 
            done, 
            info
        )

    def learn(
        self,
        observation,
        done,
        progress_remaining:float,
        episode:int = None
    ):
        sac_train_joint_action(
            policy=self.standard_policy,
            replay_buffer=self.standard_policy_buffer, 
            log_ent_coef=self.standard_policy_log_ent_coef, 
            ent_coef_optimizer=self.standard_policy_ent_coef_optimizer,
            lr_schedule=self.safe_policy_learning_rate_schedule,
            progress=progress_remaining, 
            gradient_steps=self.gradient_steps,
            target_entropy=self.standard_policy_target_entropy,
            batch_size=self.batch_size,
            gamma=self.gamma,
            tau=self.tau,
            writer=self.writer,
            episode=episode,
            policy_name='standard',
            other_player_policy=self.safe_policy,
            other_player_buffer=self.safe_policy_buffer,
            intervene_policy=self.intervene_policy,
            intervene_buffer=self.intervene_policy_buffer
        )
        print(1)
        sac_train_joint_action(
            policy=self.safe_policy,
            replay_buffer=self.safe_policy_buffer,
            log_ent_coef=self.safe_policy_log_ent_coef,
            ent_coef_optimizer=self.safe_policy_ent_coef_optimizer,
            lr_schedule=self.safe_policy_learning_rate_schedule,
            progress=progress_remaining,
            gradient_steps=self.gradient_steps,
            target_entropy=self.safe_policy_target_entropy,
            batch_size=self.batch_size,
            gamma=self.gamma,
            tau=self.tau,
            writer=self.writer,
            episode=episode,
            policy_name='safe',
            # other_player_policy=self.standard_policy,
            # other_player_buffer=self.standard_policy_buffer,
            # intervene_policy=self.intervene_policy,
            # intervene_buffer=self.intervene_policy_buffer
        )
        print(2)
        sac_train(
            policy=self.intervene_policy,
            replay_buffer=self.intervene_policy_buffer,
            log_ent_coef=self.intervene_policy_log_ent_coef,
            ent_coef_optimizer=self.intervene_policy_ent_coef_optimizer,
            lr_schedule=self.intervene_policy_learning_rate_schedule,
            progress=progress_remaining,
            gradient_steps=self.gradient_steps,
            target_entropy=self.intervene_policy_target_entropy,
            batch_size=self.batch_size,
            gamma=self.gamma,
            tau=self.tau,
            writer=self.writer,
            episode=episode,
            policy_name='intervene'
        )
        print(3)
   
