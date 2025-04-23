from typing import Dict, Tuple, Union

import gym
import torch
import numpy as np

from stable_baselines3.sac.policies import SACPolicy, DoubleCriticSACPolicy
from stable_baselines3.common.buffers import ReplayBuffer, InterventionReplayBuffer
from stable_baselines3.common.utils import get_device

from .algos import sac_train, double_critic_sac_train, two_action_sac_train
from .base_policy import Policy, ActionDict

from torch.utils.tensorboard import SummaryWriter

class TwoModelFreeOffPolicy(Policy):
    '''
        DESTA with:
            - SAC for task policy
            - SAC for safe policy
            - SAC for intervention policy
    '''
    def __init__(
        self, 
        environment:gym.Env,
        
        buffer_size:int,
        
        action_distance_threshold:float,
        intervention_cost:float,
        
        learning_rate_task_policy:float,
        learning_rate_intervention_policy:float,
        #gradient_steps:int=20,
        gradient_steps:int=1,

        network:list = [256, 256],
        batch_size:int = 100,
        gamma:float = 1.,
        tau:float = 0.005,
        intervention_bias = 0.6,
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
        
        self.intervention_bias = intervention_bias
        
        # Learning rate schedules                        
        self.task_policy_learning_rate_schedule = \
            lambda progress: learning_rate_task_policy
        self.intervene_policy_learning_rate_schedule = \
            lambda progress: learning_rate_intervention_policy

        # Buffers
        self.task_policy_buffer = \
            InterventionReplayBuffer(
                buffer_size=buffer_size, 
                observation_space=environment.observation_space,
                action_space=environment.action_space,
                device=self.device,
                handle_timeout_termination=False
        )

        self.intervene_action_space = gym.spaces.Box(-1, 1, shape=(1,))
        self.intervene_policy_buffer = \
            ReplayBuffer(
                buffer_size=buffer_size,
                observation_space=environment.observation_space,
                action_space=self.intervene_action_space,
                device=self.device,
                handle_timeout_termination=False
        )

        # Policies
        self.task_policy = \
            DoubleCriticSACPolicy(
                observation_space=environment.observation_space,
                action_space=environment.action_space,
                lr_schedule=self.task_policy_learning_rate_schedule,
                net_arch=network,
                activation_fn=torch.nn.ReLU,
        ).to(self.device)

        self.intervene_policy = \
            SACPolicy(
                observation_space=environment.observation_space,
                action_space=self.intervene_action_space,
                lr_schedule=self.intervene_policy_learning_rate_schedule,
                net_arch=network,
                activation_fn=torch.nn.ReLU,
                use_other_player_actions=True,
                action_dim2=2
        ).to(self.device)

        # Entropy
        self.task_policy_target_entropy = \
            -np.prod(environment.action_space.shape).astype(np.float32)
        self.task_policy_log_ent_coef = \
            torch.log(torch.ones(1, device=self.device)).requires_grad_(True)
        self.task_policy_ent_coef_optimizer = torch.optim.Adam(
            [self.task_policy_log_ent_coef], lr=self.task_policy_learning_rate_schedule(1))

        self.intervene_policy_target_entropy = \
            -np.prod(self.intervene_action_space.shape).astype(np.float32)
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
            Compute actions of task, and intervene for 
            current observation.

            Arguments:
                observation - current environment observation
            
            Returns:
                action - policy's action for observation
                action_dict - other useful info
        '''
        observation = torch.as_tensor([observation]).to(self.device)
        intervene_action = \
            self.intervene_policy.forward(observation, deterministic=deterministic)


        transformed_intervene_action = self.transform_intervention_output(
            intervene_action
        )

        # NOTE: The implementation of SAC used here doesn't allow for
        # discrete action spaces. Hence to 'simulate' a binary option
        # for intervening or not intervening, we simply intervene if SAC's
        # picked action is > 0 else we don't intervene. 
        # intervene = True if intervene_action >= 0 else False
        intervene = True if transformed_intervene_action.item() < 0.5 else False

        task_action = \
            self.task_policy.forward(
                observation,
                deterministic=True if intervene else deterministic
            )
        
        action = task_action
        action = action.cpu().numpy()[0, :]

        action_dict = \
            {
                'intervene': intervene,
                'intervene_action': intervene_action.cpu().numpy()[0, :],
            }
        
        return action, action_dict
    
    # action output range[-1,1],expand to allowable range[action_space_low,action_space_high]-->[0,1]
    # Then f(x) = x^k | k ∈ (0, 2],
    def transform_intervention_output(self, action):
        mapped_action = (action + 1) / 2 # Map from [-1, 1] to [0, 1]
        biased_action = mapped_action ** self.intervention_bias
        return biased_action

    def store_transition(
            self,
            observation:np.ndarray,
            action:np.ndarray,
            next_observation:np.ndarray,
            reward:float,
            done:bool,
            last_done:bool,
            info:dict,
            action_dict: ActionDict,
            deterministic:bool = False
    ):
        '''
            Store appropriate interaction data in each policy's respective
            buffer.
        '''
        intervene = action_dict['intervene']
        safe_buffer_cost = info['cost'] + self.intervention_cost
        self.task_policy_buffer.add( \
            observation,
            next_observation,
            intervene,
            action,
            -safe_buffer_cost if intervene else reward,
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
        intervene,
        progress_remaining:float,
        episode:int = None
    ):
        double_critic_sac_train(
            policy=self.task_policy,
            replay_buffer=self.task_policy_buffer, 
            log_ent_coef=self.task_policy_log_ent_coef, 
            ent_coef_optimizer=self.task_policy_ent_coef_optimizer,
            lr_schedule=self.task_policy_learning_rate_schedule,
            progress=progress_remaining, 
            gradient_steps=self.gradient_steps,
            target_entropy=self.task_policy_target_entropy,
            batch_size=self.batch_size,
            gamma=self.gamma,
            tau=self.tau,
            writer=self.writer,
            episode=episode,
            policy_name='task',
            intervene_policy=self.intervene_policy,
            intervene=intervene,
        )

        two_action_sac_train(
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
            policy_name='intervene',
            task_policy=self.task_policy
        )
   