from typing import Tuple, Dict
import gym
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)

from .buffers import RolloutBuffer
from .base_policy import Policy, ActionDict
from .trpo import trpo_step

from torch.utils.tensorboard import SummaryWriter

from collections import namedtuple
import scipy
from utils import normal_log_density, set_flat_params_to, \
    get_flat_grad_from, get_flat_params_from


def lr_schedule_fn_1(progress: float) -> float:
    if progress > 0.1:
        return 1e-3
    return 1e-3 * progress / 0.1


def lr_schedule_fn_2(progress: float) -> float:
    if progress > 0.1:
        return 5e-4
    return 5e-4 * progress / 0.1


device = get_device("cpu")


Transition = namedtuple(
    'Transition',
    (
        'state',
        'action',
        'mask',
        'next_state',
        'reward'
    )
)

class Memory(object):
    def __init__(self):
        self.memory = []

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition(*args))

    def sample(self):
        return Transition(*zip(*self.memory))

    def __len__(self):
        return len(self.memory)

class Actor(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Actor, self).__init__()
        self.affine1 = nn.Linear(num_inputs, 64)
        self.affine2 = nn.Linear(64, 64)

        self.action_mean = nn.Linear(64, num_outputs)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)

        self.action_log_std = nn.Parameter(torch.zeros(1, num_outputs))

        self.saved_actions = []
        self.rewards = []
        self.final_value = 0

    def forward(self, x):
        x = torch.tanh(self.affine1(x))
        x = torch.tanh(self.affine2(x))

        action_mean = self.action_mean(x)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std
    
class InterveneActor(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(InterveneActor, self).__init__()
        self.affine1 = nn.Linear(num_inputs, 64)
        self.affine2 = nn.Linear(64, 64)

        self.action_mean = nn.Linear(64, num_outputs)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)
        
        self.softmax = nn.Softmax()

        # self.action_log_std = nn.Parameter(torch.zeros(1, num_outputs))

        self.saved_actions = []
        self.rewards = []
        self.final_value = 0

    def forward(self, x):
        x = torch.tanh(self.affine1(x))
        x = torch.tanh(self.affine2(x))

        action = self.softmax(self.action_mean(x))

        return action


class Critic(nn.Module):
    def __init__(self, num_inputs):
        super(Critic, self).__init__()
        self.affine1 = nn.Linear(num_inputs, 64)
        self.affine2 = nn.Linear(64, 64)
        self.value_head = nn.Linear(64, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, x):
        x = torch.tanh(self.affine1(x))
        x = torch.tanh(self.affine2(x))

        state_values = self.value_head(x)
        return state_values

class ThreeTRPOPolicy(Policy):
    def __init__(
            self,
            env: gym.Env,
            network:list,
            batch_size: int,
            fixed_cost: float,
            writer: SummaryWriter,
            gamma: float = 1.0,
            tau: float = 0.97,
            l2_reg: float = 1e-3,
            max_kl: float = 1e-2,
            damping: float = 1e-1
        ):
        self.gamma = gamma
        self.tau = tau
        self.l2_reg = l2_reg
        self.max_kl = max_kl
        self.damping = damping
        self.writer = writer
        self.device = get_device("cuda")
        self.fixed_cost = fixed_cost
        
        num_inputs = env.observation_space.shape[0]
        num_actions = env.action_space.shape[0]
        
        self.standard_actor = Actor(num_inputs, num_actions)
        self.safe_actor = Actor(num_inputs, num_actions)
        self.intervene_actor = InterveneActor(num_inputs, 2)

        self.standard_critic = Critic(num_inputs)
        self.safe_critic = Critic(num_inputs)
        self.intervene_critic = Critic(num_inputs)
        
        self.standard_memory = Memory()
        self.safe_memory = Memory()
        self.intervene_memory = Memory()
        
    def start_episode(self):
        self.standard_memory = Memory()
        self.safe_memory = Memory()
        self.intervene_memory = Memory()

    @torch.no_grad()
    def get_action(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, ActionDict]:
        obs = obs.astype(float)
        standard_actions = self.get_action_for_policy(self.standard_actor, obs)
        safe_actions = self.get_action_for_policy(self.safe_actor, obs)
        intervene_actions = self.get_intervene_action_for_policy(self.intervene_actor, obs)
        intervene = bool(intervene_actions.squeeze().cpu().numpy())
        actions = safe_actions if intervene else standard_actions
        action_dict = {
            'intervene': intervene,
            'action_standard': standard_actions,
            'action_safe': safe_actions,
            'action_intervene': intervene_actions
        }
        return actions, action_dict

    def get_action_for_policy(self, policy_net: Actor, obs: np.ndarray) -> np.ndarray:
        state = torch.from_numpy(obs).unsqueeze(0).type(torch.FloatTensor)
        action_mean, _, action_std = policy_net(Variable(state))
        action = torch.normal(action_mean, action_std)
        return action
    
    def get_intervene_action_for_policy(self, policy_net: Actor, obs: np.ndarray) -> np.ndarray:
        state = torch.from_numpy(obs).unsqueeze(0).type(torch.FloatTensor)
        prob_vec = policy_net(Variable(state))
        action = torch.multinomial(prob_vec, 1).squeeze()
        return action.unsqueeze(0)

    def store_transition(
        self,
        obs: np.ndarray,
        action:np.ndarray,
        next_obs: np.ndarray,
        reward: float,
        done: bool,
        last_done: bool,
        info: Dict,
        action_dict: ActionDict
    ):
        intervene = action_dict['intervene']
        intervene_action = action_dict['action_intervene']
        cost = info['cost'] + self.fixed_cost if intervene else info['cost']
        self.standard_memory.push(obs, action, not done, next_obs, reward)
        self.safe_memory.push(obs, action, not done, next_obs, -cost)
        self.intervene_memory.push(obs, intervene_action, not done, next_obs, -cost)

    def learn(self, obs: np.ndarray, done: bool, current_progress_remaining: float, episode: int):
        self.update_params(
            self.standard_memory,
            self.standard_actor,
            self.standard_critic,
            num=0
        )
        # self.update_params(
        #     self.safe_memory,
        #     self.safe_actor,
        #     self.safe_critic,
        #     num=1
        # )
        # self.update_params(
        #     self.intervene_memory,
        #     self.intervene_actor,
        #     self.intervene_critic,
        #     discrete_actions=True,
        #     num=2
        # )
    def update_params(self, memory, policy_net, value_net, discrete_actions=False, num=-1):
        batch = memory.sample()
        rewards = torch.Tensor(batch.reward)
        masks = torch.Tensor(batch.mask)
        actions = torch.Tensor(np.concatenate(batch.action, 0))
        states = torch.Tensor(batch.state)
        values = value_net(Variable(states))
    
        returns = torch.Tensor(actions.size(0),1)
        deltas = torch.Tensor(actions.size(0),1)
        advantages = torch.Tensor(actions.size(0),1)
    
        prev_return = 0
        prev_value = 0
        prev_advantage = 0
        for i in reversed(range(rewards.size(0))):
            returns[i] = rewards[i] + self.gamma * prev_return * masks[i]
            deltas[i] = rewards[i] + self.gamma * prev_value * masks[i] - values.data[i]
            advantages[i] = deltas[i] + self.gamma * self.tau * prev_advantage * masks[i]
    
            prev_return = returns[i, 0]
            prev_value = values.data[i, 0]
            prev_advantage = advantages[i, 0]

        targets = Variable(returns)
    
        # Original code uses the same LBFGS to optimize the value loss
        def get_value_loss(flat_params):
            print('num:', num)
            set_flat_params_to(value_net, torch.Tensor(flat_params))
            for param in value_net.parameters():
                if param.grad is not None:
                    param.grad.data.fill_(0)
    
            values_ = value_net(Variable(states))
    
            value_loss = (values_ - targets).pow(2).mean()
    
            # weight decay
            for param in value_net.parameters():
                value_loss += param.pow(2).sum() * self.l2_reg
            value_loss.backward()
            return (value_loss.data.double().numpy(), get_flat_grad_from(value_net).data.double().numpy())
    
        flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss, get_flat_params_from(value_net).double().numpy(), maxiter=25)
        set_flat_params_to(value_net, torch.Tensor(flat_params))
    
        advantages = (advantages - advantages.mean()) / advantages.std()
    
        # action_means, action_log_stds, action_stds = policy_net(Variable(states))
        
        if discrete_actions:
            action_means = policy_net(Variable(states))
            action_dist = CategoricalDistribution(2).proba_distribution(action_logits=action_means)
            print('actions:', actions.shape)
            print('action_dist.log_prob(actions):', action_dist.log_prob(actions).shape)
            fixed_log_prob = action_dist.log_prob(actions).data.clone()
        else:
            action_means, action_log_stds, action_stds = policy_net(Variable(states))
            fixed_log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds).data.clone()
        
        
        # fixed_log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds).data.clone()
    
        def get_loss(volatile=False):
            # if volatile:
            #     with torch.no_grad():
            #         action_means, action_log_stds, action_stds = policy_net(Variable(states))
            # else:
            #     action_means, action_log_stds, action_stds = policy_net(Variable(states))
                
                
            if discrete_actions:
                
                if volatile:
                    with torch.no_grad():
                        action_means = policy_net(Variable(states))
                else:
                    action_means = policy_net(Variable(states))
                
                action_dist = CategoricalDistribution(2).proba_distribution(action_logits=action_means)
                log_prob = action_dist.log_prob(Variable(actions))
            else:
                
                if volatile:
                    with torch.no_grad():
                        action_means, action_log_stds, action_stds = policy_net(Variable(states))
                else:
                    action_means, action_log_stds, action_stds = policy_net(Variable(states))
                    
                log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)
                
                    
            # log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)
            action_loss = -Variable(advantages) * torch.exp(log_prob - Variable(fixed_log_prob))
            return action_loss.mean()
    
    
        def get_kl():
            if discrete_actions:
                return
            else:
                mean1, log_std1, std1 = policy_net(Variable(states))
                mean0 = Variable(mean1.data)
                log_std0 = Variable(log_std1.data)
                std0 = Variable(std1.data)
                kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
                print('kl:', kl)
                return kl.sum(1, keepdim=True)
    
        trpo_step(policy_net, get_loss, get_kl, self.max_kl, self.damping)
