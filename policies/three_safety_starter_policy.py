from typing import Tuple, Dict
import gym
import numpy as np
import torch

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import get_device

from .buffers import RolloutBuffer
from .base_policy import Policy, ActionDict

from torch.utils.tensorboard import SummaryWriter

from .safety_starter_agents.safe_rl.utils.mpi_tools import num_procs

from .safety_starter_agent import SafetyStarterAgent


def lr_schedule_fn_1(progress: float) -> float:
    if progress > 0.1:
        return 1e-3
    return 1e-3 * progress / 0.1


def lr_schedule_fn_2(progress: float) -> float:
    if progress > 0.1:
        return 5e-4
    return 5e-4 * progress / 0.1


device = get_device("cpu")

class ThreeSafetyStarterPolicy(Policy):
    """
    Algorithm implemented using three PPO agents (player1, player2 and intervention agent).

    Player 1 only learns from its own actions, and same for player 2, using interventions in
    the rollout buffer.

    The intervention policy can learn at every timestep.
    """
    def __init__(
            self,
            env: gym.Env,
            agent_fn,
            network:list,
            batch_size: int,
            fixed_cost: float,
            writer: SummaryWriter,
            algorithm,
            steps_per_epoch=4000,
            max_ep_len=1000,
            save_freq=1,
            min_episodes=2000,
            gamma: float = 1.0,
        ):
        self.max_ep_len = max_ep_len
        self.local_steps_per_epoch = int(steps_per_epoch / num_procs())
        self.save_freq = save_freq
        self.min_episodes = min_episodes
        self.algorithm = algorithm
        
        self.writer = writer
        self.device = get_device("cuda")
        self.fixed_cost = fixed_cost
        
        self.standard_agent = SafetyStarterAgent(
            env.observation_space,
            env.action_space,
            agent=agent_fn(),
            agent_type='standard',
            max_ep_len=max_ep_len,
            steps_per_epoch=steps_per_epoch,
            save_freq=save_freq,
            min_episodes=min_episodes
        )
                
        self.safe_agent = SafetyStarterAgent(
            env.observation_space,
            env.action_space,
            agent=agent_fn(),
            agent_type='safe',
            max_ep_len=max_ep_len,
            steps_per_epoch=steps_per_epoch,
            save_freq=save_freq,
            min_episodes=min_episodes
        )
                
        self.intervene_agent = SafetyStarterAgent(
            env.observation_space,
            env.action_space,
            agent=agent_fn(),
            agent_type='intervention',
            max_ep_len=max_ep_len,
            steps_per_epoch=steps_per_epoch,
            save_freq=save_freq,
            min_episodes=min_episodes
        )
                
        self.ss_agents = [self.standard_agent, self.safe_agent, self.intervene_agent]
        
    def compute_penalties(self):
        for ss_agent in self.ss_agents:
            if ss_agent.agent.use_penalty:
                ss_agent.cur_penalty = ss_agent.sess.run(ss_agent.penalty)

    def logger_save_state(self, epoch, max_epochs, env):
        for ss_agent in self.ss_agents:
            if (epoch % ss_agent.save_freq == 0) or (epoch == max_epochs-1):
                ss_agent.logger.save_state({'env': env}, None)

    def buffer_finish_path(
            self,
            done,
            ep_len,
            o,
            terminal,
            ep_ret,
            ep_cost
    ):
        for ss_agent in self.ss_agents:
            # If trajectory didn't reach terminal state, bootstrap value target(s)
            if done and not(ep_len == ss_agent.max_ep_len):
                # Note: we do not count env time out as true terminal state
                last_val, last_cval = 0, 0
            else:
                feed_dict={ss_agent.x_ph: o[np.newaxis]}
                if ss_agent.agent.reward_penalized:
                    last_val = ss_agent.sess.run(ss_agent.v, feed_dict=feed_dict)
                    last_cval = 0
                else:
                    last_val, last_cval = ss_agent.sess.run([ss_agent.v, ss_agent.vc], feed_dict=feed_dict)
            ss_agent.buf.finish_path(last_val, last_cval)
    
            # Only save EpRet / EpLen if trajectory finished
            if terminal:
                ss_agent.logger.store(EpRet=ep_ret, EpLen=ep_len, EpCost=ep_cost)
            else:
                print('Warning: trajectory cut off by epoch at %d steps.'%ep_len)

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> Tuple[np.ndarray, ActionDict]:
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs).to(self.device)
            
            a_i, v_t_i, vc_t_i, logp_t_i, pi_info_t_i = self.intervene_agent.get_action(obs_tensor)
            a_1, v_t_1, vc_t_1, logp_t_1, pi_info_t_1 = self.standard_agent.get_action(obs_tensor)
            a_2, v_t_2, vc_t_2, logp_t_2, pi_info_t_2 = self.safe_agent.get_action(obs_tensor)
            
            a_i = a_i.squeeze()
            x = a_i[0]
            y = a_i[1]
            
            if x > 0 and y > 0:
                actions = a_1
                intervene = False
            elif x < 0 and y > 0:
                actions = a_1
                intervene = False
            elif x > 0 and y < 0:
                actions = a_2
                intervene = True
            else:
                actions = a_2
                intervene = True

            actions = actions[0, :]
            # a_i = a_i.cpu().numpy()
            action_dict = {
                'intervene': intervene,
                'v_t_1': v_t_1,
                'v_t_2': v_t_2,
                'v_t_i': v_t_i,
                'vc_t_1': vc_t_1,
                'vc_t_2': vc_t_2,
                'vc_t_i': vc_t_i,
                'logp_t_1': logp_t_1,
                'logp_t_2': logp_t_2,
                'logp_t_i': logp_t_i,
                'pi_info_t_1': pi_info_t_1,
                'pi_info_t_2': pi_info_t_2,
                'pi_info_t_i': pi_info_t_i,
                'a_i': a_i,
            }
            # print('actions:', actions.shape)
            return actions, action_dict

    def store_transition(
            self,
            obs: np.ndarray,
            action:np.ndarray,
            next_obs: np.ndarray,
            r: float,
            info: Dict,
            action_dict: ActionDict
    ):
        intervene = action_dict['intervene']

        c = info['cost'] + self.fixed_cost if intervene else info['cost']
        
        # c = info['cost']
        
        
        # print(cost)
        # cost = torch.from_numpy(cost).to(self.device)
        # self.standard_rollout_buffer.add(obs, action, reward, last_done, action_dict['values_1'],
        #                            action_dict['log_prob_1'], not intervene)
        # self.safe_rollout_buffer.add(obs, action, -cost, last_done, action_dict['values_2'],
        #                            action_dict['log_prob_2'], intervene)
        # self.intervene_rollout_buffer.add(obs, action_dict['action_i'], -cost, last_done,
        #                            action_dict['values_i'], action_dict['log_prob_i'], True)

        self.standard_agent.store(
            obs,
            action,
            r,
            c,
            action_dict['v_t_1'],
            action_dict['vc_t_1'],
            action_dict['logp_t_1'],
            action_dict['pi_info_t_1'],
            not intervene
            # True
        )
        
        self.safe_agent.store(
            obs,
            action,
            -c,
            c,
            action_dict['v_t_2'],
            action_dict['vc_t_2'],
            action_dict['logp_t_2'],
            action_dict['pi_info_t_2'],
            intervene
        )
        
        self.intervene_agent.store(
            obs,
            action,
            -c,
            c,
            action_dict['v_t_i'],
            action_dict['vc_t_i'],
            action_dict['logp_t_i'],
            action_dict['pi_info_t_i'],
            True
        )

    def learn(self):
        # with torch.no_grad():
            # Compute action and value for the last timestep
            # obs_tensor = torch.as_tensor([obs]).to(self.device)
            # _, values_i, _ = self.intervene_agent.get_action(obs_tensor)
            # _, values_1, _ = self.standard_agent.get_action(obs_tensor)
            # _, values_2, _ = self.safe_agent.get_action(obs_tensor)

            # values_i = values_i.cpu().numpy()
            # values_1 = values_1.cpu().numpy()
            # values_2 = values_2.cpu().numpy()


            # self.standard_rollout_buffer.compute_returns_and_advantage(last_values=values_1, dones=done)
            # self.safe_rollout_buffer.compute_returns_and_advantage(last_values=values_2, dones=done)
            # self.intervene_rollout_buffer.compute_returns_and_advantage(last_values=values_i, dones=done)

        # ppo_train(self.standard_policy, self.standard_rollout_buffer, current_progress_remaining,
        #           lr_schedule_fn_1, self.writer, episode, 'standard')
        # ppo_train(self.safe_policy, self.safe_rollout_buffer, current_progress_remaining,
        #           lr_schedule_fn_2, self.writer, episode, 'safe')
        # ppo_train(self.intervene_policy, self.intervene_rollout_buffer, current_progress_remaining,
        #           lr_schedule_fn_2, self.writer, episode, 'intervene')
        
        # self.standard_agent.update()
        # self.safe_agent.update()
        # self.intervene_agent.update()
        
        # self.standard_agent.reset_buffer()
        # self.safe_agent.reset_buffer()
        # self.intervene_agent.reset_buffer()
        for ss_agent in self.ss_agents:
            ss_agent.update()
