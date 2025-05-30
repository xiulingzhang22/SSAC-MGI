# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 21:34:22 2022

@author: u84228923
"""

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import gym
import time
from .safety_starter_agents.safe_rl import trust_region as tro
from .safety_starter_agents.safe_rl.pg.agents import PPOAgent, TRPOAgent, CPOAgent
from .safety_starter_agents.safe_rl.pg.buffer import CPOBufferForBaseLearner
from .safety_starter_agents.safe_rl.pg.network import count_vars, \
                               get_vars, \
                               mlp_actor_critic,\
                               placeholders, \
                               placeholders_from_spaces
from .safety_starter_agents.safe_rl.pg.utils import values_as_sorted_list
from .safety_starter_agents.safe_rl.utils.logx import EpochLogger
from .safety_starter_agents.safe_rl.utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from .safety_starter_agents.safe_rl.utils.mpi_tools import mpi_fork, proc_id, num_procs, mpi_sum

from matplotlib import pyplot as plt


def run_eval_episode(env, get_action, cumulative_cost):
    # post training eval loop
    obs = env.reset()
    done = False
    interventions = []
    ep_return = 0
    ep_cost = 0
    ep_correct_goals = 0
    while not done:
        # post-learning evaluation loop
        action = get_action(obs)
        obs, reward, done, info = env.step(action)
        ep_return += reward
        ep_cost += info['cost']
        ep_correct_goals += info.get('correct_goals', -1)
    return ep_return, ep_cost, ep_correct_goals
        
def evaluate(env, get_action, episodes: int, cumulative_cost: float):
    ep_returns = []
    ep_costs = []
    ep_correct_goals_arr = []
    for episode in range(episodes):
        ep_return, ep_cost, ep_correct_goals = run_eval_episode(
            env,
            get_action,
            cumulative_cost
        )
        ep_returns.append(ep_return)
        ep_costs.append(ep_cost)
        ep_correct_goals_arr.append(ep_correct_goals)
        if None not in ep_correct_goals_arr:
            ave_correct_goals = np.mean(ep_correct_goals_arr)
        else:
            ave_correct_goals = np.array([-1] * len(ep_correct_goals_arr))
    return np.mean(ep_returns), np.mean(ep_costs), ave_correct_goals




class SafetyStarterAgent():
    
    def get_action(self, obs):
        get_action_outs = self.sess.run(
            self.get_action_ops, 
            feed_dict={self.x_ph: obs[np.newaxis]}
        )
        a = get_action_outs['pi']
        v_t = get_action_outs['v']
        vc_t = get_action_outs.get('vc', 0)  # Agent may not use cost value func
        logp_t = get_action_outs['logp_pi']
        pi_info_t = get_action_outs['pi_info']
        
        return a, v_t, vc_t, logp_t, pi_info_t
    
    def store(
        self,
        o,
        a,
        r,
        c,        
        v_t,
        vc_t,
        logp_t,
        pi_info_t,
        active
    ):
        if self.agent.reward_penalized:
            r_total = r - self.cur_penalty * c
            r_total = r_total / (1 + self.cur_penalty)
            self.buf.store(o, a, r_total, v_t, 0, 0, logp_t, pi_info_t, active)
        else:
            self.buf.store(o, a, r, v_t, c, vc_t, logp_t, pi_info_t, active)
        self.logger.store(VVals=v_t, CostVVals=vc_t)
    
    def update(self):
        cur_cost = self.logger.get_stats('EpCost')[0]
        c = cur_cost - self.cost_lim
        if c > 0 and self.agent.cares_about_cost:
            self.logger.log('Warning! Safety constraint is already violated.', 'red')
    
        #=====================================================================#
        #  Prepare feed dict                                                  #
        #=====================================================================#
        inputs = {k:v for k,v in zip(self.buf_phs, self.buf.get())}
        inputs[self.surr_cost_rescale_ph] = self.logger.get_stats('EpLen')[0]
        inputs[self.cur_cost_ph] = cur_cost
    
        #=====================================================================#
        #  Make some measurements before updating                             #
        #=====================================================================#
    
        measures = dict(LossPi=self.pi_loss,
                        SurrCost=self.surr_cost,
                        LossV=self.v_loss,
                        Entropy=self.ent)
        if not(self.agent.reward_penalized):
            measures['LossVC'] = self.vc_loss
        if self.agent.use_penalty:
            measures['Penalty'] = self.penalty
    
        pre_update_measures = self.sess.run(measures, feed_dict=inputs)
        self.logger.store(**pre_update_measures)
    
        #=====================================================================#
        #  Update penalty if learning penalty                                 #
        #=====================================================================#
        if self.agent.learn_penalty:
            self.sess.run(self.train_penalty, feed_dict={self.cur_cost_ph: cur_cost})
    
        #=====================================================================#
        #  Update policy                                                      #
        #=====================================================================#
        self.agent.update_pi(inputs)
    
        #=====================================================================#
        #  Update value function                                              #
        #=====================================================================#
        for _ in range(self.vf_iters):
            self.sess.run(self.train_vf, feed_dict=inputs)
    
        #=====================================================================#
        #  Make some measurements after updating                              #
        #=====================================================================#
    
        del measures['Entropy']
        measures['KL'] = self.d_kl
    
        post_update_measures = self.sess.run(measures, feed_dict=inputs)
        deltas = dict()
        for k in post_update_measures:
            if k in pre_update_measures:
                deltas['Delta'+k] = post_update_measures[k] - pre_update_measures[k]
        self.logger.store(KL=post_update_measures['KL'], **deltas)
    
    def reset_buffer(self):
        pass

    # Multi-purpose agent runner for policy optimization algos 
    # (PPO, TRPO, their primal-dual equivalents, CPO)
    def __init__(
        self,
        observation_space,
        action_space,
        agent=PPOAgent(),
        actor_critic=mlp_actor_critic, 
        agent_type='standard',
        ac_kwargs=dict(), 
        seed=0,
        render=False,
        # Experience collection:
        steps_per_epoch=4000, 
        max_ep_len=1000,
        # Discount factors:
        gamma=0.99, 
        lam=0.97,
        cost_gamma=0.99, 
        cost_lam=0.97, 
        # Policy learning:
        ent_reg=0.,
        # Cost constraints / penalties:
        cost_lim=25,
        penalty_init=1.,
        penalty_lr=5e-2, # 5e-2
        # KL divergence:
        target_kl=0.01, 
        # Value learning:
        vf_lr=1e-2, # 1e-3
        vf_iters=80, 
        # Logging:
        logger=None, 
        logger_kwargs=dict(), 
        save_freq=1,
        min_episodes=2000,
        algorithm='ppo'
    ):
    
    
        #=========================================================================#
        #  Prepare logger, seed, and environment in this process                  #
        #=========================================================================#
        # f = open(results_path, 'w')
        
        max_epochs = 10000
        self.cost_lim = cost_lim
        self.vf_iters = vf_iters
        self.agent = agent
        self.agent_type = agent_type
        self.max_ep_len = max_ep_len
        self.save_freq = save_freq
        self.min_episodes = min_episodes
        
        self.pi_scope = f'pi_{agent_type}'
        self.vf_scope = f'vf_{agent_type}'
        self.vc_scope = f'vc_{agent_type}'
    
        self.logger = EpochLogger(**logger_kwargs) if logger is None else logger
        self.logger.save_config(locals())
    
        # seed += 10000 * proc_id()
        tf.set_random_seed(seed)
        np.random.seed(seed)
    
        # env = env_fn()
        # eval_env = env_fn()
    
        self.agent.set_logger(self.logger)
    
        #=========================================================================#
        #  Create computation graph for actor and critic (not training routine)   #
        #=========================================================================#
    
        # Share information about action space with policy architecture
        ac_kwargs['action_space'] = action_space
    
        # Inputs to computation graph from environment spaces
        self.x_ph, a_ph = placeholders_from_spaces(observation_space, action_space)
    
        # Inputs to computation graph for batch data
        adv_ph, cadv_ph, ret_ph, cret_ph, logp_old_ph, active_ph = placeholders(*(None for _ in range(6)))
    
        # Inputs to computation graph for special purposes
        self.surr_cost_rescale_ph = tf.placeholder(tf.float32, shape=())
        self.cur_cost_ph = tf.placeholder(tf.float32, shape=())
    
        # Outputs from actor critic
        ac_kwargs['agent_type'] = agent_type
        ac_outs = actor_critic(self.x_ph, a_ph, **ac_kwargs)
        pi, logp, logp_pi, pi_info, pi_info_phs, self.d_kl, self.ent, self.v, self.vc = ac_outs
    
        # Organize placeholders for zipping with data from buffer on updates
        self.buf_phs = [self.x_ph, a_ph, adv_ph, cadv_ph, ret_ph, cret_ph, logp_old_ph, active_ph]
        self.buf_phs += values_as_sorted_list(pi_info_phs)
    
        # Organize symbols we have to compute at each step of acting in env
        self.get_action_ops = dict(pi=pi, 
                              v=self.v, 
                              logp_pi=logp_pi,
                              pi_info=pi_info)
    
        # If agent is reward penalized, it doesn't use a separate value function
        # for costs and we don't need to include it in get_action_ops; otherwise we do.
        if not(self.agent.reward_penalized):
            self.get_action_ops['vc'] = self.vc
    
        # Count variables
        var_counts = tuple(count_vars(scope) for scope in [self.pi_scope, self.vf_scope, self.vc_scope])
        self.logger.log('\nNumber of parameters: \t pi: %d, \t v: %d, \t vc: %d\n'%var_counts)
    
        # Make a sample estimate for entropy to use as sanity check
        approx_ent = tf.reduce_mean(-logp)
    
    
        #=========================================================================#
        #  Create replay buffer                                                   #
        #=========================================================================#
    
        # Obs/act shapes
        obs_shape = observation_space.shape
        act_shape = action_space.shape
    
        # Experience buffer
        local_steps_per_epoch = int(steps_per_epoch / num_procs())
        pi_info_shapes = {k: v.shape.as_list()[1:] for k,v in pi_info_phs.items()}
        self.buf = CPOBufferForBaseLearner(local_steps_per_epoch,
                        obs_shape, 
                        act_shape, 
                        pi_info_shapes, 
                        gamma, 
                        lam,
                        cost_gamma,
                        cost_lam)
    
    
        #=========================================================================#
        #  Create computation graph for penalty learning, if applicable           #
        #=========================================================================#
    
        if self.agent.use_penalty:
            with tf.variable_scope(f'penalty_{agent_type}'):
                # param_init = np.log(penalty_init)
                param_init = np.log(max(np.exp(penalty_init)-1, 1e-8))
                penalty_param = tf.get_variable('penalty_param',
                                              initializer=float(param_init),
                                              trainable=self.agent.learn_penalty,
                                              dtype=tf.float32)
            # penalty = tf.exp(penalty_param)
            self.penalty = tf.nn.softplus(penalty_param)
    
        if self.agent.learn_penalty:
            if self.agent.penalty_param_loss:
                penalty_loss = -penalty_param * (self.cur_cost_ph - self.cost_lim)
            else:
                penalty_loss = -self.penalty * (self.cur_cost_ph - self.
                                           self.cost_lim)
            self.train_penalty = MpiAdamOptimizer(learning_rate=penalty_lr).minimize(penalty_loss)
    
    
        #=========================================================================#
        #  Create computation graph for policy learning                           #
        #=========================================================================#
    
        # Likelihood ratio
        ratio = tf.exp(logp - logp_old_ph)
    
        # Surrogate advantage / clipped surrogate advantage
        if self.agent.clipped_adv:
            min_adv = tf.where(adv_ph>0, 
                               (1+self.agent.clip_ratio)*adv_ph, 
                               (1-self.agent.clip_ratio)*adv_ph
                               )
            surr_adv = tf.reduce_mean(tf.minimum(ratio * adv_ph, min_adv))
        else:
            surr_adv = tf.reduce_mean(ratio * adv_ph)
    
        # Surrogate cost
        self.surr_cost = tf.reduce_mean(ratio * cadv_ph)
    
        # Create policy objective function, including entropy regularization
        pi_objective = surr_adv + ent_reg * self.ent
    
        # Possibly include surr_cost in pi_objective
        if self.agent.objective_penalized:
            pi_objective -= self.penalty * self.surr_cost
            pi_objective /= (1 + self.penalty)
    
        # Loss function for pi is negative of pi_objective
        self.pi_loss = -pi_objective
    
        # Optimizer-specific symbols
        if self.agent.trust_region:
    
            # Symbols needed for CG solver for any trust region method
            pi_params = get_vars(self.pi_scope)
            flat_g = tro.flat_grad(self.pi_loss, pi_params)
            v_ph, hvp = tro.hessian_vector_product(self.d_kl, pi_params)
            if self.agent.damping_coeff > 0:
                hvp += self.agent.damping_coeff * v_ph
    
            # Symbols needed for CG solver for CPO only
            flat_b = tro.flat_grad(self.surr_cost, pi_params)
    
            # Symbols for getting and setting params
            get_pi_params = tro.flat_concat(pi_params)
            set_pi_params = tro.assign_params_from_flat(v_ph, pi_params)
    
            training_package = dict(flat_g=flat_g,
                                    flat_b=flat_b,
                                    v_ph=v_ph,
                                    hvp=hvp,
                                    get_pi_params=get_pi_params,
                                    set_pi_params=set_pi_params)
    
        elif self.agent.first_order:
    
            # Optimizer for first-order policy optimization
            train_pi = MpiAdamOptimizer(learning_rate=self.agent.pi_lr).minimize(self.pi_loss)
    
            # Prepare training package for agent
            training_package = dict(train_pi=train_pi)
    
        else:
            raise NotImplementedError
    
        # Provide training package to agent
        training_package.update(dict(pi_loss=self.pi_loss, 
                                     surr_cost=self.surr_cost,
                                     d_kl=self.d_kl, 
                                     target_kl=target_kl,
                                     cost_lim=self.cost_lim))
        self.agent.prepare_update(training_package)
    
        #=========================================================================#
        #  Create computation graph for value learning                            #
        #=========================================================================#
    
        # Value losses
        self.v_loss = tf.reduce_mean((ret_ph - self.v)**2)
        self.vc_loss = tf.reduce_mean((cret_ph - self.vc)**2)
    
        # If agent uses penalty directly in reward function, don't train a separate
        # value function for predicting cost returns. (Only use one vf for r - p*c.)
        if self.agent.reward_penalized:
            total_value_loss = self.v_loss
        else:
            total_value_loss = self.v_loss + self.vc_loss
    
        # Optimizer for value learning
        self.train_vf = MpiAdamOptimizer(learning_rate=vf_lr).minimize(total_value_loss)
    
    
        #=========================================================================#
        #  Create session, sync across procs, and set up saver                    #
        #=========================================================================#
    
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
            
        # Sync params across processes
        self.sess.run(sync_all_params())
    
        # Setup model saving
        self.logger.setup_tf_saver(self.sess, inputs={f'x_{self.agent_type}': self.x_ph}, outputs={self.pi_scope: pi, self.vf_scope: self.v, self.vc_scope: self.vc})
    
    
        #=========================================================================#
        #  Provide session to agent                                               #
        #=========================================================================#
        self.agent.prepare_session(self.sess)
    
    
        #=========================================================================#
        #  Create function for running update (called at end of each epoch)       #
        #=========================================================================#
    
        # def update():
        #     cur_cost = self.logger.get_stats('EpCost')[0]
        #     c = cur_cost - cost_lim
        #     if c > 0 and self.agent.cares_about_cost:
        #         self.logger.log('Warning! Safety constraint is already violated.', 'red')
    
        #     #=====================================================================#
        #     #  Prepare feed dict                                                  #
        #     #=====================================================================#
        #     inputs = {k:v for k,v in zip(self.buf_phs, self.buf.get())}
        #     inputs[surr_cost_rescale_ph] = self.logger.get_stats('EpLen')[0]
        #     inputs[cur_cost_ph] = cur_cost
    
        #     #=====================================================================#
        #     #  Make some measurements before updating                             #
        #     #=====================================================================#
    
        #     measures = dict(LossPi=pi_loss,
        #                     SurrCost=surr_cost,
        #                     LossV=v_loss,
        #                     Entropy=ent)
        #     if not(self.agent.reward_penalized):
        #         measures['LossVC'] = vc_loss
        #     if self.agent.use_penalty:
        #         measures['Penalty'] = penalty
    
        #     pre_update_measures = self.sess.run(measures, feed_dict=inputs)
        #     self.logger.store(**pre_update_measures)
    
        #     #=====================================================================#
        #     #  Update penalty if learning penalty                                 #
        #     #=====================================================================#
        #     if self.agent.learn_penalty:
        #         self.sess.run(train_penalty, feed_dict={cur_cost_ph: cur_cost})
    
        #     #=====================================================================#
        #     #  Update policy                                                      #
        #     #=====================================================================#
        #     self.agent.update_pi(inputs)
    
        #     #=====================================================================#
        #     #  Update value function                                              #
        #     #=====================================================================#
        #     for _ in range(vf_iters):
        #         self.sess.run(train_vf, feed_dict=inputs)
    
        #     #=====================================================================#
        #     #  Make some measurements after updating                              #
        #     #=====================================================================#
    
        #     del measures['Entropy']
        #     measures['KL'] = d_kl
    
        #     post_update_measures = self.sess.run(measures, feed_dict=inputs)
        #     deltas = dict()
        #     for k in post_update_measures:
        #         if k in pre_update_measures:
        #             deltas['Delta'+k] = post_update_measures[k] - pre_update_measures[k]
        #     self.logger.store(KL=post_update_measures['KL'], **deltas)
    
    
    
    
        # #=========================================================================#
        # #  Run main environment interaction loop                                  #
        # #=========================================================================#
    
        # start_time = time.time()
        # o, r, d, c, ep_ret, ep_cost, ep_len = env.reset(), 0, False, 0, 0, 0, 0
        # cur_penalty = 0
        # cum_cost = 0
        
        # all_rewards = []
        # all_costs = []
        # correct_goals = []
        # smooth_rewards = []
        # smooth_costs = []
        # smooth_correct_goals = []
        # steps_per_episode = []
        # episode = 0
        # ep_correct_goals = 0
        # steps = 0
        # all_steps = 0
    
        # for epoch in range(max_epochs):
        #     # print('buffer ptr start epoch:', buf.ptr)
    
        #     if agent.use_penalty:
        #         cur_penalty = self.sess.run(penalty)
    
        #     for t in range(local_steps_per_epoch):
                
        #         steps += 1
        #         all_steps += 1
    
        #         # Possibly render
        #         if render and proc_id()==0 and t < 1000:
        #             env.render()
                
        #         # Get outputs from policy
        #         get_action_outs = self.sess.run(self.get_action_ops, 
        #                                    feed_dict={x_ph: o[np.newaxis]})
        #         a = get_action_outs['pi']
        #         v_t = get_action_outs['v']
        #         vc_t = get_action_outs.get('vc', 0)  # Agent may not use cost value func
        #         logp_t = get_action_outs['logp_pi']
        #         pi_info_t = get_action_outs['pi_info']
    
        #         # Step in environment
        #         o2, r, d, info = env.step(a)
    
        #         # Include penalty on cost
        #         c = info.get('cost', 0)
        #         ep_correct_goals += info.get('correct_goals', 0)
    
        #         # Track cumulative cost over training
        #         cum_cost += c
    
        #         # save and log
        #         if agent.reward_penalized:
        #             r_total = r - cur_penalty * c
        #             r_total = r_total / (1 + cur_penalty)
        #             buf.store(o, a, r_total, v_t, 0, 0, logp_t, pi_info_t)
        #         else:
        #             buf.store(o, a, r, v_t, c, vc_t, logp_t, pi_info_t)
        #         logger.store(VVals=v_t, CostVVals=vc_t)
    
        #         o = o2
        #         ep_ret += r
        #         ep_cost += c
        #         ep_len += 1
                
        #         def get_action(obs):
        #             # Get outputs from policy
        #             get_action_outs = self.sess.run(self.get_action_ops, 
        #                                        feed_dict={x_ph: obs[np.newaxis]})
        #             return get_action_outs['pi']
                
        #         if all_steps % 1000 == 0:
        #             mean_eval_return, mean_eval_cost, mean_correct_goals = \
        #                 evaluate(
        #                     eval_env,
        #                     get_action,
        #                     32,
        #                     cum_cost
        #                 )
        #             f.write(f'{all_steps}, {mean_eval_return}, {mean_eval_cost}, {cum_cost}, {mean_correct_goals}\n')
        #             f.flush()
                
    
        #         terminal = d or (ep_len == max_ep_len)
        #         if terminal or (t==local_steps_per_epoch-1):
                    
        #             all_rewards.append(ep_ret)
        #             all_costs.append(ep_cost)
        #             correct_goals.append(ep_correct_goals)
        #             smooth_rewards.append(np.mean(all_rewards[-100:]))
        #             smooth_costs.append(np.mean(all_costs[-100:]))
        #             smooth_correct_goals.append(np.mean(correct_goals[-100:]))
        #             steps_per_episode.append(steps)
                    
        #             print(f'Algorithm: {algorithm}, Epoch: {epoch}, Episode: {episode}, Reward: {np.mean(all_rewards[-100:])}, Cost: {np.mean(all_costs[-100:])}, Steps: {steps}')
        #             ep_correct_goals = 0
        #             steps = 0
        #             # If trajectory didn't reach terminal state, bootstrap value target(s)
        #             if d and not(ep_len == max_ep_len):
        #                 # Note: we do not count env time out as true terminal state
        #                 last_val, last_cval = 0, 0
        #             else:
        #                 feed_dict={x_ph: o[np.newaxis]}
        #                 if agent.reward_penalized:
        #                     last_val = self.sess.run(v, feed_dict=feed_dict)
        #                     last_cval = 0
        #                 else:
        #                     last_val, last_cval = self.sess.run([v, vc], feed_dict=feed_dict)
        #             buf.finish_path(last_val, last_cval)
    
        #             # Only save EpRet / EpLen if trajectory finished
        #             if terminal:
        #                 logger.store(EpRet=ep_ret, EpLen=ep_len, EpCost=ep_cost)
        #             else:
        #                 print('Warning: trajectory cut off by epoch at %d steps.'%ep_len)
    
        #             # Reset environment
        #             o, r, d, c, ep_ret, ep_len, ep_cost = env.reset(), 0, False, 0, 0, 0, 0
    
        #             episode += 1
                    
        #     if (epoch % self.save_freq == 0) or (epoch == max_epochs-1):
        #         logger.save_state({'env': env}, None)
    
        #     #=====================================================================#
        #     #  Run RL update                                                      #
        #     #=====================================================================# 
        #     # print('all rewards:', len(all_rewards))
        #     # print('buffer ptr almost end epoch 1:', buf.ptr)
        #     update()
        #     # print('buffer ptr almost end epoch 2:', buf.ptr)
    
        #     #=====================================================================#
        #     #  Cumulative cost calculations                                       #
        #     #=====================================================================#
        #     cumulative_cost = mpi_sum(cum_cost)
        #     cost_rate = cumulative_cost / ((epoch+1)*steps_per_epoch)
        #     # print('buffer ptr almost end epoch 3:', buf.ptr)
        #     #=====================================================================#
        #     #  Log performance and stats                                          #
        #     #=====================================================================#
        #     logger.log_tabular('Epoch', epoch)
    
        #     # Performance stats
        #     logger.log_tabular('EpRet', with_min_and_max=True)
        #     logger.log_tabular('EpCost', with_min_and_max=True)
        #     logger.log_tabular('EpLen', average_only=True)
        #     logger.log_tabular('CumulativeCost', cumulative_cost)
        #     logger.log_tabular('CostRate', cost_rate)
    
        #     # Value function values
        #     logger.log_tabular('VVals', with_min_and_max=True)
        #     logger.log_tabular('CostVVals', with_min_and_max=True)
    
        #     # Pi loss and change
        #     logger.log_tabular('LossPi', average_only=True)
        #     logger.log_tabular('DeltaLossPi', average_only=True)
    
        #     # Surr cost and change
        #     logger.log_tabular('SurrCost', average_only=True)
        #     logger.log_tabular('DeltaSurrCost', average_only=True)
    
        #     # V loss and change
        #     logger.log_tabular('LossV', average_only=True)
        #     logger.log_tabular('DeltaLossV', average_only=True)
    
        #     # Vc loss and change, if applicable (reward_penalized agents don't use vc)
        #     if not(agent.reward_penalized):
        #         logger.log_tabular('LossVC', average_only=True)
        #         logger.log_tabular('DeltaLossVC', average_only=True)
    
        #     if agent.use_penalty or agent.save_penalty:
        #         logger.log_tabular('Penalty', average_only=True)
        #         logger.log_tabular('DeltaPenalty', average_only=True)
        #     else:
        #         logger.log_tabular('Penalty', 0)
        #         logger.log_tabular('DeltaPenalty', 0)
    
        #     # Anything from the agent?
        #     agent.log()
    
        #     # Policy stats
        #     logger.log_tabular('Entropy', average_only=True)
        #     logger.log_tabular('KL', average_only=True)
    
        #     # Time and steps elapsed
        #     logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        #     logger.log_tabular('Time', time.time()-start_time)
    
        #     # Show results_test!
        #     logger.dump_tabular()
            
        #     if len(all_rewards) >= self.min_episodes:
        #         break
            
        
        # tf.keras.backend.clear_session()
        # f.close()
        # return smooth_rewards, smooth_costs, smooth_correct_goals, steps_per_episode

# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--agent', type=str, default='ppo')
#     parser.add_argument('--env', type=str, default='Safexp-PointGoal1-v0')
#     parser.add_argument('--hid', type=int, default=64)
#     parser.add_argument('--l', type=int, default=2)
#     parser.add_argument('--gamma', type=float, default=0.99)
#     parser.add_argument('--cost_gamma', type=float, default=0.99)
#     parser.add_argument('--seed', '-s', type=int, default=0)
#     parser.add_argument('--cpu', type=int, default=4)
#     parser.add_argument('--steps', type=int, default=4000)
#     parser.add_argument('--epochs', type=int, default=50)
#     parser.add_argument('--len', type=int, default=1000)
#     parser.add_argument('--cost_lim', type=float, default=10)
#     parser.add_argument('--exp_name', type=str, default='runagent')
#     parser.add_argument('--kl', type=float, default=0.01)
#     parser.add_argument('--render', action='store_true')
#     parser.add_argument('--reward_penalized', action='store_true')
#     parser.add_argument('--objective_penalized', action='store_true')
#     parser.add_argument('--learn_penalty', action='store_true')
#     parser.add_argument('--penalty_param_loss', action='store_true')
#     parser.add_argument('--entreg', type=float, default=0.)
#     args = parser.parse_args()

#     try:
#         import safety_gym
#     except:
#         print('Make sure to install Safety Gym to use constrained RL environments.')

#     mpi_fork(args.cpu)  # run parallel code with mpi

#     # Prepare logger
#     from safe_rl.utils.run_utils import setup_logger_kwargs
#     logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

#     # Prepare agent
#     agent_kwargs = dict(reward_penalized=args.reward_penalized,
#                         objective_penalized=args.objective_penalized,
#                         learn_penalty=args.learn_penalty,
#                         penalty_param_loss=args.penalty_param_loss)
#     if args.agent=='ppo':
#         agent = PPOAgent(**agent_kwargs)
#     elif args.agent=='trpo':
#         agent = TRPOAgent(**agent_kwargs)
#     elif args.agent=='cpo':
#         agent = CPOAgent(**agent_kwargs)

#     run_polopt_agent(lambda : gym.make(args.env),
#                      agent=agent,
#                      actor_critic=mlp_actor_critic,
#                      ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
#                      seed=args.seed, 
#                      render=args.render, 
#                      # Experience collection:
#                      steps_per_epoch=args.steps, 
#                      epochs=args.epochs,
#                      max_ep_len=args.len,
#                      # Discount factors:
#                      gamma=args.gamma,
#                      cost_gamma=args.cost_gamma,
#                      # Policy learning:
#                      ent_reg=args.entreg,
#                      # KL Divergence:
#                      target_kl=args.kl,
#                      cost_lim=args.cost_lim, 
#                      # Logging:
#                      logger_kwargs=logger_kwargs,
#                      save_freq=1
#                      )