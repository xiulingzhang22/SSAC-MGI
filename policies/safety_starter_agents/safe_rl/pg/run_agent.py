import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import gym
import time
from . import trust_region as tro
from .agents import PPOAgent, TRPOAgent, CPOAgent
from .buffer import CPOBuffer
from .network import count_vars, \
                               get_vars, \
                               mlp_actor_critic,\
                               placeholders, \
                               placeholders_from_spaces
from .utils import values_as_sorted_list
from ..utils.logx import EpochLogger
from ..utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from ..utils.mpi_tools import mpi_fork, proc_id, num_procs, mpi_sum

from matplotlib import pyplot as plt
max_steps=100
def run_eval_episode(env, get_action, cumulative_cost):
    # post training eval loop
    obs = env.reset()
    done = False
    interventions = []
    ep_return = 0
    ep_cost = 0
    ep_correct_goals = 0
    i = 0
    while not done and i < max_steps:
        # post-learning evaluation loop
        action = get_action(obs)
        obs, reward, done, info = env.step(action)
        ep_return += reward
        ep_cost += info['cost']
        ep_correct_goals += info.get('correct_goals', -1)
        i+=1
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
        if episode == episodes-1:
            env.render()
    return np.mean(ep_returns), np.mean(ep_costs), ave_correct_goals

# Multi-purpose agent runner for policy optimization algos 
# (PPO, TRPO, their primal-dual equivalents, CPO)
def run_polopt_agent(env_fn,
                     results_path,
                     agent=CPOAgent(),
                     actor_critic=mlp_actor_critic, 
                     ac_kwargs=dict(), 
                     seed=2024,
                     render=False,
                     # Experience collection:
                     steps_per_epoch=300,
                     max_ep_len=300,
                     # Discount factors:
                     gamma=0.98,
                     lam=0.97,
                     cost_gamma=0.99, 
                     cost_lam=0.97, 
                     # Policy learning:
                     ent_reg=0.,
                     # Cost constraints / penalties:
                     cost_lim=25,#25,
                     penalty_init=1.,
                     penalty_lr=5e-3, # 5e-2
                     # KL divergence:
                     target_kl=0.001,
                     # Value learning:
                     vf_lr=4e-3, # 1e-3
                     vf_iters=80, 
                     # Logging:
                     logger=None, 
                     logger_kwargs=dict(), 
                     save_freq=1,
                     min_episodes=500,
                     algorithm=None
                     ):

    logger = EpochLogger(**logger_kwargs) if logger is None else logger
    logger.save_config(locals())

    # seed += 10000 * proc_id()
    tf.set_random_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    eval_env = env_fn()

    agent.set_logger(logger)

    #=========================================================================#
    #  Create computation graph for actor and critic (not training routine)   #
    #=========================================================================#

    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space

    # Inputs to computation graph from environment spaces
    x_ph, a_ph = placeholders_from_spaces(env.observation_space, env.action_space)
    # Inputs to computation graph for batch data
    adv_ph, cadv_ph, ret_ph, cret_ph, logp_old_ph = placeholders(*(None for _ in range(5)))

    # Inputs to computation graph for special purposes
    surr_cost_rescale_ph = tf.placeholder(tf.float32, shape=())
    cur_cost_ph = tf.placeholder(tf.float32, shape=())

    # Outputs from actor critic
    ac_outs = actor_critic(x_ph, a_ph, **ac_kwargs)
    pi, logp, logp_pi, pi_info, pi_info_phs, d_kl, ent, v, vc = ac_outs

    # Organize placeholders for zipping with data from buffer on updates
    buf_phs = [x_ph, a_ph, adv_ph, cadv_ph, ret_ph, cret_ph, logp_old_ph]
    buf_phs += values_as_sorted_list(pi_info_phs)

    # Organize symbols we have to compute at each step of acting in env
    get_action_ops = dict(pi=pi, 
                          v=v, 
                          logp_pi=logp_pi,
                          pi_info=pi_info)

    # If agent is reward penalized, it doesn't use a separate value function
    # for costs and we don't need to include it in get_action_ops; otherwise we do.
    if not(agent.reward_penalized):
        get_action_ops['vc'] = vc

    # Count variables
    var_counts = tuple(count_vars(scope) for scope in ['pi', 'vf', 'vc'])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d, \t vc: %d\n'%var_counts)

    # Make a sample estimate for entropy to use as sanity check
    approx_ent = tf.reduce_mean(-logp)


    #=========================================================================#
    #  Create replay buffer                                                   #
    #=========================================================================#

    # Obs/act shapes
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    # Experience buffer
    local_steps_per_epoch = 300#int(steps_per_epoch / num_procs())
    number_UAV=6#3#5#6
    number_user=20 #15,20,10
    pi_info_shapes = {k: v.shape.as_list()[1:] for k,v in pi_info_phs.items()}
    buf = CPOBuffer(local_steps_per_epoch*number_UAV, #5,
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

    if agent.use_penalty:
        with tf.variable_scope('penalty'):
            # param_init = np.log(penalty_init)
            param_init = np.log(max(np.exp(penalty_init)-1, 1e-8))
            penalty_param = tf.get_variable('penalty_param',
                                          initializer=float(param_init),
                                          trainable=agent.learn_penalty,
                                          dtype=tf.float32)
        # penalty = tf.exp(penalty_param)
        penalty = tf.nn.softplus(penalty_param)

    if agent.learn_penalty:
        if agent.penalty_param_loss:
            penalty_loss = -penalty_param * (cur_cost_ph - cost_lim)
        else:
            penalty_loss = -penalty * (cur_cost_ph - cost_lim)
        train_penalty = MpiAdamOptimizer(learning_rate=penalty_lr).minimize(penalty_loss)


    #=========================================================================#
    #  Create computation graph for policy learning                           #
    #=========================================================================#

    # Likelihood ratio
    ratio = tf.exp(logp - logp_old_ph)

    # Surrogate advantage / clipped surrogate advantage
    if agent.clipped_adv:
        min_adv = tf.where(adv_ph>0, 
                           (1+agent.clip_ratio)*adv_ph, 
                           (1-agent.clip_ratio)*adv_ph
                           )
        surr_adv = tf.reduce_mean(tf.minimum(ratio * adv_ph, min_adv))
    else:
        surr_adv = tf.reduce_mean(ratio * adv_ph)

    # Surrogate cost
    surr_cost = tf.reduce_mean(ratio * cadv_ph)

    # Create policy objective function, including entropy regularization
    pi_objective = surr_adv + ent_reg * ent

    # Possibly include surr_cost in pi_objective
    if agent.objective_penalized:
        pi_objective -= penalty * surr_cost
        pi_objective /= (1 + penalty)

    # Loss function for pi is negative of pi_objective
    pi_loss = -pi_objective

    # Optimizer-specific symbols
    if agent.trust_region:

        # Symbols needed for CG solver for any trust region method
        pi_params = get_vars('pi')
        flat_g = tro.flat_grad(pi_loss, pi_params)
        v_ph, hvp = tro.hessian_vector_product(d_kl, pi_params)
        if agent.damping_coeff > 0:
            hvp += agent.damping_coeff * v_ph

        # Symbols needed for CG solver for CPO only
        flat_b = tro.flat_grad(surr_cost, pi_params)

        # Symbols for getting and setting params
        get_pi_params = tro.flat_concat(pi_params)
        set_pi_params = tro.assign_params_from_flat(v_ph, pi_params)

        training_package = dict(flat_g=flat_g,
                                flat_b=flat_b,
                                v_ph=v_ph,
                                hvp=hvp,
                                get_pi_params=get_pi_params,
                                set_pi_params=set_pi_params)

    elif agent.first_order:

        # Optimizer for first-order policy optimization
        train_pi = MpiAdamOptimizer(learning_rate=agent.pi_lr).minimize(pi_loss)

        # Prepare training package for agent
        training_package = dict(train_pi=train_pi)

    else:
        raise NotImplementedError

    # Provide training package to agent
    training_package.update(dict(pi_loss=pi_loss, 
                                 surr_cost=surr_cost,
                                 d_kl=d_kl, 
                                 target_kl=target_kl,
                                 cost_lim=cost_lim))
    agent.prepare_update(training_package)

    #=========================================================================#
    #  Create computation graph for value learning                            #
    #=========================================================================#

    # Value losses
    v_loss = tf.reduce_mean((ret_ph - v)**2)
    vc_loss = tf.reduce_mean((cret_ph - vc)**2)

    # If agent uses penalty directly in reward function, don't train a separate
    # value function for predicting cost returns. (Only use one vf for r - p*c.)
    if agent.reward_penalized:
        total_value_loss = v_loss
    else:
        total_value_loss = v_loss + vc_loss

    # Optimizer for value learning
    train_vf = MpiAdamOptimizer(learning_rate=vf_lr).minimize(total_value_loss)


    #=========================================================================#
    #  Create session, sync across procs, and set up saver                    #
    #=========================================================================#

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Sync params across processes
    sess.run(sync_all_params())

    # Setup model saving
    logger.setup_tf_saver(sess, inputs={'x': x_ph}, outputs={'pi': pi, 'v': v, 'vc': vc})

    #=========================================================================#
    #  Provide session to agent                                               #
    #=========================================================================#
    agent.prepare_session(sess)

    #=========================================================================#
    #  Create function for running update (called at end of each epoch)       #
    #=========================================================================#

    def update():
        cur_cost = logger.get_stats('EpCost')[0]
        c = cur_cost - cost_lim
        if c > 0 and agent.cares_about_cost:
            logger.log('Warning! Safety constraint is already violated.', 'red')

        #=====================================================================#
        #  Prepare feed dict                                                  #
        #=====================================================================#
        inputs = {k:v for k,v in zip(buf_phs, buf.get())}
        inputs[surr_cost_rescale_ph] = logger.get_stats('EpLen')[0]
        inputs[cur_cost_ph] = cur_cost

        #=====================================================================#
        #  Make some measurements before updating                             #
        #=====================================================================#

        measures = dict(LossPi=pi_loss,
                        SurrCost=surr_cost,
                        LossV=v_loss,
                        Entropy=ent)
        if not(agent.reward_penalized):
            measures['LossVC'] = vc_loss
        if agent.use_penalty:
            measures['Penalty'] = penalty

        pre_update_measures = sess.run(measures, feed_dict=inputs)
        logger.store(**pre_update_measures)

        #=====================================================================#
        #  Update penalty if learning penalty                                 #
        #=====================================================================#
        if agent.learn_penalty:
            sess.run(train_penalty, feed_dict={cur_cost_ph: cur_cost})

        #=====================================================================#
        #  Update policy                                                      #
        #=====================================================================#
        agent.update_pi(inputs)

        #=====================================================================#
        #  Update value function                                              #
        #=====================================================================#
        for _ in range(vf_iters):
            sess.run(train_vf, feed_dict=inputs)

        #=====================================================================#
        #  Make some measurements after updating                              #
        #=====================================================================#

        del measures['Entropy']
        measures['KL'] = d_kl

        post_update_measures = sess.run(measures, feed_dict=inputs)
        deltas = dict()
        for k in post_update_measures:
            if k in pre_update_measures:
                deltas['Delta'+k] = post_update_measures[k] - pre_update_measures[k]
        logger.store(KL=post_update_measures['KL'], **deltas)

    #=========================================================================#
    #  Run main environment interaction loop                                  #
    #=========================================================================#

    start_time = time.time()
    #o, r, d, c, ep_ret, ep_cost, ep_len = env.reset(), 0, False, 0, 0, 0, 0
    cur_penalty = 0
    cum_cost = 0
    ep_len=0
    
    all_rewards = []
    all_costs = []

    smooth_rewards = []
    smooth_costs = []

    episode = 0

    steps = 0
    all_steps = 0

    for epoch in range(min_episodes):
        env.reset()
        # print('buffer ptr start epoch:', buf.ptr)
        if agent.use_penalty:
            cur_penalty = sess.run(penalty)

        ep_ret = 0
        ep_cost = 0
        for t in range(local_steps_per_epoch):
            env.go_before_step(t)
            server_ids = range(number_user, number_user+number_UAV)
            steps += 1
            all_steps += 1
            for server_id in server_ids:
                o = env.get_state(server_id)
                server_idle_state = env.get_server_state(t, server_id)
                get_action_outs = sess.run(get_action_ops,
                                       feed_dict={x_ph: o[np.newaxis]})
                if not server_idle_state:
                    #env.perform_computing(t, server_id)
                    a=np.array([[0.0, -1.0]])
                    v_t = get_action_outs['v']
                    vc_t = get_action_outs.get('vc', 0)  # Agent may not use cost value func
                    logp_t = get_action_outs['logp_pi']
                    pi_info_t = get_action_outs['pi_info']
                    a = np.clip(a, -1, 1)
                    o2, r, d, info = env.step(server_id, a[0])
                    c = info.get('cost', 0)
                else:
                    a = get_action_outs['pi']
                    v_t = get_action_outs['v']
                    vc_t = get_action_outs.get('vc', 0)  # Agent may not use cost value func
                    logp_t = get_action_outs['logp_pi']
                    pi_info_t = get_action_outs['pi_info']
                    a=np.clip(a, -1, 1)
                    o2, r, d, info = env.step(server_id,a[0])
                    c = info.get('cost', 0)

                # Track cumulative cost over training
                cum_cost += c
                # save and log
                r_total = r -cur_penalty * c
                r_total = r_total / (1 + cur_penalty)
                buf.store(o, a, r-c, v_t, 0, 0, logp_t, pi_info_t) #r_total
                """if agent.reward_penalized:
                    r_total = r - cur_penalty * c
                    r_total = r_total / (1 + cur_penalty)
                    buf.store(o, a, r_total, v_t, 0, 0, logp_t, pi_info_t)
                else:
                    buf.store(o, a, r - c, v_t, c, vc_t, logp_t, pi_info_t)
                    # buf.store(o, a, r-c, v_t, c, vc_t, logp_t, pi_info_t)"""
                logger.store(VVals=v_t, CostVVals=vc_t)
                #o = o2
                ep_ret += r
                ep_cost += c
                def get_action(obs):
                    # Get outputs from policy
                    get_action_outs = sess.run(get_action_ops,
                                               feed_dict={x_ph: obs[np.newaxis]})
                    return get_action_outs['pi']
            ep_len += 1
            """if all_steps % 1000 == 0:
                mean_eval_return, mean_eval_cost, mean_correct_goals = \
                    evaluate(
                        eval_env,
                        get_action,
                        32,
                        cum_cost
                    )
                f.write(f'{all_steps}, {mean_eval_return}, {mean_eval_cost}, {cum_cost}, {mean_correct_goals}\n')
                f.flush()"""
        env.record_results(local_steps_per_epoch)
        all_rewards.append(ep_ret)
        all_costs.append(ep_cost)
        smooth_rewards.append(np.mean(all_rewards[-100:]))
        smooth_costs.append(np.mean(all_costs[-100:]))

        print(
            f'Algorithm: {algorithm}, Episode: {epoch}, Reward: {np.mean(all_rewards[-100:])}, Cost: {np.mean(all_costs[-100:])}')
        """# If trajectory didn't reach terminal state, bootstrap value target(s)
        if d and not (ep_len == max_ep_len):
            # Note: we do not count env time out as true terminal state
            last_val, last_cval = 0, 0
        else:
            feed_dict = {x_ph: o[np.newaxis]}
            if agent.reward_penalized:
                last_val = sess.run(v, feed_dict=feed_dict)
                last_cval = 0
            else:
                last_val, last_cval = sess.run([v, vc], feed_dict=feed_dict)"""
        last_val, last_cval = 0, 0
        buf.finish_path(last_val, last_cval)

        """# Only save EpRet / EpLen if trajectory finished
        if d:
            logger.store(EpRet=ep_ret, EpLen=ep_len, EpCost=ep_cost)
        else:
            print('Warning: trajectory cut off by epoch at %d steps.' % ep_len)

        # Reset environment
        #o, r, d, c, ep_ret, ep_len, ep_cost = env.reset(), 0, False, 0, 0, 0, 0
        episode += 1"""
        logger.store(EpRet=ep_ret, EpLen=ep_len, EpCost=ep_cost)
        update()

        cumulative_cost = mpi_sum(cum_cost)
        cost_rate = cumulative_cost / ((epoch+1)*steps_per_epoch)

        """if len(all_rewards) >= min_episodes:
            break"""

    tf.keras.backend.clear_session()
    metrics_records=env.env.record_metrics
    return all_rewards,smooth_rewards, smooth_costs,metrics_records

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, default='ppo')
    parser.add_argument('--env', type=str, default='Safexp-PointGoal1-v0')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--cost_gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--len', type=int, default=1000)
    parser.add_argument('--cost_lim', type=float, default=10)
    parser.add_argument('--exp_name', type=str, default='runagent')
    parser.add_argument('--kl', type=float, default=0.01)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--reward_penalized', action='store_true')
    parser.add_argument('--objective_penalized', action='store_true')
    parser.add_argument('--learn_penalty', action='store_true')
    parser.add_argument('--penalty_param_loss', action='store_true')
    parser.add_argument('--entreg', type=float, default=0.)
    args = parser.parse_args()

    try:
        import safety_gym
    except:
        print('Make sure to install Safety Gym to use constrained RL environments.')

    mpi_fork(args.cpu)  # run parallel code with mpi

    # Prepare logger
    from safe_rl.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    # Prepare agent
    agent_kwargs = dict(reward_penalized=args.reward_penalized,
                        objective_penalized=args.objective_penalized,
                        learn_penalty=args.learn_penalty,
                        penalty_param_loss=args.penalty_param_loss)
    if args.agent=='ppo':
        agent = PPOAgent(**agent_kwargs)
    elif args.agent=='trpo':
        agent = TRPOAgent(**agent_kwargs)
    elif args.agent=='cpo':
        agent = CPOAgent(**agent_kwargs)

    run_polopt_agent(lambda : gym.make(args.env),
                     agent=agent,
                     actor_critic=mlp_actor_critic,
                     ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
                     seed=args.seed, 
                     render=args.render, 
                     # Experience collection:
                     steps_per_epoch=args.steps, 
                     epochs=args.epochs,
                     max_ep_len=args.len,
                     # Discount factors:
                     gamma=args.gamma,
                     cost_gamma=args.cost_gamma,
                     # Policy learning:
                     ent_reg=args.entreg,
                     # KL Divergence:
                     target_kl=args.kl,
                     cost_lim=args.cost_lim, 
                     # Logging:
                     logger_kwargs=logger_kwargs,
                     save_freq=1
                     )