from collections import deque
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import errno
import itertools
import copy
import csv
import numpy as np
import matplotlib.pyplot as plt
from absl import app, flags
from environment import config as cf

from environment.mulitple_server_env import Environment as Mobile_Comp_Env
from policies import SinglePlayerPolicy, SinglePlayerOffPolicy, \
                     ThreeModelFreePolicy, ThreeModelFreeOffPolicy, \
                     ThreeTRPOPolicy, \
                     ThreeSafetyStarterPolicy, TwoModelFreeOffPolicy

import sys
sys.path.append('./safety_starter_agents')
sys.path.append(os.path.abspath('./environment'))
from policies.safety_starter_agents.safe_rl.pg import run_agent
from policies.safety_starter_agents.safe_rl.pg.agents import PPOAgent, TRPOAgent, CPOAgent
from policies.safety_starter_agents.safe_rl.sac.sac import sac
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
import tensorflow as tf
FLAGS=flags.FLAGS

# Experiment
flags.DEFINE_string(
    'algorithm',
    'ALL',
    'Options: ALL, desta_sac, desta_ppo, ppo(_lagrangian), trpo(_lagrangian), sac, sac_safety_starter, sac_lagrangian, ppo_safety_starter, cpo'
)
flags.DEFINE_integer('num_seeds',2, 'Number different seeds to run all algorithms for')
flags.DEFINE_integer('episodes',600, 'Total episodes to learn')
flags.DEFINE_integer('max_rounds',300,'Total rounds of one episode')
# RL Algorithm
flags.DEFINE_integer('batch_size', 1024, 'Policy rollout length for PPO') #1024
flags.DEFINE_integer('rollout_len',1024, 'Policy rollout length for PPO')
flags.DEFINE_integer('replay_buffer_size', 1000000, 'Size of Experience Replay for SAC')

# Important to tune...
flags.DEFINE_float('intervention_cost', 0.1, 'Cost of Player 2 intervening')
flags.DEFINE_float('pi_standard_lr', 1e-4, 'Learning rate of standard policy')
flags.DEFINE_float('pi_safe_lr', 1e-3, 'Learning rate of safe policy')
flags.DEFINE_float('pi_intervene_lr', 1e-4, 'Learnin rate of intervention policy')
flags.DEFINE_float('action_dist_threshold',np.inf, 'Maximum distance between actions \
    if using off-policy for safe policy (only applicable to ThreeModelFreeOffPolicy)')

def _save_trajectory(trajectory,train_env):
    train_env.plot_uavs_trjectory(trajectory)
def ensure_directories_present(dir_name):
    if not os.path.exists(dir_name):
        try:
            os.makedirs(dir_name)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
"""def run_eval_episode(env, policy):
    env.reset()
    trajectory = dict(
        zip(list(range(cf.NB_NODES, cf.NB_NODES + cf.NB_SERVERS)), [[] for _ in range(cf.NB_SERVERS)]))
    done = False
    interventions=[]
    ep_return = 0
    ep_cost = 0
    ep_timestep=0
    while not done and ep_timestep<=cf.MAX_ROUNDS:
        env.go_before_step(ep_timestep)
        server_ids = range(cf.NB_NODES, cf.NB_SERVERS + cf.NB_NODES)
        for server_id in server_ids:
            server_obs = env.get_state(server_id)
            trajectory[server_id].append([server_obs[0], server_obs[1], server_obs[2]])
            actions, action_dict = policy.get_action(server_obs)
            interventions.append(action_dict['intervene'])
            new_obs, reward, done, info = env.step(server_id, actions)
            ep_return += reward
            ep_cost += info['cost']
        ep_timestep += 1
    return ep_return, ep_cost
def evaluate(env, get_action, episodes: int):
    ep_returns = []
    ep_costs = []
    for episode in range(episodes):
        ep_return, ep_cost = run_eval_episode(
            env,
            get_action
        )
        ep_returns.append(ep_return)
        ep_costs.append(ep_cost)
    return np.mean(ep_returns), np.mean(ep_costs)"""

def run_desta(
        train_env,
        eval_env,
        episodes,
        policy,
        algorithm: str,
        seed: int):
    train_env.seed(seed)
    train_env.action_space.seed(seed)
    train_env.observation_space.seed(seed)
    results_dir = f'results_2017_{cf.NB_NODES}_{cf.NB_SERVERS}/{algorithm}'
    ensure_directories_present(results_dir)
    total_episodes = episodes
    n_steps_learn = FLAGS.batch_size
    steps_one_episode=FLAGS.max_rounds #300
    n_steps=1

    ep_rewards = []
    ep_costs = []
    ep_violations = []

    smooth_rewards = []
    smooth_costs = []
    smooth_violations = []
    for i in range(total_episodes):
        train_env.reset() #reset network的初始化
        trajectory = dict(
            zip(list(range(cf.NB_NODES, cf.NB_NODES + cf.NB_SERVERS)), [[] for _ in range(cf.NB_SERVERS)]))
        done = False
        last_done = True
        ep_reward = 0
        ep_cost = 0
        ep_violation = 0
        ep_timestep = 0
        while not done and ep_timestep<steps_one_episode:
            train_env.go_before_step(ep_timestep)
            server_ids = range(cf.NB_NODES, cf.NB_SERVERS + cf.NB_NODES)
            for server_id in server_ids:
                server_obs = train_env.get_state(server_id)
                trajectory[server_id].append([server_obs[0], server_obs[1], server_obs[2]])
                server_idle_state=train_env.get_server_state(ep_timestep,server_id)
                if not server_idle_state:
                    #actions = np.array([0., -1.])
                    #action_dict = {'intervene': False, 'intervene_action': np.array([-1.], dtype=np.float32),
                                   #'safe_action': np.array([0., -1.], dtype=np.float32)}
                    new_obs, reward, done, info = train_env.step_no_idle(ep_timestep,server_id)
                else:
                    actions, action_dict = policy.get_action(server_obs)
                    new_obs, reward, done, info = train_env.step(server_id, actions)
                    policy.store_transition(server_obs, actions, new_obs, reward, done, last_done, info, action_dict)
                obs = new_obs
                last_done = done
                ep_reward += reward
                ep_cost += info['cost']
                ep_violation += info['cost'] > 0
            if n_steps >= n_steps_learn: #and i%2==0:
                current_progress_remaining = 1 - i / total_episodes
                if algorithm == 'desta_sac_double_critic':
                    policy.learn(obs, done, action_dict['intervene'], current_progress_remaining, i)
                else:
                    policy.learn(obs, done, current_progress_remaining, i)
                n_steps=0
            n_steps += 1
            ep_timestep += 1
        #metric_result_line = train_env.record_results(ep_timestep)
        if i >= 400 and i %20==0:
            _save_trajectory(trajectory, train_env)
        ep_violations.append(ep_violation)
        ep_rewards.append(ep_reward)
        ep_costs.append(ep_cost)
        smooth_rewards.append(np.mean(ep_rewards[-100:]))
        smooth_costs.append(np.mean(ep_costs[-100:]))
        smooth_violations.append(np.mean(ep_violations[-100:]))
        train_env.env.record_result(i)
        print(f'Algorithm: {algorithm}, Seed: {seed}, Episode: {i}, Violations: {np.mean(ep_violations[-100:])}, Cost: {np.mean(ep_costs[-100:])}, Reward: {np.mean(ep_rewards[-100:])}')
    with open(os.path.join(results_dir, f'{algorithm}_{seed}.txt'), 'w', newline='') as file:
        writer = csv.writer(file)
        assert len(smooth_rewards) == len(smooth_costs), \
            "Lengths of smooth_rewards, smooth_costs, and steps_per_episode must match"
        for i, row in enumerate(train_env.env.record_metrics):
            combined_row = [ep_rewards[i], smooth_rewards[i], smooth_costs[i]] + row
            writer.writerow(combined_row)
    return smooth_rewards, smooth_costs

def run_pg_baseline(
        env_fn,
        agent,
        episodes: int,
        seed: int,
        algorithm: str
    ):
    results_dir = f'results_2018_{cf.NB_NODES}_{cf.NB_SERVERS}/{algorithm}'
    ensure_directories_present(results_dir)
    ep_rewards,smooth_rewards, smooth_costs,metrics_record = \
        run_agent.run_polopt_agent(
            env_fn,
            os.path.join(results_dir, f'{algorithm}_{seed}'),
            agent=agent,
            min_episodes=episodes,
            seed=seed,

            algorithm=algorithm
        )
    with open(os.path.join(results_dir, f'{algorithm}_{seed}.txt'), 'w', newline='') as file:
        writer = csv.writer(file)
        assert len(smooth_rewards) == len(smooth_costs), \
            "Lengths of smooth_rewards, smooth_costs, and steps_per_episode must match"

        for i, row in enumerate(metrics_record):
            combined_row = [ep_rewards[i], smooth_rewards[i], smooth_costs[i]] + row
            writer.writerow(combined_row)
    return smooth_rewards[:episodes], smooth_costs[:episodes]


def run_desta_sac_non_resource(env_fn, episodes: int, seed: int):
    train_env = env_fn()
    eval_env = env_fn()
    # TODO 定义policy
    policy = ThreeModelFreeOffPolicy(environment=train_env,
                                     buffer_size=FLAGS.replay_buffer_size,
                                     batch_size=FLAGS.batch_size,
                                     learning_rate_standard_policy=FLAGS.pi_standard_lr,
                                     learning_rate_safe_policy=FLAGS.pi_safe_lr,
                                     learning_rate_intervention_policy=FLAGS.pi_intervene_lr,
                                     intervention_cost=FLAGS.intervention_cost,
                                     action_distance_threshold=FLAGS.action_dist_threshold,
                                     network=[32, 32],
                                     # alpha=3.,
                                     writer=writer
                                     )
    return run_desta(
        train_env,
        eval_env,
        episodes,
        policy,
        'run_desta_sac_non_resource',
        seed
    )

def run_desta_sac(env_fn, episodes: int, seed: int):
    train_env = env_fn()
    eval_env = env_fn()
    # TODO 定义policy
    # policy = ThreeModelFreeOffPolicy_JointAction(
    policy = ThreeModelFreeOffPolicy(environment=train_env,
        buffer_size=FLAGS.replay_buffer_size,
        batch_size=FLAGS.batch_size,
        learning_rate_standard_policy=FLAGS.pi_standard_lr,
        learning_rate_safe_policy=FLAGS.pi_safe_lr,
        learning_rate_intervention_policy=FLAGS.pi_intervene_lr,
        intervention_cost=FLAGS.intervention_cost,
        action_distance_threshold=FLAGS.action_dist_threshold,
        network=[64, 64],
        #alpha=3.,
        writer=writer
    )
    return run_desta(
        train_env,
        eval_env,
        episodes,
        policy,
        'desta_sac',
        seed
    )

def run_manual(env_fn, episodes: int, seed: int):
    train_env = env_fn()
    eval_env = env_fn()
    algorithm='manual'
    path_with_arrows = [
        [250, 250], [250, 200], [250, 150], [200, 150], [150, 150], [150, 200], [150, 250], [150, 300],
        [150, 350], [150, 400], [150, 450], [100, 450], [50, 450], [50, 400], [50, 350], [50, 300],
        [50, 250], [50, 200], [50, 150], [50, 100], [50, 50], [100, 50], [150, 50], [200, 50],
        [250, 50], [300, 50], [350, 50], [400, 50], [450, 50], [450, 100], [450, 150], [450, 200], [450, 250],
        [450, 300], [450, 350], [450, 400], [450, 450], [400, 450], [350, 450], [300, 450], [250, 450],
        [250, 400], [250, 350], [300, 350], [350, 350], [350, 300], [350, 250], [350, 200], [350, 150],
        [300, 150], [300, 200], [300, 250], [250, 250]
    ]
    results_dir = f'results_2018_{cf.NB_NODES}_{cf.NB_SERVERS}/{algorithm}'
    ensure_directories_present(results_dir)
    total_episodes = episodes
    steps_one_episode = FLAGS.max_rounds  # 300
    n_steps = 1

    ep_rewards = []
    ep_costs = []
    ep_violations = []

    smooth_rewards = []
    smooth_costs = []
    smooth_violations = []
    for i in range(total_episodes):
        train_env.reset()  # reset network的初始化
        trajectory = dict(
            zip(list(range(cf.NB_NODES, cf.NB_NODES + cf.NB_SERVERS)), [[] for _ in range(cf.NB_SERVERS)]))
        done = False
        ep_reward = 0
        ep_cost = 0
        ep_violation = 0
        ep_timestep = 0

        total_locations=dict(
            zip(
                range(cf.NB_NODES, cf.NB_NODES + cf.NB_SERVERS),
                [copy.deepcopy(path_with_arrows) for _ in range(cf.NB_SERVERS)]
            )
        )
        while not done and ep_timestep < steps_one_episode:
            train_env.go_before_step(ep_timestep)
            server_ids = range(cf.NB_NODES, cf.NB_SERVERS + cf.NB_NODES)
            for server_id in server_ids:
                server_obs = train_env.get_state(server_id)
                trajectory[server_id].append([server_obs[0], server_obs[1], server_obs[2]])
                server_idle_state = train_env.get_server_state(ep_timestep, server_id)
                if not server_idle_state:
                    actions = np.array([server_obs[0], server_obs[1]])
                else:
                    if not total_locations[server_id]:  # 如果路径被弹空了
                        total_locations[server_id] = copy.deepcopy(path_with_arrows)
                    if server_id % 2 == 0:
                        actions = total_locations[server_id].pop(-1)  # 从末尾弹出
                    else:
                        actions = total_locations[server_id].pop(0)
                new_obs, reward, done, info = train_env.manual_step(server_id, actions)
                obs = new_obs
                last_done = done
                ep_reward += reward
                ep_cost += info['cost']
                ep_violation += info['cost'] > 0
                n_steps = 0
            n_steps += 1
            ep_timestep += 1
        #_save_trajectory(trajectory, train_env)
        ep_violations.append(ep_violation)
        ep_rewards.append(ep_reward)
        ep_costs.append(ep_cost)
        smooth_rewards.append(np.mean(ep_rewards[-100:]))
        smooth_costs.append(np.mean(ep_costs[-100:]))
        smooth_violations.append(np.mean(ep_violations[-100:]))
        train_env.env.record_result(i)
        print(
            f'Algorithm: {algorithm}, Seed: {seed}, Episode: {i}, Violations: {np.mean(ep_violations[-100:])}, Cost: {np.mean(ep_costs[-100:])}, Reward: {np.mean(ep_rewards[-100:])}')
    with open(os.path.join(results_dir, f'{algorithm}_{seed}.txt'), 'w', newline='') as file:
        writer = csv.writer(file)
        assert len(smooth_rewards) == len(smooth_costs), \
            "Lengths of smooth_rewards, smooth_costs, and steps_per_episode must match"
        for i, row in enumerate(train_env.env.record_metrics):
            combined_row = [ep_rewards[i], smooth_rewards[i], smooth_costs[i]] + row
            writer.writerow(combined_row)
    return smooth_rewards, smooth_costs

def run_desta_sac_double_critic(env_fn, episodes: int, seed: int):
    train_env = env_fn()
    eval_env = env_fn()

    policy = TwoModelFreeOffPolicy(
        environment=train_env,
        buffer_size=FLAGS.replay_buffer_size,
        batch_size=FLAGS.batch_size,
        learning_rate_task_policy=FLAGS.pi_standard_lr,
        learning_rate_intervention_policy=FLAGS.pi_intervene_lr,
        intervention_cost=FLAGS.intervention_cost,
        action_distance_threshold=FLAGS.action_dist_threshold,
        network=[32, 32],
        writer=writer
    )

    return run_desta(
        train_env,
        eval_env,
        episodes,
        policy,
        'desta_sac_double_critic',
        seed
    )

def run_sac(env_fn, episodes: int, seed: int):
    train_env = env_fn()
    eval_env = env_fn()

    policy = SinglePlayerOffPolicy(
        env=train_env,
        buffer_size=FLAGS.replay_buffer_size,
        pi_1_lr=FLAGS.pi_standard_lr,
        network=[32, 32],
    )

    return run_desta(train_env, eval_env, episodes, policy, 'sac', seed)

def run_ppo(env_fn, episodes: int, seed: int):
    kwargs = dict(
        reward_penalized=False,
        objective_penalized=False,
        learn_penalty=False,
        penalty_param_loss=False  # Irrelevant in unconstrained
    )
    agent = PPOAgent(**kwargs)
    return run_pg_baseline(env_fn, agent, episodes, seed, 'ppo')

def run_trpo(env_fn, episodes: int, seed: int):
    kwargs = dict(
        reward_penalized=False,
        objective_penalized=False,
        learn_penalty=False,
        penalty_param_loss=False  # Irrelevant in unconstrained
    )
    agent = TRPOAgent(**kwargs)
    
    return run_pg_baseline(env_fn, agent, episodes, seed, 'trpo')

def run_cpo(env_fn, episodes: int, seed: int):
    return run_pg_baseline(env_fn, CPOAgent(), episodes, seed, 'cpo')
# 'ppo_lagrangian',
# 'trpo_lagrangian',
# 'desta_sac_double_critic',
# 'sac_lagrangian', # Too slow to run
#'desta_ss_ppo',
# 'desta_ss_trpo_lagrangian',

def main(_):
    seeds = FLAGS.num_seeds
    algos = [
        # 'manual',
        'desta_sac',
        #'desta_sac_non_resource',
        #'sac',
        #'trpo',
        #'cpo',
        # 'desta_sac_double_critic',
        #'ppo',
    ] if FLAGS.algorithm == 'ALL' else [FLAGS.algorithm]
    env_fn = lambda: Mobile_Comp_Env()
    
    results = {
        'smooth_rewards': {},
        'smooth_costs': {}
    }
    for seed in range(0,1):
        for algo in algos:
            run_fn = f'run_{algo}'
            # globals()[run_fn]调用所有名称为run_fn的自定义函数
            smooth_rewards, smooth_costs = \
                globals()[run_fn](env_fn, FLAGS.episodes, seed)
            results['smooth_rewards'][algo] = smooth_rewards
            results['smooth_costs'][algo] = smooth_costs
        
    def plot_for_all_algos(
        plot_name: str,
        results: np.ndarray,
        title: str,
        xlabel: str,
        ylabel: str
    ):
        plot_dir = './plots'
        ensure_directories_present(plot_dir)
        plt.figure()
        for algo in algos:
            plt.plot(results[algo], label=algo)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(frameon=True,edgecolor="black",fontsize='large')
        plt.subplots_adjust(right=0.8)
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, f'{plot_name}_test.png'))
        #plt.show()

    plot_for_all_algos(
        'rewards',
        results['smooth_rewards'],
        'Rewards (100 Episode Average)',
        'Episode',
        'Reward'
    )

    plot_for_all_algos(
        'costs',
        results['smooth_costs'],
        'Costs (100 Episode Average)',
        'Episode',
        'Cost'
    )

if __name__ == '__main__':
    app.run(main)


