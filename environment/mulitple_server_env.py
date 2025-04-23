import math
import numpy as np
from . import config as cf
from .env_wrapper import Env
import gym
import logging
from gym import spaces

class Environment(gym.Env):
    def __init__(self):
        logging.info('Initializing environment with gym...')
        self.env = Env()
        self.agents_n = cf.NB_SERVERS
        self.max_rounds = cf.MAX_ROUNDS
        self.done=False

        low = np.array([-np.pi/4, 0], dtype=np.float32)  # 低边界
        high = np.array([np.pi/4, cf.D_MAX], dtype=np.float32)  # 高边界
        self.action_space = spaces.Box(low=low, high=high,shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(
            11,), dtype=np.float32) #pos_x,pos_y,move_azimuth,remain_energy,resource_type1,esource_type2, load_rate_cpu,load_rate_memory
        self.state_dim = 11  #2表示全部server的load的平均：sum(load_cpu)/nb_server,sum(load_mem)/nb_server
        self.action_dim = 2  #


    def seed(self, seed=cf.GLOBAL_SEED):
        np.random.seed(seed)

    def reset(self):
        self.seed(42)
        self.round_nb = -1
        self.done=False
        self.env.reset()

    def get_state(self,server_id):
        server = self.env.get_server(server_id)
        server_obs = []
        if server.is_alive():
            server_obs.extend([server.pos_x, server.pos_y, server.move_azimuth*100,server.total_amount_move/100])
            server_obs.extend(server.instance_types)
            server_cpu_loads=server.total_cpu_utility_per_slot[-1]
            server_memory_loads=server.total_memory_utility_per_slot[-1]
            server_obs.extend([cpu*100for cpu in server_cpu_loads])
            server_obs.extend([mem*100 for mem in server_memory_loads])
            server_obs.append(len(server.waiting_tasks_queue))
        else:
            server_obs.extend([-1,-1,-1,-1, -1, -1, -1,-1,-1,-1,-1])
        return np.array(server_obs)

    def go_before_step(self,ep_timestep):
        self.round_nb=ep_timestep
        self.env.before_step(self.round_nb)

    def get_server_state(self,ep_timestep,server_id):
        return self.env.get_server_idle_state(ep_timestep,server_id)

    def perform_computing(self,ep_timestep,server_id):
        self.env.only_perform_computing(ep_timestep,server_id)

    def step_non_resource(self,server_id,actions):
        before_step_metraces = self.env.server_dealt_change(server_id)
        self.env.step_action_non_resource(self.round_nb, actions, server_id)
        after_step_metraces = self.env.server_dealt_change(server_id)
        panaty_boundry = self.env.get_panaty_boundry(server_id)
        return self.get_state(server_id), self.get_reward(server_id, before_step_metraces, after_step_metraces,
                panaty_boundry), self.has_done(self.round_nb), self.get_info(server_id, self.round_nb)

    def step_no_idle(self,ep_timestep,server_id):
        before_step_metraces=self.env.server_dealt_change(server_id)
        self.perform_computing(ep_timestep,server_id)
        after_step_metraces=self.env.server_dealt_change(server_id)
        panaty_boundry=self.env.get_panaty_boundry(server_id)
        return self.get_state(server_id),self.get_reward(server_id,before_step_metraces,after_step_metraces,panaty_boundry),self.has_done(self.round_nb),self.get_info(server_id,self.round_nb)

    def step(self,server_id,actions):
        before_step_metraces=self.env.server_dealt_change(server_id)
        self.env.step_action(self.round_nb,actions,server_id)
        after_step_metraces=self.env.server_dealt_change(server_id)
        panaty_boundry=self.env.get_panaty_boundry(server_id)
        return self.get_state(server_id),self.get_reward(server_id,before_step_metraces,after_step_metraces,panaty_boundry),self.has_done(self.round_nb),self.get_info(server_id,self.round_nb)

    def manual_step(self,server_id,actions):
        before_step_metraces=self.env.server_dealt_change(server_id)
        self.env.manual_step_action(self.round_nb,actions,server_id)
        after_step_metraces=self.env.server_dealt_change(server_id)
        panaty_boundry=self.env.get_panaty_boundry(server_id)
        return self.get_state(server_id),self.get_reward(server_id,before_step_metraces,after_step_metraces,panaty_boundry),self.has_done(self.round_nb),self.get_info(server_id,self.round_nb)

    def get_reward(self,server_id,before_step_metraces,after_step_metraces,panaty_boundry):
        #[move_energy,average_device_consumption,new_upload_request,completion_tasks,uncompletion_tasks,completion_jobs,uncompletion_jobs]
        delta_move_energy = after_step_metraces[0] - before_step_metraces[0]
        ave_delta_device_energy = after_step_metraces[1]-before_step_metraces[1]
        delta_access_requests=after_step_metraces[2]-before_step_metraces[2]
        delte_completion_tasks=after_step_metraces[3]-before_step_metraces[3]
        delte_uncompletion_tasks=after_step_metraces[4]-before_step_metraces[4]
        delte_completion_jobs=after_step_metraces[5]-before_step_metraces[5]
        delta_uncompletion_jobs=after_step_metraces[6]-before_step_metraces[6]

        uav_energy_cost=delta_move_energy/(79.86*(1+3*cf.D_MAX**2/14400)+88.63*np.sqrt(np.sqrt(1+cf.D_MAX**4/1055.08)-cf.D_MAX**2/32.48)+0.0092*cf.D_MAX**3+1e-8)
        #panaty_access_request=0.5 if delta_access_requests==0 else 0
        device_energy_cost=ave_delta_device_energy / (cf.POWER_MAX + 1e-8)
        completion_task_rate=delte_completion_tasks /(delte_completion_tasks+delte_uncompletion_tasks+1e-8)
        #uncompletion_job_rate=delta_uncompletion_jobs/(delte_completion_jobs+delta_uncompletion_jobs+1e-8)

        server_total_miss_job=self.env.get_miss_job_server(server_id)
        uncompletion_job_rate=server_total_miss_job/(self.env.amount_requests_receive+1e-8)
        delta_access_requests_rate=delta_access_requests/(delta_access_requests+(self.env.amount_requests_receive-self.env.before_request)+1e-8)
        if delta_access_requests > 0:
            reward = delta_access_requests_rate
        else:
            reward=-(0.40 * uncompletion_job_rate + 0.3 * uav_energy_cost + 0.3 * ave_delta_device_energy)

        """if delta_access_requests > 0:
            reward = completion_task_rate-device_energy_cost#-uav_energy_cost*0.5+completion_task_rate
        else:
            reward = completion_task_rate-uav_energy_cost#delta_access_requests/(delta_access_requests+delte_uncompletion_tasks+1e-8)-uav_energy_cost#completion_task_rate"""

        return reward

    def has_done(self,round_nb):
        msg = ''
        if round_nb + 1 == self.max_rounds:
            msg += 'Finished with max round'
        """elif len(self.env.get_alive_servers()) == 0:
            msg += 'Finished with no charger alive'"""
        if len(msg) > 1:
            self.done = True
        return self.done

    def get_info(self,serve_id,round_nb):
        server_collisions = []
        current_server_location = self.get_state(serve_id)[:2]
        # 获取所有服务器的位置，避免重复调用 get_state
        all_server_positions = {
            server_index: self.get_state(server_index)[:2]
            for server_index in range(cf.NB_NODES, cf.NB_NODES + cf.NB_SERVERS)
        }
        for server_index, server_location in all_server_positions.items():
            if server_index == serve_id:
                continue
            distance = self._get_distance(current_server_location, server_location)
            if distance <= cf.SAFE_DISTANCE:
                server_collisions.append(cf.SAFE_DISTANCE - distance)
        for obstacle in self.env.obstacles_location:
            distance = self._get_distance(current_server_location, obstacle[:2])
            if distance <= cf.SAFE_DISTANCE_OBS:
                server_collisions.append(cf.SAFE_DISTANCE_OBS - distance)

        # 计算成本
        cost_coll = np.sum(server_collisions) * 0.01
        #total_requests = self.env.amount_requests_receive + 1e-8  # 避免除以零
        #cost_job_miss_rate = self.env.get_missed_job() / total_requests

        info = {'cost': cost_coll} #+ cost_job_miss_rate}
        return info

    def _get_distance(self, point1, point2):
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1] ) ** 2)

    def record_results(self,round_nb):
        self.env.record_result(round_nb)

    def plot_uavs_trjectory(self,trajectory):
        self.env.last_uavs_trajectory(trajectory)

#env_fn = Environment()