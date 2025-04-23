import logging
import numpy as np
import random
import os
import pandas as pd
from collections import defaultdict
from . import config as cf
from .end_device import EndDevice
from .mobile_server import Mobile_Server
from .obstacle_setting import Obstacles
from .alibaba_2018_trace.job_csv_reader import CSVReader as Get_Jobs
user_file_path_static = "D:/Code/MEC_Desta_MASAC/desta_masac/environment/UE_location_Alibaba_data/Static_UE_locations_20.csv"
job_file_path = 'D:/Code/MEC_Desta_MASAC/desta_masac/environment/UE_location_Alibaba_data/alibaba2017_V2.csv' #alibaba2018_duration200.csv
class Network(list):
    def __init__(self):
        logging.debug('Initializing network...')
        self.round_time=0
        if os.path.splitext(os.path.basename(user_file_path_static))[0]=='Static_UE_locations_20':
            self.mobile=False
        else:
            self.mobile=True
        edge_nodes = [EndDevice(self.mobile,user_file_path_static,i, self) for i in range(0, cf.NB_NODES)]
        self.extend(edge_nodes)
        servers = [Mobile_Server(i, self) for i in range(cf.NB_NODES, cf.NB_NODES + cf.NB_SERVERS)]
        self.extend(servers)
        self._dict = {}
        for i in self:
            self._dict[i.id] = i
        self.obstacles = np.array([
        (282.05, 364.13), (182.25, 443.65),
        (246.05, 358.31), (240.13, 180.58),
        (420.55, 249.15)
        ])#Obstacles().static_obstacles
        self.seed=42


    def reset(self):
        self.seed = 42
        for edgenode in self:
            edgenode.reactivate()
        for server in self[cf.NB_NODES : cf.NB_NODES + cf.NB_SERVERS]:
            server.reactivate()

        self.total_jobs = Get_Jobs(job_file_path).get_total_jobs()  # 得到所有的job 对象
        self.jobs_per_time = []
        self.job_workflow_arrival_model(len(self.total_jobs))

        self.round_time = 0
        self.amount_requests_receive = 0
        self.requests_list = []

        self.missed_job_in_pool = 0
        self.missed_task_in_pool = 0

        self.send_jobs_each_timeslot=[]  #len(self.send_jobs_each_timeslot)值应该和self.amount_requests_receive 的值相等
        self.send_tasks_each_timeslot=[]

    #从这个逻辑开始整合network网络
    def get_edge_nodes(self):
        return [node for node in self[0: cf.NB_NODES]]

    def update_nodes_location(self,nb_round):
        for node in self.get_edge_nodes():
            node._update_location(nb_round)

    def get_server(self, server_id):
        return self[server_id]

    def get_device(self, device_id):
        return self[device_id]

    def get_servers(self):
        return [server for server in self[cf.NB_NODES: cf.NB_NODES + cf.NB_SERVERS]]

    def get_energy_move(self,server):
        return server.total_amount_move

    def get_energy_serve(self,server):
        return server.total_amount_serve

    def get_completion_tasks(self,server):
        return server.amount_completed_tasks

    def get_completion_jobs(self,server):
        return server.amount_completed_jobs

    def get_uncompletion_tasks(self,server):
        return server.amount_tot_miss_tasks+self.missed_task_in_pool#+self.get_remain_tasks_from_request()

    def get_uncompletion_jobs(self,server):
        return server.amount_tot_miss_jobs+self.missed_job_in_pool

    def get_waiting_taks(self,server):
        return len(server.waiting_tasks_queue)

    def _update_devices(self, round_nb):
        self.update_request_pool(round_nb)
        self._send_request_phase(round_nb)

    def update_request_pool(self,round_nb):
        del_job_in_pool=[]
        for request in self.requests_list[:]:  # 使用切片复制列表以安全地进行迭代
            missed_job=False
            for task in request.jobconfig.task_configs:
                if (round_nb - request.job_sub_time) >= task.duration:
                    missed_job=True
                    break
            if missed_job:
                self.missed_task_in_pool+=request.numtasks
                del_job_in_pool.append(request)

        for request in del_job_in_pool:
            self.requests_list.remove(request)
            self.missed_job_in_pool += 1

    def job_workflow_arrival_model(self,job_num):
        TOTAL_JOBS = job_num#len(self.total_jobs)  # 总Job数量
        TOTAL_TIME = 300#600  # 总时间（每分钟一个时刻）
        DISTRIBUTION = 'poisson'  # 可选: 'gaussian', 'poisson', 'uniform'

        # Step 1: 生成每个时刻的Job数量
        np.random.seed(self.seed)  # 保证可复现
        if DISTRIBUTION == 'gaussian':
            mean_jobs = TOTAL_JOBS / TOTAL_TIME  # 每个时刻的平均Job数量
            std_dev = mean_jobs * 0.5  # 标准差
            self.jobs_per_time = np.random.normal(loc=mean_jobs, scale=std_dev, size=TOTAL_TIME)
            self.jobs_per_time = np.clip(self.jobs_per_time, 0, None)  # 确保没有负数

        elif DISTRIBUTION == 'poisson':
            lambda_jobs = TOTAL_JOBS / TOTAL_TIME  # 泊松分布的λ参数
            self.jobs_per_time = np.random.poisson(lam=lambda_jobs, size=TOTAL_TIME)

        elif DISTRIBUTION == 'uniform':
            self.jobs_per_time = np.random.uniform(low=5, high=20, size=TOTAL_TIME)  # 均匀分布
            self.jobs_per_time = self.jobs_per_time.astype(int)

        # Step 2: 调整总数以匹配 TOTAL_JOBS
        self.jobs_per_time = (self.jobs_per_time / self.jobs_per_time.sum()) * TOTAL_JOBS
        self.jobs_per_time = np.round(self.jobs_per_time).astype(int)

        # 调整误差
        while self.jobs_per_time.sum() != TOTAL_JOBS:
            diff = TOTAL_JOBS - self.jobs_per_time.sum()
            adjustment_index = np.random.randint(0, TOTAL_TIME)
            self.jobs_per_time[adjustment_index] += np.sign(diff)

    def _send_request_phase(self,round_nb):
        if self.mobile:
            self.update_nodes_location(round_nb)
        number_requests = self.jobs_per_time[round_nb]
        self.send_jobs_each_timeslot.append(number_requests)

        popped_jobs = self.total_jobs[:number_requests]  # 取前 number_requests 个元素
        self.total_jobs = self.total_jobs[number_requests:]
        self.send_tasks_each_timeslot.append(np.sum([job.num_tasks for job in popped_jobs]))
        edge_devices = self.get_edge_nodes()

        for job in popped_jobs:
            chosen_device = random.choice(edge_devices)  # 随机选择一个设备
            chosen_device.send_request(round_nb, job)

        """if round_nb<270:
            number_requests=self.jobs_per_time[round_nb]
            self.send_jobs_each_timeslot.append(number_requests)
            if number_requests > len(self.total_jobs):
                print("Warning: 请求的数量超过了剩余的 Job 数量，返回所有剩余的 Job。")
                number_requests = len(self.total_jobs)
            popped_jobs = self.total_jobs[:number_requests]  # 取前 number_requests 个元素
            self.total_jobs = self.total_jobs[number_requests:]

            self.send_tasks_each_timeslot.append(np.sum([job.num_tasks for job in popped_jobs]))
            edge_devices = self.get_edge_nodes()

            device_jobs_map = defaultdict(list)
            # 随机分配任务给设备
            for job in popped_jobs:
                chosen_device = random.choice(edge_devices)  # 随机选择一个设备
                device_jobs_map[chosen_device].append(job)

            # 每个设备调用 send_request 方法多次发送任务
            for device, jobs in device_jobs_map.items():
                for job in jobs:
                    device.send_request(round_nb, job)"""

    def _get_panaty_boundry(self,server_id):
        server = self.get_server(server_id)
        if server.pos_x==cf.INITIAL_X or server.pos_x==cf.AREA_WIDTH \
                or server.pos_y==cf.INITIAL_X or server.pos_y==cf.AREA_LENGTH:
            panaty=-0.5
        else:
            panaty=0
        return panaty

    def _update_server(self, round_nb, server_action,server_id):
        server=self.get_server(server_id)
        server.update(round_nb, server_action)

    def _manual_update_server(self, round_nb, server_action,server_id):
        server = self.get_server(server_id)
        server.manual_update(round_nb, server_action)

    def _update_server_non_resource(self,round_nb, server_action,server_id):
        server = self.get_server(server_id)
        server.update_non_resource(round_nb, server_action)

    def _only_perform_computing(self,ep_timestep,server_id):
        server = self.get_server(server_id)
        server.only_computing(ep_timestep)

    def _get_server_idle_state(self,round_nb,server_id):
        server = self.get_server(server_id)
        server.if_idle(round_nb)
        return server.idle_state

    """def get_alive_servers(self):
        return [server for server in self[cf.NB_NODES : cf.NB_NODES + cf.NB_SERVERS] if server.is_alive()]"""

    def get_missed_job(self):
        total_miss_jobs=0
        for server in self.get_servers():
            total_miss_jobs+=server.amount_tot_miss_jobs
        return self.missed_job_in_pool+total_miss_jobs

    def get_missed_task(self):
        total_miss_tasks=0
        for server in self.get_servers():
            total_miss_tasks+=server.amount_tot_miss_tasks
        return self.missed_task_in_pool+total_miss_tasks