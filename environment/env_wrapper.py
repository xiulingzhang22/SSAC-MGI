import logging
import numpy as np
from . import config as cf
from .network import Network
import matplotlib.cm as cm
import pandas as pd
from matplotlib import pyplot as plt

class Env(Network):
    def __init__(self):
        logging.info(f'Initializing environment')
        super(Env, self).__init__()   #初始化network
        self.max_rounds = cf.MAX_ROUNDS
        self.record_metrics = []
        self.obstacles_location=None
        self.init_node_pos()

    def init_node_pos(self):
        pos_x, pos_y = [], []
        for node in self.get_edge_nodes():  # 获取生成的节点位置坐标
            pos_x.append(node.pos_x)
            pos_y.append(node.pos_y)
        plt.figure()
        plt.scatter(pos_x, pos_y, alpha=0.6, label='Nodes')
        self.obstacles_location = self.obstacles
        obs_x = [obs[0] for obs in self.obstacles_location]
        obs_y = [obs[1] for obs in self.obstacles_location]
        plt.scatter(obs_x, obs_y, label='Obstacles', c='red',marker="^")
        # 添加图例和标签
        plt.xlim([cf.INITIAL_X, cf.AREA_WIDTH])
        plt.ylim([cf.INITIAL_Y, cf.AREA_LENGTH])
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"user_map_{cf.NB_NODES}.png")

    def last_uavs_trajectory(self, trajectory):
        # 设置画布大小
        fig, ax = plt.subplots(figsize=(12, 10))

        ## 🎯 **1. 绘制 UAV 轨迹**
        colormap = cm.get_cmap('tab10')  # 使用 matplotlib 的 'tab10' 调色板
        for i, (key, points) in enumerate(trajectory.items()):
            x_vals = [p[0] for p in points]
            y_vals = [p[1] for p in points]

            # 绘制轨迹
            ax.plot(
                x_vals, y_vals,
                marker='o',
                label=f'Trajectory {key}',
                color=colormap(i % 10)
            )

            # 🔸 保存轨迹到 CSV 文件
            df = pd.DataFrame({'x': x_vals, 'y': y_vals})
            csv_filename = f'Trajectory_{key}.csv'
            df.to_csv(csv_filename, index=False)

        ## 🧍 **2. 绘制用户位置**
        pos_x, pos_y = [], []
        for node in self.get_edge_nodes():  # 获取生成的节点位置坐标
            pos_x.append(node.pos_x)
            pos_y.append(node.pos_y)
        ax.scatter(
            pos_x, pos_y,
            alpha=0.8,
            label='Users',
            c='blue',
            marker='s',
            s=100,
            edgecolor='black'
        )  # 用户用蓝色方形标记

        ## 🚧 **3. 绘制障碍物位置**
        self.obstacles_location = self.obstacles
        obs_x = [obs[0] for obs in self.obstacles_location]
        obs_y = [obs[1] for obs in self.obstacles_location]
        ax.scatter(
            obs_x, obs_y,
            label='Obstacles',
            c='red',
            marker='^',
            s=160,
            edgecolor='black'
        )  # 障碍物用红色三角形标记

        ## 📝 **4. 图表设置**
        ax.legend()
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.grid(True)

        ## 📥 **5. 保存图像**
        plt.savefig('UAV_trajectories_with_users_and_obstacles_colored.png', dpi=300)
        plt.show()

    def reset(self):
        super().reset()
        self.done = False
        self.before_request=0

    def before_step(self,round_nb):
        self.before_request=self.amount_requests_receive
        self._update_devices(round_nb)
        self.total_miss_tasks=self.get_missed_task()
        self.total_miss_jobs =self.get_missed_job()

    def server_dealt_change(self,server_id):
        server=self.get_server(server_id)
        move_energy=self.get_energy_move(server)
        average_device_consumption=0 if len(server.device_power_comsumption)==0 else np.average(server.device_power_comsumption)
        new_upload_request= server.access_request_num
        #serve_energy=self.get_energy_serve(server)
        completion_tasks=self.get_completion_tasks(server)
        uncompletion_tasks=self.get_uncompletion_tasks(server)
        #waiting_tasks=self.get_waiting_taks(server)
        completion_jobs = self.get_completion_jobs(server)
        uncompletion_jobs=self.get_uncompletion_jobs(server)
        return [move_energy,average_device_consumption,new_upload_request,completion_tasks,uncompletion_tasks,completion_jobs,uncompletion_jobs] #serve_energy,

    def get_panaty_boundry(self,server_id):
        return self._get_panaty_boundry(server_id)

    def step_action(self,round_nb,action,server_id):
        self._update_server(round_nb,action,server_id)

    def manual_step_action(self,round_nb,action,server_id):
        self._manual_update_server(round_nb, action, server_id)

    def step_action_non_resource(self,round_nb,action,server_id):
        self._update_server_non_resource(round_nb,action,server_id)

    def only_perform_computing(self,ep_timestep,server_id):
        self._only_perform_computing(ep_timestep,server_id)

    def get_server_idle_state(self,round_nb,server_id):
        server_state=self._get_server_idle_state(round_nb,server_id)
        return server_state

    def record_result(self,episode):
        total_completion_jobs,total_uncompletion_jobs = self.get_total_jobs()  # completion rate of taks
        total_completion_tasks,total_uncompletion_tasks = self.get_total_tasks()
        average_move_energy,average_devices_energy=self.average_energy_utility()

        #average_completion_time_task=self.average_completion_time_task()
        #average_task_slowdown=self.average_slowdown()
        totaol_access_requests=self.get_access_request()
        average_server_cpu_utility,average_server_memory_utility=self.average_resources_utility()
        result_line = [total_completion_jobs,total_uncompletion_jobs,totaol_access_requests,self.amount_requests_receive,total_uncompletion_jobs/self.amount_requests_receive,
                      average_move_energy,average_devices_energy,average_server_cpu_utility,average_server_memory_utility]
        self.record_metrics.append(result_line)
        #return result_line

    def get_miss_job_server(self,server_id):
        server=self.get_server(server_id)
        return (server.amount_tot_miss_jobs+self.missed_job_in_pool)

    def get_total_jobs(self):
        total_completion_jobs=0
        total_uncompletion_jobs = 0
        serving_job_list_uavs=[]
        for server in self.get_servers():
            serving_job_list_uavs.append(len(server.serving_job_list))
            total_completion_jobs+=server.amount_completed_jobs
            total_uncompletion_jobs+=server.amount_tot_miss_jobs
        return total_completion_jobs,total_uncompletion_jobs+self.missed_job_in_pool #+len(self.requests_list)+np.sum(serving_job_list_uavs)

    def get_total_tasks(self):
        total_completion_tasks=0
        total_uncompletion_tasks = 0
        for server in self.get_servers():
            total_completion_tasks+=server.amount_completed_tasks
            total_uncompletion_tasks+=server.amount_tot_miss_tasks
        return total_completion_tasks,total_uncompletion_tasks+self.missed_task_in_pool+self.get_remain_tasks_from_request()

    def get_remain_tasks_from_request(self):
        remain_taks=0
        for request in self.requests_list:
            remain_taks+=request.numtasks
        return remain_taks

    def average_energy_utility(self):
        total_move_energy=[]
        total_devices_energy=[]
        for server in self.get_servers():
            total_move_energy.append(server.total_amount_move)
            total_devices_energy.extend(server.device_power_comsumption)
        return np.average(total_move_energy)/382500,np.average(total_devices_energy) #average uav, device during T

    def average_completion_time_task(self):
        total_completion_time = 0
        total_completion_tasks = 0
        for server in self.get_servers():
            total_completion_time += server.total_tasks_time
            total_completion_tasks += server.amount_completed_tasks
        return total_completion_time/(total_completion_tasks+1e-8)

    def average_slowdown(self):
        total_slowdown = []
        for server in self.get_servers():
            total_slowdown.extend(server.total_slowdown)
        if len(total_slowdown)!=0:
            average_slowdown = np.average(total_slowdown)
        else :
            average_slowdown = 0
        return average_slowdown

    def get_access_request(self):
        total_access_request=[]
        for server in self.get_servers():
            total_access_request.append(server.access_request_num)
        return np.sum(total_access_request)

    def average_resources_utility(self):
        average_cpu_utility = []
        average_memory_utility = []

        for server in self.get_servers():
            data_array_cpu = np.array(server.total_cpu_utility_per_slot[1:])

            non_zero_col1_cpu = data_array_cpu[:, 0][data_array_cpu[:, 0] != 0]
            non_zero_col2_cpu = data_array_cpu[:, 1][data_array_cpu[:, 1] != 0]

            non_zero_mean_col1_cpu = np.mean(non_zero_col1_cpu)
            non_zero_mean_col2_cpu = np.mean(non_zero_col2_cpu)

            average_cpu_utility.extend((non_zero_mean_col1_cpu,non_zero_mean_col2_cpu))  # average slot

            data_array_mem = np.array(server.total_memory_utility_per_slot[1:])
            non_zero_col1_mem = data_array_mem[:, 0][data_array_mem[:, 0] != 0]
            non_zero_col2_mem = data_array_mem[:, 1][data_array_mem[:, 1] != 0]

            non_zero_mean_col1_mem = np.mean(non_zero_col1_mem)
            non_zero_mean_col2_mem = np.mean(non_zero_col2_mem)

            average_memory_utility.extend((non_zero_mean_col1_mem,non_zero_mean_col2_mem))  # average slot

        # 计算平均值，如果列表为空则返回0
        average_server_cpu_utility = np.average(average_cpu_utility) if len(average_cpu_utility)!=0 else 0
        average_server_memory_utility = np.average(average_memory_utility) if len(average_memory_utility)!=0 else 0
        return average_server_cpu_utility, average_server_memory_utility  # average time slot resources utility


    def get_finished_job_rate(self):
        finished_jobs=len(self.server.finished_jobs)  #
        finished_undelay_jobs=self.server.amount_completed_jobs
        return finished_undelay_jobs/(finished_jobs+self.missed_job_in_pool+1)

    def get_finished_task_rate(self):
        finished_undelay_tasks=self.server.amount_completed_tasks
        return finished_undelay_tasks/(self.server.amount_tot_miss_tasks+self.missed_task_in_pool+1)
