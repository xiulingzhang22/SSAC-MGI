from collections import namedtuple
from typing import List, Dict
from .import config as cf
import random
from datetime import datetime
import math
import pandas as pd
Request = namedtuple("Request", "id location job_sub_time instance_type jobconfig numtasks amount_upload")

class EndDevice:
    def __init__(self,is_mobile,user_file_path,node_id,parent=None):
        self.filename=user_file_path
        self.round=0
        self.id=node_id #根据id 找到该用户的所有移动位置，并按照时间戳排序
        self.trajectory=[]
        self.load_trajectory_for_user(is_mobile)  #获得用户的轨迹（按照时间戳排序的轨迹）
        self.network_handler = parent
        self.job_queue = []
        self.task_queue=[]
        self.reactivate()

    def load_trajectory_for_user(self,is_mobile):
        data = pd.read_csv(self.filename)
        if is_mobile:
            # 取当前用户的 300 个轨迹点
            start_idx = self.id * cf.MAX_ROUNDS
            end_idx = (self.id + 1) * cf.MAX_ROUNDS
            user_data = data.iloc[start_idx:end_idx]
        else:
            user_data = data.iloc[[self.id]]  # 用双中括号保持 DataFrame 格式
        self.trajectory = list(zip(user_data["X"], user_data["Y"]))

    def reactivate(self):
        # 初始坐标
        self.round = 0
        self.pos_x = self.trajectory[0][0]
        self.pos_y = self.trajectory[0][1]
        self.energy_upload=0

    def _update_location(self,nb_round):
        self.pos_x=self.trajectory[nb_round][0]
        self.pos_y=self.trajectory[nb_round][1]

    def send_request(self,nb_round,jobconfig):  # 每次调用send_request时计数器加1
        amount_upload= cf.UPLOAD_DATA_TH#random.sample(1,5) #[1,5]Mbits
        jobconfig.submit_time=nb_round
        request = Request(
            self.id,
            (self.pos_x, self.pos_y),
            nb_round,
            jobconfig.instance_type,
            jobconfig,
            jobconfig.num_tasks,
            amount_upload
        )
        self.job_queue.append(jobconfig)
        self.task_queue.extend([task for task in jobconfig.task_configs])
        self.network_handler.requests_list.append(request)
        self.network_handler.amount_requests_receive += 1

    def _get_distance(self, point1, point2):
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

#file_path = 'D:/Code/MEC_Desta_MASAC/desta_masac/environment/User_data_Twitter/4interpolate_user_movements.csv'  # File path redefined due to environment reset
#device = EndDevice(filename=file_path, node_id=0)

