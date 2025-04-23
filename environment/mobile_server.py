import numpy as np
from .import config as cf
import math

class Mobile_Server:
    def __init__(self, id, parent=None):
        self.id = id
        self.network_handler = parent
        self.reactivate()

    def reactivate(self):
        self.pos_x = cf.INITIAL_X+cf.AREA_WIDTH/2  #np.random.uniform(0, cf.AREA_WIDTH)
        self.pos_y = cf.INITIAL_Y+cf.AREA_LENGTH/2    # np.random.uniform(0, cf.AREA_LENGTH)
        self.move_azimuth=cf.INITIAL_AZIMUTH
        self.high=cf.HOVER_HIGH
        self.location = [(self.pos_x,self.pos_y,self.move_azimuth)]
        self.device_power_comsumption=[]
        # 初始化资源可用性状态，默认所有资源都是可用的
        self.resource=self.heterogeneous_resource()  #cpu/core 和memory
        self.remained_resouces=[]
        self.instance_types=list(self.resource.keys())  #[]得到当前server中的instance types flag= True if 1 in self.resource.keys() else Fale
        self.if_available={key: True for key in self.instance_types}
        self.energy= cf.INITIAL_ENERGY
        self.waiting_tasks_queue=[]
        self.alive = True
        self.serving_job_list = []
        self.finished_jobs = []
        self.access_request_num=0

        self.total_amount_move = 0
        self.total_amount_serve = 0

        self.amount_completed_tasks = 0  # 在延迟内满足服务请求deadline的task总数
        self.amount_completed_jobs = 0  # 在延迟内满足jobdedline的job总数
        self.amount_tot_miss_jobs = 0
        self.amount_tot_miss_tasks = 0

        self.total_cpu_utility_per_slot=[[0,0]]
        self.total_memory_utility_per_slot=[[0,0]]

        self.timeslot_access_number_onslot =0
        self.idle_state=True
        self.last_location_time=0

    def if_idle(self, round_nb):
        if self.timeslot_access_number_onslot>1:
            self.timeslot_access_number_onslot-=1
            self.idle_state=False
        else:
            self.idle_state = True

    def heterogeneous_resource(self):
        if self.id-cf.NB_NODES==0 or self.id-cf.NB_NODES==3 or self.id-cf.NB_NODES==6 : #每个SERVER异构包含两种instance类型
            resource = {  #c6a.2xlarge, m6a.12xlarge
                  3: [1856,2001],# 代表 2 GHz 的 cpu_capacity
                  8:  [3520,3795]# 代表 4 GB 的 memory_capacity
            }
        elif self.id-cf.NB_NODES==1 or self.id-cf.NB_NODES==4 or self.id-cf.NB_NODES==7:
            resource = { #c6a.4xlarge, m6a.8xlarge
                10: [2048, 2208],
                12: [3584, 3864]
            }
        elif self.id-cf.NB_NODES==2 or self.id-cf.NB_NODES==5 or self.id-cf.NB_NODES==8:
            resource = { #c6a.8xlarge, m6a.4xlarge
                10: [2048, 2208],
                3: [1856, 2001]
            }
        return resource

    def resource_load(self):
        resource_value=list(self.resource.values())
        if self.id-cf.NB_NODES==0 or self.id-cf.NB_NODES==3 or self.id-cf.NB_NODES==6: #每个SERVER异构包含两种instance类型
            cpu1_load=(1856-resource_value[0][0])/1856
            mem1_load=(2001-resource_value[0][1])/2001
            cpu2_load = (3520 - resource_value[1][0]) / 3520
            mem2_load = (3795 - resource_value[1][1]) / 3795
        elif self.id-cf.NB_NODES==1 or self.id-cf.NB_NODES==4 or self.id-cf.NB_NODES==7:
            cpu1_load = (2048 - resource_value[0][0]) / 2048
            mem1_load = (2208 - resource_value[0][1]) / 2208
            cpu2_load = (3584 - resource_value[1][0]) / 3584
            mem2_load = (3864 - resource_value[1][1]) / 3864
        elif self.id-cf.NB_NODES==2 or self.id-cf.NB_NODES==5 or self.id-cf.NB_NODES==8:
            cpu1_load = (2048 - resource_value[0][0]) / 2048
            mem1_load = (2208 - resource_value[0][1]) / 2208
            cpu2_load = (1856 - resource_value[1][0]) / 1856
            mem2_load = (2001 - resource_value[1][1]) / 2001
        resource_load=[cpu1_load,mem1_load,cpu2_load,mem2_load]
        return resource_load

    def is_alive(self):
        return self.alive

    def only_computing(self,round_nb):
        self.total_amount_move += 168.49
        new_timeslot_access_number = self.update_servering_list(round_nb)
        if new_timeslot_access_number>self.timeslot_access_number_onslot:
            self.timeslot_access_number_onslot=new_timeslot_access_number
        #考虑覆盖范围和单位时隙内的最大接入数量，实现用户关联，把request放入server的waiting queue
        if  len(self.serving_job_list)>0: #self.is_alive() and
            sorted_request = self.check_resort_task(round_nb) #得到所有的job的task队列（按照deadline排序）
            self.perform_computing_loop(sorted_request,round_nb) #分配instance运行task
        self.compute_resource_utility()
        self.resource=self.heterogeneous_resource() #重新释放资源
        self.if_available = {key: True for key in self.instance_types}

    def manual_move_next_location(self,move_location):
        delta_x = move_location[0]-self.pos_x
        delta_y= move_location[1]-self.pos_y
        speed=np.sqrt(delta_x**2+delta_y**2)
        self.pos_x = move_location[0]
        self.pos_y = move_location[1]
        if speed==0:
            move_energy=168.49
        else:
            move_energy=79.86*(1+3*speed**2/14400)+88.63*np.sqrt(np.sqrt(1+speed**4/1055.08)-speed**2/32.48)+0.0092*speed**3
        self.total_amount_move += move_energy#self.consume(move_energy)
        self.move_azimuth = 0
        self.location.append((self.pos_x,self.pos_y,self.move_azimuth))

    def manual_update(self, round_nb, action):
        self.last_location_time = round_nb
        self.manual_move_next_location(action)  # 验证：tot_length=speed
        if len(self.network_handler.requests_list) > 0:  # self.is_alive() and
            self.timeslot_access_number_onslot = self.update_servering_list(
                round_nb)  # 考虑覆盖范围和单位时隙内的最大接入数量，实现用户关联，把request放入server的waiting queue
        if len(self.serving_job_list) > 0:  # self.is_alive() and
            #sorted_request = self.check_resort_task(round_nb)  # 得到所有的job的task队列（按照deadline排序）
            #self.perform_computing_loop(sorted_request, round_nb)  # 分配instance运行task
            self.check_task_non_resource(round_nb)  # 得到所有的job的task队列（按照deadline排序）
            self.perform_computing_FIFO(round_nb)
        self.compute_resource_utility()
        self.resource = self.heterogeneous_resource()  # 重新释放资源
        self.if_available = {key: True for key in self.instance_types}

    def update(self, round_nb, action):
        self.last_location_time=round_nb
        move_action=self.action_mapping(action)
        self.move_next_location(move_action)  # 验证：tot_length=speed
        if len(self.network_handler.requests_list)>0: #self.is_alive() and
            self.timeslot_access_number_onslot=self.update_servering_list(round_nb)  #考虑覆盖范围和单位时隙内的最大接入数量，实现用户关联，把request放入server的waiting queue
        if len(self.serving_job_list)>0: #self.is_alive() and
            sorted_request = self.check_resort_task(round_nb) #得到所有的job的task队列（按照deadline排序）
            self.perform_computing_loop(sorted_request,round_nb) #分配instance运行task
        self.compute_resource_utility()
        self.resource=self.heterogeneous_resource() #重新释放资源
        self.if_available = {key: True for key in self.instance_types}

    def update_non_resource(self, round_nb, action):
        self.last_location_time = round_nb
        move_action = self.action_mapping(action)
        self.move_next_location(move_action)  # 验证：tot_length=speed
        if len(self.network_handler.requests_list) > 0:  # self.is_alive() and
            self.timeslot_access_number_onslot = self.update_servering_list(
                round_nb)  # 考虑覆盖范围和单位时隙内的最大接入数量，实现用户关联，把request放入server的waiting queue
        if len(self.serving_job_list) > 0:  # self.is_alive() and
            self.check_task_non_resource(round_nb)  # 得到所有的job的task队列（按照deadline排序）
            self.perform_computing_FIFO(round_nb)  # 分配instance运行task
        self.compute_resource_utility()
        self.resource = self.heterogeneous_resource()  # 重新释放资源
        self.if_available = {key: True for key in self.instance_types}

    def check_task_non_resource(self, round_nb):
        remove_jobs = []
        for request in self.serving_job_list:  # 1. 遍历所有的 Job，判断是否存在 Missed 任务
            waiting_time = round_nb - request.jobconfig.submit_time
            job_missed = False  # 标记当前 job 是否 missed

            for task in request.jobconfig.task_configs:
                if task.finish_time is None and task.duration - waiting_time <= 0:
                    job_missed = True
                    request.jobconfig.job_finished = 2
                    break  # 如果任何任务 Missed，该 Job 视为 Missed

            if job_missed:
                remove_jobs.append(request)
                self.amount_tot_miss_tasks += request.jobconfig.num_tasks
                self.amount_tot_miss_jobs += 1
        # 2. 统计 Missed Jobs 并移除它们
        for job in remove_jobs:
            self.serving_job_list.remove(job)

    def perform_computing_FIFO(self, round_nb):
        avalibal_resource = True
        while avalibal_resource and self.serving_job_list:
            unique_instance_types = list({int(request.instance_type) for request in self.serving_job_list})
            for request in self.serving_job_list[:]:
                if len(self.serving_job_list) == 0 \
                        or (len(unique_instance_types) == 2 and not any(self.if_available.values())) \
                        or (len(unique_instance_types) == 1 and not self.if_available[unique_instance_types[0]]):
                    avalibal_resource = False
                    break
                if not self.if_available[request.instance_type]:
                    continue
                request.jobconfig.task_configs.sort(
                    key=lambda task: task.duration - (round_nb - request.jobconfig.submit_time))
                for task in request.jobconfig.task_configs:
                    if task.finish_time is not None:
                        continue
                    else:
                        allcation_instances = self.computing_resource(task.task_type,
                                                                      task.cpu_unit, task.memory_unit,
                                                                      task.instances_number, task.instances_number)
                        if (task.instances_number > 0 and allcation_instances <= 0):  # allocation_instance <= 0 or
                            self.if_available[task.task_type] = False
                            break
                        task.instances_number = max(0, task.instances_number - allcation_instances)
                    if task.instances_number == 0:
                        task.finish_time = round_nb
                finished_task = 0
                for task in request.jobconfig.task_configs:
                    if task.finish_time is not None:
                        finished_task += 1
                        self.amount_completed_tasks += 1
                if finished_task == request.jobconfig.num_tasks:
                    self.serving_job_list.remove(request)
                    self.amount_completed_jobs += 1


    def action_mapping(self,action): #区间 把[-1,1][-1,-1]区间内的值 映射到 [-np.pi/4,np.pi/4][0,cf.D_MAX]
        h_angle, speed = action[0], action[1]
        horizental_angle = h_angle*math.pi/4#(h_angle + 1) * (math.pi / 2)
        speed = (speed + 1) / 2 * cf.D_MAX
        return np.array([horizental_angle, speed])

    def move_next_location(self,move_action):
        h_angle=move_action[0]
        speed  =move_action[1]
        a_new = self.move_azimuth + h_angle # 更新方位角，归一化到 [0, 2π]
        new_angle = a_new % (2 * math.pi)
        x = self.pos_x + speed * math.cos(new_angle)
        y = self.pos_y + speed * math.sin(new_angle)
        if speed==0:
            move_energy=168.49
        else:
            move_energy=79.86*(1+3*speed**2/14400)+88.63*np.sqrt(np.sqrt(1+speed**4/1055.08)-speed**2/32.48)+0.0092*speed**3
        self.total_amount_move += move_energy#self.consume(move_energy)
        #if self.is_alive():
        self.pos_x = np.clip(x, cf.INITIAL_X, cf.AREA_WIDTH)
        self.pos_y = np.clip(y, cf.INITIAL_Y, cf.AREA_LENGTH)
        self.move_azimuth = new_angle
        self.location.append((self.pos_x,self.pos_y,self.move_azimuth))

    def get_devices_upload_energy(self,distance):
        theta = (180 / math.pi) * math.asin(cf.HOVER_HIGH / distance)  # 仰角 θ_k,i (单位：度)
        P_LoS = 1 / (1 + cf.C * math.exp(-cf.D * (theta - cf.C)))
        P_NLos = 1 - P_LoS
        h_m_n = 1.425 * 10 ** (-4) * (1 / (distance ** 2)) * (1 / (cf.mu_LoS * P_LoS + cf.mu_NLoS * P_NLos))
        current_device_power_cost = 10 ** (-14) * (2 ** (cf.UPLOAD_DATA_TH / (cf.BANDWIDTH)) - 1) / h_m_n
        return current_device_power_cost

    def sort_requests_by_min_deadline(self, round_nb):
        def calculate_request_deadline(request):
            task_deadlines = [
                task.duration - (round_nb - request.job_sub_time)
                for task in request.jobconfig.task_configs
            ]
            return min(task_deadlines) if task_deadlines else float('inf')

        request_deadline_map = {
            request: calculate_request_deadline(request)
            for request in self.network_handler.requests_list
        }

        self.network_handler.requests_list.sort(
            key=lambda req: request_deadline_map[req]
        )

    def update_servering_list(self,round_nb):
        self.sort_requests_by_min_deadline(round_nb) #按照deadline 优先级上传job
        user_request_temp_count = {}
        for user_id in self.network_handler.get_edge_nodes():
            user_request_temp_count[user_id.id] = user_request_temp_count.get(user_id.id, 0)
            for request in self.network_handler.requests_list:  # 先来先服务
                if request.id == user_id.id:
                    hori_distance = self._get_distance2D(request.location, (self.pos_x, self.pos_y))
                    if request.instance_type in self.instance_types and hori_distance <= cf.R_MAX and \
                        user_request_temp_count[user_id.id] <100:  # In UAV's coverage and satisfy the require of accese number
                        user_request_temp_count[request.id] = user_request_temp_count.get(request.id, 0) + 1
                        distance = self._get_distance3D(request.location, (self.pos_x, self.pos_y))
                        current_device_power_cost = self.get_devices_upload_energy(distance)  # 加判断if current_device_power_cost<=5#p_max
                        self.device_power_comsumption.append(current_device_power_cost)
                        current_device = self.network_handler.get_device(request.id)
                        current_device.energy_upload += current_device_power_cost
                        self.serving_job_list.append(request)
                        self.network_handler.requests_list.remove(request)
                        self.access_request_num += 1
        request_count_list = [count for count in user_request_temp_count.values() if count > 0]
        max_request_count = max(request_count_list) if request_count_list else 0
        return max_request_count

    def check_resort_task(self,round_nb):
        request_waiting_queue = []
        remove_jobs = []
        for request in self.serving_job_list: # 1. 遍历所有的 Job，判断是否存在 Missed 任务
            waiting_time = round_nb - request.jobconfig.submit_time
            job_missed = False  # 标记当前 job 是否 missed

            for task in request.jobconfig.task_configs:
                if task.finish_time is None and task.duration - waiting_time <=0:
                    job_missed = True
                    request.jobconfig.job_finished=2
                    break  # 如果任何任务 Missed，该 Job 视为 Missed

            if job_missed:
                remove_jobs.append(request)
                self.amount_tot_miss_tasks += request.jobconfig.num_tasks
                self.amount_tot_miss_jobs += 1
        # 2. 统计 Missed Jobs 并移除它们
        for job in remove_jobs:
            self.serving_job_list.remove(job)
        # 3. 对剩余的任务按照 deadline 排序
        for request in self.serving_job_list:
            waiting_time = round_nb - request.jobconfig.submit_time
            task_dedline=[]
            for task in request.jobconfig.task_configs:
                if task.finish_time is None:
                    deadline = task.duration - waiting_time
                    task_dedline.append(deadline)
                    self.waiting_tasks_queue.append(task)
            request_waiting_queue.append((np.min(task_dedline), request))
        # 4. 按照 Deadline 排序
        request_waiting_queue.sort(key=lambda x: x[0])
        last_request_waiting_queue = [request[1] for request in request_waiting_queue]
        return last_request_waiting_queue

    def perform_computing_loop(self, sorted_request, round_nb): # check servering_list and sort for tasks
        avalibal_resource = True
        loop_index = 0
        while avalibal_resource and sorted_request:
            remove_jobs = []
            unique_instance_types = list({int(request.instance_type) for request in sorted_request})
            for request in sorted_request:
                if len(self.serving_job_list) == 0 \
                        or (len(unique_instance_types) == 2 and not any(self.if_available.values())) \
                        or (len(unique_instance_types) == 1 and not self.if_available[unique_instance_types[0]]):
                    avalibal_resource = False
                    break
                if not self.if_available[request.instance_type]:
                    continue
                if loop_index == 0:  #在第一次分配的时候先按照deadline为每个job确定分配的实例数
                    request.jobconfig.task_configs.sort(
                        key=lambda task: task.duration - (round_nb - request.jobconfig.submit_time)
                    )
                    for task in request.jobconfig.task_configs:
                        if task.finish_time is not None:
                            continue
                        deadline = task.duration - (round_nb - request.jobconfig.submit_time)
                        if deadline < math.floor(task.duration/1.5):  # 如果剩余时间小于duration,那么应该分配得实例数量大于原规定得并行需求才能在deadline之前完成计算
                            instances_number = math.ceil(task.instances_number / deadline)
                        else:
                            instances_number = task.instances_number_request  # 确保至少为 1
                        task.allocation_instance = instances_number
                for task in request.jobconfig.task_configs:
                    if task.finish_time is not None:
                        continue
                    allocation_instance = self.computing_resource(
                        task.task_type,
                        task.cpu_unit,
                        task.memory_unit, task.allocation_instance, task.instances_number
                    )
                    if (task.instances_number>0 and task.allocation_instance>0 and allocation_instance <= 0):  # allocation_instance <= 0 or
                        self.if_available[task.task_type]= False
                        break
                    task.instances_number -= allocation_instance
                    if task.instances_number < 0:
                        task.instances_number = 0  # 确保不会变为负数
                    if task.instances_number == 0:
                        task.finish_time = round_nb
                        self.amount_completed_tasks += 1
                finished_time = 0
                for task in request.jobconfig.task_configs:
                    if task.finish_time is not None:
                        finished_time += 1
                if finished_time == request.jobconfig.num_tasks:
                    remove_jobs.append(request)
                    self.serving_job_list.remove(request)
                    self.amount_completed_jobs += 1

            for job in remove_jobs:
                sorted_request.remove(job)
            loop_index += 1

    def computing_resource(self, task_type, task_cpu, task_memory, instances_number, task_instance_num):
        new_instances_number = np.min([math.floor(self.resource[task_type][0] / task_cpu),
                                       math.floor(self.resource[task_type][1] / task_memory), instances_number,
                                       task_instance_num])
        if self.resource[task_type][0] >= task_cpu * new_instances_number and self.resource[task_type][
            1] >= task_memory * new_instances_number:
            self.resource[task_type][0] -= task_cpu * new_instances_number
            self.resource[task_type][1] -= task_memory * new_instances_number
        return new_instances_number

    def consume(self, amount):
        if self.energy >= amount:
            self.energy -= amount
            return amount
        else:
            spend_energy = self.energy
            self.energy = 0
            self.alive = False
            return spend_energy

    def compute_resource_utility(self):
        resource_load=self.resource_load() #[cpu1_load,mem1_load,cpu2_load,mem2_load]
        self.total_cpu_utility_per_slot.append([resource_load[0],resource_load[2]])
        self.total_memory_utility_per_slot.append([resource_load[1],resource_load[3]])

    def _get_distance2D(self, point1, point2):
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def _get_distance3D(self, point1, point2):
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2+cf.HOVER_HIGH**2)


"""import numpy as np
from .import config as cf
import math

class Mobile_Server:
    def __init__(self, id, parent=None):
        self.id = id
        self.network_handler = parent
        self.reactivate()

    def reactivate(self):
        self.pos_x = cf.INITIAL_X+cf.AREA_WIDTH/2  #np.random.uniform(0, cf.AREA_WIDTH)
        self.pos_y = cf.INITIAL_Y+cf.AREA_LENGTH/2    # np.random.uniform(0, cf.AREA_LENGTH)
        self.move_azimuth=cf.INITIAL_AZIMUTH
        self.high=cf.HOVER_HIGH
        self.location = [(self.pos_x,self.pos_y,self.move_azimuth)]
        self.device_power_comsumption=[]
        # 初始化资源可用性状态，默认所有资源都是可用的
        self.resource=self.heterogeneous_resource()  #cpu/core 和memory
        self.remained_resouces=[]
        self.instance_types=list(self.resource.keys())  #[]得到当前server中的instance types flag= True if 1 in self.resource.keys() else Fale
        self.energy= cf.INITIAL_ENERGY
        self.waiting_tasks_queue=[]
        self.alive = True
        self.serving_job_list = []
        self.finished_jobs = []
        self.access_request_num=0

        self.total_amount_move = 0
        self.total_amount_serve = 0

        self.amount_completed_tasks = 0  # 在延迟内满足服务请求deadline的task总数
        self.amount_completed_jobs = 0  # 在延迟内满足jobdedline的job总数
        self.amount_tot_miss_jobs = 0
        self.amount_tot_miss_tasks = 0

        self.total_cpu_utility_per_slot=[[0,0]]
        self.total_memory_utility_per_slot=[[0,0]]

        #self.total_slowdown=[]

    def heterogeneous_resource(self):
        if self.id-cf.NB_NODES==0 or self.id-cf.NB_NODES==3 or self.id-cf.NB_NODES==6 : #每个SERVER异构包含两种instance类型
            resource = {  #c6a.2xlarge, m6a.12xlarge
                  10: [1600,1920],# 代表 2 GHz 的 cpu_capacity
                  8:  [640, 1280]# 代表 4 GB 的 memory_capacity
            }
        elif self.id-cf.NB_NODES==1 or self.id-cf.NB_NODES==4 or self.id-cf.NB_NODES==7:
            resource = { #c6a.4xlarge, m6a.8xlarge
                  10: [960,1920],
                  3: [1280,1280]
            }
        elif self.id-cf.NB_NODES==2 or self.id-cf.NB_NODES==5 or self.id-cf.NB_NODES==8:
            resource = { #c6a.8xlarge, m6a.4xlarge
                  10: [1664,1920],# 代表 2 GHz 的 CPU
                  12: [576,1280]# 代表 4 GB 的 GPU
            }
        return resource

    def is_alive(self):
        return self.alive

    def update(self, round_nb, action):
        move_action=self.action_mapping(action)
        #if self.is_alive():
        self.move_next_location(move_action)  # 验证：tot_length=speed
        if  len(self.network_handler.requests_list)>0: #self.is_alive() and
            self.update_servering_list(round_nb)  #考虑覆盖范围和单位时隙内的最大接入数量，实现用户关联，把request放入server的waiting queue
        if  len(self.serving_job_list)>0: #self.is_alive() and
            sorted_request = self.check_resort_task(round_nb) #得到所有的job的task队列（按照deadline排序）
            self.perform_computing_loop(sorted_request,round_nb) #分配instance运行task
        self.compute_resource_utility()
        self.resource=self.heterogeneous_resource() #重新释放资源

    def resource_load(self):
        resource_value=list(self.resource.values())
        #print('the value of resources is :',resource_value)
        if self.id-cf.NB_NODES==0 or self.id-cf.NB_NODES==3 or self.id-cf.NB_NODES==6: #每个SERVER异构包含两种instance类型
            cpu1_load=(1600-resource_value[0][0])/1600
            mem1_load=(1920-resource_value[0][1])/1920
            cpu2_load = (640 - resource_value[1][0]) / 640
            mem2_load = (1280 - resource_value[1][1]) / 1280
        elif self.id-cf.NB_NODES==1 or self.id-cf.NB_NODES==4 or self.id-cf.NB_NODES==7:
            cpu1_load = (960 - resource_value[0][0]) / 960
            mem1_load = (1920 - resource_value[0][1]) / 1920
            cpu2_load = (1280 - resource_value[1][0]) / 1280
            mem2_load = (1280 -resource_value[1][1]) / 1280
        elif self.id-cf.NB_NODES==2 or self.id-cf.NB_NODES==5 or self.id-cf.NB_NODES==8:
            cpu1_load = (1664 - resource_value[0][0]) / 1664
            mem1_load = (1920 - resource_value[0][1]) / 1920
            cpu2_load = (576 - resource_value[1][0]) / 576
            mem2_load = (1280 - resource_value[1][1]) / 1280
        resource_load=[cpu1_load,mem1_load,cpu2_load,mem2_load]
        return resource_load

    def action_mapping(self,action): #区间 把[-1,1][-1,-1]区间内的值 映射到 [-np.pi/4,np.pi/4][0,cf.D_MAX]
        h_angle, speed = action[0], action[1]
        horizental_angle = h_angle*math.pi/4#(h_angle + 1) * (math.pi / 2)
        speed = (speed + 1) / 2 * cf.D_MAX
        return np.array([horizental_angle, speed])

    def move_next_location(self,move_action):
        h_angle=move_action[0]
        speed  =move_action[1]
        a_new = self.move_azimuth + h_angle # 更新方位角，归一化到 [0, 2π]
        new_angle = a_new % (2 * math.pi)
        x = self.pos_x + speed * math.cos(new_angle)
        y = self.pos_y + speed * math.sin(new_angle)
        if speed==0:
            move_energy=168.49
        else:
            move_energy=79.86*(1+3*speed**2/14400)+88.63*np.sqrt(np.sqrt(1+speed**4/1055.08)-speed**2/32.48)+0.0092*speed**3
        self.total_amount_move += move_energy#self.consume(move_energy)
        #if self.is_alive():
        self.pos_x = np.clip(x, cf.INITIAL_X, cf.AREA_WIDTH)
        self.pos_y = np.clip(y, cf.INITIAL_Y, cf.AREA_LENGTH)
        self.move_azimuth = new_angle
        self.location.append((self.pos_x,self.pos_y,self.move_azimuth))

    def get_devices_upload_energy(self,distance):
        theta = (180 / math.pi) * math.asin(cf.HOVER_HIGH / distance)  # 仰角 θ_k,i (单位：度)
        P_LoS = 1 / (1 + cf.C * math.exp(-cf.D * (theta - cf.C)))
        P_NLos = 1 - P_LoS
        h_m_n = 1.425 * 10 ** (-4) * (1 / (distance ** 2)) * (1 / (cf.mu_LoS * P_LoS + cf.mu_NLoS * P_NLos))
        current_device_power_cost = 10 ** (-14) * (2 ** (cf.UPLOAD_DATA_TH / (cf.BANDWIDTH)) - 1) / h_m_n
        return current_device_power_cost

    def sort_requests_by_min_deadline(self, round_nb):
        def calculate_request_deadline(request):
            task_deadlines = [
                task.duration - (round_nb - request.job_sub_time)
                for task in request.jobconfig.task_configs
            ]
            return min(task_deadlines) if task_deadlines else float('inf')

        request_deadline_map = {
            request: calculate_request_deadline(request)
            for request in self.network_handler.requests_list
        }

        self.network_handler.requests_list.sort(
            key=lambda req: request_deadline_map[req]
        )

    def update_servering_list(self,round_nb):
        self.sort_requests_by_min_deadline(round_nb) #按照deadline 优先级上传job
        for user_id in self.network_handler.get_edge_nodes():
            access_number_onslot = 0
            for request in self.network_handler.requests_list: #先来先服务
                hori_distance = self._get_distance2D(request.location, (self.pos_x, self.pos_y))
                if request.instance_type in self.instance_types and hori_distance <= cf.R_MAX:  # In UAV's coverage and satisfy the require of accese number
                    if user_id.id == request.id and access_number_onslot == 1:
                        break
                    access_number_onslot += 1
                    distance = self._get_distance3D(request.location, (self.pos_x, self.pos_y))
                    current_device_power_cost = self.get_devices_upload_energy(distance)
                    self.device_power_comsumption.append(current_device_power_cost)
                    current_device = self.network_handler.get_device(request.id)
                    current_device.energy_upload += current_device_power_cost
                    self.serving_job_list.append(request)
                    self.network_handler.requests_list.remove(request)
                    self.access_request_num += 1

    def check_resort_task(self,round_nb):
        request_waiting_queue = []
        remove_jobs = []
        for request in self.serving_job_list: # 1. 遍历所有的 Job，判断是否存在 Missed 任务
            waiting_time = round_nb - request.jobconfig.submit_time
            job_missed = False  # 标记当前 job 是否 missed

            for task in request.jobconfig.task_configs:
                if task.finish_time is None and task.duration - waiting_time <=0:
                    job_missed = True
                    request.jobconfig.job_finished=2
                    break  # 如果任何任务 Missed，该 Job 视为 Missed

            if job_missed:
                remove_jobs.append(request)
                self.amount_tot_miss_tasks += request.jobconfig.num_tasks
                self.amount_tot_miss_jobs += 1
        # 2. 统计 Missed Jobs 并移除它们
        for job in remove_jobs:
            self.serving_job_list.remove(job)
        # 3. 对剩余的任务按照 deadline 排序
        for request in self.serving_job_list:
            waiting_time = round_nb - request.jobconfig.submit_time
            task_dedline=[]
            for task in request.jobconfig.task_configs:
                if task.finish_time is None:
                    deadline = task.duration - waiting_time
                    task_dedline.append(deadline)
                    self.waiting_tasks_queue.append(task)
            request_waiting_queue.append((np.min(task_dedline), request))
        # 4. 按照 Deadline 排序
        request_waiting_queue.sort(key=lambda x: x[0])
        last_request_waiting_queue=[request[1] for request in request_waiting_queue]
        return last_request_waiting_queue

    def panduan_access(self,sorted_request,round_nb):
        loop=True
        iterations=0
        while loop and sorted_request and iterations<1000:
            iterations+=1
            if iterations>=999:
                print('panduan_sixunhuan')
            last_request=self.serving_job_list[-1]
            first_request=sorted_request[0]
            task_on_time=0
            witing_time =  round_nb - first_request.job_sub_time
            for task in first_request.jobconfig.task_configs:
                if task.finish_time is not None:
                    task_on_time += 1
                else:
                    deadline = task.duration - witing_time
                    if deadline < task.duration:  # 如果剩余时间小于duration,那么应该分配得实例数量大于原规定得并行需求才能在deadline之前完成计算
                        instances_number = math.ceil(task.instances_number / deadline)
                    else:
                        instances_number = task.instances_number_request  # 确保至少为 1
                    first_request_if_delete = self.panduan_computing_resource(task.task_type,
                                                                              task.cpu_unit,
                                                                              task.memory_unit, instances_number)
                    if first_request_if_delete:
                        task_on_time += 1
                    else:
                        if last_request.id == first_request.id:
                            self.access_request_num -= 1
                            self.network_handler.requests_list.append(last_request)
                        else:
                            self.amount_tot_miss_jobs += 1
                            self.amount_completed_tasks += first_request.jobconfig.num_tasks
                        sorted_request.remove(sorted_request[0])
                        self.serving_job_list.remove(last_request)
                        break
                if task_on_time == first_request.numtasks or len(sorted_request)==0:
                    loop = False
                    break

        return sorted_request

    def perform_computing_loop(self, sorted_request, round_nb): # check servering_list and sort for tasks
        new_sorted_request=sorted_request#self.panduan_access(sorted_request, round_nb)
        avalibal_resource=True
        loop_index=0
        while avalibal_resource and new_sorted_request and loop_index<1000:
            if loop_index>=999:
                print('perform_computing_loop_sixunhuan')
            remove_jobs=[]
            for request in new_sorted_request:
                if loop_index == 0:
                    request.jobconfig.task_configs.sort(
                        key=lambda task: task.duration - (round_nb - request.jobconfig.submit_time)
                    )
                    for task in request.jobconfig.task_configs:
                        if task.finish_time is not None:
                            continue
                        deadline = task.duration - (round_nb - request.jobconfig.submit_time)
                        if deadline < task.duration:  # 如果剩余时间小于duration,那么应该分配得实例数量大于原规定得并行需求才能在deadline之前完成计算
                            instances_number = math.ceil(task.instances_number / deadline)
                        else:
                            instances_number = task.instances_number_request  # 确保至少为 1
                        task.allocation_instance = instances_number
                for task in request.jobconfig.task_configs:
                    if task.finish_time is not None:
                        continue
                    allocation_instance = self.computing_resource(
                        task.task_type,
                        task.cpu_unit,
                        task.memory_unit, task.allocation_instance,task.instances_number
                    )
                    task.instances_number -= allocation_instance
                    if task.instances_number < 0:
                        task.instances_number = 0  # 确保不会变为负数
                    if task.instances_number == 0:
                        task.finish_time = round_nb
                        self.amount_completed_tasks += 1
                finished_time=0
                for task in request.jobconfig.task_configs:
                    if task.finish_time is not None:
                        finished_time+=1
                    if finished_time == request.jobconfig.num_tasks and request in self.serving_job_list:
                        remove_jobs.append(request)
                        self.serving_job_list.remove(request)
                        self.amount_completed_jobs += 1
                if allocation_instance <= 0 or len(self.serving_job_list)==0:
                    avalibal_resource = False
                    break
            for job in remove_jobs:
                new_sorted_request.remove(job)
            loop_index += 1
            if not avalibal_resource:
                break

    def computing_resource(self, task_type, task_cpu, task_memory, instances_number,task_instance_num):
        new_instances_number = np.min([math.floor(self.resource[task_type][0] / task_cpu),
                                       math.floor(self.resource[task_type][1] / task_memory), instances_number,task_instance_num])
        if self.resource[task_type][0] >= task_cpu * new_instances_number and self.resource[task_type][
            1] >= task_memory * new_instances_number:
            self.resource[task_type][0] -= task_cpu * new_instances_number
            self.resource[task_type][1] -= task_memory * new_instances_number
        return new_instances_number

    def panduan_computing_resource(self,task_type,task_cpu,task_memory,instances_number):
        need_cpu_resource=task_cpu*instances_number
        need_mem_resource=task_memory*instances_number
        if self.resource[task_type][0]>need_cpu_resource and self.resource[task_type][1]>need_mem_resource:
            return True
        else:
            return False

    def consume(self, amount):
        if self.energy >= amount:
            self.energy -= amount
            return amount
        else:
            spend_energy = self.energy
            self.energy = 0
            self.alive = False
            return spend_energy

    def compute_resource_utility(self):
        resource_load=self.resource_load() #[cpu1_load,mem1_load,cpu2_load,mem2_load]
        self.total_cpu_utility_per_slot.append([resource_load[0],resource_load[2]])
        self.total_memory_utility_per_slot.append([resource_load[1],resource_load[3]])

    def _get_distance2D(self, point1, point2):
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def _get_distance3D(self, point1, point2):
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2+cf.HOVER_HIGH**2)



"""