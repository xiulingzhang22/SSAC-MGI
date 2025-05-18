import random
from .import config as cf
class Obstacles:
    def __init__(self):
        self.static_area = (cf.INITIAL_X,cf.AREA_WIDTH,cf.INITIAL_Y,cf.AREA_LENGTH)  # (x_min, x_max, y_min, y_max)
        self.num_static = cf.NUM_STATIC
        self.static_obstacles = self.generate_static_obstacles()

    def generate_static_obstacles(self):
        random.seed(2024)  # 固定随机种子
        obstacles = []
        for _ in range(self.num_static):
            x = random.uniform(self.static_area[0], self.static_area[1])
            y = random.uniform(self.static_area[2], self.static_area[3])
            z = cf.HOVER_HIGH
            obstacles.append((x, y,z))
        return obstacles
