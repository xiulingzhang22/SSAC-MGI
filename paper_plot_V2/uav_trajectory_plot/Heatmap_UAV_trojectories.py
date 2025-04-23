import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LogNorm

# 读取用户位置
user_data = pd.read_csv('D:/Code/MEC_Desta_MASAC/desta_masac/paper_plot_V2/uav_trajectory_plot/Static_UE_locations_20.csv')
user_x = user_data['x'].values
user_y = user_data['y'].values

# 障碍物坐标
obstacle_coords = np.array([
    (282.05, 364.13), (182.25, 443.65),
    (246.05, 358.31), (240.13, 180.58),
    (420.55, 249.15)
])
obstacle_x = obstacle_coords[:, 0]
obstacle_y = obstacle_coords[:, 1]

# UAV轨迹数据
trajectory_data = pd.read_csv('D:/Code/MEC_Desta_MASAC/desta_masac/paper_plot_V2/uav_trajectory_plot/static_ue_uavs_trajectories.csv')
x_coords = trajectory_data['x'].values
y_coords = trajectory_data['y'].values

# 网格参数
x_min, x_max = 0, 500
y_min, y_max = 0, 500
num_bins_x = 60
num_bins_y = 60

# 生成热力图数据
heatmap, xedges, yedges = np.histogram2d(x_coords, y_coords, bins=[num_bins_x, num_bins_y],
                                         range=[[x_min, x_max], [y_min, y_max]])

# 绘图
fig, ax = plt.subplots(figsize=(10, 8))

# 使用 LogNorm 进行对数压缩显示
im = ax.imshow(heatmap.T + 1e-2, origin='lower', aspect='auto',
               extent=[x_min, x_max, y_min, y_max], cmap='YlGnBu',
               norm=LogNorm(vmin=1e-2, vmax=heatmap.max()))

cbar = plt.colorbar(im, ax=ax, pad=0.02)

# 移除默认 label
cbar.set_label('')

# 设置标签为 log₁₀(z)，位于色条下方
cbar.set_label('log₁₀(z)', fontsize=12, labelpad=4)

# 设置刻度
x_ticks = np.arange(x_min, x_max + 1, 50)
y_ticks = np.arange(y_min, y_max + 1, 50)
ax.set_xticks(x_ticks)
ax.set_yticks(y_ticks)
ax.set_xticklabels([f'{tick:.0f}' for tick in x_ticks], rotation=45)
ax.set_yticklabels([f'{tick:.0f}' for tick in y_ticks])

# 加粗刻度线
ax.tick_params(width=2, length=6)

# 去掉四边框
for spine in ax.spines.values():
    spine.set_visible(False)

# 绘制用户与障碍物
ax.scatter(user_x, user_y, marker='*', s=150, color='yellow', edgecolor='black', label='Users')
ax.scatter(obstacle_x, obstacle_y, marker='^', s=80, color='red', edgecolor='black', label='Obstacles')

ax.legend()
#ax.set_xlabel('X Coordinate')
#ax.set_ylabel('Y Coordinate')
#ax.set_title('Heatmap with User and Obstacle Positions (Log Scale)')
# 设置显示范围
ax.set_xlim(x_min-5, x_max+5)
ax.set_ylim(y_min-5, y_max+5)
ax.grid(False)
plt.savefig('static_UE_UAV_trajectory.pdf')
plt.show()
