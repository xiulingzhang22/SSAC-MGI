import matplotlib.pyplot as plt

# UAV 路径数据（闭环）
path_with_arrows = [
    [250, 250], [250, 200], [250, 150], [200, 150], [150, 150], [150, 200], [150, 250], [150, 300],
    [150, 350], [150, 400], [150, 450], [100, 450], [50, 450], [50, 400], [50, 350], [50, 300],
    [50, 250], [50, 200], [50, 150], [50, 100], [50, 50], [100, 50], [150, 50], [200, 50],
    [250, 50], [300, 50], [350, 50], [400, 50], [450, 50], [450, 100], [450, 150], [450, 200], [450, 250],
    [450, 300], [450, 350], [450, 400], [450, 450], [400, 450], [350, 450], [300, 450], [250, 450],
    [250, 400], [250, 350], [300, 350], [350, 350], [350, 300], [350, 250], [350, 200], [350, 150],
    [300, 150], [300, 200], [300, 250], [250, 250]
]

# 顺序路径和倒序路径
forward_path = path_with_arrows
reversed_path = path_with_arrows[::-1]

# 箭头颜色分别设置
arrow_colors = ['#cc5500', '#0072B2']  # 深橙色 + 蓝色

# 创建子图
fig = plt.figure(figsize=(10, 4))
gs = fig.add_gridspec(1, 2, wspace=0)
axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]

titles = ["Clockwise UAV Trajectory", "Counterclockwise UAV Trajectory"]
labels = ["(a) x (m)", "(b) x (m)"]

for ax, path, title, x_label, arrow_color in zip(axs, [forward_path, reversed_path], titles, labels, arrow_colors):
    x_arrow, y_arrow = zip(*path)
    start_point = path[0]
    end_point = path[-1]

    ax.plot(x_arrow, y_arrow, '-k', linewidth=1, label="UAV Trajectory")

    for i in range(len(path) - 1):
        x_start, y_start = path[i]
        x_end, y_end = path[i + 1]
        dx = x_end - x_start
        dy = y_end - y_start
        ax.arrow(x_start, y_start, dx, dy,
                 head_width=10, head_length=10, fc=arrow_color, ec=arrow_color, length_includes_head=True)

    ax.scatter(x_arrow, y_arrow, c='blue', s=10, marker='o', zorder=5)
    ax.scatter(*start_point, c='purple', s=40, marker='^', label='Start/End', zorder=6)
    ax.scatter(*end_point, c='purple', s=40, marker='^', zorder=6)

    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(True, linestyle='--')
    ax.set_xticks(range(0, 501, 50))
    ax.set_yticks(range(0, 501, 50))
    ax.set_aspect('equal')
    ax.set_xlabel(x_label)
    ax.set_ylabel("y (m)")
    ax.legend(loc='upper right', fontsize=8, frameon=True)

plt.savefig("Two_manual_uav_trajectory.pdf", bbox_inches='tight')
plt.show()
