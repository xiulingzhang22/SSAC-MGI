import matplotlib.pyplot as plt
import numpy as np

# UAV 数量（x轴）
uav_numbers = [2, 3, 4, 5, 6]
methods = ["SSAC-MGI", "SSAC-MGI-FIFS", "SSAC", "STRPO", "SCPO", "MANUAL"]
marker_label = ['o', '*', 'd', 's', '^', 'p']

# 2017平均值数据
data_dict = {
    "SSAC-MGI": {
        'total_costs': [0.2441, 0.7354, 1.1661, 1.6473, 2.0501] #3UAV 1.1354
    },
    "SSAC-MGI-FIFS": {
        'total_costs': [0.2172, 0.9130, 1.1556, 1.2488, 2.7339]
    },
    "SSAC": {
        'total_costs': [0.2376, 1.2491, 1.9923, 3.4647, 4.9428] #1.0491
    },
    "STRPO": {
        'total_costs': [0.3728, 0.8601, 1.6398, 2.5251, 4.2319]
    },
    "SCPO": {
        'total_costs': [0.1699, 1.3163, 1.4370, 1.7692, 6.0031] #0.6163
    },
    "MANUAL": {
        'total_costs': [1.5438, 26.1029, 34.4740, 83.3681, 143.4175]
    }
}

# 方差数据（标准差 = sqrt(variance)）
variance_dict = {
    "SSAC-MGI": {
        'total_costs': [0.0107, 0.0641, 0.0789, 0.1729, 0.0556]
    },
    "SSAC-MGI-FIFS": {
        'total_costs': [0.0122, 0.0388, 0.0184, 0.0272, 0.0687]
    },
    "SSAC": {
        'total_costs': [0.0171, 0.2364, 0.2216, 0.1426, 0.1858]
    },
    "STRPO": {
        'total_costs': [0.0180, 0.0272, 0.0558, 0.1320, 0.4455]
    },
    "SCPO": {
        'total_costs': [0.0089, 0.0378, 0.2205, 0.0603, 0.3073]
    },
    "MANUAL": {
        'total_costs': [0.0474, 0.3397, 0.5216, 2.2430, 5.1446]
    }
}

"""# 2018
data_dict = {
    "SSAC-MGI": {
        'total_costs': [0.2302,0.4759,1.1522,1.8570, 1.9439] #2uav 0.5759
    },
    "SSAC-MGI-FIFS": {
        'total_costs': [0.3231,1.1893, 1.4232,1.9839,2.2994] #2uav 1.7893, 5uav 0.9839
    },
    "SSAC": {
        'total_costs': [0.4378,1.4077,1.0091,1.9999,3.6730] #0.9077,
    },
    "STRPO": {
        'total_costs': [0.2671,1.2022,1.6818,1.9279,3.8857] #3.2022
    },
    "SCPO": {
        'total_costs': [0.2972,1.3117,3.5477,6.0892,8.5249] #0.6117
    },
    "MANUAL": {
        'total_costs': [1.5693,16.0586,11.1131,65.3326,122.0424]
    }
}

# 方差数据（标准差 = sqrt(variance)）
variance_dict = {
    "SSAC-MGI": {
        'total_costs': [0.0084,0.0175,0.0492,0.0763,0.0222]
    },
    "SSAC-MGI-FIFS": {
        'total_costs': [0.0157,0.2377,0.0386,0.0386,0.1488]
    },
    "SSAC": {
        'total_costs': [0.1360,0.1009,0.0304,0.2651,0.0710]
    },
    "STRPO": {
        'total_costs': [0.0187,0.5655,0.0988,0.0755,0.0802]
    },
    "SCPO": {
        'total_costs': [0.0256,0.0445,0.4696,0.5753,0.3245]
    },
    "MANUAL": {
        'total_costs': [0.0376,0.2408,0.2973,2.6706,3.9794]
    }
}"""

fig, ax = plt.subplots(figsize=(5,4))
ymax = 7#7#10

for i, method in enumerate(methods):
    means = np.array(data_dict[method]['total_costs'])
    stds = np.sqrt(variance_dict[method]['total_costs'])

    lower = means - stds
    upper = means + stds

    # 主线
    ax.plot(uav_numbers, means, marker=marker_label[i], label=method, linewidth=2)

    # 阴影区域
    ax.fill_between(
        uav_numbers,
        np.clip(lower, 0, ymax),
        np.clip(upper, 0, ymax),
        alpha=0.2
    )

    # 箭头 + 标签
    for x, y, y_upper in zip(uav_numbers, means, upper):
        if y_upper > 10:
            ax.annotate(
                f"{y_upper:.1f}",
                xy=(x, ymax),
                xytext=(x, ymax - 1),
                textcoords="data",
                ha='center',
                fontsize=10,
                arrowprops=dict(arrowstyle='->', color='saddlebrown', linewidth=2)
            )

# 加粗坐标轴标题和刻度
ax.set_xlabel("Number of UAVs", fontsize=14, fontweight='bold')
ax.set_ylabel("Total Safety Costs", fontsize=14, fontweight='bold')
ax.set_xticks(uav_numbers)
ax.tick_params(axis='both', labelsize=12, width=2)

# 设置四周边框线加粗
for spine in ax.spines.values():
    spine.set_linewidth(2)

ax.set_xlim(2,6.2)
ax.set_ylim(0, ymax)
ax.grid(True, linestyle='--', linewidth=0.8)
legend = ax.legend(loc='upper left', bbox_to_anchor=(0, 0.87), fontsize=8, framealpha=0.5)
plt.tight_layout()
plt.savefig("total_costs_2017.pdf")
plt.show()
