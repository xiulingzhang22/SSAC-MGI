import os
import numpy as np
import matplotlib.pyplot as plt

# ✅ 使用真实数据文件路径
base_path = "D:/Code/MEC_Desta_MASAC/desta_masac/paper_plot_V2/convergence_analysis/converagece_metrics_manual"

file_paths_1 = [
    os.path.join(base_path, "desta_sac_test_2017", "desta_sac_0.txt"),
    os.path.join(base_path, "desta_sac_test_2017", "desta_sac_1.txt"),
]

file_paths_2 = [
    os.path.join(base_path, "desta_sac_test_2018", "desta_sac_0.txt"),
    os.path.join(base_path, "desta_sac_test_2018", "desta_sac_1.txt"),
]

file_paths_3 = [
    os.path.join(base_path, "manual_2017", "manual_0.txt"),
    os.path.join(base_path, "manual_2017", "manual_1.txt"),
]

file_paths_4 = [
    os.path.join(base_path, "manual_2018", "manual_0.txt"),
    os.path.join(base_path, "manual_2018", "manual_1.txt"),
]

# 所有列名（和之前一样）
columns = [
    'Training Rewards', 'smooth_rewards', 'Training Costs', 'completion_jobs',
    'uncompletion_jobs', 'total_access_requests', 'amount_requests_receive',
    'uncompletion_rates', 'average_move_energy', 'average_devices_energy',
    'cpu_utility', 'memory_utility'
]

target_columns = [
    ('Training Costs','Traning Safety Cost'),
    ('uncompletion_rates', "Total Miss Rate"),
    ('average_move_energy', "Average UAV Energy Cost"),
    ('average_devices_energy', "Average UE Energy Cost")
]

# 数据读取
def load_dataset(file_paths):
    dataset = []
    for path in file_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing: {path}")
        data = np.loadtxt(path, delimiter=',', max_rows=400)
        dataset.append(data)
    return np.array(dataset)  # shape: (2, 400, 12)

# 滑动平均
def moving_average_stride_full(data, window_size=10, stride=10):
    smoothed = []
    positions = []
    i = 0
    while i + window_size <= len(data):
        smoothed.append(np.mean(data[i:i + window_size]))
        positions.append(i)
        i += stride
    if i < len(data):
        smoothed.append(np.mean(data[i:]))
        positions.append(i)
    return np.array(smoothed), np.array(positions)

# 加载数据
data1 = load_dataset(file_paths_1)
data2 = load_dataset(file_paths_2)
data3 = load_dataset(file_paths_3)
data4 = load_dataset(file_paths_4)

# 绘图
fig, axs = plt.subplots(1, 4, figsize=(16, 3))
labels = ['V2017 SSAC-MGI', 'V2018 SSAC-MGI','V2017 MANUAL','V2018 MANUAL']
window_size = 10
stride = 10

for idx, (col_name, ylabel_text) in enumerate(target_columns):
    col_idx = columns.index(col_name)
    ax = axs[idx]
    ax.set_xlim(0, 400)

    for i, data in enumerate([data1, data2,data3,data4]):
        col_data = data[:, :, col_idx]
        mean_vals = np.mean(col_data, axis=0)
        std_vals = np.std(col_data, axis=0)

        # 原始数据线
        ax.plot(mean_vals, color='lightgray', linestyle='--', linewidth=0.8)

        # 平滑曲线
        smooth_mean, x_smooth = moving_average_stride_full(mean_vals, window_size, stride)
        smooth_std, _ = moving_average_stride_full(std_vals, window_size, stride)

        line = ax.plot(x_smooth, smooth_mean, label=labels[i])[0]
        line.set_linewidth(2)
        ax.fill_between(x_smooth, smooth_mean - smooth_std, smooth_mean + smooth_std, alpha=0.45)
    if col_name == 'average_devices_energy':
        ax.set_ylim(0.4,0.5)

    ax.set_xlabel(f"({chr(ord('a') + idx)}) Episodes", fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel_text, fontsize=11, fontweight='bold')
    ax.grid(True, linestyle='--')
    ax.legend(fontsize=8,loc='best') #, loc='upper right'
    ax.tick_params(axis='both', which='major', labelsize=10, width=2)
    for spine in ax.spines.values():
        spine.set_linewidth(2)

plt.tight_layout()
plt.subplots_adjust(wspace=0.25)
plt.savefig("ssac_mgi_metrics_compare.pdf", dpi=300, transparent=True)
plt.show()
