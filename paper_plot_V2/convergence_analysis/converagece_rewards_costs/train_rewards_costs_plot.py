import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#"IPPO": r"D:\Code\MEC_Desta_MASAC\desta_masac\results_2018_20_3\ppo",
# 定义算法路径 D:\Code\MEC_Desta_MASAC\desta_masac\results_2018_20_3
algorithms = {
    "SSAC-MGI": r"D:\Code\MEC_Desta_MASAC\desta_masac\paper_plot_V2\convergence_analysis\results_2017_20_3\desta_sac_test",
    "SSAC-MGI-\nFIFS": r"D:\Code\MEC_Desta_MASAC\desta_masac\paper_plot_V2\convergence_analysis\results_2017_20_3\run_desta_sac_non_resource_test",
    "SSAC": r"D:\Code\MEC_Desta_MASAC\desta_masac\paper_plot_V2\convergence_analysis\results_2017_20_3\sac_test",
    "STRPO": r"D:\Code\MEC_Desta_MASAC\desta_masac\paper_plot_V2\convergence_analysis\results_2017_20_3\trpo_test",
    "SCPO": r"D:\Code\MEC_Desta_MASAC\desta_masac\paper_plot_V2\convergence_analysis\results_2017_20_3\cpo_test",
    }
"""algorithms ={
"SSAC-MGI": r"D:\Code\MEC_Desta_MASAC\desta_masac\results_2018_20_3\desta_sac_test",
    "SSAC-MGI-\nFIFS": r"D:\Code\MEC_Desta_MASAC\desta_masac\results_2018_20_3\run_desta_sac_non_resource_test",
    "ISAC": r"D:\Code\MEC_Desta_MASAC\desta_masac\results_2018_20_3\sac_test",
    "ICPO": r"D:\Code\MEC_Desta_MASAC\desta_masac\results_2018_20_3\cpo_test",
    "ITRPO": r"D:\Code\MEC_Desta_MASAC\desta_masac\results_2018_20_3\trpo_test",
}"""

# 列名
columns = [
    'Training Rewards', 'smooth_rewards', 'Training Costs', 'completion_jobs',
    'uncompletion_jobs', 'total_access_requests', 'amount_requests_receive',
    'uncompletion_rates', 'average_move_energy', 'average_devices_energy',
    'cpu_utility', 'memory_utility'
]

def smooth_data(data, window_size):
    return data.rolling(window=window_size, min_periods=1).mean()

def load_algorithm_data(base_path):
    datasets = []
    files = sorted([f for f in os.listdir(base_path) if f.endswith(".txt")])
    for file in files:
        file_path = os.path.join(base_path, file)
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        try:
            df = pd.read_csv(file_path, sep=",", names=columns)
            df = df.apply(pd.to_numeric, errors='coerce')
            df.fillna(0, inplace=True)
            df = df.iloc[:400]
            df = df.apply(lambda col: smooth_data(col, window_size=60), axis=0)
            datasets.append(df)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    return datasets

all_data = {}
for algo, path in algorithms.items():
    all_data[algo] = load_algorithm_data(path)

def compute_mean_std(dataframes):
    data_matrix = np.array([df.values for df in dataframes])  # (num_files, num_rows, num_cols)
    mean_values = np.mean(data_matrix, axis=0)
    std_values  = np.std(data_matrix, axis=0)
    return mean_values, std_values

# 只绘制第1列(索引0)和第3列(索引2)
columns_to_plot = [0, 2]

fig, axs = plt.subplots(1, 2, figsize=(8, 3))

import string

for i, col_index in enumerate(columns_to_plot):
    col_name = columns[col_index]
    ax = axs[i]

    # 子图标号（用于 xlabel）
    subplot_label = f"({string.ascii_lowercase[i]})"
    #subplot_label = f"({string.ascii_lowercase[i + 2]})"

    for algo, data_list in all_data.items():
        if not data_list:
            print(f"No valid data for algorithm: {algo}")
            continue

        col_data = [df.iloc[:, col_index] for df in data_list]
        mean_values, std_values = compute_mean_std(col_data)
        x = np.arange(len(mean_values))

        ax.plot(x, mean_values, label=f"{algo}", linewidth=2)
        ax.fill_between(x, mean_values - std_values, mean_values + std_values, alpha=0.2)

    ax.set_xlim(0, 400)

    if col_name == 'Training Rewards' or col_name == 'smooth_rewards':
        ax.set_ylim(-180, -75)
        #ax.set_ylim(-220, -100)
    elif col_name == 'Training Costs':
        ax.set_ylim(0, 8)

    # 将标号加到 x 轴标签中
    ax.set_xlabel(f"{subplot_label} Episodes", fontsize=12, fontweight='bold')

    if col_name == 'Training Costs':
        ax.set_ylabel('Training Safety Costs', fontsize=12, fontweight='bold')
    else:
        ax.set_ylabel(col_name, fontsize=12, fontweight='bold')

    ax.tick_params(axis='both', which='major', labelsize=10, width=2)

    for spine in ax.spines.values():
        spine.set_linewidth(2)

    ax.grid(True, linestyle='--')
    ax.legend(fontsize=10)

    if col_name == 'Training Costs':
        inset_ax = ax.inset_axes([0.25, 0.65, 0.3, 0.3])
        for algo, data_list in all_data.items():
            if not data_list:
                continue
            col_data = [df.iloc[:, col_index] for df in data_list]
            mean_values, std_values = compute_mean_std(col_data)
            x = np.arange(len(mean_values))

            x_zoom = x[350:400]
            mean_zoom = mean_values[350:400]
            std_zoom = std_values[350:400]

            inset_ax.plot(x_zoom, mean_zoom, linewidth=2)
            inset_ax.fill_between(x_zoom, mean_zoom - std_zoom, mean_zoom + std_zoom, alpha=0.2)

        inset_ax.set_xlim(350, 400)
        inset_ax.set_ylim(0.5, 2)

        for side in ["top", "bottom", "left", "right"]:
            inset_ax.spines[side].set_linewidth(1)
            inset_ax.spines[side].set_linestyle('--')
            inset_ax.spines[side].set_color('gray')

plt.tight_layout()
plt.subplots_adjust(wspace=0.16)
plt.savefig('v2017_rewards_test.pdf')
plt.show()
