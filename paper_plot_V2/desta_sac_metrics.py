import os
import numpy as np

def read_data(filepath):
    return np.loadtxt(filepath, delimiter=',', max_rows=500)

file_name = "D:/Code/MEC_Desta_MASAC/desta_masac/paper_plot_V2/data_v2017/results_2017_20_6"

algorithms = {
    'manual': os.path.join(file_name, 'manual', 'manual_0.txt'),
    'desta_sac': os.path.join(file_name, 'desta_sac_test', 'desta_sac_0.txt'),
    'no_resorce': os.path.join(file_name, 'run_desta_sac_non_resource_test', 'run_desta_sac_non_resource_0.txt'),
    'sac': os.path.join(file_name, 'sac_test', 'sac_0.txt'),
    'trpo': os.path.join(file_name, 'trpo_test', 'trpo_0.txt'),
    'cpo': os.path.join(file_name, 'cpo_test', 'cpo_0.txt'),
}

columns = [
    'Training Rewards', 'smooth_rewards', 'Training Costs', 'completion_jobs',
    'uncompletion_jobs', 'total_access_requests', 'amount_requests_receive',
    'uncompletion_rates', 'average_move_energy', 'average_devices_energy',
    'cpu_utility', 'memory_utility'
]

# 需要的列名和其对应索引
target_columns = ['Training Costs', 'uncompletion_rates', 'average_move_energy',
                  'average_devices_energy', 'cpu_utility', 'memory_utility']
target_indices = [columns.index(col) for col in target_columns]

# 创建结果存储结构
results = []

# 遍历每个算法并计算目标列的均值和标准差
for algo_name, filepath in algorithms.items():
    data = read_data(filepath)
    row = []
    for idx in target_indices:
        last_100 = data[-350:-250, idx]
        avg = np.mean(last_100)
        std = np.std(last_100)
        row.extend([avg, std])
    results.append(row)

# 保存到文本文件
output_path = "v2017_UAV6.txt"
np.savetxt(output_path, results, fmt='%.4f', delimiter='\t',
           header='\t'.join([f"{col}_avg\t{col}_std" for col in target_columns]),
           comments='')

output_path
