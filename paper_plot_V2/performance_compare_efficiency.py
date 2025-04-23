import matplotlib.pyplot as plt
import numpy as np

# UAV 数量（x轴）
uav_numbers = [2, 3, 4, 5, 6]
methods = ["SSAC-MGI", "SSAC-MGI-FIFS", "SSAC", "STRPO", "SCPO", "MANUAL"]
marker_label = ['o', '*', 'd', 's', '^', 'p']

# 2017平均值数据
data_dict = {
    "SSAC-MGI": {
        'weight_missrate_uav_ue_energy': [0.5794,0.6132,0.6257,0.6434,0.6479] #4uav0.6057
    },
    "SSAC-MGI-FIFS": {
        'weight_missrate_uav_ue_energy': [0.5831,0.6057,0.6064,0.6655,0.6301]
    },
    "SSAC": {
        'weight_missrate_uav_ue_energy': [0.5736,0.5981,0.5943,0.6407,0.6206]
    },
    "STRPO": {
        'weight_missrate_uav_ue_energy': [0.5510,0.5760,0.6147,0.6184,0.6131]
    },
    "SCPO": {
        'weight_missrate_uav_ue_energy': [0.5630,0.5630,0.6046,0.5933,0.5686]
    },
    "MANUAL": {
        'weight_missrate_uav_ue_energy': [0.5153,0.5373,0.5463,0.5521,0.5510]
    }
}

variance_dict = {
    "SSAC-MGI": {
        'weight_missrate_uav_ue_energy': [0.0186,0.0155,0.0150,0.0174,0.0168]
    },
    "SSAC-MGI-FIFS": {
        'weight_missrate_uav_ue_energy': [0.0166,0.0137,0.0132,0.0148,0.0140]
    },
    "SSAC": {
        'weight_missrate_uav_ue_energy': [0.0180,0.0177,0.0146,0.0146,0.0153]
    },
    "STRPO": {
        'weight_missrate_uav_ue_energy': [0.0323,0.0233,0.0194,0.0208,0.0179]
    },
    "SCPO": {
        'weight_missrate_uav_ue_energy': [0.0184,0.0272,0.0156,0.0141,0.0153]
    },
    "MANUAL": {
        'weight_missrate_uav_ue_energy': [0.0038,0.0043,0.0041,0.0034,0.0031]
    }
}

"""# 2018平均值数据
data_dict = {
    "SSAC-MGI": {
        'weight_missrate_uav_ue_energy': [0.5205,0.5568,0.5890,0.5947,0.6268] # 4uav 0.5590
    },
    "SSAC-MGI-FIFS": {
        'weight_missrate_uav_ue_energy': [0.5166,0.5556,0.5639,0.6057,0.6228] #5uav 0.6257
    },
    "SSAC": {
        'weight_missrate_uav_ue_energy': [0.5370,0.5516,0.5580,0.5845,0.6262]
    },
    "STRPO": {
        'weight_missrate_uav_ue_energy': [0.5083,0.5360,0.5738,0.5906,0.6095]
    },
    "SCPO": {
        'weight_missrate_uav_ue_energy': [0.4938,0.5201,0.5237,0.5492,0.5766]
    },
    "MANUAL": {
        'weight_missrate_uav_ue_energy': [0.4616,0.5437,0.5410,0.5485,0.5460]
    }
}

variance_dict = {
    "SSAC-MGI": {
        'weight_missrate_uav_ue_energy': [0.0124,0.0145,0.0157,0.0176,0.0147]
    },
    "SSAC-MGI-FIFS": {
        'weight_missrate_uav_ue_energy': [0.0160,0.0136,0.0158,0.0150,0.0175]
    },
    "SSAC": {
        'weight_missrate_uav_ue_energy': [0.0145,0.0150,0.0139,0.0181,0.0138]
    },
    "STRPO": {
        'weight_missrate_uav_ue_energy': [0.0189,0.0169,0.0260,0.0221,0.0214]
    },
    "SCPO": {
        'weight_missrate_uav_ue_energy': [0.0246,0.0184,0.0234,0.0191,0.0250]
    },
    "MANUAL": {
        'weight_missrate_uav_ue_energy': [0.0040,0.0041,0.0033,0.0030,0.0025]
    }
}"""
# 绘图
fig, ax = plt.subplots(figsize=(5, 4))

for i, method in enumerate(methods):
    means = np.array(data_dict[method]['weight_missrate_uav_ue_energy'])
    stds = variance_dict[method]['weight_missrate_uav_ue_energy']

    ax.plot(uav_numbers, means, marker=marker_label[i], label=method, linewidth=2)
    ax.fill_between(uav_numbers, means - stds, means + stds, alpha=0.2)

# 坐标轴与图例
ax.set_xlabel("Number of UAVs", fontsize=14, fontweight='bold')
ax.set_ylabel("Weighted Performance", fontsize=14, fontweight='bold')
ax.set_xticks(uav_numbers)
ax.tick_params(axis='both', labelsize=12, width=2)

# 加粗边框线
for spine in ax.spines.values():
    spine.set_linewidth(2)

ax.set_xlim(2,6)
#ax.set_ylim(0.45, 0.64)
ax.set_ylim(0.5, 0.7)
ax.grid(True, linestyle='--', linewidth=0.8)
ax.legend(loc='upper left', fontsize=9, framealpha=0.5)

plt.tight_layout()
plt.savefig("utility_vs_uavs_2017.pdf")
plt.show()