import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 数据准备
methods = ["Total Miss Rate", "UAV Energy Cost", "UE Energy Cost"]#,"CPU Utility"]
scenarios = ["2 UAV", "3 UAV", "4 UAV", "5 UAV", "6 UAV"]

"""UAV2_AVG = [
    [0.4408, 0.3733, 0.3150, 0.2679, 0.2267],  # MISS RATE
    [0.3961, 0.3312, 0.3848, 0.3630, 0.3900],  # UAV ENERGY
    [0.4377, 0.4676, 0.4345, 0.4498, 0.4504],  # UE ENERGY
]
UAV2_AVG_STD = [
    [0.0326, 0.0375, 0.0317, 0.0320, 0.0293],  # MISS RATE
    [0.0392, 0.0222, 0.0286, 0.0392, 0.0390],  # UAV ENERGY
    [0.0241, 0.0174, 0.0157, 0.0146, 0.0150],  # UE ENERGY
]"""

UAV2_AVG = [
    [0.6324,0.5220,0.4928,0.4080,0.3396],  # MISS RATE
    [0.3898,0.3885,0.4141,0.3810,0.3436],  # UAV ENERGY
    [0.4307,0.4325,0.3386,0.4391,0.4476],  # UE ENERGY
    #[0.8956,0.9156,0.8910,0.8826,0.8794],
]
UAV2_AVG_STD = [
    [0.0224,0.0323,0.0381,0.0360,0.0387],  # MISS RATE
    [0.0201,0.0211,0.0222,0.0364,0.0179],  # UAV ENERGY
    [0.0223,0.0212,0.0161,0.0146,0.0133],  # UE ENERGY
    #[0.0201,0.0174,0.0137,0.0121,0.0109]
]

# 构建 DataFrame
df_total = []
for i, method in enumerate(methods):
    for j, scenario in enumerate(scenarios):
        df_total.append({
            "Method": method,
            "Scenario": scenario,
            "Value": UAV2_AVG[i][j],
            "Std": UAV2_AVG_STD[i][j]
        })

df_total = pd.DataFrame(df_total)

# 设置主题
sns.set_theme(style="whitegrid")

# 创建画布
fig, ax = plt.subplots(figsize=(5, 2.2))

# 绘制柱状图
sns.barplot(data=df_total, x="Method", y="Value", hue="Scenario", ci=None, ax=ax)

# 设置y轴标签和限制
ax.set_ylabel("")
#ax.set_ylabel("Value (V2017)", fontweight='bold')
ax.set_ylim(0, 0.95)
ax.set_xlabel("")
ax.tick_params(axis='y', labelsize=10, pad=0.5)

# 添加误差条
for container, scenario in zip(ax.containers, scenarios):
    stds = df_total[df_total["Scenario"] == scenario]["Std"].values
    for bar, err in zip(container, stds):
        x = bar.get_x() + bar.get_width() / 2
        height = bar.get_height()
        ax.errorbar(x, height, yerr=err, fmt='none',
                    ecolor='black', capsize=2, linewidth=1)

# 美化边框和字体
for spine in ax.spines.values():
    spine.set_linewidth(2)
    spine.set_color('black')

ax.set_xticklabels(ax.get_xticklabels(), fontweight='bold', fontsize=10)

# 图例设置
handles, labels = ax.get_legend_handles_labels()
ax.get_legend().remove()
fig.legend(
    handles=handles,
    labels=labels,
    loc="upper center",
    bbox_to_anchor=(0.526, 1.02),
    ncol=5,
    fontsize=10,
    columnspacing=1.15,
    handletextpad=0.3
)

plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.savefig('v2018_ssac_mgi.pdf')
plt.show()
