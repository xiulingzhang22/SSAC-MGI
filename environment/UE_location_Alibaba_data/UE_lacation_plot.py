import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

import pandas as pd
import folium

"""#file_path = 'D:/Code/MEC_Desta_MASAC/desta_masac/environment/UE_location_Alibaba_data/Static_UE_locations_twitter_20.csv'
file_path = 'D:/Code/MEC_Desta_MASAC/desta_masac/environment/UE_location_Alibaba_data/Mobile_UE_trajectories_twitter_20.csv'
data = pd.read_csv(file_path)

# 为每个用户编号（从 1 开始）
data['User_Number'] = data.groupby('Username').ngroup() + 1
top_users = data['Username'].unique()[:20]

# 只保留前 20 个用户的数据
filtered_data = data[data['Username'].isin(top_users)].copy()

# 创建地图，以 Oxford Street 为中心
map_center = [51.50895752, -0.12737984]
user_map = folium.Map(location=map_center, zoom_start=15)

# 添加中心点标记
folium.Marker(
    location=map_center,
    popup="Oxford Street Center",
    tooltip="Map Center",
    icon=folium.Icon(color="red", icon="flag")
).add_to(user_map)

# 遍历每个点并用用户编号标注，同时添加用户名 popup
for _, row in filtered_data.iterrows():

    lat = row['Latitude']
    lon = row['Longitude']
    time_slot = row['User_Number']
    username = row['Username']

    #popup_html = f"""
    #<div style="width:50px;">
        #<p><b>User:</b> {username}</p>
        #<p><b>Location:</b> {lat}, {lon}</p>
    #</div>
    #"""

    #popup = folium.Popup(popup_html, max_width=50)

    #folium.Marker(
        #location=[lat, lon],
        #popup=popup,  # 使用 Popup 对象，可以自定义大小
        #tooltip=f"Time slot {time_slot}",  # 悬浮提示
        #icon=folium.DivIcon(
            #html=f"""
            #<div style="background-color: #007bff;
                        #color: white;
                       # font-size: 10px; #15
                        #font-weight: bold;
                        #text-align: center;
                        #border-radius: 50%;
                        #width: 20px;  #30
                        #height: 20px; #30
                        #line-height: 20px;"> #30
                #{time_slot}
           # </div>"""
        #)
    #).add_to(user_map)


#map_file_path = 'D:/Code/MEC_Desta_MASAC/desta_masac/environment/UE_location_Alibaba_data/Static_UE_locations_twitter_20.html'
#map_file_path = 'D:/Code/MEC_Desta_MASAC/desta_masac/environment/UE_location_Alibaba_data/Mobile_UE_trajectories_twitter_20.html'
#user_map.save(map_file_path)

#读取数据映射到500*500二维坐标系内
#twitter_file_path = "D:/Code/MEC_Desta_MASAC/desta_masac/environment/UE_location_data/Static_UE_locations_twitter_20.csv"
twitter_file_path = "D:/Code/MEC_Desta_MASAC/desta_masac/environment/UE_location_Alibaba_data/Mobile_UE_trajectories_twitter_20.csv"
df = pd.read_csv(twitter_file_path)

# 强制转换为数值，防止因字符串类型导致计算错误
df["Latitude"] = pd.to_numeric(df["Latitude"], errors='coerce')
df["Longitude"] = pd.to_numeric(df["Longitude"], errors='coerce')

# 经纬度边界
min_lat, max_lat = 51.5057, 51.51462684
min_lon, max_lon = -0.13991274, -0.12107491

# 映射到 [0, 500]
df["X"] = ((df["Longitude"] - min_lon) / (max_lon - min_lon)) * 500
df["Y"] = ((df["Latitude"] - min_lat) / (max_lat - min_lat)) * 500

# 可视化：每个用户用不同颜色
plt.figure(figsize=(10, 10))

usernames = df["Username"].unique()
colors = cm.get_cmap("tab20", len(usernames))  # tab20色彩映射，支持最多20种颜色，超过也可以自动分配

for idx, (username, user_df) in enumerate(df.groupby("Username")):
    plt.scatter(user_df["X"], user_df["Y"], s=50, alpha=0.6,
                label=username, color=colors(idx))

plt.title("User Trajectories on 2D Plane")
plt.xlabel("X (Normalized Longitude)")
plt.ylabel("Y (Normalized Latitude)")
plt.xlim(0, 500)
plt.ylim(0, 500)
plt.grid(True)
plt.tight_layout()
plt.legend(fontsize='small', loc='lower left', ncol=2)
plt.show()

# 保存归一化后的完整轨迹数据
output_path = "D:/Code/MEC_Desta_MASAC/desta_masac/environment/UE_location_Alibaba_data/Mobile_UE_trajectories_20.csv"
df[["X", "Y"]].to_csv(output_path, index=False)
