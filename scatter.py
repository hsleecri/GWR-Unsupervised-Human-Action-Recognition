import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import string

# 데이터프레임 생성 (예시)
df = pd.read_csv('C:\\Users\\zjkee\\OneDrive\\바탕 화면\\rpy\\video_cut\\4_front_2\\1\\1.mp4_pose_world_face(preprocessed)_gammagwr(0.225,3080).csv')

# 그래프 크기 설정
plt.figure(figsize=(16, 4))

# SOP에 따른 배경 색상 지정
sop_values = df['SOP'].values
norm = plt.Normalize(vmin=np.min(sop_values), vmax=np.max(sop_values))
cmap = plt.cm.get_cmap('Pastel1')
colors = cmap(norm(sop_values))
'''
'''
# SOP 값에 따라 배경 색상 채우기
for sop in range(1, 6):
    sop_data = df[df['SOP'] == sop]
    frames = sop_data['FRAME'].unique()
    for frame in frames:
        plt.axvline(x=frame, color=cmap(sop - 1), alpha=0.2)

#CLUSTER scatter plot 그리기
#plt.scatter(df['FRAME'], df['CLUSTER'], color='blue', label='CLUSTER', zorder=10)

#plt.plot(df['FRAME'], df['CLUSTER'], linestyle='solid', color='C0')

# # CLUSTER 값에 따른 라인 플롯 생성
# cluster_changes = np.where(np.diff(df['CLUSTER']) != 0)[0] + 1  # 값이 변경되는 지점 인덱스
# start_frame = 0  # 라인 플롯 시작 프레임

# clusters = df['CLUSTER'].unique()
# colors = ['blue', 'green', 'red', 'purple', 'orange', 'black', 'magenta']  # CLUSTER 값에 따른 색상 설정
# # colors = ['blue', 'green', 'red', 'purple']
# # colors = ['tab:red', 'tab:orange', 'tab:pink', 'tab:green', 'tab:blue', 'tab:brown', 'tab:purple']

# for idx in cluster_changes:
#     plt.plot(df['FRAME'].values[start_frame:idx], df['CLUSTER'].values[start_frame:idx], color=colors[df['CLUSTER'].values[start_frame]], label='CLUSTER', linewidth = 5)
#     start_frame = idx  # 다음 라인 플롯 시작 프레임 설정

# # 마지막 세그먼트 그리기
# plt.plot(df['FRAME'].values[start_frame:], df['CLUSTER'].values[start_frame:], color=colors[df['CLUSTER'].values[start_frame]], label='CLUSTER', linewidth = 5)

# # 축 레이블 설정
plt.xlabel('FRAME')
# plt.ylabel('CLUSTER_INDEX_GWR')

# # 알파벳 대문자 리스트 생성
# alphabet_labels = list(string.ascii_uppercase)

# # y축 레이블 설정
# plt.yticks(np.arange(int(min(df['CLUSTER'])), int(max(df['CLUSTER'])) + 1, 1), alphabet_labels[:max(df['CLUSTER'])+1])

# y축 눈금 비워주기
plt.yticks([])

# x축 범위 설정
plt.xlim(0, max(df['FRAME']))

# x축 눈금 설정
plt.xticks(np.arange(0, max(df['FRAME']) + 1, 500))


# 그래프 표시
plt.show()

