import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 데이터프레임 생성 (예시)
df = pd.read_csv('C:\\Users\\zjkee\\OneDrive\\바탕 화면\\rpy\\video_cut\\4_front_2\\1\\1.mp4_pose_world_face(preprocessed)_gammagwr(0.2,3080).csv')

# 그래프 크기 설정
plt.figure(figsize=(16, 2))

# SOP에 따른 배경 색상 지정
sop_values = df['SOP'].values
norm = plt.Normalize(vmin=np.min(sop_values), vmax=np.max(sop_values))
cmap = plt.cm.get_cmap('Pastel1')
colors = cmap(norm(sop_values))

# SOP 값에 따라 배경 색상 채우기
for sop in range(1, 6):
    sop_data = df[df['SOP'] == sop]
    frames = sop_data['FRAME'].unique()
    for frame in frames:
        plt.axvline(x=frame, color=cmap(sop - 1), alpha=0.2)

# 축 레이블 설정
plt.xlabel('FRAME')
plt.ylabel('GROUND_TRUTH')

# y축 눈금 비워주기
plt.yticks([])

# 그래프 표시
plt.show()