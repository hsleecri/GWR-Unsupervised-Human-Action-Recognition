import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

# CSV 파일을 DataFrame으로 로드
file_path = "C:\\Users\\hslee\\Desktop\\dataset\\HYEONSU\\PROCESS4_FRONT_PART1_CSV\\Part1.mp4pose_world_visibility사라짐_하반신제거_결측치처리됨_첫행feature없음.csv"
df = pd.read_csv(file_path)

# 2. 피처 스케일링
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)

# 주성분 분석(PCA)을 이용하여 차원 축소
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)

# K-means 클러스터링 (여러 번 반복하여 군집 중심 업데이트)
num_clusters = 5
num_iterations = 2
for _ in range(num_iterations):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(df_pca)
    df['cluster'] = clusters

    # 주변 샘플의 클러스터 결과와 다른 샘플의 클러스터를 주변 샘플과 동일하게 업데이트
    for i in range(1, len(df)):
        if df.at[i, 'cluster'] != df.at[i - 1, 'cluster']:
            df.at[i, 'cluster'] = df.at[i - 1, 'cluster']

# 원래 샘플 인덱스를 기준으로 DataFrame 정렬
df.sort_index(inplace=True)

# 샘플 인덱스를 기준으로 클러스터링 결과를 시각화
plt.scatter(df.index, df['cluster'], s=50, c=df['cluster'], cmap='viridis', edgecolors='k')
plt.xlabel('frame')
plt.ylabel('cluster')
plt.title('result')
plt.show()
