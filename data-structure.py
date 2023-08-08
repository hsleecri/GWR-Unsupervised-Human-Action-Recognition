# 데이터 구조를 좀 보자
# import libraries
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

file_path = "C:\\Users\\hslee\\Desktop\\dataset\\HYEONSU\\PROCESS4_FRONT_PART1_CSV\\Part1.mp4pose_world_visibility사라짐_하반신제거_결측치처리됨.csv"
df = pd.read_csv(file_path)
df=df.iloc[:,33:75] 
print(df.head(0))
#2d

# PCA 모델 생성 및 학습
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df)

# UMAP 모델 생성 및 학습
umap_model = umap.UMAP(n_components=2, n_neighbors=5, min_dist=0.3)
umap_result = umap_model.fit_transform(df)

# 시각화를 위해 2차원 데이터프레임 생성
df_pca = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
df_umap = pd.DataFrame(data=umap_result, columns=['UMAP1', 'UMAP2'])

# 그라데이션 색상 맵 설정
cmap = plt.cm.get_cmap('viridis')  # 'viridis' 색상 맵 사용, 다른 맵도 사용 가능

# 시각화
plt.figure(figsize=(18, 6))

plt.subplot(131)
sns.scatterplot(x='PC1', y='PC2', data=df_pca, hue=df_pca.index, palette=cmap)
plt.title('PCA 2D')

plt.subplot(133)
sns.scatterplot(x='UMAP1', y='UMAP2', data=df_umap, hue=df_umap.index, palette=cmap)
plt.title('UMAP 2D')

plt.tight_layout()
plt.show()
