#preprocess the data
#import librarires
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# CSV 파일을 읽어옵니다. 파일 경로는 알맞게 수정해주세요.
file_path = "C:\\Users\\hslee\\Desktop\\dataset\\HYEONSU\\PROCESS4_FRONT_PART1_CSV\\Part1.mp4pose_world_visibility사라짐_하반신제거_결측치처리됨.csv"
df = pd.read_csv(file_path)

# visibility columns 제거, 하반신 제거는 걍 엑셀에서 했음.

'''
# DataFrame의 열(컬럼) 개수를 가져옵니다.
num_columns = len(df.columns)
print(num_columns)

# 4의 배수인 인덱스의 열(컬럼)을 모두 제거합니다.
# 열 인덱스는 0부터 시작하므로, 3, 7, 11, ... 인덱스를 제거하면 됩니다.
# 즉, 4로 나누었을 때 나머지가 3인 열을 제거합니다.
columns_to_drop = [col_index for col_index in range(num_columns) if col_index % 4 == 3]
df.drop(df.columns[columns_to_drop], axis=1, inplace=True)

# 전처리된 DataFrame을 CSV 파일로 저장합니다.
# 저장할 파일 경로는 알맞게 수정해주세요.
output_file_path = "C:\\Users\\hslee\\Desktop\\dataset\\HYEONSU\\PROCESS4_FRONT_PART1_CSV\\Part1.mp4pose_world_visibility사라짐.csv"
print(len(df.columns))
print(df.head())
# df.to_csv(output_file_path, index=False)
'''

# 결측치 처리

'''
# CSV 파일을 읽어옵니다. 파일 경로는 알맞게 수정해주세요.
file_path = "C:\\Users\\hslee\\Desktop\\dataset\\HYEONSU\\PROCESS4_FRONT_PART1_CSV\\Part1.mp4pose_world_visibility사라짐_하반신제거.csv"
df = pd.read_csv(file_path)
df.replace(-1, np.nan, inplace=True)
print(df.info)

df_interpolated=df.interpolate()
print(df_interpolated.shape)
# 전처리된 DataFrame을 CSV 파일로 저장합니다.
# 저장할 파일 경로는 알맞게 수정해주세요.
output_file_path = "C:\\Users\\hslee\\Desktop\\dataset\\HYEONSU\\PROCESS4_FRONT_PART1_CSV\\Part1.mp4pose_world_visibility사라짐_하반신제거_결측치처리됨.csv"
df_interpolated.to_csv(output_file_path, index=False)
'''


# correlation matrix
df_face=df.iloc[:,:33] #33 face 전체
print(df_face.head(1))
correlation_matrix=df_face.corr()


clustergrid=sns.clustermap(correlation_matrix, 
                           cmap = 'RdYlBu_r', 
            annot = False,   # 실제 값을 표시한다
            linewidths=.5,  # 경계면 실선으로 구분하기
            cbar_kws={"shrink": .5},# 컬러바 크기 절반으로 줄이기
            vmin = -1,vmax = 1   # 컬러바 범위 -1 ~ 1
              )

# sns.heatmap(correlation_matrix, 
#                            cmap = 'RdYlBu_r', 
#             annot = False,   # 실제 값을 표시한다
#             linewidths=.5,  # 경계면 실선으로 구분하기
#             cbar_kws={"shrink": .5},# 컬러바 크기 절반으로 줄이기
#             vmin = -1,vmax = 1   # 컬러바 범위 -1 ~ 1
#               )
plt.show()

# 클러스터링 된 행과 열의 레이블을 텍스트로 확인합니다.
print("Clustered row labels:")
print(clustergrid.dendrogram_row.reordered_ind)

print("\nClustered column labels:")
print(clustergrid.dendrogram_col.reordered_ind)