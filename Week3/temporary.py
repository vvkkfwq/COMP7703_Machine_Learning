import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 读取数据
data = pd.read_csv('heightWeightData.csv', header=None, names=['label', 'height', 'weight'])

# 提取特征
X = data[['height', 'weight']]

# 应用k-means聚类算法
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
data['cluster'] = kmeans.labels_

# 绘制数据点
plt.figure(figsize=(10, 6))
colors = {0: 'red', 1: 'blue'}
for label in data['label'].unique():
    subset = data[data['label'] == label]
    plt.scatter(subset['height'], subset['weight'], c=subset['cluster'].map(colors), label=f'Class {label}', alpha=0.5)

# 绘制聚类中心
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.75, marker='X', label='Centers')

plt.xlabel('Height')
plt.ylabel('Weight')
plt.title('K-means Clustering of Height and Weight Data')
plt.legend()
plt.show()