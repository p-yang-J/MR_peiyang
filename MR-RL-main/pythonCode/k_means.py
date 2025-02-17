import numpy as np
from sklearn.cluster import DBSCAN

# 数据已经被标准化
data = np.load('features.npy')

# 使用DBSCAN进行聚类
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(data)

# 获取类别标签
labels = dbscan.labels_

np.savetxt('kmeans.txt',labels)