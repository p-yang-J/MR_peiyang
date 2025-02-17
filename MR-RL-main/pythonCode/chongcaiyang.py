import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial import distance

# 定义函数进行重采样
def resample_track(track, num_points=100):
    old_points = np.linspace(0, 1, track.shape[0])
    new_points = np.linspace(0, 1, num_points)

    new_track = []
    for dim in range(track.shape[1]):
        interpolator = interp1d(old_points, track[:, dim], kind='linear')
        new_track.append(interpolator(new_points))

    return np.array(new_track).T

def calculate_error(ground_truth, prediction):
    """
    计算预测点到轨迹的最短距离
    :param ground_truth: Ground truth轨迹, shape为(n, 3)的numpy array，n为点的数量
    :param prediction: 预测的轨迹点, shape为(1, 3)的numpy array
    :return: 预测点到轨迹的最短距离
    """
    dists = distance.cdist(ground_truth, prediction, 'euclidean')
    min_dist = np.min(dists)
    return min_dist

# 读取文件
x = np.loadtxt('dx_list.txt')
y = np.loadtxt('dy_list.txt')
z = np.loadtxt('dz_list.txt')

# 组合成轨迹
track = np.column_stack((x, y, z))

# 重新采样
resampled_track = resample_track(track)
print(resampled_track)

prediction = np.array([[4.173244325373208774e+00
, 6, -1]])  # 预测的轨迹点
print(calculate_error(resampled_track, prediction))  # 输出预测点到轨迹的最短距离