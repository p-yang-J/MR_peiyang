import cv2
import numpy as np
from sklearn.cluster import KMeans

def calculate_distances(image_path, output_path):
    # 读取图像
    image = cv2.imread(image_path)

    # 定义颜色阈值（BGR颜色空间），根据你的标记颜色进行修改
    blue = np.array([255, 0, 0])
    yellow = np.array([0, 255, 255])
    green = np.array([0, 255, 0])

    # 使用OpenCV的inRange函数来找出图像中对应颜色的点
    blue_points = cv2.inRange(image, blue, blue)
    yellow_points = cv2.inRange(image, yellow, yellow)
    green_points = cv2.inRange(image, green, green)

    # 使用OpenCV的findNonZero函数来找出图像中非零点的坐标
    blue_coords = cv2.findNonZero(blue_points)
    yellow_coords = cv2.findNonZero(yellow_points)
    green_coords = cv2.findNonZero(green_points)

    # 为K-means调整坐标数组的形状
    blue_coords_reshaped = blue_coords.reshape(-1, 2)

    # 使用K-means聚类算法找出两个蓝点的中心
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(blue_coords_reshaped)
    blue_coord1, blue_coord2 = kmeans.cluster_centers_

    # 计算所有绿点和黄点到两个蓝点的距离，并取最短的那一个，同时在图像上画出距离线段
    green_distances = []
    yellow_distances = []
    for coord in green_coords:
        coord = coord[0]
        dist1 = np.linalg.norm(coord - blue_coord1)
        dist2 = np.linalg.norm(coord - blue_coord2)
        if dist1 < dist2:
            cv2.line(image, tuple(coord), tuple(blue_coord1.astype(int)), (0, 0, 255), 1)
            green_distances.append(dist1)
        else:
            cv2.line(image, tuple(coord), tuple(blue_coord2.astype(int)), (0, 0, 255), 1)
            green_distances.append(dist2)
    for coord in yellow_coords:
        coord = coord[0]
        dist1 = np.linalg.norm(coord - blue_coord1)
        dist2 = np.linalg.norm(coord - blue_coord2)
        if dist1 < dist2:
            cv2.line(image, tuple(coord), tuple(blue_coord1.astype(int)), (0, 0, 255), 1)
            yellow_distances.append(dist1)
        else:
            cv2.line(image, tuple(coord), tuple(blue_coord2.astype(int)), (0, 0, 255), 1)
            yellow_distances.append(dist2)

    # 保存线段图像
    cv2.imwrite(output_path, image)

    # 计算并返回平均距离和方差
    return (np.mean(green_distances), np.var(green_distances)), (np.mean(yellow_distances), np.var(yellow_distances))

# 使用函数
image_path = 'G:\Cybercall\PycharmProjects\pythonProject\dis_error.png'  # 换成你图片的路径
output_path = 'output.jpg'  # 输出图片的路径
print(calculate_distances(image_path, output_path))





# 使用函数
# image_path = 'D:\PycharmProjects\pythonProject\dis_error.png'  # 换成你图片的路径
# print(calculate_distances(image_path))
