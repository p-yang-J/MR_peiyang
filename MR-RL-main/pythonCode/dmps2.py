import time

import numpy as np
import matplotlib.pyplot as plt
from pydmps import dmp_discrete
from sensapex import SensapexDevice, UMP
ump = UMP.get_ump()
dev_ids = ump.list_devices()


devs = {i: SensapexDevice(i) for i in dev_ids}
dev = devs[1]

# 读取所有轨迹数据
trajectories_x = [np.loadtxt(f'x_values{i+1}.txt') for i in range(3)]
trajectories_y = [np.loadtxt(f'y_values{i+1}.txt') for i in range(3)]
trajectories_z = [np.loadtxt(f'z_values{i+1}.txt') for i in range(3)]

# 计算最长的轨迹长度
max_len = max(len(t) for t in trajectories_x)

# 创建一个DMPs对象，100个基函数，1个维度
dmp_x = dmp_discrete.DMPs_discrete(n_dmps=1, n_bfs=100)
dmp_y = dmp_discrete.DMPs_discrete(n_dmps=1, n_bfs=100)
dmp_z = dmp_discrete.DMPs_discrete(n_dmps=1, n_bfs=100)

# 对每条轨迹，执行imitate_path，并将结果累加
weights_x, weights_y, weights_z = 0, 0, 0
for x, y, z in zip(trajectories_x, trajectories_y, trajectories_z):
    # 使用线性插值调整轨迹长度
    x = np.interp(np.linspace(0, 1, max_len), np.linspace(0, 1, len(x)), x)
    y = np.interp(np.linspace(0, 1, max_len), np.linspace(0, 1, len(y)), y)
    z = np.interp(np.linspace(0, 1, max_len), np.linspace(0, 1, len(z)), z)

    # 使用调整长度后的轨迹训练DMP
    dmp_x.imitate_path(y_des=x)
    dmp_y.imitate_path(y_des=y)
    dmp_z.imitate_path(y_des=z)

    weights_x += dmp_x.w
    weights_y += dmp_y.w
    weights_z += dmp_z.w

# 计算平均权
dmp_x.w = weights_x/4
dmp_y.w = weights_y/4
dmp_z.w = weights_z/4

# 创造一个新的轨迹
y_track_x, _, _ = dmp_x.rollout()
y_track_y, _, _ = dmp_y.rollout()
y_track_z, _, _ = dmp_z.rollout()

# for i in range(len(y_track_x)):
#     pos = [9999 - y_track_y[i], 9999 - y_track_x[i], 9999 - y_track_z[i], 9999]
#     dev.goto_pos(pos, 1000)

# 创建绘图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=30., azim=120)

# 绘制原始轨迹
# for x, y, z in zip(trajectories_x, trajectories_y, trajectories_z):
#     ax.plot(x, y, z, 'r--')
for x, y, z in zip(trajectories_x, trajectories_y, trajectories_z):
    # 使用线性插值调整轨迹长度
    x = np.interp(np.linspace(0, 1, max_len), np.linspace(0, 1, len(x)), x)
    y = np.interp(np.linspace(0, 1, max_len), np.linspace(0, 1, len(y)), y)
    z = np.interp(np.linspace(0, 1, max_len), np.linspace(0, 1, len(z)), z)

    ax.plot(x, y, z, 'r--')

# 绘制生成的DMP轨迹
ax.plot(y_track_x, y_track_y, y_track_z, 'b-')
print(y_track_x[1][0])
for i in range(100):
    dev.goto_pos([9999 - y_track_y[i][0], 9999 - y_track_x[i][0], 9999 - y_track_z[i][0], 0], 1000)
    time.sleep(0.05)


# 显示图像
plt.show()



