import socket
import time
from sensapex import SensapexDevice, UMP
import matplotlib.pyplot as plt
import numpy as np
import serial
from pynput import keyboard
from scipy.interpolate import interp1d
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
import d3rlpy
from d3rlpy.preprocessing import MinMaxActionScaler
import torch
import gym
from gym import spaces

ump = UMP.get_ump()
dev_ids = ump.list_devices()
#ser = serial.Serial('COM4', 9600, timeout=1)
print("Connection established")

devs = {i: SensapexDevice(i) for i in dev_ids}

dev = devs[1]

server_ip = '10.167.156.160'  # 服务器IP，和Unity脚本中的服务器IP一致
server_port = 16306  # 服务器端口，和Unity脚本中的服务器端口一致
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # 创建UDP socket
sock.bind((server_ip, server_port))  # 绑定到指定的IP和端口

#host, port = "127.0.0.1", 16308
#host,port ="172.16.2.76",16308
#host, port = "10.167.104.130", 16308
host, port = "10.167.110.20", 16308
#host, port = "10.167.156.160", 16308
#joint[0.268+-0.0275]
startPos = [0.268, -0.002440972, 0.013553]  # Vector3   x = 0, y = 0, z = 0
x = 0
y = 0
z = 0
x_pre = 0
y_pre = 0
z_pre = 0
eulerX_pre = 0
first_run = True
dx_list = []
dy_list = []
dz_list = []
eulerX_list = []



#初始化电机
currentRead = 1
cmd = ''
currentThreshold = 250
motorstep = 115000
cycle = 0
data = [0] * 10000
velocityMax = 200000
position = -20000
#ser.write(b'ANSW0\n')
time.sleep(0.05)

cmd = 'SP' + str(velocityMax) + '\n'
#ser.write(cmd.encode())
time.sleep(0.05)


# G和S计算初始值
new_points = []
points_hand =[]
points = []
sigma = 1
vp = 1
G_values = []
S_values = []
scale1_list =[]
scale2_list =[]
scale3_list =[]
#define ada ratio
# def ada_ratio():
#
#     ratio = 10
#     return ratio

max_steps = 500  # Set a maximum number of steps
step = 0
move_ahead = 0

#计算GheS
def calculate_G(points):
    # Calculate the velocity and acceleration
    velocity = np.diff(points, axis=0)
    acceleration = np.diff(velocity, axis=0)

    # Calculate the norm of the velocity
    velocity_norm = np.linalg.norm(velocity[:-1], axis=1)

    # Check if the norm of the velocity is zero
    velocity_norm[velocity_norm == 0] = np.finfo(float).eps

    # Calculate the curvature
    curvature = np.linalg.norm(np.cross(velocity[:-1], acceleration), axis=1) / velocity_norm ** 3

    # Check if the curvature is zero or negative
    curvature[curvature <= 0] = np.finfo(float).eps

    # Calculate G
    G = np.median(np.log10(curvature))

    return G


def calculate_S(points, sigma, vp):
    # Calculate the velocity and acceleration
    velocity = np.diff(points, axis=0)
    acceleration = np.diff(velocity, axis=0)

    # Calculate the jerk
    jerk = np.diff(acceleration, axis=0)

    # Check if vp is zero
    if vp == 0:
        vp = np.finfo(float).eps

    # Calculate the dimensionless jerk
    dimensionless_jerk = sigma ** 5 / vp ** 2 * np.sum(jerk ** 2, axis=1)

    # Check if the dimensionless jerk is zero or negative
    dimensionless_jerk[dimensionless_jerk <= 0] = np.finfo(float).eps

    # Calculate S
    S = np.median(np.log10(dimensionless_jerk))

    return S

# This will become False when Space is pressed, and the while-loop will break
keep_running = True
cluster = False
timestamps = []
effiency_list = []

def on_space(key):
    global keep_running
    if key == keyboard.Key.space:
        keep_running = False
#


#add filter

#卡尔曼滤波
class SimpleKalmanFilter:
    def __init__(self, initial_state, initial_estimate_error, measurement_noise, process_noise):
        self.state_estimate = initial_state
        self.estimate_error = initial_estimate_error
        self.measurement_noise = measurement_noise
        self.process_noise = process_noise

    def update(self, measurement):
        # prediction
        self.estimate_error += self.process_noise

        # kalman gain
        kalman_gain = self.estimate_error / (self.estimate_error + self.measurement_noise)

        # correction
        self.state_estimate += kalman_gain * (measurement - self.state_estimate)
        self.estimate_error = (1 - kalman_gain) * self.estimate_error

        return self.state_estimate
dx_filter = SimpleKalmanFilter(initial_state=0, initial_estimate_error=1, measurement_noise=1, process_noise=0.01)
dy_filter = SimpleKalmanFilter(initial_state=0, initial_estimate_error=1, measurement_noise=1, process_noise=0.01)
dz_filter = SimpleKalmanFilter(initial_state=0, initial_estimate_error=1, measurement_noise=1, process_noise=0.01)
deulerX_filter = SimpleKalmanFilter(initial_state=0, initial_estimate_error=1, measurement_noise=1, process_noise=0.01)
###
addx = 0
addy = 0
addz = 0
x_dist = 0
y_dist = 0
z_dist = 0

position = 0
trajectory_error =0

#生成ground truth轨迹
def resample_track(track, num_points=100):
    old_points = np.linspace(0, 1, track.shape[0])
    new_points = np.linspace(0, 1, num_points)

    new_track = []
    for dim in range(track.shape[1]):
        interpolator = interp1d(old_points, track[:, dim], kind='linear')
        new_track.append(interpolator(new_points))

    return np.array(new_track).T


# 读取文件
x = np.loadtxt('x_values.txt')
y = np.loadtxt('y_values.txt')
z = np.loadtxt('z_values.txt')

# 组合成轨迹
track = np.column_stack((x, y, z))

# 重新采样
resampled_track = resample_track(track)

def calculate_error(ground_truth, prediction):
    """
    计算预测点到轨迹的最短距离和最近点在 x、y、z 方向上的距离
    :param ground_truth: Ground truth轨迹, shape为(n, 3)的numpy array，n为点的数量
    :param prediction: 预测的轨迹点, shape为(1, 3)的numpy array
    :return: 预测点到轨迹的最短距离和最近点在 x、y、z 方向上的距离（以元组形式返回）
    """
    dists = distance.cdist(ground_truth, prediction, 'euclidean')
    min_dist = np.min(dists)
    nearest_point_index = np.argmin(dists)
    nearest_point = ground_truth[nearest_point_index]
    x_dist = nearest_point[0] - prediction[0, 0]
    y_dist = nearest_point[1] - prediction[0, 1]
    z_dist = nearest_point[2] - prediction[0, 2]
    return min_dist, x_dist, y_dist, z_dist


total_distance = 0
total_distance_hand = 0.0001

total_distance_hand_list =[]

listener = keyboard.Listener(on_release=on_space)
listener.start()

class CustomEnv(gym.Env):

    def __init__(self):
        super(CustomEnv, self).__init__()
        # self.action_space = spaces.Box(low=np.array([-3000, -3000, -4000]), high=np.array([3000, 3000, 4000]),
        #                                dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,))

        # 定义动作空间的最小值和最大值
        self.action_min = np.array([-3000, -3000, -4000])
        self.action_max = np.array([3000, 3000, 4000])

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        # Initialize other required variables
        self.first_run = True
        self.cluster = False
        done = False
        self.i = 0
        self.G = 0
        self.S = 0
        self.x_dist = 0
        self.y_dist = 0
        self.z_dist = 0
        self.trajectory_error = 0
        self.x = 0
        self.y = 0
        self.z = 0
        self.dx_filtered = 0
        self.dy_filtered = 0
        self.dz_filtered = 0
        self.x_pre = 0
        self.y_pre = 0
        self.z_pre = 0
        self.eulerX_pre = 0
        self.scale_addx = 0
        self.scale_addy = 0
        self.scale_addz = 0
        self.scale_ada_x = 10000
        self.scale_ada_y = 10000
        self.scale_ada_z = 10000
        self.addx = 0
        self.addy = 0
        self.addz = 0
        self.disx = 0
        self.disy = 0
        self.disz = 0
        self.dx = 0
        self.dy = 0
        self.dz = 0
        self.total_distance = 0
        self.total_distance_hand = 0.0001
        self.observations = np.zeros((1000, 6))
        self.actions = np.zeros((1000, 3))
        self.rewards = np.zeros(1000)
        self.terminals = np.random.randint(2, size=1000)
        self.reward = 0

    def step(self, action):
        action = self.scale_action(action)
        self.scale_addx, self.scale_addy, self.scale_addz = action
        data, addr = sock.recvfrom(1024)  # 接收最大1024字节的数据
        message_data = data.decode('utf-8')
        if "HandPosition:" in message_data:
            message_data = message_data.replace("HandPosition:", "")
            self.x, self.y, self.z, eulerX, eulerY, eulerZ = map(lambda s: float(s.replace(',', '')), message_data.split())
            # print('Hand position x:', x, 'y:', y, 'z:', z)
            # print('Hand rotation Euler angles x:', eulerX, 'y:', eulerY, 'z:', eulerZ)
            # eye tracking data
        # elif "GazeOrigin:" in message_data and "GazeDirection:" in message_data:
        #     gaze_data = message_data.split("GazeDirection:")
        #     gaze_origin_data = gaze_data[0].replace("GazeOrigin:", "").split()
        #     gaze_direction_data = gaze_data[1].split()
        #     x_origin, y_origin, z_origin = map(lambda s: float(s.replace(',', '')), gaze_origin_data)
        #     x_direction, y_direction, z_direction = map(lambda s: float(s.replace(',', '')), gaze_direction_data)
        #     print('Gaze origin x:', x_origin, 'y:', y_origin, 'z:', z_origin)
        #     print('Gaze direction x:', x_direction, 'y:', y_direction, 'z:', z_direction)
        elif message_data == "1":
            self.cluster = True
            self.first_run = False
            print('Received command: cluster')
        # if self.x == 0:
        #     self.first_run = True
        if self.first_run or self.cluster:
            if not self.first_run:
                self.cluster = False
                self.first_run = True
                self.x_pre = 0
                self.y_pre = 0
                self.z_pre = 0
            else:
                self.first_run = False
        else:
            self.dx = self.x - self.x_pre
            self.dy = self.y - self.y_pre
            self.dz = self.z - self.z_pre
            #deulerX = eulerX - eulerX_pre
            timestamps.append(time.time())
            # print('dx:', dx, 'dy:', dy, 'dz:', dz)
            # dx_filtered = dx_filter.update(dx)
            # dy_filtered = dy_filter.update(dy)
            # dz_filtered = dz_filter.update(dz)
            self.dx_filtered = dx_filter.update(self.dx)
            self.dy_filtered = dy_filter.update(self.dy)
            self.dz_filtered = dz_filter.update(self.dz)
            #deulerX_filtered = deulerX_filter.update(deulerX)
            points_hand.append([self.dx_filtered, self.dy_filtered, self.dz_filtered])

            self.scale_ada_x = 6000 + self.scale_addx + 10000 * 1 / (1 + np.exp(-abs(self.x_dist) / 500))
            self.scale_ada_y = 6000 + self.scale_addy + 10000 * 1 / (1 + np.exp(-abs(self.y_dist) / 500))
            self.scale_ada_z = 6000 + self.scale_addz + 10000 * 1 / (1 + np.exp(-abs(self.z_dist) / 500))
            print("动作：", self.scale_addx)
            scale1_list.append(self.scale_ada_x)
            scale2_list.append(self.scale_ada_y)
            scale3_list.append(self.scale_ada_z)
            self.disx = self.dx_filtered * self.scale_ada_x
            self.disy = self.dy_filtered * self.scale_ada_y
            self.disz = self.dz_filtered * self.scale_ada_z
            print("dx_filtered:",self.dx)
            #diseulerX = deulerX_filtered * 600
            # print("电机的转角：", diseulerX)
            self.addx = self.addx + self.disx
            self.addy = self.addy + self.disy
            self.addz = self.addz + self.disz
            dx_list.append(self.addx)
            dy_list.append(self.addy)
            dz_list.append(self.addz)
            #eulerX_list.append(diseulerX)
            #dev.goto_pos([9999 - self.addy, 9999 - self.addx, 9999 - self.addz, 19999], 10000)
            dev.goto_pos([9999 - self.addy, 9999 + self.addz, 9999 - self.addx, 19999], 10000)
            print("机器人位置：",9999 - self.addy, 9999 - self.addx, 9999 - self.addz)
            # position = position + diseulerX
            # cmd = 'LA' + str(position) + '\n'
            # ser.write(cmd.encode())
            # time.sleep(0.05)
            # cmd = 'M' + '\n'
            # ser.write(cmd.encode())
            #  time.sleep(0.05)
            # dev.goto_pos([9999 + addx, 9999 - addy, 9999 - addz, 19999], 10000)
            print(self.addx, self.addy, self.addz)
            point_now = np.array([[self.addx, self.addy, self.addz]])
            self.trajectory_error, self.x_dist, self.y_dist, self.z_dist = calculate_error(resampled_track, point_now)
            print("轨迹误差值：", self.trajectory_error)
            print("最近点在 x 方向上的距离：", self.x_dist)
            print("最近点在 y 方向上的距离：",self.y_dist)
            print("最近点在 z 方向上的距离：", self.z_dist)

            startPos[0] += (self.disx / 9999) * 0.0275  # increase z by one
            startPos[1] += (self.disy / 9999) * 0.0275
            startPos[2] += (self.disz / 9999) * 0.0275
            posString = '({})'.format(
                ','.join(map(str, startPos)))  # Adding parentheses to the string, example "(0,0,0)"
            print(posString)
            # Add the new point to the list
            points.append([self.addy, self.addx, self.addz])
            if len(points) >= 10:  # Replace 4 with the actual number of points you need
                points_array = np.array(points)
                self.G = calculate_G(points_array)
                self.S = calculate_S(points_array, sigma, vp)
                G_values.append(self.G)
                S_values.append(self.S)
            # print("G:", G)
            # print("S:", S)
            if len(points) >= 2:
                # Get the last two points
                last_point = points[-1]
                second_last_point = points[-2]

                # Calculate the Euclidean distance between the two points
                distance_robot = ((last_point[0] - second_last_point[0]) ** 2 +
                                  (last_point[1] - second_last_point[1]) ** 2 +
                                  (last_point[2] - second_last_point[2]) ** 2) ** 0.5

                # Add the distance to the total distance
                self.total_distance += distance_robot
            if len(points_hand) >= 2:
                # Get the last two points
                last_point_hand = points_hand[-1]
                second_last_point_hand = points_hand[-2]

                # Calculate the Euclidean distance between the two points
                distance_hand = ((last_point_hand[0] - second_last_point_hand[0]) ** 2 +
                                 (last_point_hand[1] - second_last_point_hand[1]) ** 2 +
                                 (last_point_hand[2] - second_last_point_hand[2]) ** 2) ** 0.5

                # Add the distance to the total distance
                self.total_distance_hand += distance_hand
                total_distance_hand_list.append(self.total_distance_hand)
            effiency = self.total_distance / self.total_distance_hand
            effiency_list.append(effiency)
            print("机器人轨迹长度：", self.total_distance)
            print("手轨迹长度：", self.total_distance_hand)
            print("效率：", effiency)

            self.reward = (200000-effiency)/50000
            #self.reward = (-self.G-self.S)


            sock.sendto(posString.encode("UTF-8"), (host, port))  # Sending string to Unity
            #endtime = time.time()
            # timing = endtime - startime
            #
            # # 手部运动距离作为reward一部分
            # if timing > 10:
            #     reward += (0.04 - total_distance_hand) * 300
            #     startime = time.time()
        self.rewards[self.i] = self.reward
        print("奖励：",self.rewards[self.i])
        self.observations[self.i] = [self.addx, self.addy, self.addz, self.scale_ada_x, self.scale_ada_y, self.scale_ada_z]
        self.actions[self.i] = [self.scale_addx, self.scale_addy, self.scale_addz]
        if self.i >= 200:
            done = True
        else:
            done = False

        #sock.sendto(posString.encode("UTF-8"), (host, port))  # Sending string to Unity
            # startPos = [0.2955, 0.027, 0.04105353]
        self.x_pre = self.x
        self.y_pre = self.y
        self.z_pre = self.z
        self.i += 1
        #print(111)
        #eulerX_pre = eulerX

    # np.save('rewards.npy', rewards)
    # np.save('observations.npy', observations)
    # np.save('actions.npy', actions)
    # np.save('terminals.npy', terminals)

        return self.observations[self.i], self.rewards[self.i], done, {}

    def reset(self):
        # Here, you'd normally reset the environment to its initial state
        self.first_run = True
        self.cluster = False
        done = False
        self.i = 0
        self.G = 0
        self.S = 0
        #self.x_dist = 0
        #self.y_dist = 0
        #self.z_dist = 0
        #self.trajectory_error = 0
        #self.scale_addx = 0
        #self.scale_addy = 0
        #self.scale_addz = 0
       # self.scale_ada_x = 10000
        #self.scale_ada_y = 10000
        #self.scale_ada_z = 10000
        self.total_distance = 0
        self.total_distance_hand = 0.00001
        # self.observations = np.zeros((1000, 6))
        # self.actions = np.zeros((1000, 3))
        # self.rewards = np.zeros(1000)
        # self.terminals = np.random.randint(2, size=1000)

        return self.observations[self.i]

    def scale_action(self, action):
        # 对每一维的动作分别进行缩放
        scaled_action = []
        for a, min_val, max_val in zip(action, self.action_min, self.action_max):
            # 将a从[-1, 1]缩放到[min_val, max_val]
            a_scaled = min_val + (a + 1) / 2.0 * (max_val - min_val)
            scaled_action.append(a_scaled)
        return np.array(scaled_action)

    def render(self, mode='human'):
        pass

    def close(self):
        # np.save('rewards.npy', self.rewards)
        # np.save('observations.npy', self.observations)
        # np.save('actions.npy', self.actions)
        # np.save('terminals.npy', self.terminals)
        pass


# bcq_new = d3rlpy.algos.BCQ()
#
# # Use the environment to build (initialize) the model
# dataset = d3rlpy.dataset.MDPDataset(
#         observations=np.load('observations.npy'),
#         actions=np.load('actions.npy'),
#         rewards=np.load('rewards.npy'),
#         terminals=np.load('terminals.npy'),
#     )
# bcq_new.build_with_dataset(dataset)
#
# bcq_new.load_model('bcq_new.pt')
# # 使用BC克隆BCQ模型
# bc = d3rlpy.algos.BC()
# bc.fit(dataset.episodes, n_epochs=1000)  # 使用BCQ生成的数据集训练BC模型
#
# # Set action scaler
# #action_scaler = d3rlpy.preprocessing.MinMaxActionScaler(minimum=[-3000, -3000, -4000], maximum=[3000, 3000, 4000])
#
# #cql = d3rlpy.algos.BCQ(use_gpu=False)
# TD3 = d3rlpy.algos.TD3(use_gpu=False,actor_learning_rate=0.05,critic_learning_rate=0.03)
# # Use the environment to build (initialize) the SAC model
#
# TD3.build_with_env(env)
#
# TD3.copy_q_function_from(bcq_new)
# TD3.copy_policy_from(bc)
#
# # Setup experience replay buffer and exploration strategy
env = CustomEnv()
TD3 = d3rlpy.algos.TD3(use_gpu=False,actor_learning_rate=0.05,critic_learning_rate=0.03)

buffer = d3rlpy.online.buffers.ReplayBuffer(maxlen=2000, env=env)
explorer = d3rlpy.online.explorers.LinearDecayEpsilonGreedy(0.1)
# # Train the model
TD3.fit_online(
    env,
    buffer,
    explorer,
    n_steps=2000,
    n_steps_per_epoch=100,
    update_start_step=100
)
#bcq_new.save_model('TD3_online.pt')

# # 创建一个新的SAC模型实例
# sac_new = d3rlpy.algos.BCQ(use_gpu=False)
#
#
# # 使用环境来构建（初始化）模型
# sac_new.build_with_env(env)
#
# # 加载先前保存的模型
# sac_new.load_model('bcq_online.pt')
#
# # 现在你可以使用sac_new来进行验证
# obs = env.reset()
# done = False
# total_reward = 0
#
# while not done:
#     action = sac_new.predict([obs])
#     print("动作：", action)
#     obs, reward, done, _ = env.step(action)
#     print("奖励2：",reward)
#     total_reward += reward
#
# print("Total reward:", total_reward)
#
#
np.savetxt('G_values.txt', G_values)
np.savetxt('S_values.txt', S_values)
np.savetxt('effiency_list.txt',effiency_list)
effiency_list_loaded = np.loadtxt('effiency_list.txt')
G_values_loaded = np.loadtxt('G_values.txt')
S_values_loaded = np.loadtxt('S_values.txt')
# # np.savetxt('G_values.txt', G_values)
# # np.savetxt('S_values.txt', S_values)












