import socket
import time
from sensapex import SensapexDevice, UMP
import matplotlib.pyplot as plt
import numpy as np
import serial
from pynput import keyboard
from scipy.interpolate import interp1d
from scipy.spatial import distance


ump = UMP.get_ump()
dev_ids = ump.list_devices()
ser = serial.Serial('COM4', 9600, timeout=1)
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
ser.write(b'ANSW0\n')
time.sleep(0.05)

cmd = 'SP' + str(velocityMax) + '\n'
ser.write(cmd.encode())
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

startime = time.time()
total_distance = 0
total_distance_hand = 0.0001



listener = keyboard.Listener(on_release=on_space)
listener.start()
while keep_running:
    data, addr = sock.recvfrom(1024)  # 接收最大1024字节的数据
    message_data = data.decode('utf-8')
    if "HandPosition:" in message_data:
        message_data = message_data.replace("HandPosition:", "")
        x, y, z, eulerX, eulerY, eulerZ = map(lambda s: float(s.replace(',', '')), message_data.split())
        #print('Hand position x:', x, 'y:', y, 'z:', z)
        #print('Hand rotation Euler angles x:', eulerX, 'y:', eulerY, 'z:', eulerZ)
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
        cluster = True
        first_run = False
        print('Received command: cluster')
    if x == 0:
        first_run = True
    if first_run or cluster:
        if not first_run:
            cluster = False
            first_run = True
            x_pre = 0
            y_pre = 0
            z_pre = 0
        else:
            first_run = False
    else:
        dx = x - x_pre
        dy = y - y_pre
        dz = z - z_pre
        deulerX = eulerX - eulerX_pre
        timestamps.append(time.time())
        #print('dx:', dx, 'dy:', dy, 'dz:', dz)
        # dx_filtered = dx_filter.update(dx)
        # dy_filtered = dy_filter.update(dy)
        # dz_filtered = dz_filter.update(dz)
        dx_filtered = dx_filter.update(dx)
        dy_filtered = dy_filter.update(dy)
        dz_filtered = dz_filter.update(dz)
        deulerX_filtered = deulerX_filter.update(deulerX)
        points_hand.append([dx_filtered, dy_filtered, dz_filtered])
        scale_ada_x = 8000 + abs(x_dist) * 5
        scale_ada_y = 8000 + abs(y_dist) * 5
        scale_ada_z = 8000 + abs(z_dist) * 5
        scale1_list.append(scale_ada_x)
        scale2_list.append(scale_ada_y)
        scale3_list.append(scale_ada_z)
        disx = dx_filtered * scale_ada_x
        disy = dy_filtered * scale_ada_y
        disz = dz_filtered * scale_ada_z
        diseulerX = deulerX_filtered*600
        print("电机的转角：",diseulerX)
        # if move_ahead ==1:
        #     disz = disz + 500
        #     move_ahead = 0
        #real robot
        # if dx_filtered > 0.014284:
        #     disx = 9999
        # else:
          #     disx = dx_filtered*700000
        # if dy_filtered > 0.014284:
        #     disy = 9999
        # else:
        #     disy = dy_filtered*700000
        # if dz_filtered > 0.014284:
        #     disz = 9999
        # else:
        #     disz = dz_filtered*700000
        addx = addx + disx
        addy = addy + disy
        addz = addz + disz
        dx_list.append(addx)
        dy_list.append(addy)
        dz_list.append(addz)
        eulerX_list.append(diseulerX)
        #dev.goto_pos([9999+addx, 9999-addz, 9999-addy, 19999], 10000 )
        dev.goto_pos([9999 -addy, 9999 - addx, 9999 - addz, 19999], 10000)
        position = position + diseulerX
        cmd = 'LA' + str(position) + '\n'
        ser.write(cmd.encode())
        #time.sleep(0.05)
        cmd = 'M' + '\n'
        ser.write(cmd.encode())
        #  time.sleep(0.05)
        #dev.goto_pos([9999 + addx, 9999 - addy, 9999 - addz, 19999], 10000)
        print(addx,addy,addz)
        point_now = np.array([[addx, addy, addz]])
        trajectory_error, x_dist, y_dist, z_dist = calculate_error(resampled_track, point_now)
        print("轨迹误差值：", trajectory_error)
        print("最近点在 x 方向上的距离：", x_dist)
        print("最近点在 y 方向上的距离：", y_dist)
        print("最近点在 z 方向上的距离：", z_dist)
        #time.sleep(0.5)  # sleep 0.5 sec
        #[+-0.0275]
        #(disx/9999)*0.0275
        # startPos[0] += (dz_filtered/9999)  # increase z by one
        # startPos[1] += (dx_filtered/20)
        # startPos[2] += (dy_filtered/20)
        startPos[0] += (disx/9999)*0.0275 # increase z by one
        startPos[1] += (disy/9999)*0.0275
        startPos[2] += (disz/9999)*0.0275
        posString = '({})'.format(','.join(map(str, startPos)))  # Adding parentheses to the string, example "(0,0,0)"
        print(posString)
        # Add the new point to the list
        points.append([addy, addx, addz])
        if len(points) >= 10:  # Replace 4 with the actual number of points you need
            points_array = np.array(points)
            G = calculate_G(points_array)
            S = calculate_S(points_array, sigma, vp)
            G_values.append(G)
            S_values.append(S)
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
            total_distance += distance_robot
        if len(points_hand) >= 2:
            # Get the last two points
            last_point_hand = points_hand[-1]
            second_last_point_hand = points_hand[-2]

            # Calculate the Euclidean distance between the two points
            distance_hand = ((last_point_hand[0] - second_last_point_hand[0]) ** 2 +
                             (last_point_hand[1] - second_last_point_hand[1]) ** 2 +
                             (last_point_hand[2] - second_last_point_hand[2]) ** 2) ** 0.5

            # Add the distance to the total distance
            total_distance_hand += distance_hand
        endtime = time.time()
        timing = endtime - startime
        effiency = total_distance / total_distance_hand
        print("机器人轨迹长度：", total_distance)
        print("手轨迹长度：", total_distance_hand)
        print("效率：", effiency)
        print("时间：", timing)


        sock.sendto(posString.encode("UTF-8"), (host, port))  # Sending string to Unity
    #startPos = [0.2955, 0.027, 0.04105353]
    x_pre = x
    y_pre = y
    z_pre = z
    eulerX_pre = eulerX
    step = step + 1
    if step == max_steps:
        break
    pass

#listener.stop()

# Save G and S values
np.savetxt('G_values.txt', G_values)
np.savetxt('S_values.txt', S_values)

# Load G and S values
G_values_loaded = np.loadtxt('G_values.txt')
S_values_loaded = np.loadtxt('S_values.txt')

# Save the lists
np.savetxt('dx_list.txt', dx_list)
np.savetxt('dy_list.txt', dy_list)
np.savetxt('dz_list.txt', dz_list)
np.savetxt('eulerX.txt',eulerX_list)
np.savetxt('scale1.txt', scale1_list)
np.savetxt('scale2.txt', scale2_list)
np.savetxt('scale3.txt', scale3_list)
# Load the lists
dx_list_loaded = np.loadtxt('dx_list.txt')
dy_list_loaded = np.loadtxt('dy_list.txt')
dz_list_loaded = np.loadtxt('dz_list.txt')
eulerX_list_load = np.loadtxt('eulerX.txt')
scale1_load =  np.loadtxt('scale1.txt')
scale1_load =  np.loadtxt('scale2.txt')
scale1_load =  np.loadtxt('scale3.txt')
np.savetxt('timestamps.txt', timestamps)

plt.figure()
plt.plot(dx_list, label='dx')
plt.plot(dy_list, label='dy')
plt.plot(dz_list, label='dz')
plt.xlabel('Time steps')
plt.ylabel('Displacement')
plt.legend()
plt.show()



