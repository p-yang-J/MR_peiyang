import socket
import time
from sensapex import SensapexDevice, UMP
import matplotlib.pyplot as plt
import numpy as np
#import serial
from pynput import keyboard

from scipy import interpolate
from pydmps import dmp_discrete
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
scale_ada =19998
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
points_hand =[]
points = []
sigma = 1
vp = 1
G_values = []
S_values = []
#define ada ratio
# def ada_ratio():
#
#     ratio = 10
#     return ratio

max_steps = 200  # Set a maximum number of steps
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



# 读取所有轨迹数据
trajectories_x = [np.loadtxt(f'x_values{i+1}.txt') for i in range(6)]
trajectories_y = [np.loadtxt(f'y_values{i+1}.txt') for i in range(6)]
trajectories_z = [np.loadtxt(f'z_values{i+1}.txt') for i in range(6)]

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

# 计算平均权重
dmp_x.w = weights_x / 6
dmp_y.w = weights_y / 6
dmp_z.w = weights_z / 6


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
position = 0

startime = time.time()
total_distance = 0
total_distance_hand = 0.0001



listener = keyboard.Listener(on_release=on_space)
listener.start()
#while keep_running:
for i in range(1000):
    data, addr = sock.recvfrom(1024)  # 接收最大1024字节的数据
    message_data = data.decode('utf-8')
    if "HandPosition:" in message_data:
        message_data = message_data.replace("HandPosition:", "")
        x, y, z, eulerX, eulerY, eulerZ = map(lambda s: float(s.replace(',', '')), message_data.split())
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
        disx = dx_filtered * scale_ada
        disy = dy_filtered * scale_ada
        disz = dz_filtered * scale_ada
        diseulerX = deulerX_filtered*600
        print("电机的转角：",diseulerX)

        y_track_x, _, _ = dmp_x.step()
        y_track_y, _, _ = dmp_y.step()
        y_track_z, _, _ = dmp_z.step()
        y_track_x = y_track_x + disx
        y_track_y = y_track_y + disy
        y_track_z = y_track_z + disz

        # 更新DMP的初始位置
        dmp_x.y0 = y_track_x
        dmp_y.y0 = y_track_y
        dmp_z.y0 = y_track_z

        addx = addx + disx
        addy = addy + disy
        addz = addz + disz
        dx_list.append(addx)
        dy_list.append(addy)
        dz_list.append(addz)
        eulerX_list.append(diseulerX)
        #dev.goto_pos([9999+addx, 9999-addz, 9999-addy, 19999], 10000 )
        #dev.goto_pos([9999 -addy, 9999 - addx, 9999 - addz, 19999], 10000)
        dev.goto_pos([9999 - y_track_y, 9999 - y_track_x, 9999 - y_track_z, 19999], 1000)
        position = position + diseulerX
        cmd = 'LA' + str(position) + '\n'
        #ser.write(cmd.encode())
        #time.sleep(0.05)
        cmd = 'M' + '\n'
        #ser.write(cmd.encode())
        #  time.sleep(0.05)
        #dev.goto_pos([9999 + addx, 9999 - addy, 9999 - addz, 19999], 10000)
        print(addx,addy,addz)
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
            distance = ((last_point[0] - second_last_point[0]) ** 2 +
                        (last_point[1] - second_last_point[1]) ** 2 +
                        (last_point[2] - second_last_point[2]) ** 2) ** 0.5

            # Add the distance to the total distance
            total_distance += distance
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
    # step = step + 1
    # if step == max_steps:
    #     break
    # pass

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
# Load the lists
dx_list_loaded = np.loadtxt('dx_list.txt')
dy_list_loaded = np.loadtxt('dy_list.txt')
dz_list_loaded = np.loadtxt('dz_list.txt')
eulerX_list_load = np.loadtxt('eulerX.txt')
np.savetxt('timestamps.txt', timestamps)

plt.figure()
plt.plot(dx_list, label='dx')
plt.plot(dy_list, label='dy')
plt.plot(dz_list, label='dz')
plt.xlabel('Time steps')
plt.ylabel('Displacement')
plt.legend()
plt.show()