import socket
import time
from sensapex import SensapexDevice, UMP
import matplotlib.pyplot as plt
import numpy as np
import cv2
from pynput import keyboard
ump = UMP.get_ump()
dev_ids = ump.list_devices()

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

x_pre = 0
y_pre = 0
z_pre = 0
first_run = True

dx_list = []
dy_list = []
dz_list = []


#define ada ratio
# def ada_ratio():
#
#     ratio = 10
#     return ratio

max_steps = 2000  # Set a maximum number of steps
step = 0
move_ahead = 0

# This will become False when Space is pressed, and the while-loop will break
keep_running = True


# 调用usb摄像头
cap = cv2.VideoCapture(1)

# 获取视频宽度和高度
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 定义编码器并创建VideoWriter对象，这里保存为MP4
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

# 创建一个可以调整大小的窗口
cv2.namedWindow("window1", cv2.WINDOW_NORMAL)


# def on_space(key):
#     global keep_running
#     if key == keyboard.Key.space:
#         keep_running = False
#
# listener = keyboard.Listener(on_release=on_space)
# listener.start()

#add filter
# 平滑滤波
class MovingAverageFilter:
    def __init__(self, window_size):
        self.window_size = window_size
        self.values = []

    def update(self, value):
        if len(self.values) >= self.window_size:
            self.values.pop(0)
        self.values.append(value)
        return sum(self.values) / len(self.values)

dx_filter = MovingAverageFilter(window_size=5)
dy_filter = MovingAverageFilter(window_size=5)
dz_filter = MovingAverageFilter(window_size=5)
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
###
addx = 0
addy = 0
addz = 0
while True:
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
    # elif message_data == "1":
    #     move_ahead = 1
    #     print('Received command: 向前运动')
    ret, frame = cap.read()
    if ret:
        # 将帧写入输出文件
        out.write(frame)

        cv2.imshow("window1", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break
    if first_run:
        first_run = False
    else:
        dx = x - x_pre
        dy = y - y_pre
        dz = z - z_pre
        #print('dx:', dx, 'dy:', dy, 'dz:', dz)
        # dx_filtered = dx_filter.update(dx)
        # dy_filtered = dy_filter.update(dy)
        # dz_filtered = dz_filter.update(dz)
        dx_filtered = dx_filter.update(dx)
        dy_filtered = dy_filter.update(dy)
        dz_filtered = dz_filter.update(dz)
        disx = dx_filtered * 19998
        disy = dy_filtered * 19998
        disz = dz_filtered * 19998
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
        dev.goto_pos([9999+addx, 9999+addz, 9999-addy, 19999], 10000 )
        print(addx,addy,addz)
        #time.sleep(0.5)  # sleep 0.5 sec
        #[+-0.0275]
        #(disx/9999)*0.0275
        # startPos[0] += (dz_filtered/9999)  # increase z by one
        # startPos[1] += (dx_filtered/20)
        # startPos[2] += (dy_filtered/20)
        startPos[0] += (disz/9999)*0.0275 # increase z by one
        startPos[1] += (disx/9999)*0.0275
        startPos[2] += (disy/9999)*0.0275
        posString = '({})'.format(','.join(map(str, startPos)))  # Adding parentheses to the string, example "(0,0,0)"
        print(posString)

        sock.sendto(posString.encode("UTF-8"), (host, port))  # Sending string to Unity
    #startPos = [0.2955, 0.027, 0.04105353]
    x_pre = x
    y_pre = y
    z_pre = z
    step = step + 1
    if step == max_steps:
        break

#listener.stop()

# Save the lists
np.savetxt('dx_list.txt', dx_list)
np.savetxt('dy_list.txt', dy_list)
np.savetxt('dz_list.txt', dz_list)

# Load the lists
dx_list_loaded = np.loadtxt('dx_list.txt')
dy_list_loaded = np.loadtxt('dy_list.txt')
dz_list_loaded = np.loadtxt('dz_list.txt')

plt.figure()
plt.plot(dx_list, label='dx')
plt.plot(dy_list, label='dy')
plt.plot(dz_list, label='dz')
plt.xlabel('Time steps')
plt.ylabel('Displacement')
plt.legend()
plt.show()

cap.release()
out.release()
cv2.destroyAllWindows()