import socket
import time
import numpy as np
server_ip = '10.167.156.160'  # 服务器IP，和Unity脚本中的服务器IP一致
server_port = 16306  # 服务器端口，和Unity脚本中的服务器端口一致
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # 创建UDP socket
sock.bind((server_ip, server_port))  # 绑定到指定的IP和端口

#host, port = "127.0.0.1", 16308
#host,port ="172.16.2.76",16308
host, port = "10.167.104.130", 16308
#host, port = "10.167.156.160", 16308

#用于计算G和S
points = []
sigma = 1
vp = 1

beta_1 = 0
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


#unity初始
startPos = [0.268, -0.002440972, 0.013553]  # Vector3   x = 0, y = 0, z = 0

x_pre = 0
y_pre = 0
z_pre = 0
first_run = True

#define ada ratio
# def ada_ratio():
#
#     ratio = 10
#     return ratio


while True:
    data, addr = sock.recvfrom(1024)  # 接收最大1024字节的数据
   #print('Received message:', data.decode('utf-8'))  # 将接收到的字节数据解码成字符串并打印
    message_data = data.decode('utf-8')
    message_data = message_data.replace("HandPosition:", "")
    x, y, z = map(lambda s: float(s.replace(',', '')), message_data.split())
    if first_run:
        first_run = False
    else:
        dx = x - x_pre
        dy = y - y_pre
        dz = z - z_pre
        print('dx:', dx, 'dy:', dy, 'dz:', dz)
        #time.sleep(0.5)  # sleep 0.5 sec
        startPos[0] += (dz/(6+beta_1))  # increase z by one
        startPos[1] += (dx/(6+beta_1))
        startPos[2] += (dy/(6+beta_1))
        posString = '({})'.format(','.join(map(str, startPos)))  # Adding parentheses to the string, example "(0,0,0)"
        print(posString)

        sock.sendto(posString.encode("UTF-8"), (host, port))  # Sending string to Unity

        # Add the new point to the list
        points.append([startPos[0], startPos[1], startPos[2]])
        if len(points) >= 10:  # Replace 4 with the actual number of points you need
            points_array = np.array(points)
            G = calculate_G(points_array)
            S = calculate_S(points_array, sigma, vp)
            print("G:", G)
            print("S:", S)


    x_pre = x
    y_pre = y
    z_pre = z