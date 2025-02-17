import socket
import time

server_ip = '10.167.156.160'  # 服务器IP，和Unity脚本中的服务器IP一致
server_port = 16306  # 服务器端口，和Unity脚本中的服务器端口一致
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # 创建UDP socket
sock.bind((server_ip, server_port))  # 绑定到指定的IP和端口

#host, port = "127.0.0.1", 16308
#host,port ="172.16.2.76",16308
host, port = "10.167.104.130", 16308
#host, port = "10.167.156.160", 16308
#joint[0.268+-0.0275]
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
        startPos[0] += (dz/10)  # increase z by one
        startPos[1] += (dx/10)
        startPos[2] += (dy/10)
        posString = '({})'.format(','.join(map(str, startPos)))  # Adding parentheses to the string, example "(0,0,0)"
        print(posString)

        sock.sendto(posString.encode("UTF-8"), (host, port))  # Sending string to Unity
    x_pre = x
    y_pre = y
    z_pre = z





