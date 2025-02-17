import socket


server_ip = '10.167.156.160'  # 服务器IP，和Unity脚本中的服务器IP一致
server_port = 16306  # 服务器端口，和Unity脚本中的服务器端口一致
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # 创建UDP socket
sock.bind((server_ip, server_port))  # 绑定到指定的IP和端口

x_pre = 0
y_pre = 0
z_pre = 0
first_run = True
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

    x_pre = x
    y_pre = y
    z_pre = z



