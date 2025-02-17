import socket

server_ip = '10.167.156.160'  # 服务器IP，和Unity脚本中的服务器IP一致
server_port = 16306  # 服务器端口，和Unity脚本中的服务器端口一致
# server_ip = '10.167.105.72'
# server_port = 16308
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # 创建UDP socket
sock.bind((server_ip, server_port))  # 绑定到指定的IP和端口

while True:
    data, addr = sock.recvfrom(1024)  # 接收最大1024字节的数据
    message_data = data.decode('utf-8')
    if "HandPosition:" in message_data:
        message_data = message_data.replace("HandPosition:", "")
        x, y, z, eulerX, eulerY, eulerZ = map(lambda s: float(s.replace(',', '')), message_data.split())
        print('Hand position x:', x, 'y:', y, 'z:', z)
        print('Hand rotation Euler angles x:', eulerX, 'y:', eulerY, 'z:', eulerZ)
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
    elif message_data == "2":
        #env.good = 1
        print('Received command: good')
    elif message_data == "3":
        #env.bad = 1
        print('Received command: bad')
    elif message_data == "4":
        #env.finished = 1
        print('Received command: done')
    elif message_data == "increase":
        print('Received command: increase')
    elif message_data =="decrease":
        print('Received command: decrease')
    elif "deepvalue:" in message_data:
        value = float(message_data.replace("deepvalue:", ""))
        print('deepvalue:',value)

