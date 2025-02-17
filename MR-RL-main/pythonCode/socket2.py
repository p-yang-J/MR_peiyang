import socket
import time

#host, port = "127.0.0.1", 16308
#host,port ="172.16.2.76",16308
#host, port = "10.167.104.130", 16308
host, port = "10.167.110.20", 16308
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

startPos = [0.268, -0.002440972, 0.013553]  # Vector3   x = 0, y = 0, z = 0
while True:
    time.sleep(0.5)  # sleep 0.5 sec
    startPos[0] += 0.0005  # increase x by one
    startPos[1] += 0.0005
    startPos[2] += 0.0005
    posString = '({})'.format(','.join(map(str, startPos)))  # Adding parentheses to the string, example "(0,0,0)"
    print(posString)

    sock.sendto(posString.encode("UTF-8"), (host, port))  # Sending string to Unity








