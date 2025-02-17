import cv2
import time
import numpy as np
import socket
import struct
import pickle

# Open the webcam
cap = cv2.VideoCapture(2)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error opening video stream or file")
    exit()

# Create a socket object
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Get the local machine name
host = '10.167.105.72'  # IP of the remote computer

# Reserve a port for your service
port = 16309

# Bind to the port
try:
    s.connect((host, port))
except Exception as e:
    print("Could not connect to server:", str(e))
    exit()

try:
    while True:
        ret, frame = cap.read()
        if ret:
            # Serialize frame
            data = pickle.dumps(frame)

            # Send the length of the data first
            message_size = struct.pack("L", len(data))

            # Then data
            try:
                s.sendall(message_size + data)
                print("data:", len(data))
            except Exception as e:
                print("Could not send data:", str(e))
                break

            # Show the frame
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Camera could not read frame")
            break
except KeyboardInterrupt:
    print('Interrupted')
finally:
    cap.release()
    if s:
        s.close()
    cv2.destroyAllWindows()  # Added to close the frame window

