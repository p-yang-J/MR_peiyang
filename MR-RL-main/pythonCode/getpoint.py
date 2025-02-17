import cv2
import numpy as np
import matplotlib.pyplot as plt
import pyrealsense2 as rs
from sensapex import UMP
from sensapex import SensapexDevice, UMP

## open cam
# 调用usb摄像头
#cap = cv2.VideoCapture(3)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

#cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc("M", "J", "P", "G"))  # ?1为啥要重设

# 显示
##
# code about manipulator
ump = UMP.get_ump()
dev_ids = ump.list_devices()


devs = {i: SensapexDevice(i) for i in dev_ids}

print("SDK version:", ump .sdk_version())
print("Found device IDs:", dev_ids)

#stage = ump.get_device(1)
#stage.calibrate_zero_position()

def print_pos(timeout=None):
    line = ""
    for i in dev_ids:
        dev = devs[i]
        try:
            pos = str(dev.get_pos(timeout=timeout))
        except Exception as err:
            pos = str(err.args[0])
        pos = pos + " " * (30 - len(pos))
        line += f"{i:d}:  {pos}"
    print(line)

print_pos()
dev = devs[1]
print(dev.get_pos())


dev.goto_pos([9999.837890625, 9999.900390625, 9999.837890625, 9999.7841796875], 5000)
#####

pipe = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
profile = pipe.start(config)

frameset = pipe.wait_for_frames()
color_frame = frameset.get_color_frame()
color_init = np.asanyarray(color_frame.get_data())

font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10, 500)
fontScale = 1
fontColor = (255, 255, 255)
lineType = 2

try:
    while True:
        ## get the state number from txt(matlab)
        with open("statandstop.txt", mode='r', encoding='utf-8') as notes:
            state_number = notes.read()
        #####
        ## open cam
        #ret, frame = cap.read()
        #cv2.imshow("window1", frame)
        ##
        # Store next frameset for later processing:
        frameset = pipe.wait_for_frames()
        color_frame = frameset.get_color_frame()
        depth_frame = frameset.get_depth_frame()

        color = np.asanyarray(color_frame.get_data())
        res = color.copy()
        hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)

        l_b = np.array([100, 43, 46])
        u_b = np.array([124, 255, 255])

        mask = cv2.inRange(hsv, l_b, u_b)
        color = cv2.bitwise_and(color, color, mask=mask)

        colorizer = rs.colorizer()
        colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())

        # Create alignment primitive with color as its target stream:
        align = rs.align(rs.stream.color)
        frameset = align.process(frameset)

        # Update color and depth frames:
        aligned_depth_frame = frameset.get_depth_frame()
        colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())

        ### motion detector
        d = cv2.absdiff(color_init, color)
        gray = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, th = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(th, np.ones((3, 3), np.uint8), iterations=3)
        (c, _) = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(color, c, -1, (0, 255, 0), 2)
        color_init = color

        depth = np.asanyarray(aligned_depth_frame.get_data())

        for contour in c:
            if cv2.contourArea(contour) < 1500:
                continue
            (x, y, w, h) = cv2.boundingRect(contour)
            disx = (x+w/2)*40
            disy = (y +h/2)*40
            print("x")
            print(disx)
            print("y")
            print(disy)
            bottomLeftCornerOfText = (x, y)

            # Crop depth data:
            depth = depth[x:x + w, y:y + h].astype(float)

            depth_crop = depth.copy()

            if depth_crop.size == 0:
                continue
            depth_res = depth_crop[depth_crop != 0]

            # Get data scale from the device and convert to meters
            depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
            depth_res = depth_res * depth_scale

            if depth_res.size == 0:
                continue

            dist = min(depth_res)*30000
            print("dist")
            print(dist)

            ### code about maipulator
            #manipulator.goto_pos((y1 + disy, z1 + dist, x1 + disx), speed=200)
            dev.goto_pos([999.837890625+disx, 999.900390625+dist, 999.837890625+disy, 999.7841796875], 5000)
            ######
            ## 存储数据
            q1 = str(disx)
            q2 = str(disy)
            q3 = str(dist)
            q4 = str(0)
            q5 = str(0)
            qq = q1 + ' ' + q2 + ' ' + q3 + ' ' + q4 + ' ' + q5 + '\n'
            with open("shujux.txt", mode='a',encoding='utf-8') as notex:
                notex.write(qq)
                notex.close()
            ################
            cv2.rectangle(res, (x, y), (x + w, y + h), (0, 255, 0), 3)
            text = "Depth: " + str("{0:.2f}").format(dist) + "X:" + str("{0:.2f}").format(disx) +"Y:" +str("{0:.2f}").format(disy)
            cv2.putText(res,
                        text,
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        lineType)

        cv2.namedWindow('RBG', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RBG', res)
        cv2.namedWindow('Depth', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Depth', colorized_depth)
        cv2.namedWindow('mask', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('mask', mask)

        cv2.waitKey(1)

finally:
    pipe.stop()