# import cv2
# import time
# # 调用usb摄像头
# cap = cv2.VideoCapture(1)
# # 获取视频宽度和高度
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# # 定义编码器并创建VideoWriter对象，这里保存为MP4
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))
# # 创建一个可以调整大小的窗口
# cv2.namedWindow("window1", cv2.WINDOW_NORMAL)
# # 创建一个空的列表用于存储每帧的时间戳
# timestamps = []
# try:
#     # 显示
#     while True:
#         ret, frame = cap.read()
#         if ret:
#             # 记录当前时间戳
#             timestamps.append(time.time())
#
#             # 将帧写入输出文件
#             out.write(frame)
#
#             cv2.imshow("window1", frame)
#
#             if cv2.waitKey(1) & 0xFF == ord("q"):
#                 break
#         else:
#             break
# except KeyboardInterrupt:
#     print('Interrupted')
# finally:
#     # 保存时间戳到文件
#     with open('timestamps_video.txt', 'w') as f:
#         for ts in timestamps:
#             f.write(f'{ts}\n')
#
#     # 关闭
#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()

import cv2
import time

# 调用usb摄像头
cap = cv2.VideoCapture(0)

# 获取视频宽度和高度
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 定义编码器并创建VideoWriter对象，这里保存为AVI
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (frame_width, frame_height))

# 创建一个可以调整大小的窗口
cv2.namedWindow("window1", cv2.WINDOW_NORMAL)

# 创建一个空的列表用于存储每帧的时间戳
timestamps = []
frame_count = 0

try:
    # 显示
    while True:
        ret, frame = cap.read()
        if ret:
            # 记录当前时间戳
            timestamps.append(time.time())

            # 将帧写入输出文件
            out.write(frame)
            frame_count += 1

            cv2.imshow("window1", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
except KeyboardInterrupt:
    print('Interrupted')

finally:
    # 保存时间戳到文件
    with open('timestamps_video.txt', 'w') as f:
        for ts in timestamps:
            f.write(f'{ts}\n')
    print(f'Total frames written: {frame_count}')
    # 关闭
    cap.release()
    out.release()
    cv2.destroyAllWindows()