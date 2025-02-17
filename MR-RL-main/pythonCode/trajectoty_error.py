import cv2
import numpy as np

# 读取模板
template = cv2.imread('moban.png',0)

# 获取模板的宽度和高度
w, h = template.shape[::-1]

# 读取视频
cap = cv2.VideoCapture('error.mov')

while(cap.isOpened()):
    # 读取一帧
    ret, frame = cap.read()
    if ret == True:
        # 转换到HSV图像
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 定义褐色的HSV范围
        lower_brown = np.array([40, 40, 20])
        upper_brown = np.array([60, 50, 30])

        # 根据定义的HSV颜色范围，创建一个褐色的掩膜
        mask = cv2.inRange(hsv, lower_brown, upper_brown)

        # 对原图像和掩膜进行位运算，过滤掉除褐色以外的部分
        brown_only = cv2.bitwise_and(frame, frame, mask=mask)
        # 转换到灰度图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 进行模板匹配
        res = cv2.matchTemplate(gray,template,cv2.TM_CCOEFF_NORMED)

        # 设置阈值
        threshold = 0.8

        # 找到匹配的位置
        loc = np.where( res >= threshold)

        # 在匹配的位置上画矩形
        for pt in zip(*loc[::-1]):
            cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

        # 显示结果
        cv2.imshow('Detected',frame)

        # 按q退出
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
