import asyncio
import websockets
import cv2
import numpy as np
import base64

async def server(websocket, path):
    cap = cv2.VideoCapture(0)  # 打开摄像头

    while True:
        ret, frame = cap.read()  # 读取一帧图像
        _, buffer = cv2.imencode('.jpg', frame)  # 将图像编码为JPEG格式
        jpeg_as_text = base64.b64encode(buffer)  # 将JPEG图像编码为base64格式

        await websocket.send(jpeg_as_text)  # 通过WebSocket发送图像

start_server = websockets.serve(server, 'localhost', 8765)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
