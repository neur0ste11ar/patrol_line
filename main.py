# coding:utf-8
# 加入摄像头模块，让小车实现自动循迹行驶
# 思路为:摄像头读取图像，进行二值化，将白色的赛道凸显出来
# 选择下方的一行像素，红色为 0
# 找到0值的中点
# 目标中点与标准中点(320)进行比较得出偏移量
# 根据偏移量来控制小车左右轮的转速
# 考虑了偏移过多失控->停止;偏移量在一定范围内->高速直行(这样会速度不稳定，已删)

# import pandas as pd
# from scipy import linalg
# import tflite_runtime.interpreter as tflite
# import threading  
# import threading  # 导入 threading 库

import cv2
import numpy as np
from new_driver import driver
import time
from threading import Thread
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image

STRAIGHT_POWER = 90

# Define VideoStream class to handle streaming of video from webcam in separate processing thread
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=20):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        #ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        #ret = self.stream.set(3,resolution[0])
        #ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

    # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
    # Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
    # Return the most recent frame
        return self.frame

    def stop(self):
    # Indicate that the camera and thread should be stopped
        self.stopped = True

car = driver()
# 定义看到crosswalk的次数
crosswalk_num = 0
# 初始化输入网络的图像尺寸
image_size=(28,28)
# 打开摄像头，图像尺寸 640*480(长*高)，opencv 存储值为 480*640(行*列) 
videostream = VideoStream(resolution=(480,640),framerate=10).start()
time.sleep(1)

# upload calibration matrix
# data = np.load('calibration.npz')
# cameraMatrix = data['cameraMatrix']
# distCoeffs = data['distCoeffs']

import os
import torch
import torchvision
from torchvision import transforms
from PIL import Image
from torch import nn
i=0 #识别图片计数

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.Conv2d(16, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3),
            nn.Conv2d(16, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(32*4*4, 100),
            nn.ReLU(),
            nn.Linear(100, 4)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 32*4*4)
        x = self.fc(x)
        # x = nn.functional.normalize(x)
        return x

def process_frame(frame):
    hsv = cv2.cvtColor(frame,cv2.COLOR_RGB2HSV)
    lower_white = np.array([0,0,235])
    upper_white = np.array([180,10,255])
    mask = cv2.inRange(hsv,lower_white,upper_white)
    return mask

def find_track_center(binary_frame):
    bottom_row = binary_frame[-230, 80:560]
    white_pixels = np.where(bottom_row == 255)[0]
    if len(white_pixels) > 0:
        track_center = int(np.mean(white_pixels))
        return track_center
    else:
        return None


try:
    while True:
        sign=-1

        frame = videostream.read()
        
        frame = cv2.resize(frame, (640, 480))
        binary_frame = process_frame(frame)

        track_center = find_track_center(binary_frame)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_color = np.array([100, 120, 0])
        upper_color = np.array([140, 255, 255])
        mask = cv2.inRange(hsv, lower_color, upper_color)
        part = cv2.bitwise_and(frame, frame, mask=mask)
        gray = cv2.cvtColor(part, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 2)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=100,
            param2=20,
            minRadius=13,
            maxRadius=300
        )
        circle = frame
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center_x, center_y, radius = i[0], i[1], i[2]
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                #cv2.circle(frame, (center_x, center_y), radius, 255, -1)
                circle = frame[center_y-radius:center_y+radius, center_x-radius:center_x+radius]
            # Convert the NumPy array to a PIL Image
            image = Image.fromarray(circle)
            
            data_class=['left','pause','right','straight']   #按文件索引顺序排列
            transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),  # 将图像转换为单通道灰度图
                transforms.Resize((28, 28)),                 # 调整图像大小为 28x28
                transforms.ToTensor(),                       # 将图像转换为张量，值缩放到 [0, 1]
                transforms.Normalize((0.5,), (0.5,))         # 归一化到 [-1, 1]，均值和标准差为 0.5
            ])
            image=transform(image)
            
            model_ft=Net()     #需要使用训练时的相同模型
            # print(model_ft)

            model=torch.load("best_model.pth",map_location=torch.device("cpu")) #选择训练后得到的模型文件
            # print(model)
            image=torch.reshape(image,(1,1,28,28))      #修改待预测图片尺寸，需要与训练时一致
            model.double()
            model.eval()
            with torch.no_grad():
                output=model(image.double())
            print(output)               #输出预测结果
            # print(int(output.argmax(1)))
            sign=int(output.argmax(1))
            print(int(output.argmax(1)))
        if sign == 0:
            car.set_speed(STRAIGHT_POWER, 0, 60)
            time.sleep(0.8)
            continue
        elif sign == 1:
            car.set_speed(0, 0, 0)
            time.sleep(1)
            continue
        elif sign == 2:
            car.set_speed(STRAIGHT_POWER, 0, -60)
            time.sleep(0.8)
            continue
        elif sign == 3:
            car.set_speed(STRAIGHT_POWER, 0, 0)
            time.sleep(1)
            continue
        if track_center is not None:
            offset = track_center - 480//2
            print(offset)
            #k = 0.1*abs(offset)
            
            turn_power = STRAIGHT_POWER*0.8
        
            if abs(offset) <= 10:
                car.set_speed(STRAIGHT_POWER, 0, 0)
            elif offset > 60:
                car.set_speed(STRAIGHT_POWER-30, 0, -30)
            elif offset < -60:
                car.set_speed(STRAIGHT_POWER-30, 0, 30)
            elif offset > 50:
                car.set_speed(STRAIGHT_POWER-10, 0, -20)
            elif offset < -50:
                car.set_speed(STRAIGHT_POWER-10, 0, 20)
            elif offset > 40:
                car.set_speed(STRAIGHT_POWER, 0, -10)
            elif offset < -40:
                car.set_speed(STRAIGHT_POWER, 0, 10)
            elif offset > 20:
                car.set_speed(STRAIGHT_POWER, 0, -5)
            elif offset < -20:
                car.set_speed(STRAIGHT_POWER, 0, 5)
            elif offset > 10:
                car.set_speed(STRAIGHT_POWER, 0, 0)
            elif offset < -10:
                car.set_speed(STRAIGHT_POWER, 0, 0)
        else:
            print("No track")
            car.set_speed(STRAIGHT_POWER, 0, 0)
        
        cv2.imshow("frame",binary_frame)
        
        # 按q键可以退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 确保在程序退出前停止小车
    print("Stopping the vehicle...")
    car.set_speed(0, 0, 0)
    videostream.stop()
    cv2.destroyAllWindows()


