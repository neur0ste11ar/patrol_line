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
#from new_driver import driver
import time
from threading import Thread
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image

STRAIGHT_POWER = 100

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

#car = driver()
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
root_path="./model/test"         #待测试文件夹
names=os.listdir(root_path)

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

try:
    while True:

        frame = videostream.read()
        
        frame = cv2.resize(frame, (640, 480)) 
        
        # Convert the NumPy array to a PIL Image
        image = Image.fromarray(frame)
        
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
        model.eval()
        with torch.no_grad():
            output=model(image)
        print(output)               #输出预测结果
        # print(int(output.argmax(1)))
        print("预测为：{}".format(data_class[int(output.argmax(1))]))   #对结果进行处理，使直接显示出预测的植物种类print(image.shape)
        
        
        cv2.imshow("frame",frame)
        
        # 按q键可以退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 确保在程序退出前停止小车
    print("Stopping the vehicle...")
    #car.set_speed(0, 0, 0)
    videostream.stop()
    cv2.destroyAllWindows()

