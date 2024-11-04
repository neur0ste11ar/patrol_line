# coding:utf-8
# ��������ͷģ�飬��С��ʵ���Զ�ѭ����ʻ
# ˼·Ϊ:����ͷ��ȡͼ�񣬽��ж�ֵ��������ɫ������͹�Գ���
# ѡ���·���һ�����أ���ɫΪ 0
# �ҵ�0ֵ���е�
# Ŀ���е����׼�е�(320)���бȽϵó�ƫ����
# ����ƫ����������С�������ֵ�ת��
# ������ƫ�ƹ���ʧ��->ֹͣ;ƫ������һ����Χ��->����ֱ��(�������ٶȲ��ȶ�����ɾ)

# import pandas as pd
# from scipy import linalg
# import tflite_runtime.interpreter as tflite
# import threading  
# import threading  # ���� threading ��

import cv2
import numpy as np
from new_driver import driver
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

car = driver()
# ���忴��crosswalk�Ĵ���
crosswalk_num = 0
# ��ʼ�����������ͼ��ߴ�
image_size=(28,28)
# ������ͷ��ͼ��ߴ� 640*480(��*��)��opencv �洢ֵΪ 480*640(��*��) 
videostream = VideoStream(resolution=(480,640),framerate=10).start()
time.sleep(1)

# upload calibration matrix
# data = np.load('calibration.npz')
# cameraMatrix = data['cameraMatrix']
# distCoeffs = data['distCoeffs']


def process_frame_white(frame):
    hsv = cv2.cvtColor(frame,cv2.COLOR_RGB2HSV)
    lower_white = np.array([0,0,215])
    upper_white = np.array([180,30,255])
    mask = cv2.inRange(hsv,lower_white,upper_white)
    return mask

def process_frame_yellow(frame):
    hsv = cv2.cvtColor(frame,cv2.COLOR_RGB2HSV)
    lower_white = np.array([80,43,46])
    upper_white = np.array([120,255,255])
    mask = cv2.inRange(hsv,lower_white,upper_white)
    return mask

def find_track_center(binary_frame):
    bottom_row = binary_frame[-100, 40:600]
    white_pixels = np.where(bottom_row == 255)[0]
    if len(white_pixels) > 0:
        track_center = int(np.mean(white_pixels))
        return track_center
    else:
        return None

try:
    while True:

        frame = videostream.read()
        
        frame = cv2.resize(frame, (640, 480)) 
        # ����ͼ�񣬽��ж�ֵ�� 
        binary_frame_white = process_frame_white(frame)
        binary_frame_yellow = process_frame_yellow(frame)
        combined_frame = cv2.bitwise_or(binary_frame_white,binary_frame_yellow)

        track_center_white = find_track_center(binary_frame_white)
        track_center_yellow = find_track_center(binary_frame_yellow)
        if track_center_white == None or track_center_yellow == None:
            track_center = None
        else:
            track_center = (track_center_white +track_center_yellow)//2
    #frame = cv2.undistort(frame, cameraMatrix, distCoeffs, None, cameraMatrix)

        #frame = cv2.resize(frame, None, fx = 0.25, fy = 0.25, interpolation= cv2.INTER_NEAREST) #���� 160*120
        
        if track_center is not None:
            offset = track_center - 560//2
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
                car.set_speed(STRAIGHT_POWER-10, 0, -25)
            elif offset < -50:
                car.set_speed(STRAIGHT_POWER-10, 0, 25)
            elif offset > 40:
                car.set_speed(STRAIGHT_POWER, 0, -20)
            elif offset < -40:
                car.set_speed(STRAIGHT_POWER, 0, 20)
            elif offset > 20:
                car.set_speed(STRAIGHT_POWER, 0, -10)
            elif offset < -20:
                car.set_speed(STRAIGHT_POWER, 0, 10)
            elif offset > 10:
                car.set_speed(STRAIGHT_POWER, 0, -10)
            elif offset < -10:
                car.set_speed(STRAIGHT_POWER, 0, 10)
        else:
            if track_center_white == None:
                car.set_speed(STRAIGHT_POWER, 0, 45)
            if track_center_white == None:
                car.set_speed(STRAIGHT_POWER, 0, -45)
            else:
                car.set_speed(STRAIGHT_POWER, 0, 0)
            print("No track")
            
        
        cv2.imshow("frame",binary_frame_white)
        
        # ��q�������˳�
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # ȷ���ڳ����˳�ǰֹͣС��
    print("Stopping the vehicle...")
    car.set_speed(0, 0, 0)
    videostream.stop()
    cv2.destroyAllWindows()


        





        


