from __future__ import absolute_import

import os
import glob
import numpy as np

from siamfc import TrackerSiamFC

import cv2

def mouse_callback(event, x, y, flags, param):
    global box_start, drawing_box, box_coordinates

    if event == cv2.EVENT_LBUTTONDOWN:
        box_start = (x, y)
        drawing_box = True
    elif event == cv2.EVENT_LBUTTONUP:
        drawing_box = False
        img = frame.copy()
        cv2.rectangle(img, box_start, (x, y), (255, 0, 0), 2)
        cv2.imshow("Frame", img)
        # 获取框的坐标
        box_coordinates = [box_start[0], box_start[1], x-box_start[0], y-box_start[1]]
        print("Box Coordinates:", box_coordinates)
    elif drawing_box == True:
        img = frame.copy()
        cv2.rectangle(img, box_start, (x, y), (255, 0, 0), 2)
        cv2.imshow("Frame", img)

if __name__ == '__main__':
    # 替换成你的test视频文件路径
    video_path = "~/2_dataset/football_my_dataset/IMG_0773.MOV"
    video_path = os.path.expanduser(video_path)
    
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if ret:
        drawing_box = False
        box_start = None
        
        cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Frame', width, height)
        cv2.imshow('Frame', frame)
        cv2.setMouseCallback("Frame", mouse_callback)
        while True:
            key = cv2.waitKey(1) & 0xFF
    	    #当按或者回车时结束循环
            if key == ord('q') or key == ord('\r') :
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Failed to read video file.")
    
    #if use groundtruth_rect.txt cancel next two lines
    #seq_dir = os.path.expanduser('~/2_dataset/football_my_dataset/')
    #anno = np.loadtxt(seq_dir + 'groundtruth_rect.txt', delimiter=',')
    
    net_path = 'pretrained/siamfc_alexnet_e50.pth'
    tracker = TrackerSiamFC(net_path=net_path)
    tracker.track_video(video_path, box_coordinates, visualize=True)
