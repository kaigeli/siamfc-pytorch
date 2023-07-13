from __future__ import absolute_import

import argparse
import os
import glob
import numpy as np

from siamfc import TrackerSiamFC

import cv2

def parse_arguments():
    parser = argparse.ArgumentParser(description="Training script")
    # 替换成你的test视频文件路径
    parser.add_argument('-v', '--video_path', default='~/2_dataset/football_my_dataset/IMG_0773.MOV', type=str,
                        help="you will track the video path.")
    parser.add_argument('-m', '--model_path', default='./pretrained/siamfc_alexnet_e50.pth', type=str,
                        help="the pretrained model path.")
    args = parser.parse_args()
    return args

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
        # 获取框的坐标[x, y, w, h]
        box_coordinates = [box_start[0], box_start[1], x-box_start[0], y-box_start[1]]
        print("Box Coordinates:", box_coordinates)
    elif event == cv2.EVENT_MOUSEMOVE and flags==cv2.EVENT_FLAG_LBUTTON:
        img = frame.copy()
        cv2.rectangle(img, box_start, (x, y), (255, 0, 0), 2)
        cv2.imshow("Frame", img)

def get_box(video_path):
    global frame

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if ret:
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

if __name__ == '__main__':
    args = parse_arguments()
    video_path = os.path.expanduser(args.video_path)
    get_box(video_path)
    
    tracker = TrackerSiamFC(net_path=args.model_path)
    tracker.track_video(video_path, box_coordinates, visualize=True)
