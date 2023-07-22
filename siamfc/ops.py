from __future__ import absolute_import, division

import torch.nn as nn
import cv2
import numpy as np


def init_weights(model, gain=1):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def read_image(img_file, cvt_code=cv2.COLOR_BGR2RGB):
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    if cvt_code is not None:
        img = cv2.cvtColor(img, cvt_code)
    return img


def show_image(img, boxes=None, box_fmt='ltwh', colors=None,
               thickness=3, fig_n=1, delay=1, visualize=True,
               cvt_code=cv2.COLOR_RGB2BGR):
    if cvt_code is not None:
        img = cv2.cvtColor(img, cvt_code)
    
    # resize img if necessary
    max_size = 1920
    if max(img.shape[:2]) > max_size:
        scale = max_size / max(img.shape[:2])
        out_size = (
            int(img.shape[1] * scale),
            int(img.shape[0] * scale))
        img = cv2.resize(img, out_size)
        if boxes is not None:
            boxes = np.array(boxes, dtype=np.float32) * scale
    
    if boxes is not None:
        assert box_fmt in ['ltwh', 'ltrb']
        boxes = np.array(boxes, dtype=np.int32)
        if boxes.ndim == 1:
            boxes = np.expand_dims(boxes, axis=0)
        if box_fmt == 'ltrb':
            boxes[:, 2:] -= boxes[:, :2]
        
        # clip bounding boxes
        bound = np.array(img.shape[1::-1])[None, :]
        boxes[:, :2] = np.clip(boxes[:, :2], 0, bound)
        boxes[:, 2:] = np.clip(boxes[:, 2:], 0, bound - boxes[:, :2])
        
        if colors is None:
            colors = [
                (0, 0, 255),
                (0, 255, 0),
                (255, 0, 0),
                (0, 255, 255),
                (255, 0, 255),
                (255, 255, 0),
                (0, 0, 128),
                (0, 128, 0),
                (128, 0, 0),
                (0, 128, 128),
                (128, 0, 128),
                (128, 128, 0)]
        colors = np.array(colors, dtype=np.int32)
        if colors.ndim == 1:
            colors = np.expand_dims(colors, axis=0)
        
        for i, box in enumerate(boxes):
            color = colors[i % len(colors)]
            pt1 = (box[0], box[1])
            pt2 = (box[0] + box[2], box[1] + box[3])
            img = cv2.rectangle(img, pt1, pt2, color.tolist(), thickness)
    
    if visualize:
        winname = 'window_{}'.format(fig_n)
        cv2.imshow(winname, img)
        cv2.waitKey(delay)

    return img

def resize_with_padding(image, target_size, padding_color=(0, 0, 0)):
    original_height, original_width = image.shape[:2]
    target_width, target_height = target_size

    # 计算缩放比例
    width_ratio = target_width / original_width
    height_ratio = target_height / original_height

    # 根据较小的比例进行缩放，保持原始宽高比
    if width_ratio < height_ratio:
        new_width = target_width
        new_height = int(original_height * width_ratio)
    else:
        new_height = target_height
        new_width = int(original_width * height_ratio)

    # 使用cv2.resize()函数进行缩放
    resized_image = cv2.resize(image, (new_width, new_height))

    # 计算填充边缘的尺寸
    top = (target_height - new_height) // 2
    bottom = target_height - new_height - top
    left = (target_width - new_width) // 2
    right = target_width - new_width - left

    # 使用cv2.copyMakeBorder()函数进行填充
    padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)

    return padded_image

def crop_and_resize(img, center, size, out_size,
                    border_type=cv2.BORDER_CONSTANT,
                    border_value=(0, 0, 0),
                    interp=cv2.INTER_LINEAR):
    # convert box to corners (0-indexed)
    # round(size) 对size四舍五入取整，round(size, 2)保留两位小数
    size = np.round(size)
    # print(f'size = {size}')
    # print(f'center = {center}')
    corners = np.concatenate((
        np.round(center - (size - 1) / 2),
        np.round(center - (size - 1) / 2) + size))
    #  print(f'corners = {corners}')[816., 1729., 858., 1771.]
    corners = np.round(corners).astype(int)
    # print(f'corners = {corners}') # [816, 1729, 858, 1771]
    # pad image if necessary
    pads = np.concatenate((
        -corners[:2], corners[2:] - img.shape[:2]))
    # print(pads) #[-816, -1729, -222, -149]
    npad = max(0, int(pads.max()))
    # print(f'img.shape = {img.shape}')
    if npad > 0:
        img = cv2.copyMakeBorder(
            img, npad, npad, npad, npad,
            border_type, value=border_value)
        print(f'in img.shape = {img.shape}')
    # print(f'out img.shape = {img.shape}')
    # crop image patch
    corners = (corners + npad).astype(int)
    # print(f'corners = {corners}')
    patch = img[corners[0]:corners[2], corners[1]:corners[3]]
    # print(f'patch.shape = {patch.shape}, out_size = {out_size}')
    # cv2.imshow("Image with Trajectory", patch)
    # cv2.waitKey(0)
    # resize to out_size
    # 方案二：
    patch = cv2.resize(patch, (out_size, out_size),
                       interpolation=interp)
    # 方案一：
    # patch = resize_with_padding(patch, (out_size, out_size))
    # cv2.imshow("Image with Trajectory", patch)
    # cv2.waitKey(0)
    return patch
