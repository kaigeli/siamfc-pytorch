from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import cv2
import sys
import os
from collections import namedtuple
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from got10k.trackers import Tracker

from . import ops
from .backbones import AlexNetV1
from .heads import SiamFC
from .losses import BalancedLoss
from .datasets import Pair
from .transforms import SiamFCTransforms
from tqdm import tqdm
import matplotlib.pyplot as plt

__all__ = ['TrackerSiamFC']


class Net(nn.Module):

    def __init__(self, backbone, head):
        super(Net, self).__init__()
        self.backbone = backbone
        self.head = head
    
    def forward(self, z, x):
        z = self.backbone(z)
        x = self.backbone(x)
        return self.head(z, x)


class TrackerSiamFC(Tracker):

    def __init__(self, net_path=None, **kwargs):
        super(TrackerSiamFC, self).__init__('SiamFC', True)
        self.cfg = self.parse_args(**kwargs)

        # setup GPU device if available
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        # setup model
        self.net = Net(
            backbone=AlexNetV1(),
            head=SiamFC(self.cfg.out_scale))
        ops.init_weights(self.net)
        
        # load checkpoint if provided
        if net_path is not None:
            self.net.load_state_dict(torch.load(
                net_path, map_location=lambda storage, loc: storage))
        self.net = self.net.to(self.device)

        # setup criterion
        self.criterion = BalancedLoss()

        # setup optimizer
        self.optimizer = optim.SGD(
            self.net.parameters(),
            lr=self.cfg.initial_lr,
            weight_decay=self.cfg.weight_decay,
            momentum=self.cfg.momentum)
        
        # setup lr scheduler
        gamma = np.power(
            self.cfg.ultimate_lr / self.cfg.initial_lr,
            1.0 / self.cfg.epoch_num)
        self.lr_scheduler = ExponentialLR(self.optimizer, gamma)

    def parse_args(self, **kwargs):
        # default parameters
        cfg = {
            # basic parameters
            'out_scale': 0.001,
            'exemplar_sz': 127,
            'instance_sz': 255,
            'context': 0.5,
            # inference parameters
            'scale_num': 3,
            'scale_step': 1.0375,
            'scale_lr': 0.59,
            'scale_penalty': 0.9745,
            'window_influence': 0.176,
            'response_sz': 17,
            'response_up': 16,
            'total_stride': 8,
            # train parameters
            'epoch_num': 500,
            'batch_size': 64,
            'num_workers': 16,
            'initial_lr': 1e-1,
            'ultimate_lr': 1e-6,
            'weight_decay': 5e-4,
            'momentum': 0.9,
            'r_pos': 16,
            'r_neg': 0}
        
        for key, val in kwargs.items():
            if key in cfg:
                cfg.update({key: val})
        return namedtuple('Config', cfg.keys())(**cfg)
    
    @torch.no_grad()
    def init(self, img, box):
        # set to evaluation mode
        self.net.eval()

        # convert box to 0-indexed and center based [y, x, h, w]
        box = np.array([
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]], dtype=np.float32)
        self.center, self.target_sz = box[:2], box[2:]

        # create hanning window 272 = 16*17
        self.upscale_sz = self.cfg.response_up * self.cfg.response_sz
        self.hann_window = np.outer(
            np.hanning(self.upscale_sz),
            np.hanning(self.upscale_sz))
        # self.hann_window.sum() = 18360.25
        self.hann_window /= self.hann_window.sum()
        # 0~1/18360.25,中间大，四周小

        # search scale factors
        # in paper scale_factors = 1.025{-2,-1,0,1,2},这里用的1.035
        # np.linspace(-2,2,5) = array([-2., -1.,  0.,  1.,  2.])
        # 1.035** np.linspace(-2,2,5) = array([0.9335107, 0.96618357, 1., 1.035, 1.071225])
        self.scale_factors = self.cfg.scale_step ** np.linspace(
            -(self.cfg.scale_num // 2),
            self.cfg.scale_num // 2, self.cfg.scale_num)

        # exemplar and search sizes
        # self.target_sz = 21,20
        # context = 0.5*41 = 20.5 (football 宽高大概为25，25)
        # self.z_sz = np.sqrt(1680.75) = 40.997
        # self.x_sz = 82.3167
        # x_sz 一般为z_sz的两倍大小，因为256/128 = 2
        # z_sz 和x_sz为crop的大小，即crop出来一个z_sz*z_sz的图片，再resize到exemplar_sz即127*127
        # crop和resize方法可以优化
        # np.prod是计算元素相乘
        context = self.cfg.context * np.sum(self.target_sz)
        self.z_sz = np.sqrt(np.prod(self.target_sz+context))
        self.x_sz = self.z_sz * \
            self.cfg.instance_sz / self.cfg.exemplar_sz
        print(f'context={context},z_sz = {self.z_sz}, z_xz = {self.x_sz}')
        # exemplar image
        self.avg_color = np.mean(img, axis=(0, 1))
        # z_sz for crop, exemplar_sz 为输出大小，首帧为127
        z = ops.crop_and_resize(
            img, self.center, self.z_sz,
            out_size=self.cfg.exemplar_sz,
            border_value=self.avg_color)

        #print(f'z.shape = {z.shape}') out z.shape = (127, 127, 3)
        # exemplar features
        # .permute(2, 0, 1)作用是将 (127, 127, 3)转为（3，127，127）
        # .unsqueeze(0)作用是为Tensor增加一个维度
        z = torch.from_numpy(z).to(
            self.device).permute(2, 0, 1).unsqueeze(0).float()
        # print(z.shape) #out torch.Size([1, 3, 127, 127])
        self.kernel = self.net.backbone(z)
        # print(self.kernel.shape)
        # self.kernel.shape = torch.Size([1, 256, 6, 6])
    @torch.no_grad()
    def update(self, img):
        # set to evaluation mode
        self.net.eval()

        # search images
        x = [ops.crop_and_resize(
            img, self.center, self.x_sz * f,
            out_size=self.cfg.instance_sz,
            border_value=self.avg_color) for f in self.scale_factors]

        x = np.stack(x, axis=0)
        x = torch.from_numpy(x).to(
            self.device).permute(0, 3, 1, 2).float()
        
        # responses
        x = self.net.backbone(x)
        responses = self.net.head(self.kernel, x)
        responses = responses.squeeze(1).cpu().numpy()

        # upsample responses and penalize scale changes
        responses = np.stack([cv2.resize(
            u, (self.upscale_sz, self.upscale_sz),
            interpolation=cv2.INTER_CUBIC)
            for u in responses])

        responses[:self.cfg.scale_num // 2] *= self.cfg.scale_penalty
        responses[self.cfg.scale_num // 2 + 1:] *= self.cfg.scale_penalty
        # peak scale
        scale_id = np.argmax(np.amax(responses, axis=(1, 2)))
        # peak location
        response = responses[scale_id]
        response -= response.min()
        response /= response.sum() + 1e-16
        response = (1 - self.cfg.window_influence) * response + \
            self.cfg.window_influence * self.hann_window
        loc = np.unravel_index(response.argmax(), response.shape)

        # locate target center
        disp_in_response = np.array(loc) - (self.upscale_sz - 1) / 2
        disp_in_instance = disp_in_response * \
            self.cfg.total_stride / self.cfg.response_up
        disp_in_image = disp_in_instance * self.x_sz * \
            self.scale_factors[scale_id] / self.cfg.instance_sz
        self.center += disp_in_image

        # update target size
        scale =  (1 - self.cfg.scale_lr) * 1.0 + \
            self.cfg.scale_lr * self.scale_factors[scale_id]
        self.target_sz *= scale
        self.z_sz *= scale
        self.x_sz *= scale

        # return 1-indexed and left-top based bounding box
        box = np.array([
            self.center[1] + 1 - (self.target_sz[1] - 1) / 2,
            self.center[0] + 1 - (self.target_sz[0] - 1) / 2,
            self.target_sz[1], self.target_sz[0]])

        return box
    
    def track(self, img_files, box, visualize=False):
        frame_num = len(img_files)
        boxes = np.zeros((frame_num, 4))
        boxes[0] = box
        times = np.zeros(frame_num)

        for f, img_file in enumerate(img_files):
            img = ops.read_image(img_file)

            begin = time.time()
            if f == 0:
                self.init(img, box)
            else:
                boxes[f, :] = self.update(img)
            times[f] = time.time() - begin

            if visualize:
                ops.show_image(img, boxes[f, :])

        return boxes, times

    def track_video(self, video_path, box, visualize=False):
        cap = cv2.VideoCapture(video_path)
        # 初始化帧计数器
        frame_count = 0
        # 获取视频的帧数, 宽高
        frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # 初始化输出视频对象编码，与原视频一致，注：ffmpeg不支持h264，即：875967080,所以用MJPEG4替代
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        if fourcc == 875967080:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # 可以查看fourcc格式
        # print(chr(fourcc&0xFF) + chr((fourcc>>8)&0xFF) + chr((fourcc>>16)&0xFF) + chr((fourcc>>24)&0xFF))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        # print(f'height, width:{(height, width)}')
        boxes = np.zeros((frame_num, 4))
        boxes[0] = box
        read_times = np.zeros(frame_num)
        track_times = np.zeros(frame_num)
        show_times = np.zeros(frame_num)
        out_dir = './out/'
        os.makedirs(out_dir, exist_ok=True)
        out_path = video_path.split('/')[-1].split('.')
        out_path = out_dir+out_path[0]+'_box_'+str(box[0])+"_"+str(box[1])+"_"+str(box[2])+"_"+str(box[3])+"."+out_path[1]
        print(out_path)
        img_out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        # 循环读取视频帧
        while cap.isOpened():
            begin = time.time()
            # 读取视频帧
            ret, img = cap.read()

            if not ret:
                break

            read_times[frame_count] = time.time() - begin

            if frame_count == 0:
                self.init(img, box)
            else:
                boxes[frame_count, :] = self.update(img)

            track_times[frame_count] = time.time() - read_times[frame_count] - begin

            if visualize:
                image = ops.show_image(img, boxes[frame_count, :],cvt_code=None) 
                img_out.write(image)
            show_times[frame_count] = time.time() - track_times[frame_count] - read_times[frame_count] - begin

            # 递增帧计数器
            frame_count += 1

        # 释放视频捕获资源
        cap.release()
        # 关闭输出视频对象
        img_out.release()
        # 打开视频文件
        print(f'average frame time: read:{read_times.sum()*1000/frame_count:.2f} ms, track:{track_times.sum()*1000/frame_count:.2f} ms, show:{show_times.sum()*1000/frame_count:.2f} ms')
        #print(f'len: read_times:{len(read_times)}, track_time:{len(track_times)}, show_time:{len(show_times)}')
        print(f'all time : {show_times.sum()+read_times.sum()+track_times.sum():.2f} s')
        # 打开文件以写入数据
        box_path = out_path.replace("."+video_path.split('.')[-1],".txt")
        print(f'boxes out path: {box_path}')
        with open(box_path, "w") as file:
        # 按行保存列表数据
            for row in boxes:
                # 将每一行转换为字符串形式，并以,分隔元素
                row_str = ",".join(str(element) for element in row)
                # 写入当前行数据并换行
                file.write(row_str + "\n")

        return boxes

    def train_step(self, batch, backward=True):
        # set network mode
        self.net.train(backward)

        # parse batch data
        z = batch[0].to(self.device, non_blocking=self.cuda)
        x = batch[1].to(self.device, non_blocking=self.cuda)

        with torch.set_grad_enabled(backward):
            # inference
            responses = self.net(z, x)

            # calculate loss
            labels = self._create_labels(responses.size())
            loss = self.criterion(responses, labels)
            
            if backward:
                # back propagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        return loss.item()

    @torch.enable_grad()
    def train_over(self, seqs, val_seqs=None,
                   save_dir='pretrained'):
        # set to train mode
        self.net.train()

        # create save_dir folder
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # setup dataset
        transforms = SiamFCTransforms(
            exemplar_sz=self.cfg.exemplar_sz,
            instance_sz=self.cfg.instance_sz,
            context=self.cfg.context)
        dataset = Pair(
            seqs=seqs,
            transforms=transforms)
        
        # setup dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cuda,
            drop_last=True)
        
        # loop over epochs
        for epoch in range(self.cfg.epoch_num):
            pbar = tqdm(total=len(dataloader)) 
            # loop over dataloader
            losss = np.zeros(len(dataloader))
            for it, batch in enumerate(dataloader):
                losss[it] = self.train_step(batch, backward=True)
                pbar.update(1)
                pbar.set_description('Epoch: {} [{}/{}] Average Loss: {:.5f}'.format(
                    epoch + 1, it + 1, len(dataloader), losss.sum()/(it+1)))
            pbar.close()
            # update lr at each epoch
            self.lr_scheduler.step(epoch=epoch)
            # save checkpoint
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            net_path = os.path.join(
                save_dir, 'siamfc_alexnet_e%d.pth' % (epoch + 1))
            torch.save(self.net.state_dict(), net_path)
    
    def _create_labels(self, size):
        # skip if same sized labels already created
        if hasattr(self, 'labels') and self.labels.size() == size:
            return self.labels

        def logistic_labels(x, y, r_pos, r_neg):
            dist = np.abs(x) + np.abs(y)  # block distance
            labels = np.where(dist <= r_pos,
                              np.ones_like(x),
                              np.where(dist < r_neg,
                                       np.ones_like(x) * 0.5,
                                       np.zeros_like(x)))
            return labels

        # distances along x- and y-axis
        n, c, h, w = size
        x = np.arange(w) - (w - 1) / 2
        y = np.arange(h) - (h - 1) / 2
        x, y = np.meshgrid(x, y)

        # create logistic labels
        r_pos = self.cfg.r_pos / self.cfg.total_stride
        r_neg = self.cfg.r_neg / self.cfg.total_stride
        labels = logistic_labels(x, y, r_pos, r_neg)

        # repeat to size
        labels = labels.reshape((1, 1, h, w))
        labels = np.tile(labels, (n, c, 1, 1))

        # convert to tensors
        self.labels = torch.from_numpy(labels).to(self.device).float()
        
        return self.labels
