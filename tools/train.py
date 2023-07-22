from __future__ import absolute_import

import os
from got10k.datasets import *

from siamfc import TrackerSiamFC
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Training script")
    # 预训练模型
    parser.add_argument('-m', '--pretrain_model_path', default='./pretrained/siamfc_alexnet_e50.pth', type=str,
                        help="the pretrained model path.")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_arguments()
    root_dir = os.path.expanduser('~/2_dataset/GOT-10k-full_data')
    seqs = GOT10k(root_dir, subset='train', return_meta=True)

    tracker = TrackerSiamFC(net_path=args.pretrain_model_path)
    tracker.train_over(seqs)
