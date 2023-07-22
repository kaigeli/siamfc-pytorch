# SiamFC - PyTorch

> Highlights of this update:
> - Higher scores with more stable training performance.
> - Faster training (~11 minutes to train one epoch on GOT-10k on a single GPU).
> - Added MIT LICENSE.
> - Organized code.
> - Uploaded pretrained weights. ([Google Drive](https://drive.google.com/file/d/1UdxuBQ1qtisoWYFZxLgMFJ9mJtGVw6n4/view?usp=sharing) or [Baidu Yun](https://pan.baidu.com/s/1MTVXylPrSqpqmVD4iBwbpg) (password: wbek))

A clean PyTorch implementation of SiamFC tracker described in paper [Fully-Convolutional Siamese Networks for Object Tracking](https://www.robots.ox.ac.uk/~luca/siamese-fc.html). The code is evaluated on 7 tracking datasets ([OTB (2013/2015)](http://cvlab.hanyang.ac.kr/tracker_benchmark/index.html), [VOT (2018)](http://votchallenge.net), [DTB70](https://github.com/flyers/drone-tracking), [TColor128](http://www.dabi.temple.edu/~hbling/data/TColor-128/TColor-128.html), [NfS](http://ci2cv.net/nfs/index.html) and [UAV123](https://ivul.kaust.edu.sa/Pages/pub-benchmark-simulator-uav.aspx)), using the [GOT-10k toolkit](https://github.com/got-10k/toolkit).

## Performance (the scores are not updated yet)

### GOT-10k

| Dataset | AO    | SR<sub>0.50</sub> | SR<sub>0.75</sub> |
|:------- |:-----:|:-----------------:|:-----------------:|
| GOT-10k | 0.355 | 0.390             | 0.118             |

The scores are comparable with state-of-the-art results on [GOT-10k leaderboard](http://got-10k.aitestunion.com/leaderboard).

### OTB / UAV123 / DTB70 / TColor128 / NfS

| Dataset       | Success Score    | Precision Score |
|:-----------   |:----------------:|:----------------:|
| OTB2013       | 0.589            | 0.781            |
| OTB2015       | 0.578            | 0.765            |
| UAV123        | 0.523            | 0.731            |
| UAV20L        | 0.423            | 0.572            |
| DTB70         | 0.493            | 0.731            |
| TColor128     | 0.510            | 0.691            |
| NfS (30 fps)  | -                | -                |
| NfS (240 fps) | 0.520            | 0.624            |

### VOT2018

| Dataset       | Accuracy    | Robustness (unnormalized) |
|:-----------   |:-----------:|:-------------------------:|
| VOT2018       | 0.502       | 37.25                     |

## Installation

Install Anaconda, then install dependencies:

```bash
# install PyTorch >= 1.0
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
# intall OpenCV using menpo channel (otherwise the read data could be inaccurate)
conda install -c menpo opencv tqdm
# install GOT-10k toolkit
pip install got10k
```

[GOT-10k toolkit](https://github.com/got-10k/toolkit) is a visual tracking toolkit that implements evaluation metrics and tracking pipelines for 9 popular tracking datasets.

## Training the tracker

1. Setup the training dataset in `tools/train.py`. Default is the GOT-10k dataset located at `~/data/GOT-10k`.

2. Run:

```
python tools/train.py
```

## Evaluate the tracker

1. Setup the tracking dataset in `tools/test.py`. Default is the OTB dataset located at `~/data/OTB`.

2. Setup the checkpoint path of your pretrained model. Default is `pretrained/siamfc_alexnet_e50.pth`.

3. Run:

```
python tools/test.py
```

## Running the demo

1. Setup the sequence path in `tools/demo.py`. Default is `~/data/OTB/Crossing`.

2. Setup the checkpoint path of your pretrained model. Default is `pretrained/siamfc_alexnet_e50.pth`.

3. Run:

```
python tools/demo.py
```
## 其他相关知识点
Hanning窗口的数学表达式如下：

$$w(n)=0.5(1-\cos\frac{2\pi n}{N-1})$$

其中：

- $w(n)$ 是Hanning窗口在n时刻的值。
- $N$ 是窗口的长度，通常为一段信号的采样点数。

SiamFC (Siamese Fully Convolutional) 是一种用于目标跟踪的深度学习网络。它的模板图片尺寸的设定与目标跟踪算法的设计有关。

在SiamFC中，模板图片的尺寸是通过以下方式计算得到的：

$${template\_size} = \sqrt{(w + \frac{w+h}{2}) \cdot (h + \frac{w+h}{2})}$$

其中，\(w\) 是目标的宽度，\(h\) 是目标的高度。

这个尺寸计算的方式主要是为了保证在目标跟踪过程中，模板图片的尺寸能够兼顾目标的大小，并且避免因为目标尺寸较大或较小而导致跟踪性能下降。

假设目标的宽度和高度都相等（\(w = h\)），那么计算模板图片尺寸的公式可以简化为：

\[ \text{template\_size} = \sqrt{w \cdot 2 \cdot w} = \sqrt{2} \cdot w \]

也就是说，模板图片的尺寸是目标边长的约 1.414 倍。这样的设计能够比较好地适应不同大小的目标。

总之，SiamFC中模板图片尺寸的设定是为了提高目标跟踪算法的适应性，使其能够在不同尺寸的目标上表现良好。