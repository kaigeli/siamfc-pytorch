from __future__ import absolute_import

import torch.nn as nn
import torch.nn.functional as F


__all__ = ['SiamFC']


class SiamFC(nn.Module):

    def __init__(self, out_scale=0.001):
        super(SiamFC, self).__init__()
        self.out_scale = out_scale
    
    def forward(self, z, x):
        return self._fast_xcorr(z, x) * self.out_scale
    
    def _fast_xcorr(self, z, x):
        # fast cross correlation
        # x.shape = torch.Size([3, 256, 22, 22]), 3是因为scale_num = 3
        # z.shape = torch.Size([1, 256, 6, 6])
        nz = z.size(0)
        # nz = 1
        nx, c, h, w = x.size()
        x = x.view(-1, nz * c, h, w)
        # x.shape = torch.Size([3, 256, 22, 22])
        out = F.conv2d(x, z, groups=nz)
        # out.shape = torch.Size([3, 1, 17, 17])
        out = out.view(nx, -1, out.size(-2), out.size(-1))
        # print(out.shape)
        # out.shape = torch.Size([3, 1, 17, 17])
        return out
