import argparse
import contextlib
import os
import sys
from copy import deepcopy
import platform
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.Models.research import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args
from utils.plots import feature_visualization
from utils.torch_utils import (fuse_conv_and_bn, initialize_weights, model_info, profile, scale_img, select_device,
                               time_sync)



class ASFF_Detect4(nn.Module):  # add ASFFV5 layer and Rfb
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=1, anchors=(), ch=(), multiplier=0.5, rfb=False, inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.l0_fusion = ASFFV54(level=0, multiplier=multiplier, rfb=rfb)
        self.l1_fusion = ASFFV54(level=1, multiplier=multiplier, rfb=rfb)
        self.l2_fusion = ASFFV54(level=2, multiplier=multiplier, rfb=rfb)
        self.l3_fusion = ASFFV54(level=3, multiplier=multiplier, rfb=rfb)
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output
        result = []

        result.append(self.l3_fusion(x))
        result.append(self.l2_fusion(x))
        result.append(self.l1_fusion(x))
        result.append(self.l0_fusion(x))
        x = result
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4] or self.onnx_dynamic:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy, wh, conf = y.tensor_split((2, 4), 4)
                    xy = (xy * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        shape = 1, self.na, ny, nx, 2  # grid shape
        if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid(torch.arange(ny).to(d), torch.arange(nx).to(d), indexing='ij')
        else:
            yv, xv = torch.meshgrid(torch.arange(ny, device=d), torch.arange(nx, device=d))
        grid = torch.stack((xv, yv), 2).expand(shape).float()
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape).float()
        return grid, anchor_grid


# common.py
class ASFFV54(nn.Module):
    def __init__(self, level, multiplier=1, rfb=False, vis=False, act_cfg=True):
        """
        ASFF version for YoloV5 .
        different than YoloV3
        multiplier should be 1, 0.5
        which means, the channel of ASFF can be
        512, 256, 128 -> multiplier=1
        256, 128, 64 -> multiplier=0.5
        For even smaller, you need change code manually.
        """
        super(ASFFV54, self).__init__()
        self.level = level
        self.dim = [int(1024 * multiplier), int(512 * multiplier),
                    int(256 * multiplier), int(128 * multiplier)]

        print(self.dim)

        self.inter_dim = self.dim[self.level]
        if level == 0:
            self.stride_level_1 = Conv(int(512 * multiplier), self.inter_dim, 3, 2)

            self.stride_level_2 = Conv(int(256 * multiplier), self.inter_dim, 3, 2)

            self.stride_level_3 = Conv(int(128 * multiplier), self.inter_dim, 3, 2)

            self.expand = Conv(self.inter_dim, int(1024 * multiplier), 3, 1)
        elif level == 1:
            self.compress_level_0 = Conv(int(1024 * multiplier), self.inter_dim, 1, 1)

            self.stride_level_2 = Conv(int(256 * multiplier), self.inter_dim, 3, 2)

            self.stride_level_3 = Conv(int(128 * multiplier), self.inter_dim, 3, 2)

            self.expand = Conv(self.inter_dim, int(512 * multiplier), 3, 1)
        elif level == 2:
            self.compress_level_0 = Conv(int(1024 * multiplier), self.inter_dim, 1, 1)

            self.compress_level_1 = Conv(int(512 * multiplier), self.inter_dim, 1, 1)

            self.stride_level_3 = Conv(int(128 * multiplier), self.inter_dim, 3, 2)

            self.expand = Conv(self.inter_dim, int(256 * multiplier), 3, 1)
        elif level == 3:
            self.compress_level_0 = Conv(int(1024 * multiplier), self.inter_dim, 1, 1)

            self.compress_level_1 = Conv(int(512 * multiplier), self.inter_dim, 1, 1)

            self.compress_level_2 = Conv(int(256 * multiplier), self.inter_dim, 1, 1)

            self.expand = Conv(self.inter_dim, int(128 * multiplier), 3, 1)
        # when adding rfb, we use half number of channels to save memory
        compress_c = 8 if rfb else 16
        self.weight_level_0 = Conv(
            self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = Conv(
            self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = Conv(
            self.inter_dim, compress_c, 1, 1)
        self.weight_level_3 = Conv(
            self.inter_dim, compress_c, 1, 1)
        self.weight_levels = Conv(
            compress_c * 3, 3, 1, 1)
        self.vis = vis

    def forward(self, x):  # l,m,s
        """
        # 128, 256, 512
        512, 256, 128
        from small -> large
        """
        x_level_0 = x[3]  # l
        x_level_1 = x[2]  # m
        x_level_2 = x[1]  # s
        x_level_3 = x[0]  # s

        print('x_level_0: ', x_level_0.shape)
        print('x_level_1: ', x_level_1.shape)
        print('x_level_2: ', x_level_2.shape)
        print('x_level_3: ', x_level_3.shape)

        if self.level == 0:
            level_0_resized = x_level_0

            level_1_resized = self.stride_level_1(x_level_1)

            level_2_downsampled_inter = F.max_pool2d(x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)

            level_3_downsampled_inter = F.max_pool2d(x_level_3, 3, stride=4, padding=1)
            level_3_resized = self.stride_level_3(level_3_downsampled_inter)

        elif self.level == 1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=2, mode='nearest')

            level_1_resized = x_level_1

            level_2_resized = self.stride_level_2(x_level_2)

            level_3_downsampled_inter = F.max_pool2d(x_level_3, 3, stride=2, padding=1)
            level_3_resized = self.stride_level_3(level_3_downsampled_inter)

        elif self.level == 2:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=4, mode='nearest')

            x_level_1_compressed = self.compress_level_1(x_level_1)
            level_1_resized = F.interpolate(x_level_1_compressed, scale_factor=2, mode='nearest')

            level_2_resized = x_level_2

            level_3_resized = self.stride_level_3(x_level_3)

        elif self.level == 3:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=8, mode='nearest')

            level_1_compressed = self.compress_level_1(x_level_1)
            level_1_resized = F.interpolate(level_1_compressed, scale_factor=4, mode='nearest')

            x_level_2_compressed = self.compress_level_2(x_level_2)
            level_2_resized = F.interpolate(x_level_2_compressed, scale_factor=2, mode='nearest')

            level_3_resized = x_level_3

        print('level: {}, l1_resized: {}, l2_resized: {}, l3_resized: {}'.format(self.level, level_1_resized.shape,
                                                                                 level_2_resized.shape,
                                                                                 level_3_resized.shape))

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        level_3_weight_v = self.weight_level_3(level_3_resized)

        print('level_0_weight_v: ', level_0_weight_v.shape)
        print('level_1_weight_v: ', level_1_weight_v.shape)
        print('level_2_weight_v: ', level_2_weight_v.shape)
        print('level_3_weight_v: ', level_3_weight_v.shape)

        levels_weight_v = torch.cat(
            (level_0_weight_v, level_1_weight_v, level_2_weight_v, level_3_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + \
                            level_1_resized * levels_weight[:, 1:2, :, :] + \
                            level_2_resized * levels_weight[:, 2:3:, :] + \
                            level_3_resized * levels_weight[:, 3:, :, :]

        out = self.expand(fused_out_reduced)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out
