import argparse
import logging
import sys
from copy import deepcopy
from pathlib import Path
import math

import torch
import torch.nn as nn

sys.path.append('./')  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)

from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import make_divisible, check_file, set_logging
from utils.plots import feature_visualization
from utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr

try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None


class Detect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor (x,y,w,h,obj + classes)
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors per layer
        self.grid = [torch.zeros(1)] * self.nl  # init grid placeholder
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl, na, 2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv layers

    def forward(self, x):
        z = []  # inference output list
        self.training |= self.export  # training mode with export flag
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # apply conv for this detection layer
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()  # reshape

            if not self.training:  # inference mode
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                # xy
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]
                # wh
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, classes
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:
            import yaml
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.SafeLoader)

        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            logger.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override number of classes

        if anchors:
            # anchors should be a list of lists, e.g. [[10,13,16,30,33,23], [...], [...]]
            logger.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = anchors  # correct way to override anchors

        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # build model
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default class names

        # Build strides and normalize anchors
        m = self.model[-1]  # Detect layer
        if isinstance(m, Detect):
            s = 256  # base image size for stride calculation
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # initialize detection biases

        # Initialize weights and print info
        initialize_weights(self)
        self.info()
        logger.info('')

    def forward(self, x, augment=False, profile=False, visualize=False):
        if augment:
            img_size = x.shape[-2:]
            scales = [1, 0.83, 0.67]
            flips = [None, 3, None]
            y = []
            for si, fi in zip(scales, flips):
                xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
                yi = self.forward_once(xi, visualize=visualize)[0]
                yi[..., :4] /= si  # scale coords back
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # vertical flip undo
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # horizontal flip undo
                y.append(yi)
            return torch.cat(y, 1), None
        else:
            return self.forward_once(x, profile, visualize)

    def forward_once(self, x, profile=False, visualize=False):
        y, dt = [], []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]

            if profile:
                o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2 if thop else 0
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                print(f'{o:10.1f}{m.np:10.0f}{dt[-1]:10.1f}ms {m.type:<40}')

            x = m(x)
            y.append(x if m.i in self.save else None)

            if visualize:
                feature_visualization(x, m.type, m.i)

        if profile:
            print(f'{sum(dt):.1f}ms total')
        return x

    def _initialize_biases(self, cf=None):
        m = self.model[-1]  # Detect layer
        for mi, s in zip(m.m, m.stride):
            b = mi.bias.view(m.na, -1)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # objectness bias
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def fuse(self):
        print('Fusing Conv2d and BatchNorm2d layers...')
        for m in self.model.modules():
            if isinstance(m, Conv) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, 'bn')
                m.forward = m.fuseforward
        self.info()
        return self

    def nms(self, mode=True):
        present = isinstance(self.model[-1], NMS)
        if mode and not present:
            print('Adding NMS module...')
            m = NMS()
            m.f = -1
            m.i = self.model[-1].i + 1
            self.model.add_module(name=f'{m.i}', module=m)
            self.eval()
        elif not mode and present:
            print('Removing NMS module...')
            self.model = self.model[:-1]
        return self

    def autoshape(self):
        print('Adding autoShape module...')
        m = autoShape(self)
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())
        return m

    def info(self, verbose=False, img_size=640):
        model_info(self, verbose, img_size)


def parse_model(d, ch):  # d: model config dict, ch: input channels list
    anchors, nc, gd, gw = d.get('anchors', []), d['nc'], d['depth_multiple'], d['width_multiple']
    layers, save, c2 = [], [], ch[-1]
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):
        m = eval(m) if isinstance(m, str) else m  # eval string if necessary
        n = max(round(n * gd), 1) if n > 1 else n  # depth scaling
        if m in [Conv, DWConv, Bottleneck, BottleneckCSP, SPP, Focus, CrossConv, BottleneckTransformer, C3]:
            c1 = c2
            c2 = args[0] if isinstance(args, list) else args  # output channels
            c2 = make_divisible(c2 * gw, 8)
            args = [c1, c2, *args[1:]] if isinstance(args, list) else [c1, c2]
            for _ in range(n):
                layers.append(m(*args))
            save.extend(x for x in (f if isinstance(f, list) else [f]) if x != -1)
        elif m is Detect:
            args.append([make_divisible(x * gw, 8) for x in ch])  # ch input for Detect
            m_ = m(*args)
            layers.append(m_)
            save.extend(x for x in (f if isinstance(f, list) else [f]) if x != -1)
            c2 = nc
        else:
            for _ in range(n):
                layers.append(m(*args))
            save.extend(x for x in (f if isinstance(f, list) else [f]) if x != -1)

    return nn.Sequential(*layers), sorted(set(save))


def attempt_load(weights, map_location=None, inplace=True, fuse=True):
    model = Model(weights if isinstance(weights, dict) else weights)
    ckpt = torch.load(weights, map_location=map_location)
    model.load_state_dict(ckpt['model'].float().state_dict(), strict=False)
    if fuse:
        model.fuse()
    return model


class autoShape(nn.Module):
    # Wrapper for inference with multiple image types and auto pre/post-processing
    def __init__(self, model):
        super(autoShape, self).__init__()
        self.model = model
        copy_attr(self, model, include=('names', 'stride', 'device'))

    def forward(self, imgs, size=640, augment=False, profile=False):
        # Process input images (numpy, PIL, tensor), auto resize, normalize
        if isinstance(imgs, (list, tuple)):
            imgs = [self.prep(img, size) for img in imgs]
            imgs = torch.cat(imgs, 0)
        else:
            imgs = self.prep(imgs, size)
        imgs = imgs.to(self.model.device)
        if imgs.ndimension() == 3:
            imgs = imgs.unsqueeze(0)
        pred = self.model(imgs, augment=augment, profile=profile)[0]
        return pred

    def prep(self, im, size):
        # preprocess image to tensor
        # Implement resizing, padding, normalization etc.
        # (Add your preprocessing here as needed)
        pass
