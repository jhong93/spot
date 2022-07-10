import math
import torch
import torch.nn as nn
import torchvision
import timm

from .impl.tsm import TemporalShift
from .impl.gsm import _GSM


class GatedShift(nn.Module):
    def __init__(self, net, n_segment, n_div):
        super(GatedShift, self).__init__()

        if isinstance(net, torchvision.models.resnet.BasicBlock):
            channels = net.conv1.in_channels
        elif isinstance(net, torchvision.ops.misc.ConvNormActivation):
            channels = net[0].in_channels
        elif isinstance(net, timm.models.layers.conv_bn_act.ConvBnAct):
            channels = net.conv.in_channels
        elif isinstance(net, nn.Conv2d):
            channels = net.in_channels
        else:
            raise NotImplementedError(type(net))

        self.fold_dim = math.ceil(channels // n_div / 4) * 4
        self.gsm = _GSM(self.fold_dim, n_segment)
        self.net = net
        self.n_segment = n_segment
        print('=> Using GSM, fold dim: {} / {}'.format(
            self.fold_dim, channels))

    def forward(self, x):
        y = torch.zeros_like(x)
        y[:, :self.fold_dim, :, :] = self.gsm(x[:, :self.fold_dim, :, :])
        y[:, self.fold_dim:, :, :] = x[:, self.fold_dim:, :, :]
        return self.net(y)


# Adapted from: https://github.com/mit-han-lab/temporal-shift-module/blob/master/ops/temporal_shift.py
def make_temporal_shift(net, clip_len, is_gsm=False):

    def _build_shift(net):
        if is_gsm:
            return GatedShift(net, n_segment=clip_len, n_div=4)
        else:
            return TemporalShift(net, n_segment=clip_len, n_div=8)

    if isinstance(net, torchvision.models.ResNet):
        n_round = 1
        if len(list(net.layer3.children())) >= 23:
            n_round = 2
            print('=> Using n_round {} to insert temporal shift'.format(n_round))

        def make_block_temporal(stage):
            blocks = list(stage.children())
            print('=> Processing stage with {} blocks residual'.format(len(blocks)))
            for i, b in enumerate(blocks):
                if i % n_round == 0:
                    blocks[i].conv1 = _build_shift(b.conv1)
            return nn.Sequential(*blocks)

        net.layer1 = make_block_temporal(net.layer1)
        net.layer2 = make_block_temporal(net.layer2)
        net.layer3 = make_block_temporal(net.layer3)
        net.layer4 = make_block_temporal(net.layer4)

    elif isinstance(net, timm.models.regnet.RegNet):
        n_round = 1
        # if len(list(net.trunk_output.block3.children())) >= 23:
        #     n_round = 2
        #     print('=> Using n_round {} to insert temporal shift'.format(n_round))

        def make_block_temporal(stage):
            blocks = list(stage.children())
            print('=> Processing stage with {} blocks residual'.format(
                len(blocks)))
            for i, b in enumerate(blocks):
                if i % n_round == 0:
                    blocks[i].conv1 = _build_shift(b.conv1)

        make_block_temporal(net.s1)
        make_block_temporal(net.s2)
        make_block_temporal(net.s3)
        make_block_temporal(net.s4)

    elif isinstance(net, timm.models.convnext.ConvNeXt):
        n_round = 1
        # if len(list(net.stages[2].children())) >= 23:
        #     n_round = 2
        #     print('=> Using n_round {} to insert temporal shift'.format(n_round))

        def make_block_temporal(stage):
            blocks = list(stage.blocks)
            print('=> Processing stage with {} blocks residual'.format(
                len(blocks)))

            for i, b in enumerate(blocks):
                if i % n_round == 0:
                    blocks[i].conv_dw = _build_shift(b.conv_dw)
            return nn.Sequential(*blocks)

        make_block_temporal(net.stages[0])
        make_block_temporal(net.stages[1])
        make_block_temporal(net.stages[2])
        make_block_temporal(net.stages[3])

    else:
        raise NotImplementedError('Unsupported architecture')
