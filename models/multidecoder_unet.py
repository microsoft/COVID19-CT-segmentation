#!/usr/bin/env python
'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_network import BaseNetwork
from options.train_options import TrainOptions
from models.unet import ConvBlock, UpBlock
import copy


class MultidecoderUnetModel(BaseNetwork):
    def __init__(self, opts):
        super().__init__()
        self.downblocks = nn.ModuleList()
        self.upblocks_contours = nn.ModuleList()
        self.upblocks_glass_shadow = nn.ModuleList()
        self.upblocks_consolidation = nn.ModuleList()
        self.upblocks_fibrosis = nn.ModuleList()
        self.upblocks_pleural_effusion = nn.ModuleList()
        self.upblocks_thickening = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        self.in_channels = opts.input_channels
        self.out_channels = opts.first_layer_filters
        self.net_depth = opts.net_depth

        # down transformations
        for _ in range(self.net_depth):
            conv = ConvBlock(self.in_channels, self.out_channels)
            self.downblocks.append(conv)
            self.in_channels, self.out_channels = self.out_channels, 2 * self.out_channels

        # midpoint
        self.middle_conv = ConvBlock(self.in_channels, self.out_channels)

        # up transformations
        self.in_channels, self.out_channels = self.out_channels, int(self.out_channels / 2)
        for _ in range(self.net_depth):
            self.upblocks_glass_shadow.append(UpBlock(self.in_channels, self.out_channels))
            self.upblocks_consolidation.append(UpBlock(self.in_channels, self.out_channels))
            self.upblocks_fibrosis.append(UpBlock(self.in_channels, self.out_channels))
            self.upblocks_thickening.append(UpBlock(self.in_channels, self.out_channels))
            self.in_channels, self.out_channels = self.out_channels, int(self.out_channels / 2)

        self.ground_glass_shadow_seg_layer = nn.Conv2d(2 * self.out_channels, opts.num_classes, kernel_size=1)
        self.consolidation_shadow_seg_layer = nn.Conv2d(2 * self.out_channels, opts.num_classes, kernel_size=1)
        self.fibrosis_seg_layer = nn.Conv2d(2 * self.out_channels, opts.num_classes, kernel_size=1)
        self.interstitial_thickening_seg_layer = nn.Conv2d(2 * self.out_channels, opts.num_classes, kernel_size=1)


    def forward(self, x):
        decoder_outputs = []
        decoder_outputs2 = []

        for op in self.downblocks:
            decoder_outputs.append(op(x))
            x = self.pool(decoder_outputs[-1])

        x = self.middle_conv(x)

        x_shadow = copy.copy(x)
        for op in self.upblocks_glass_shadow:
            dec = decoder_outputs.pop()
            x_shadow = op(x_shadow, dec)
            decoder_outputs2.insert(0, dec)

        x_consolidation = copy.copy(x)
        for op in self.upblocks_consolidation:
            dec = decoder_outputs2.pop()
            x_consolidation = op(x_consolidation, dec)
            decoder_outputs.insert(0, dec)

        x_fibrosis = copy.copy(x)
        for op in self.upblocks_fibrosis:
            dec = decoder_outputs.pop()
            x_fibrosis = op(x_fibrosis, dec)
            decoder_outputs2.insert(0, dec)

        x_thickening = copy.copy(x)
        for op in self.upblocks_thickening:
            dec = decoder_outputs2.pop()
            x_thickening = op(x_thickening, dec)
        
        # Segmentation output
        ground_glass = self.ground_glass_shadow_seg_layer(x_shadow)
        consolidation = self.consolidation_shadow_seg_layer(x_consolidation)
        fibrosis = self.fibrosis_seg_layer(x_fibrosis)
        thickening = self.interstitial_thickening_seg_layer(x_thickening)

        return torch.cat([ground_glass, consolidation, fibrosis, thickening], 1)

# Test with mock data
if __name__ == "__main__":
    # A full forward pass
    opts = TrainOptions().parse()
    im = torch.randn(2, 1, 512, 512)
    model = MultitaskUnetModel(opts)
    x = model(im)
    print(x.shape)
    del model
    del x