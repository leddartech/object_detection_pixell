from object_detection_pixell.utils import to_box3d_package
from object_detection_pixell import layers

import numpy as np
import torch
import torch.nn as nn



class PCloudToBox3D(nn.Module):
    def __init__(self, cfg, in_channels=7):
        super(PCloudToBox3D, self).__init__()
        self.cfg = cfg
        self.nb_classes = len(self.cfg['PREPROCESSING']['BOX_3D']['CLASSIFICATION'])

        self.feature_extract = layers.DenseBlock(in_channels=in_channels, **self.cfg['NEURAL_NET']['FEATURE_EXTRACT'])

        C = self.feature_extract.out_channels
        for i, layer in enumerate(self.cfg['NEURAL_NET']['LAYERS']):

            if i == len(self.cfg['NEURAL_NET']['LAYERS'])-1:
                self.cfg['NEURAL_NET']['LAYERS'][layer]['args']['out_channels'] = self.nb_classes + 8

            setattr(self, layer, getattr(layers, self.cfg['NEURAL_NET']['LAYERS'][layer]['type'])(in_channels=C, **self.cfg['NEURAL_NET']['LAYERS'][layer]['args']))
            C = getattr(self, layer).out_channels

    def forward(self, x):

        pillars, indices = x

        features = self.feature_extract(pillars)
        features = torch.max(features, dim=-1)[0]

        grid = torch.zeros((
            features.shape[0],
            features.shape[1],
            self.cfg['PREPROCESSING']['POINT_CLOUD']['PILLARS']['GRID']['x'][2],
            self.cfg['PREPROCESSING']['POINT_CLOUD']['PILLARS']['GRID']['y'][2],
        ), device = features.device)

        for i in range(grid.shape[0]):
            grid[i,:,indices[i,:,0],indices[i,:,1]] = features[i]

        for layer in self.cfg['NEURAL_NET']['LAYERS']:
            grid = getattr(self, layer)(grid)

        return grid

    def post_process(self, raw_output):
        return to_box3d_package(raw_output, self.cfg)

    @property
    def size_of_net(self):
        out = 0
        for key in list(self.state_dict()):
            out += np.product(self.state_dict()[key].shape)
        return out