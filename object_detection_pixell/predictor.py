from object_detection_pixell import models
from object_detection_pixell.utils import get_state_dict
from object_detection_pixell.utils.point_cloud_utils import create_pillars

import numpy as np
import torch
import os
from ruamel import yaml

FILEPATH = os.path.dirname(os.path.abspath(__file__))
CFG = f'{FILEPATH}/configs/pixell_to_box3d_v2.yml'
STATE = f'{FILEPATH}/states/state.pt'


class Predictor:

    def __init__(self, cfg=CFG, state=STATE, in_channels:int=7):

        with open(cfg, 'r') as f:
            self.cfg = yaml.safe_load(f)

        self.state_file = state
        self.in_channels = in_channels

        self.__load_model()

    def __load_model(self):
        self.model = getattr(models, self.cfg['NEURAL_NET']['NAME'])(self.cfg, self.in_channels)
        self.model.to(self.cfg['TRAINING']['DEVICE'])

        state_dict = get_state_dict(self.state_file, device=self.cfg['TRAINING']['DEVICE'])
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def __call__(self, pcloud, amplitudes):

        _amplitudes = np.copy(amplitudes)
        if self.cfg['PREPROCESSING']['POINT_CLOUD']['INTENSITY']['ABSOLUTE']:
            distances_sq = pcloud[:,0]**2 + pcloud[:,1]**2 + pcloud[:,2]**2
            _amplitudes *= distances_sq
        if self.cfg['PREPROCESSING']['POINT_CLOUD']['INTENSITY']['LOG']:
            _amplitudes = np.log(_amplitudes+1)
        _amplitudes *= self.cfg['PREPROCESSING']['POINT_CLOUD']['INTENSITY']['NORM_FACTOR']

        _pcloud = np.vstack([pcloud.T, _amplitudes]).T

        grid = self.cfg['PREPROCESSING']['POINT_CLOUD']['PILLARS']['GRID']

        pillars, indices = create_pillars(
            _pcloud,
            self.cfg['PREPROCESSING']['POINT_CLOUD']['PILLARS']['MAX_POINTS_PER_PILLAR'],
            self.cfg['PREPROCESSING']['POINT_CLOUD']['PILLARS']['NUMBER_PILLARS'],
            (grid['x'][1] - grid['x'][0])/grid['x'][2],
            (grid['y'][1] - grid['y'][0])/grid['y'][2],
            grid['x'][0],
            grid['x'][1],
            grid['y'][0],
            grid['y'][1],
            grid['z'][0],
            grid['z'][1],
        )

        # channels first
        pillars = np.moveaxis(pillars, 2, 0)

        input = (torch.from_numpy(pillars).float(), torch.from_numpy(indices).type(torch.long))
        input = (array.unsqueeze(0).to(torch.device(self.cfg['TRAINING']['DEVICE'])) for array in input)

        raw_output = self.model(input)

        return self.model.post_process(raw_output)