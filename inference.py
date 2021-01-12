from dataloader import LeddartechDatasetInference
from utils import get_state_dict, to_box3d_package
import models

from pioneer.common.platform import extract_sensor_id
from pioneer.das.api.datasources import VirtualDatasource
from pioneer.das.api.platform import Platform
from pioneer.das.api.samples import Box3d, Sample

try:
    from pioneer.das.view.viewer import Viewer
    HAS_DASVIEW = True
except:
    HAS_DASVIEW = False

import argparse
import glob
import numpy as np
import os
import time
import torch
import yaml

FILEPATH = os.path.dirname(os.path.abspath(__file__))



class DasPredictor(VirtualDatasource):

    def __init__(self, cfg, state, input_datasource='same_as_training'):

        self.cfg = cfg
        self.dataloader = LeddartechDatasetInference(cfg)
        self.state = state
        self.model = None
        self.ds_type = self.cfg['POSTPROCESSING']['DATASOURCE_TYPE']
        self.input_datasource = self.cfg['DATASET']['LIDAR'] if input_datasource == 'same_as_training' else input_datasource

        if self.ds_type.split('-')[0] == 'box3d':
            self.sample_class = Box3d
        else:
            self.sample_class = Sample
        
        super(DasPredictor, self).__init__(self.ds_type, [self.input_datasource], None)


    def _load_model(self, in_channels):

        self.model = getattr(models, cfg['NEURAL_NET']['NAME'])(self.cfg, in_channels)
        self.model.to(self.cfg['TRAINING']['DEVICE'])

        state_dict = get_state_dict(self.state, device=self.cfg['TRAINING']['DEVICE'])
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def get_at_timestamp(self, timestamp):
        sample = self.datasources[self.input_datasource].get_at_timestamp(timestamp)
        return self[int(np.round(sample.index))]

    def __getitem__(self, key):

        s = time.time()
        lidar_sample = self.datasources[self.input_datasource][key]
        torched_lidar = self.dataloader.torch_lidar(lidar_sample)

        if isinstance(torched_lidar, tuple):
            torched_lidar = (input_data.unsqueeze(0).to(torch.device(self.cfg['TRAINING']['DEVICE'])) for input_data in torched_lidar)
        else:
            torched_lidar = torched_lidar.unsqueeze(0) #unsqueeze first dimension because batch size = 1 for inference
            torched_lidar = torched_lidar.to(torch.device(self.cfg['TRAINING']['DEVICE']))
        t1 = time.time()-s

        if self.model is None:
            in_channels = self.dataloader.check_number_channels(lidar_sample)
            self._load_model(in_channels)

        s = time.time()
        raw_output = self.model(torched_lidar)
        t2 = time.time()-s

        s = time.time()
        package = self.model.post_process(raw_output)
        t3 = time.time()-s

        print(f'Time: {1e3*(t1+t2+t3):.1f}ms ({int(1/(t1+t2+t3)):d} fps)| preprocessing: {1e3*t1:.1f}ms | inference: {1e3*t2:.1f}ms | postprocessing: {1e3*t3:.1f}ms.')

        return self.sample_class(key, self, package, lidar_sample.timestamp)




def main(cfg, state, dataset, input_datasource):

    # FIXME: crash if test set is a directory of multiple datasets
    dataset = cfg['DATASET']['TEST_SET'][0] if dataset == 'test_set' else dataset

    pf = Platform(dataset)

    if 'VIRTUAL_DATASOURCES' in cfg['DATASET']:
        pf.add_virtual_datasources(cfg['DATASET']['VIRTUAL_DATASOURCES'])

    vds = DasPredictor(cfg, state, input_datasource)

    sensor_name = extract_sensor_id(cfg['DATASET']['LABEL'])
    pf[sensor_name].add_datasource(vds, vds.ds_type)

    v = Viewer(None, pf)
    v.run()




if __name__ == "__main__":

    if not HAS_DASVIEW:
        raise Exception('pioneer.das.view must be installed.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg')
    parser.add_argument('--state')
    parser.add_argument('--dataset', default='test_set')
    parser.add_argument('--input', default='same_as_training')
    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)

    main(cfg, args.state, args.dataset, args.input)
