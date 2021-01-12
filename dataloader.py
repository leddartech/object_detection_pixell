from utils import filter_training_frames
from utils import generate_random_augmentation_state
from utils import preprocess_pcloud, preprocess_box3d
from utils import prevent_collisions_from_inserts

from pioneer.common.platform import extract_sensor_id
from pioneer.das.api import platform
from pioneer.das.api.samples import Echo
from pioneer.das.api.samples import Box3d

import numpy as np
import os
import pickle
import time
import torch
from torch.utils.data import Dataset



class LeddartechDataset(Dataset):

    def __init__(self, cfg:dict, use_test_set:bool=False):
        self.cfg = cfg
        self.use_test_set = use_test_set

        datasets = self.cfg['DATASET']['TRAIN_SET'] if not use_test_set else self.cfg['DATASET']['TEST_SET']
        cfg_sync = self.cfg['DATASET']['SYNCHRONIZATION']
        sensors = [extract_sensor_id(ds) for ds in cfg_sync['sync_labels']+cfg_sync['interp_labels']]

        cfg_vd = None
        if 'VIRTUAL_DATASOURCES' in self.cfg['DATASET']:
            cfg_vd = self.cfg['DATASET']['VIRTUAL_DATASOURCES']

        self.platform = platform.SynchronizedGroup(datasets, **cfg_sync, include=sensors, preload=True, virtual_datasources_config=cfg_vd)

        if 'TRAIN_FRAME_SELECTION' in self.cfg['DATASET'] and not use_test_set:
            indices = filter_training_frames(self.cfg)
            self.platform = platform.Filtered(self.platform, indices)
        
        self.data_augmentation = False
        self.insert_files = {}

    def __len__(self):
        return len(self.platform)

    def __getitem__(self, index):
        
        samples = self.platform[index]

        lidar_sample = samples[self.cfg['DATASET']['LIDAR']][0]
        label_sample = samples[self.cfg['DATASET']['LABEL']][0]

        data_augmentation_state = None
        if 'AUGMENTATION' in self.cfg and self.data_augmentation:
            data_augmentation_state, self.insert_files = generate_random_augmentation_state(self.cfg, self.insert_files)
            if 'INSERT' in data_augmentation_state:
                # TODO: also prevent adding a box where the point cloud is already populated (terrain, building...)
                data_augmentation_state = prevent_collisions_from_inserts(data_augmentation_state, label_sample)

        torched_lidar = self.torch_lidar(lidar_sample, data_augmentation_state)
        torched_label = self.torch_label(label_sample, data_augmentation_state)

        return torched_lidar, torched_label

    def torch_lidar(self, lidar_sample, data_augmentation_state=None):

        if isinstance(lidar_sample, Echo):
            pillars, indices = preprocess_pcloud(lidar_sample, self.cfg, data_augmentation_state)
            return (torch.from_numpy(pillars).float(), torch.from_numpy(indices).type(torch.long))
        else:
            raise NotImplementedError

    def torch_label(self, label_sample, data_augmentation_state=None):

        if isinstance(label_sample, Box3d):
            data_array, lost_gt = preprocess_box3d(label_sample, self.cfg, data_augmentation_state)
            return (torch.from_numpy(data_array).float(), torch.from_numpy(lost_gt).float())
        else:
            raise NotImplementedError

        return torch.from_numpy(data_array).float()
    
    def check_number_channels(self, lidar_sample=None):
        lidar_sample = self.platform[0][self.cfg['DATASET']['LIDAR']][0] if lidar_sample is None else lidar_sample
        torched_lidar = self.torch_lidar(lidar_sample)
        return torched_lidar[0].shape[0]


class LeddartechDatasetInference(LeddartechDataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.data_augmentation = False

    def __getitem__(self, index):
        with torch.no_grad():
            return self[index]
        