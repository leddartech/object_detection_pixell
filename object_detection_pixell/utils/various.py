from pioneer.das.api.samples import Box3d

import copy
import glob
import numpy as np
import os
import pickle
import time
import torch

FILEPATH = os.path.dirname(os.path.abspath(__file__))
INSERTS_PATH = f"{FILEPATH}/../inserts_data"


def get_state_dict(path, device):
    state = torch.load(path, map_location=device)

    # If trained with multiple GPUs, but infering on a single one, we have to remove the '.module' in the state dict
    if device == 'cuda' and torch.cuda.device_count() == 1:
        for key in list(state.keys()):
            if 'module' in key:
                state[key[7:]] = state.pop(key)

    return state


def to_percentage_string(float_value):
    return f"{100*float(float_value):.2f}%"


def train_valid_indices(total_number_frames, cfg_training):

    if 'SEED' in cfg_training:
        np.random.seed(cfg_training['SEED'])

    indices = np.arange(total_number_frames)
    np.random.shuffle(indices)
    N_valid = int(total_number_frames*cfg_training['VALIDATION_FRAMES_PROPORTION'])
    valid_indices = indices[:N_valid]
    train_indices = indices[N_valid:]

    if 'N_TRAIN' in cfg_training:
        if cfg_training['N_TRAIN'] < train_indices.size:
            train_indices = train_indices[:cfg_training['N_TRAIN']]
    if 'N_VALID' in cfg_training:
        if cfg_training['N_VALID'] < valid_indices.size:
            valid_indices = valid_indices[:cfg_training['N_VALID']]
            
    return train_indices, valid_indices


def filter_training_frames(cfg):
    arg = cfg['DATASET']['TRAIN_FRAME_SELECTION']
    if isinstance(arg, str):
        try:
            return list(np.load(arg))
        except: pass
    elif isinstance(arg, dict):
        try:
            return np.arange(**arg)
        except: pass
    raise ValueError('TRAIN_FRAME_SELECTION must be either a path to a .npy file or a dict with keys start, stop and step.')


def generate_random_augmentation_state(cfg, insert_files={}):
    #TODO: use seed to have reproductible results

    data_augmentation_state = {}

    if 'INSERT' in cfg['AUGMENTATION']:
        data_augmentation_state['INSERT'] = []
        
        for category in cfg['AUGMENTATION']['INSERT']:
            if category not in insert_files:
                insert_files[category] = glob.glob(f"{INSERTS_PATH}/{category}_*.pkl")

            if len(insert_files[category]) > 0:
                insert_files_random = np.random.choice(insert_files[category], cfg['AUGMENTATION']['INSERT'][category])

                for filename in insert_files_random:
                    with open(filename, 'rb') as f:
                        data = pickle.load(f)
                    data_augmentation_state['INSERT'].append(data)
            else:
                raise Exception('Insert data not found (data augmentation). Please run python3 prepare_inserts_data.py --cfg=CONFIG_FILE.') 

    # Left/right flip (50% chance)
    data_augmentation_state['FLIP_LR'] = bool(np.random.randint(2)) if 'LR_FLIP' in cfg['AUGMENTATION'] else False

    if 'TRANSLATION' in cfg['AUGMENTATION']:
        data_augmentation_state['TRANSLATION'] = {c:np.random.uniform(*cfg['AUGMENTATION']['TRANSLATION'][c]) for c in cfg['AUGMENTATION']['TRANSLATION']}

    if 'ROTATION' in cfg['AUGMENTATION']:
        data_augmentation_state['ROTATION'] = np.random.uniform(*cfg['AUGMENTATION']['ROTATION'])

    if 'SCALING' in cfg['AUGMENTATION']:
        data_augmentation_state['SCALING'] = np.random.uniform(*cfg['AUGMENTATION']['SCALING'])

    return data_augmentation_state, insert_files


def prevent_collisions_from_inserts(data_augmentation_state, label_sample):

    if isinstance(label_sample, Box3d):
        boxes3d = copy.deepcopy(label_sample.raw['data'])
        inserts = data_augmentation_state['INSERT']
        keep = []
        for insert in inserts:
            relative_positions = boxes3d['c'] - insert['box']['c']
            distances = (relative_positions[:,0]**2 + relative_positions[:,1]**2 + relative_positions[:,2]**2)**0.5
            margin = 0.2
            if all(distances > (np.max(boxes3d['d']/(2**0.5), axis=1) + np.max(insert['box']['d']/(2**0.5)) + margin)):
                keep.append(insert)
                boxes3d = np.append(boxes3d, insert['box'])
        data_augmentation_state['INSERT'] = keep

    else:
        raise NotImplementedError

    return data_augmentation_state

