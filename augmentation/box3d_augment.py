import numpy as np
import transforms3d


def global_flip_lr(boxes3d):
    """Flip y axis (y coordinate of the center position and yaw angle)"""
    boxes3d['c'][:,1] *= -1
    boxes3d['r'][:,2] *= -1
    return boxes3d


def global_translation(boxes3d, translation):
    """Apply global translations"""
    boxes3d['c'][:,0] += translation['x']
    boxes3d['c'][:,1] += translation['y']
    boxes3d['c'][:,2] += translation['z']
    return boxes3d


def global_rotation(boxes3d, rotation):
    """Apply global rotation"""
    boxes3d['c'] = boxes3d['c'] @ transforms3d.euler.euler2mat(0,0,np.deg2rad(rotation))
    boxes3d['r'][:,2] -= np.deg2rad(rotation)
    return boxes3d


def global_scaling(boxes3d, scaling):
    """Apply global scaling"""
    boxes3d['c'] *= scaling
    boxes3d['d'] *= scaling
    return boxes3d