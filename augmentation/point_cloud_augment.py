import numpy as np
import transforms3d


def insert(xyzi, inserts):
    for insert in inserts:
        xyzi = np.vstack([xyzi, insert['pcloud']])
    return xyzi


def global_flip_lr(xyz):
    """Flip y axis"""
    xyz[:,1] *= -1
    return xyz



def global_translation(xyz, translation):
    """Apply global translations"""
    xyz[:,0] += translation['x']
    xyz[:,1] += translation['y']
    xyz[:,2] += translation['z']
    return xyz


def global_rotation(xyz, rotation):
    """Apply global rotation"""
    xyz = xyz @ transforms3d.euler.euler2mat(0,0,np.deg2rad(rotation))
    return xyz


def global_scaling(xyz, scaling):
    """Apply global scaling"""
    xyz *= scaling
    return xyz