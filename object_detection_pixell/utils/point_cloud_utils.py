from object_detection_pixell.augmentation import point_cloud_augment

import numba
import numpy as np


def get_pcloud_data(pcloud_sample, cfg):

    pcloud = pcloud_sample.point_cloud(
        undistort=cfg['PREPROCESSING']['POINT_CLOUD']['MOTION_COMPENSATION'])

    if 'INTENSITY' in cfg['PREPROCESSING']['POINT_CLOUD']:
        intensity = pcloud_sample.amplitudes
        if cfg['PREPROCESSING']['POINT_CLOUD']['INTENSITY']['ABSOLUTE']:
            intensity *= pcloud_sample.distances**2
        if cfg['PREPROCESSING']['POINT_CLOUD']['INTENSITY']['LOG']:
            intensity = np.log(intensity+1)
        intensity *= cfg['PREPROCESSING']['POINT_CLOUD']['INTENSITY']['NORM_FACTOR']
        # print(intensity.min(), intensity.mean(), intensity.max())
        pcloud = np.vstack([pcloud.T, intensity]).T

    if 'DISTANCES' in cfg['PREPROCESSING']['POINT_CLOUD']:
        distances = pcloud_sample.distances
        distances *= cfg['PREPROCESSING']['POINT_CLOUD']['DISTANCES']['NORM_FACTOR']
        pcloud = np.vstack([pcloud.T, distances]).T

    return pcloud


def preprocess_pcloud(pcloud_sample, cfg, data_augmentation_state=None):

    pcloud = get_pcloud_data(pcloud_sample, cfg)

    if data_augmentation_state is not None:
        if 'INSERT' in data_augmentation_state:
            pcloud = point_cloud_augment.insert(
                pcloud, data_augmentation_state['INSERT'])
        if data_augmentation_state['FLIP_LR']:
            pcloud[:, [0, 1, 2]] = point_cloud_augment.global_flip_lr(
                pcloud[:, [0, 1, 2]])
        if 'TRANSLATION' in data_augmentation_state:
            pcloud[:, [0, 1, 2]] = point_cloud_augment.global_translation(
                pcloud[:, [0, 1, 2]], data_augmentation_state['TRANSLATION'])
        if 'ROTATION' in data_augmentation_state:
            pcloud[:, [0, 1, 2]] = point_cloud_augment.global_rotation(
                pcloud[:, [0, 1, 2]], data_augmentation_state['ROTATION'])
        if 'SCALING' in data_augmentation_state:
            pcloud[:, [0, 1, 2]] = point_cloud_augment.global_scaling(
                pcloud[:, [0, 1, 2]], data_augmentation_state['SCALING'])

    if 'SHUFFLE' in cfg['PREPROCESSING']['POINT_CLOUD']:
        if cfg['PREPROCESSING']['POINT_CLOUD']['SHUFFLE']:
            order = np.arange(0, pcloud.shape[0])
            np.random.shuffle(order)
            pcloud = pcloud[order]

    grid = cfg['PREPROCESSING']['POINT_CLOUD']['PILLARS']['GRID']

    # s = time.time()
    pillars, indices = create_pillars(
        pcloud,
        cfg['PREPROCESSING']['POINT_CLOUD']['PILLARS']['MAX_POINTS_PER_PILLAR'],
        cfg['PREPROCESSING']['POINT_CLOUD']['PILLARS']['NUMBER_PILLARS'],
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

    # print('python: ', 1e3*(time.time()-s))
    return pillars, indices


@numba.njit
def create_pillars(pcloud, max_points_per_pillar, max_number_pillars, x_step, y_step, x_min, x_max, y_min, y_max, z_min, z_max):

    # filter out of range points
    keep = np.where(
        (pcloud[:, 0] > x_min) & (pcloud[:, 0] < x_max) &
        (pcloud[:, 1] > y_min) & (pcloud[:, 1] < y_max) &
        (pcloud[:, 2] > z_min) & (pcloud[:, 2] < z_max)
    )
    pcloud = pcloud[keep]

    # map pcloud to grid
    pcloud_grid_index_xy = np.zeros((pcloud.shape[0], 2), dtype=np.int64)
    pcloud_grid_index_xy[:, 0] = np.floor((pcloud[:, 0] - x_min)/x_step)
    pcloud_grid_index_xy[:, 1] = np.floor((pcloud[:, 1] - y_min)/y_step)

    # pillars and indices arrays
    pillars = np.zeros(
        (max_number_pillars, max_points_per_pillar, pcloud.shape[1]+3))
    indices = np.zeros((max_number_pillars, 2))

    occupancy_lookup_grid = np.zeros(
        (pcloud_grid_index_xy[:, 0].max()+1, pcloud_grid_index_xy[:, 1].max()+1))

    # thrown_away_points = np.full(pcloud.shape[0], True)

    # loop over pillars
    i_pillar = 0
    for pillar_grid_index_xy in pcloud_grid_index_xy:

        if i_pillar >= max_number_pillars:
            break

        if occupancy_lookup_grid[pillar_grid_index_xy[0], pillar_grid_index_xy[1]] == 1:
            continue
        occupancy_lookup_grid[pillar_grid_index_xy[0],
                              pillar_grid_index_xy[1]] = 1

        grid_cell_corner_position = np.array([
            pillar_grid_index_xy[0]*x_step + x_min,
            pillar_grid_index_xy[1]*y_step + y_min,
            (z_max - z_min)/2
        ])

        in_pillar = np.where(
            (pcloud[:, 0] > grid_cell_corner_position[0]) & (pcloud[:, 0] < grid_cell_corner_position[0] + x_step) &
            (pcloud[:, 1] > grid_cell_corner_position[1]) & (pcloud[:, 1] < grid_cell_corner_position[1] + y_step) &
            (pcloud[:, 2] > z_min) & (pcloud[:, 2] < z_max)
        )[0]

        if pcloud[in_pillar].shape[0] < 1:
            continue

        pillar_center_position = np.array([
            pcloud[in_pillar, 0].mean(),
            pcloud[in_pillar, 1].mean(),
            pcloud[in_pillar, 2].mean(),
        ])

        if pcloud[in_pillar].shape[0] > max_points_per_pillar:
            in_pillar = in_pillar[:max_points_per_pillar]

        pcloud_in_pillar = pcloud[in_pillar]
        # thrown_away_points[in_pillar] = False

        indices[i_pillar] = pillar_grid_index_xy
        pillars[i_pillar, :pcloud_in_pillar.shape[0],
                0:3] = pcloud_in_pillar[:, :3] - grid_cell_corner_position
        pillars[i_pillar, :pcloud_in_pillar.shape[0],
                3:6] = pcloud_in_pillar[:, :3] - pillar_center_position
        pillars[i_pillar, :pcloud_in_pillar.shape[0],
                6:] = pcloud_in_pillar[:, 3:]

        i_pillar += 1

    # print(np.sum(thrown_away_points), i_pillar)

    pillars[:,:,3] /= x_step
    pillars[:,:,4] /= y_step
    pillars[:,:,5] /= np.abs(z_max - z_min)

    return pillars, indices
