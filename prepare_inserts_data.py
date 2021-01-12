from dataloader import LeddartechDataset
from utils import filter_occluded, get_pcloud_data

from pioneer.common.linalg import pcloud_inside_box

import argparse
import numpy as np
import os
import pickle
import tqdm
import yaml

# Too prevent "too many open files" error
import resource
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))

FILEPATH = os.path.dirname(os.path.abspath(__file__))
INSERTS_PATH = f"{FILEPATH}/inserts_data"


def main(cfg):

    os.makedirs(INSERTS_PATH, exist_ok=True)

    dataset = LeddartechDataset(cfg)

    counts = {}

    for index in tqdm.tqdm(range(len(dataset)), 'Extracting inserts data'):

        samples = dataset.platform[index]
        pcloud_sample = samples[cfg['DATASET']['LIDAR']][0]
        box3d_sample = samples[cfg['DATASET']['LABEL']][0]

        boxes3d = box3d_sample.raw['data']
        category_names = np.array(box3d_sample.label_names())

        boxes3d, category_names = filter_occluded(boxes3d, category_names, box3d_sample, occlusion_threshold=2)

        # Filter boxes outside range
        grid = cfg['PREPROCESSING']['BOX_3D']['GRID']
        keep = np.where((boxes3d['c'][:,0] >= grid['x'][0]) & (boxes3d['c'][:,0] < grid['x'][1]) & (boxes3d['c'][:,1] >= grid['y'][0]) & (boxes3d['c'][:,1] < grid['y'][1]))
        boxes3d, category_names = boxes3d[keep], category_names[keep]

        pcloud = get_pcloud_data(pcloud_sample, cfg)

        for i, box in enumerate(boxes3d):
            inside = pcloud_inside_box(pcloud[:,[0,1,2]], box, margin=0.1)
            if len(inside) < 2:
                continue

            data = {'box': box, 'pcloud': pcloud[inside], 'category_name': category_names[i]}

            if category_names[i] not in counts:
                counts[category_names[i]] = 0

            with open(f"{INSERTS_PATH}/{category_names[i]}_{counts[category_names[i]]:08d}.pkl", 'wb') as f:
                pickle.dump(data, f)

            counts[category_names[i]] += 1

        for _, sample in samples.items():
            sample[0].datasource.invalidate_caches()

    print(counts)




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg')
    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)

    main(cfg)

    

    