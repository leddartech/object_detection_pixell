from object_detection_pixell.inference import DasPredictor

from pioneer.das.api.platform import Platform
from pioneer.common.platform import extract_sensor_id

from ruamel import yaml

import argparse
import os
import pickle
import numpy as np
import tqdm


def main(cfg, state, dataset, input_datasource):

    # FIXME: crash if test set is a directory of multiple datasets
    dataset = cfg['DATASET']['TEST_SET'][0] if dataset == 'test_set' else dataset

    pf = Platform(dataset)

    if 'VIRTUAL_DATASOURCES' in cfg['DATASET']:
        pf.add_virtual_datasources(cfg['DATASET']['VIRTUAL_DATASOURCES'])

    vds = DasPredictor(cfg, state, input_datasource, verbose=False)

    sensor_name = extract_sensor_id(cfg['DATASET']['LABEL'])
    pf[sensor_name].add_datasource(vds, vds.ds_type)

    prediction_datasource = f'{sensor_name}_{vds.ds_type}'
    output_directory = f'{dataset}/{prediction_datasource}'
    os.makedirs(output_directory, exist_ok=True)

    for frame in tqdm.tqdm(range(len(pf[vds.input_datasource])), 'Inference'):
        prediction_sample = pf[prediction_datasource][frame]

        with open(f"{output_directory}/{frame:08d}.pkl", "wb") as f:
            pickle.dump(prediction_sample.raw, f)

    np.savetxt(f"{output_directory}/timestamps.csv", pf[vds.input_datasource].timestamps, fmt='%.d')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg')
    parser.add_argument('--state')
    parser.add_argument('--dataset', default='test_set')
    parser.add_argument('--input', default='same_as_training')
    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)

    main(cfg, args.state, args.dataset, args.input)