from dataloader import LeddartechDataset
import metrics
import models
from utils import get_state_dict

from ignite.contrib.handlers import tqdm_logger
from ignite.engine import create_supervised_evaluator

import torch
from torch.utils.data import DataLoader

import argparse
import glob
import os
import yaml

FILEPATH = os.path.dirname(os.path.abspath(__file__))



def main(cfg, state, plot=False):

    # Dataloaders
    dataset = LeddartechDataset(cfg, use_test_set=True)
    test_loader = DataLoader(dataset, batch_size=cfg['TRAINING']['BATCH_SIZE'], num_workers=cfg['TRAINING']['NUM_WORKERS'])
    print(f"Dataset size: {len(dataset)}")
    
    # Model
    in_channels = dataset.check_number_channels()
    model = getattr(models, cfg['NEURAL_NET']['NAME'])(cfg, in_channels)
    print(f"Model size: {model.size_of_net}")
    if cfg['TRAINING']['DEVICE'] == 'cuda' and torch.cuda.device_count() > 1: #Multi GPUs
        model = torch.nn.DataParallel(model)
    model.to(cfg['TRAINING']['DEVICE']) 
    print(f"Device set to: {cfg['TRAINING']['DEVICE']}")

    # Load model state
    state_dict = get_state_dict(state, device=cfg['TRAINING']['DEVICE'])
    model.load_state_dict(state_dict)
    model.eval()

    # Evaluator engine
    eval_metrics = {}
    for metric in cfg['TRAINING']['METRICS']:
        eval_metrics[metric] = getattr(metrics, metric)(cfg, **cfg['TRAINING']['METRICS'][metric])
    evaluator = create_supervised_evaluator(model, metrics=eval_metrics, device=cfg['TRAINING']['DEVICE'])
    pbar2 = tqdm_logger.ProgressBar(persist=True, desc='Testing')
    pbar2.attach(evaluator)

    # Start testing
    evaluator.run(test_loader)
    print('Test results: ', evaluator.state.metrics)

    if plot:
        for metric in cfg['TRAINING']['METRICS']:
            if hasattr(eval_metrics[metric], 'make_plot'):
                eval_metrics[metric].make_plot(evaluator.state.metrics)

    return evaluator.state.metrics


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg')
    parser.add_argument('--state')
    parser.add_argument('--plot', type=bool, default=False)
    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)

    main(cfg, state=args.state, plot=args.plot)
