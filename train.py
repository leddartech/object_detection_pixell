from dataloader import LeddartechDataset
from utils import train_valid_indices
import losses
import metrics
import models

from ignite.contrib.handlers import tqdm_logger
from ignite.engine import create_supervised_trainer, create_supervised_evaluator
from ignite.engine import Events
from ignite.metrics import Loss as ignite_loss

import torch
from torch.utils.data import DataLoader, Subset

import argparse
import datetime
import numpy as np
import os
import yaml

from shutil import copyfile

FILEPATH = os.path.dirname(os.path.abspath(__file__))


def main(cfg, resume_state=None):

    if 'AUGMENTATION' in cfg:
        if 'INSERT' in cfg['AUGMENTATION']:
            if not os.path.exists(f"{FILEPATH}/inserts_data/"):
                raise Exception('Insert data not found (data augmentation). Please run python3 prepare_inserts_data.py --cfg=CONFIG_FILE.') 

    # Prepare output data directory
    time_str = datetime.datetime.today().strftime("%Y_%m_%d_%H_%M_%S")
    results_directory = f"{FILEPATH}/results/{time_str}"
    os.makedirs(f"{results_directory}/states/")
    cfg['STATES_DIRECTORY'] = f"{results_directory}/states"
    with open(f"{results_directory}/config.yml", "w") as f:
        yaml.dump(cfg, f)

    # Random seed
    if 'SEED' in cfg['TRAINING']:
        torch.manual_seed(cfg['TRAINING']['SEED'])

    # Dataloaders
    dataset = LeddartechDataset(cfg)
    train_indices, valid_indices = train_valid_indices(len(dataset), cfg['TRAINING'])
    train_subset = Subset(dataset, train_indices)
    valid_subset = Subset(dataset, valid_indices)
    train_loader = DataLoader(train_subset, batch_size=cfg['TRAINING']['BATCH_SIZE'], num_workers=cfg['TRAINING']['NUM_WORKERS'], shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_subset, batch_size=cfg['TRAINING']['BATCH_SIZE'], num_workers=cfg['TRAINING']['NUM_WORKERS'], drop_last=True)
    print(f"Dataset size: {len(dataset)} | training set: {len(train_subset)} | validation set: {len(valid_subset)}")

    # Model
    in_channels = dataset.check_number_channels()
    model = getattr(models, cfg['NEURAL_NET']['NAME'])(cfg, in_channels)
    print(f"Model size: {model.size_of_net}")
    if cfg['TRAINING']['DEVICE'] == 'cuda' and torch.cuda.device_count() > 1: #Multi GPUs
        model = torch.nn.DataParallel(model)
    model.to(cfg['TRAINING']['DEVICE']) 
    print(f"Device set to: {cfg['TRAINING']['DEVICE']}")

    # Loss
    loss_function = list(cfg['TRAINING']['LOSS'].keys())[0]
    loss = getattr(losses, loss_function)(cfg, **cfg['TRAINING']['LOSS'][loss_function])

    # Optimizer
    optimizer_function = list(cfg['TRAINING']['OPTIMIZER'].keys())[0]
    optimizer = getattr(torch.optim, optimizer_function)(model.parameters(), **cfg['TRAINING']['OPTIMIZER'][optimizer_function])

    # Trainer engine
    trainer = create_supervised_trainer(model, optimizer, loss, device=cfg['TRAINING']['DEVICE'])
    pbar = tqdm_logger.ProgressBar(persist=True)
    pbar.attach(trainer, output_transform=lambda x: {'loss': x})

    # Evaluator engine
    eval_metrics = {'loss': ignite_loss(loss, device=cfg['TRAINING']['DEVICE'])}
    if 'METRICS' in cfg['TRAINING']:
        for metric in cfg['TRAINING']['METRICS']:
            eval_metrics[metric] = getattr(metrics, metric)(cfg, **cfg['TRAINING']['METRICS'][metric])
    evaluator = create_supervised_evaluator(model, metrics=eval_metrics, device=cfg['TRAINING']['DEVICE'])
    pbar2 = tqdm_logger.ProgressBar(persist=True, desc='Validation')
    pbar2.attach(evaluator)

    # Check for gradient explosion
    def check_grad(_):
        if not np.isfinite(trainer.state.output):
            print(loss.log)
            raise ValueError("Loss is not finite.")
    trainer.add_event_handler(Events.ITERATION_COMPLETED, check_grad)

    # Learning rate decay
    optimizer.lr_decay_factor = 1
    def lr_decay(_):
        for param_group in optimizer.param_groups:
            ep = trainer.state.epoch
            N = cfg['TRAINING']['SCHEDULER']['DECAY']['n_epochs']
            f = cfg['TRAINING']['SCHEDULER']['DECAY']['factor']
            optimizer.lr_decay_factor = np.exp(-ep/N) + f*(1-np.exp(-ep/N))
            param_group['lr'] = optimizer.lr_decay_factor*cfg['TRAINING']['OPTIMIZER'][optimizer_function]['lr']
            print(f"learning rate set to: {param_group['lr']}")
    if 'SCHEDULER' in cfg['TRAINING']:
        if 'DECAY' in cfg['TRAINING']['SCHEDULER']:
            trainer.add_event_handler(Events.EPOCH_STARTED, lr_decay)
 
    def handle_epoch_completed(_):
        torch.save(model.state_dict(), f"{cfg['STATES_DIRECTORY']}/{cfg['NEURAL_NET']['STATE_ID']}_{trainer.state.epoch:03d}.pt")
        dataset.data_augmentation = False
        evaluator.run(valid_loader)
        dataset.data_augmentation = True
        print('Validation results: ', evaluator.state.metrics)
        with open(f"{results_directory}/{cfg['NEURAL_NET']['STATE_ID']}.yml", "a") as f:
            yaml.dump({f'Epoch {trainer.state.epoch:03d}':evaluator.state.metrics}, f)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, handle_epoch_completed)

    # Resume training
    def resume_training(trainer):
        if resume_state is not None:
            resume_epoch = int(resume_state.split('_')[-1].split('.')[0])
            model.load_state_dict(torch.load(resume_state))
            trainer.state.iteration = resume_epoch * len(trainer.state.dataloader)
            trainer.state.epoch = resume_epoch
        else:
            with open(f"{results_directory}/{cfg['NEURAL_NET']['STATE_ID']}.yml", "w") as f:
                pass
    trainer.add_event_handler(Events.STARTED, resume_training)

    # Start training
    dataset.data_augmentation = True
    trainer.run(train_loader, max_epochs=cfg['TRAINING']['EPOCHS'])

    return results_directory



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg')
    parser.add_argument('--resume_state', default=None)
    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)

    main(cfg, resume_state=args.resume_state)

