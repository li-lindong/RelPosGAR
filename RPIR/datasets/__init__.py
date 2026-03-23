import torch
from torchvision import transforms
import numpy
import random

from RPIR.datasets.volleyball import VolleyballDataset
from RPIR.datasets.nba import NBA_Dataset

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

def dataset_entry(config, dataset_name, evaluate=False, test_img=False):
    if dataset_name == 'volleyball':
        train_set = VolleyballDataset(config.train)
        val_set = VolleyballDataset(config.val)
    elif dataset_name == 'nba':
        train_set = NBA_Dataset(config.train)
        val_set = NBA_Dataset(config.val)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.train.batch_size, shuffle=True,
                                               num_workers=config.workers, pin_memory=True, worker_init_fn=seed_worker)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=config.val.batch_size, shuffle=False,
                                             num_workers=config.workers, pin_memory=True, worker_init_fn=seed_worker)
    return train_loader, val_loader
