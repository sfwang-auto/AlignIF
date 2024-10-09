import yaml
import torch
import random
import argparse
import numpy as np


class LossRecorder():
    def __init__(self, keys):
        self.total_loss = {key: 0 for key in keys}
        self.total_count = {key: 0 for key in keys}
        self.avg_loss = {key: 0 for key in keys}
    
    def update(self, key, loss, count=1):
        self.total_loss[key] += loss * count
        self.total_count[key] += count
        self.avg_loss[key] = self.total_loss[key] / self.total_count[key]


def set_seed(seed=0):
    """
    Sets random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args(config_file):
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '-c', '--config_dict',
        type=str,
        default='configs/', 
        help="Dict of the YAML configuration file"
    )
    
    args = parser.parse_args()
    args.config = args.config_dict + config_file

    # read yaml
    with open(args.config, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    
    # merge config
    for key, value in config.items():
        setattr(args, key, value)

    return args