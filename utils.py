import random

import numpy as np
import torch
import yaml

def load_config(config_file):
    with open(config_file) as file:
        config = yaml.safe_load(file)
    return config

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)