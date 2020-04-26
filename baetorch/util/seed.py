import torch
import numpy as np

def bae_set_seed(seed_value=100):
    """
    For ensuring reproducibility, sets the seed across torch, torch.cuda, and numpy
    """
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True  #tested - needed for reproducibility
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed(seed_value)
