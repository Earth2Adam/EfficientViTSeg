import torch
import cv2
import numpy as np
from PIL import Image

from tqdm import tqdm




__all__ = ["AverageMeter"]


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.sum = 0
        self.count = 0

    def update(self, val: torch.Tensor or int or float, delta_n=1):
        self.count += delta_n
        self.sum += val * delta_n

    def get_count(self) -> torch.Tensor or int or float:
        return self.count.item() if isinstance(self.count, torch.Tensor) and self.count.numel() == 1 else self.count

    @property
    def avg(self):
        avg = -1 if self.count == 0 else self.sum / self.count
        return avg.item() if isinstance(avg, torch.Tensor) and avg.numel() == 1 else avg

    
    


