"""
Dataset file for the Rellis-3D. Used to preprocess the data and 
prepare it for the data loader. The Rellis dataset can be found here:
https://unmannedlab.github.io/research/RELLIS-3D
"""
import os
import pathlib
import numpy as np
from PIL import Image
import glob
import cv2
import math
import random
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import argparse
import torchvision.transforms.functional as F
from torch.nn.functional import pad

from efficientvit.models.utils import resize




class Resize(object):
    def __init__(
        self,
        crop_size: tuple[int, int] or None,
        interpolation: int or None = cv2.INTER_CUBIC,
    ):
        self.crop_size = crop_size
        self.interpolation = interpolation

    def __call__(self, feed_dict: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self.crop_size is None or self.interpolation is None:
            return feed_dict

        image, target = feed_dict["data"], feed_dict["label"]
        height, width = self.crop_size

        h, w, _ = image.shape
        if width != w or height != h:
            image = cv2.resize(
                image,
                dsize=(width, height),
                interpolation=self.interpolation,
            )
        return {
            "data": image,
            "label": target,
        }

    
class ToTensor(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, feed_dict: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        image, mask = feed_dict["data"], feed_dict["label"]
        image = image.transpose((2, 0, 1))  # (H, W, C) -> (C, H, W)
        image = torch.as_tensor(image, dtype=torch.float32).div(255.0)
        mask = torch.as_tensor(mask, dtype=torch.int64)
        image = F.normalize(image, self.mean, self.std, self.inplace)
        # Pad the tensor
        # The pad argument is in the form (pad_left, pad_right, pad_top, pad_bottom)
        image = pad(image, (0, 0, 8, 8), mode='constant', value=0)


        return {
            "data": image,
            "label": mask,
        }

    

class RellisDataset(Dataset):
    classes = (
        'void', 'grass', 'tree', 'pole', 'water', 'sky', 'vehicle', 'object',
        'asphalt', 'building', 'log', 'person', 'fence', 'bush', 'concrete',
        'barrier', 'puddle', 'mud', 'rubble'
       )
    class_colors = (
        (0, 0, 0), # void
        (0, 102, 0), # grass
        (0, 255, 0), # tree
        (0, 153, 153), # pole
        (0, 128, 255), # water
        (0, 0, 255), # sky
        (255, 255, 0), # vehicle
        (255, 0, 127), # object
        (64, 64, 64), # asphalt
        (255, 0, 0), # building
        (102, 0, 0), # log
        (204, 153, 255), # person
        (102, 0, 204), # fence
        (255, 153, 204), # bush
        (170, 170, 170), # concrete
        (41, 121, 255), # barrier
        (134, 255, 239), # puddle
        (99, 66, 34), # mud
        (110, 22, 138) # rubble
    )
    
    '''
    # this can be changed if we go back to the original, non-fixed dataset, which may be nice.
    label_map = np.array(
        (
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
        )
    )
    '''
    
    # for use with the original, non-fixed dataset
    label_map = np.array(
        (
            0, # void 0
            0, # dirt 1, # changed to 0 because so low samples (none actually i believe)
            0,
            1, # grass 3
            2, # tree 4
            3, # pole 5
            4, # water 6
            5,  # sky 7
            6,  # vehicle 8
            7, # object 9
            8, # asphalt 10
            0,  
            9,  # building 12
            0, 
            0,
            10, # log 15
            0,
            11,  # person 17
            12,  # fence 18
            13,  # bush 19
            0,  
            0,  
            0, 
            14,  # concrete 23
            0,  
            0,  
            0,  
            15,  # barrier 27
            0,  
            1, # from rellis github
            1, # from rellis github
            16,  # puddle 31
            4,  # from rellis github
            17,  # mud 33
            18, # rubble 34
        )
    )

    def __init__(self, data_dir):
        super().__init__()

        # load samples
        samples = []
        for dirpath, _, fnames in os.walk(os.path.join(data_dir, 'rgb')):
            if '.ipynb' in dirpath:
                continue
                
            for fname in sorted(fnames):
                suffix = pathlib.Path(fname).suffix
                if suffix not in [".jpg"]:
                    continue
                image_path = os.path.join(dirpath, fname)
                mask_path = image_path.replace("/rgb/", "/id/")
                if not mask_path.endswith(".png"):
                    mask_path = ".".join([*mask_path.split(".")[:-1], "png"])
                samples.append((image_path, mask_path))
        self.samples = samples

        # build transform
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(), # kinda just a placeholder for data_provider override
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, any]:
        image_path, mask_path = self.samples[index]
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))
        mask = self.label_map[mask]
        mask = torch.from_numpy(mask)
        feed_dict = {
            "data": image,
            "label": mask,
        }
        
        feed_dict["data"] = self.transform(feed_dict["data"])
        feed_dict["data"] = pad(feed_dict["data"] , (0, 0, 8, 8), mode='constant', value=0)


        
        return {
            "index": index,
            "image_path": image_path,
            "mask_path": mask_path,
            **feed_dict,
        }

