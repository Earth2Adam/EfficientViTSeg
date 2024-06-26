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
import math
import random
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import argparse
import torchvision.transforms.functional as F
from torch.nn.functional import pad

from efficientvit.models.utils import resize




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

    def __init__(self, data_dir, transform=None):
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

        self.transform = transform

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
        
        # EfficientViT only works with data size divisible by 64, so image size 1200x1920 is padded to 1216x1920
        feed_dict["data"] = pad(feed_dict["data"] , (0, 0, 8, 8), mode='constant', value=0)


        
        return {
            "index": index,
            "image_path": image_path,
            "mask_path": mask_path,
            **feed_dict,
        }

