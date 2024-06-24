import copy
import math
import os

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from efficientvit.apps.data_provider import CityBaseDataProvider
from efficientvit.apps.data_provider.augment import RandAug
from efficientvit.apps.data_provider.random_resolution import MyRandomResizedCrop, get_interpolate
from efficientvit.apps.utils import partial_update_config
from efficientvit.models.utils import val2list



from datasets.cityscapes import CityscapesDataset

__all__ = ["CityscapesDataProvider"]


class CityscapesDataProvider(CityBaseDataProvider):
    name = "cityscapes"

    data_dir = '/scratch/apicker/cityscapes/leftImg8bit/'
    n_classes = 19
    def __init__(
        self,
        data_dir: str or None = None,
        data_aug: dict or list[dict] or None = None,
        ###########################################
        train_batch_size=4,
        test_batch_size=4,
        n_worker=8,
    ):
        self.data_dir = data_dir or self.data_dir
        self.data_aug = data_aug

        super().__init__(
            train_batch_size,
            test_batch_size,
            n_worker,
        )

    def build_valid_transform(self):
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(**self.mean_std),
            ]
        )

    def build_train_transform(self):

        train_transforms = [
           # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
            #transforms.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random'),
            #transforms.RandomHorizontalFlip(),
        ]

        # data augmentation
        post_aug = []
        
        if self.data_aug is not None:
            for aug_op in val2list(self.data_aug):
                if aug_op["name"] == "randaug":
                    data_aug = RandAug(aug_op, mean=self.mean_std["mean"])
                elif aug_op["name"] == "erase":
                    from timm.data.random_erasing import RandomErasing

                    random_erase = RandomErasing(aug_op["p"], device="cpu")
                    post_aug.append(random_erase)
                    data_aug = None
                else:
                    raise NotImplementedError
                if data_aug is not None:
                    train_transforms.append(data_aug)
        
        train_transforms = [
            *train_transforms,
            transforms.ToTensor(),
            transforms.Normalize(**self.mean_std),
            *post_aug,
        ]
        return transforms.Compose(train_transforms)

    def build_datasets(self) -> tuple[any, any, any]:
        train_transform = self.build_train_transform()
        valid_transform = self.build_valid_transform()


        train_dataset = CityscapesDataset(os.path.join(self.data_dir,'train'))
        test_dataset = CityscapesDataset(os.path.join(self.data_dir,'val'))
        
        
        # this is IMPORTANT!!
        train_dataset.transform = train_transform
        test_dataset.transform = valid_transform

        val_dataset = test_dataset

        return train_dataset, val_dataset, test_dataset
