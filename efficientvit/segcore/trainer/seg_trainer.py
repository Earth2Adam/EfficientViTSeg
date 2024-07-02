import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientvit.models.utils import resize
from tqdm import tqdm

from efficientvit.apps.trainer import Trainer
from efficientvit.apps.utils import AverageMeter
from efficientvit.segcore.trainer.utils import accuracy, apply_mixup, label_smooth, eval_IoU
from efficientvit.models.utils import list_join, list_mean, torch_random_choices
__all__ = ["SegTrainer"]




class SegTrainer(Trainer):
    def __init__(
        self,
        path: str,
        model: nn.Module,
        data_provider,
    ) -> None:
        super().__init__(
            path=path,
            model=model,
            data_provider=data_provider,
        )
        
        

    def _validate(self, model, data_loader, epoch):
        val_loss = AverageMeter()

        with torch.no_grad():
            for feed_dict in data_loader:
                images = feed_dict["data"]
                masks = feed_dict["label"]

                images, masks = images.cuda(), masks.cuda()

                # compute output
                output = model(images)
                if output.shape[-2:] != masks.shape[-2:]:
                    output = resize(output, size=masks.shape[-2:])

                loss = self.test_criterion(output, masks)
                val_loss.update(loss, images.shape[0])

        return val_loss.avg
 

    def before_step(self, feed_dict: dict[str, any]) -> dict[str, any]:
        images = feed_dict["data"].cuda()
        masks = feed_dict["label"].cuda()

        return {
            "data": images,
            "label": masks,
        }

    def run_step(self, feed_dict: dict[str, any]) -> dict[str, any]:
        images = feed_dict["data"]
        masks = feed_dict["label"]
        
        # this section used to autocast to fp16, removed for now
        output = self.model(images)
        if output.shape[-2:] != masks.shape[-2:]:
            output = resize(output, size=masks.shape[-2:])


        loss = self.train_criterion(output, masks)
                 
        self.scaler.scale(loss).backward()


        return {
            "loss": loss,
        }

    def _train_one_epoch(self, epoch: int) -> dict[str, any]:
        train_loss = AverageMeter()
        self.model.train()

        for feed_dict in self.data_provider.train:

            # preprocessing
            feed_dict = self.before_step(feed_dict)

            # clear gradient
            self.optimizer.zero_grad()

            # forward & backward
            output_dict = self.run_step(feed_dict)

            # update: optimizer, lr_scheduler
            self.after_step()

            # update train metrics
            train_loss.update(output_dict["loss"], 1024)


        return {
            "train_loss": train_loss.avg,
        }

    def train(self, eval_save_freq=25) -> None:

        self.train_criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.test_criterion = self.train_criterion
        self.best_mIoU = 0.0
        
        
        for epoch in range(self.start_epoch, self.run_config.n_epochs + self.run_config.warumup_epochs):
            train_info_dict = self.train_one_epoch(epoch)
            
            # eval and save model
            if (epoch + 1) % eval_save_freq == 0:

                # eval
                val_loss = self._validate(self.model, self.data_provider.valid, epoch=epoch)

                # start IOU section 
                val_mIoU = eval_IoU(self.model, self.data_provider.valid)

                is_best = val_mIoU > self.best_mIoU
                self.best_mIoU = min(val_mIoU, self.best_mIoU)

                self.save_model(
                    only_state_dict=True,
                    epoch=epoch,
                    model_name=f"model_best_{epoch+1}.pt" if is_best else f"checkpoint{epoch+1}.pt",
                )
                
                info = f"Epoch {epoch+1} train loss {train_info_dict['train_loss']:.3f} val loss {val_loss:.3f} val_mIoU {val_mIoU:.2f}%"
                
            else:
                info = f"Epoch {epoch+1} train loss {train_info_dict['train_loss']:.3f}"
            
            # log
            self.write_log(info, print_log=False)
            