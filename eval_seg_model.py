import argparse
import math
import os
import pathlib

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from tqdm import tqdm




from efficientvit.apps.utils import AverageMeter, SegIoU
from efficientvit.models.utils import resize

from efficientvit.seg_model_zoo import create_seg_model


# local datasets
from datasets.rellis import RellisDataset
from datasets.cityscapes import CityscapesDataset




def get_canvas(
    image: np.ndarray,
    mask: np.ndarray,
    colors: tuple or list,
    opacity=0.5,
) -> np.ndarray:
    image_shape = image.shape[:2]
    mask_shape = mask.shape
    if image_shape != mask_shape:
        mask = cv2.resize(mask, dsize=(image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
    seg_mask = np.zeros_like(image, dtype=np.uint8)
    for k, color in enumerate(colors):
        seg_mask[mask == k, :] = color
    canvas = seg_mask * opacity + image * (1 - opacity)
    canvas = np.asarray(canvas, dtype=np.uint8)
    return canvas


def main():
    parser = argparse.ArgumentParser()    
    
   # parser.add_argument("--path", type=str, default="/scratch/apicker/cityscapes/leftImg8bit/val")   my cityscapes default image path
    parser.add_argument("--path", type=str, default="/scratch/apicker/rellis3d-nonfixed/test")  # my rellis default image path
    parser.add_argument("--dataset", type=str, default="rellis", choices=["cityscapes", "rellis"])
    parser.add_argument("--batch_size", help="batch size per gpu", type=int, default=1)
    parser.add_argument("-j", "--workers", help="number of workers", type=int, default=4)
    parser.add_argument("--crop_size", type=int, default=1024)
    parser.add_argument("--model", type=str, default='b0')
    parser.add_argument("--weight_url", type=str, default=None)
    parser.add_argument("--save_path", type=str, default=None)

    args = parser.parse_args()

    
    valid_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    if args.dataset == "cityscapes":
        dataset = CityscapesDataset(args.path, (args.crop_size, args.crop_size * 2))
    elif args.dataset == "rellis":
        dataset = RellisDataset(args.path, transform=valid_transform)
    else:
        raise NotImplementedError
        
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )
    
    print(f'len of dataset: {len(dataset)}')
    print(f'Length of dataloader: {len(data_loader)}')

    model = create_seg_model(args.model, args.dataset, weight_url=args.weight_url)
    model = torch.nn.DataParallel(model).cuda()
    model.eval()

    if args.save_path is not None:
        os.makedirs(args.save_path, exist_ok=True)

    interaction = AverageMeter()
    union = AverageMeter()
    iou = SegIoU(len(dataset.classes))

    with torch.inference_mode():
        with tqdm(total=len(data_loader), desc=f"Eval {args.model} on {args.dataset}") as t:
            for feed_dict in data_loader:
                images, mask = feed_dict["data"].cuda(), feed_dict["label"].cuda()
                
                # compute output
                output = model(images)
                
                # resize the output to match the shape of the mask
               # if output.shape[-2:] != mask.shape[-2:]:
                #    output = resize(output, size=mask.shape[-2:])
                output = torch.argmax(output, dim=1)
                stats = iou(output, mask)
                
                interaction.update(stats["i"])
                union.update(stats["u"])

                t.set_postfix(
                    {
                        "mIOU": (interaction.sum / union.sum).cpu().mean().item() * 100,
                        "image_size": list(images.shape[-2:]),
                    }
                )
                t.update()

                if args.save_path is not None:
                    with open(os.path.join(args.save_path, "summary.txt"), "a") as fout:
                        for i, (idx, image_path) in enumerate(zip(feed_dict["index"], feed_dict["image_path"])):
                            pred = output[i].cpu().numpy()
                            raw_image = np.array(Image.open(image_path).convert("RGB"))
                            canvas = get_canvas(raw_image, pred, dataset.class_colors)
                            canvas = Image.fromarray(canvas)
                            canvas.save(os.path.join(args.save_path, f"{idx}.png"))
                            fout.write(f"{idx}:\t{image_path}\n")

    mIoU = (interaction.sum / union.sum).cpu().mean().item() * 100
    class_IoUs = (interaction.sum / union.sum).cpu().numpy() * 100
    print(f"mIoU = {mIoU:.3f}")
    print("Class IoUs:")
    for i, class_iou in enumerate(class_IoUs):
        print(f"Class {i}: {class_iou:.2f}")

if __name__ == "__main__":
    main()
