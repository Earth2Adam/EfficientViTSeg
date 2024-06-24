import torch
from efficientvit.apps.utils import AverageMeter
from efficientvit.models.utils import resize

__all__ = ["accuracy", "eval_IOU"]


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)) -> list[torch.Tensor]:
    """Computes the precision@k for the specified values of k."""
    maxk = max(topk)
    batch_size = target.shape[0]

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res




class SegIOU:
    def __init__(self, num_classes: int, ignore_index: int = -1) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def __call__(self, outputs: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = (outputs + 1) * (targets != self.ignore_index)
        targets = (targets + 1) * (targets != self.ignore_index)
        intersections = outputs * (outputs == targets)

        outputs = torch.histc(
            outputs,
            bins=self.num_classes,
            min=1,
            max=self.num_classes,
        )
        targets = torch.histc(
            targets,
            bins=self.num_classes,
            min=1,
            max=self.num_classes,
        )
        intersections = torch.histc(
            intersections,
            bins=self.num_classes,
            min=1,
            max=self.num_classes,
        )
        unions = outputs + targets - intersections

        return {
            "i": intersections,
            "u": unions,
        }



def eval_IOU(model, val_loader):

    model.eval()
    interaction = AverageMeter()
    union = AverageMeter()
    
    iou = SegIOU(num_classes=19, ignore_index=-1)
    
    num = 1
    with torch.inference_mode():
        for feed_dict in val_loader:
            images, mask = feed_dict["data"].cuda(), feed_dict["label"].cuda()
            
            
            # compute output
            output = model(images)
            
    
            
            # resize the output to match the shape of the mask
            if output.shape[-2:] != mask.shape[-2:]:
                output = resize(output, size=mask.shape[-2:])
            output = torch.argmax(output, dim=1)
            if num == 1:
                print(f'size of i,m,o (after argmax and upsample) {images.shape} {mask.shape} {output.shape}')
                num = 0
            stats = iou(output, mask)
            interaction.update(stats["i"])
            union.update(stats["u"])

             
    #print(f"mIoU = {(interaction.sum / union.sum).cpu().mean().item() * 100:.3f}")
    
    return (interaction.sum / union.sum).cpu().mean().item() * 100
    
