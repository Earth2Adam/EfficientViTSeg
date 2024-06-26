import torch
from efficientvit.apps.utils import AverageMeter, SegIoU
from efficientvit.models.utils import resize

__all__ = ["accuracy", "eval_IoU"]


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


def eval_IoU(model, val_loader):

    model.eval()
    interaction = AverageMeter()
    union = AverageMeter()
    
    iou = SegIoU(num_classes=19, ignore_index=-1)
    
    with torch.inference_mode():
        for feed_dict in val_loader:
            images, mask = feed_dict["data"].cuda(), feed_dict["label"].cuda()
            
            
            # compute output
            output = model(images)
            
            
            # resize the output to match the shape of the mask
            if output.shape[-2:] != mask.shape[-2:]:
                output = resize(output, size=mask.shape[-2:])
            output = torch.argmax(output, dim=1)
            stats = iou(output, mask)
            interaction.update(stats["i"])
            union.update(stats["u"])

                 
    return (interaction.sum / union.sum).cpu().mean().item() * 100
    