from torch import nn as nn
from torch.nn import functional as F

upsample = lambda x, size: F.interpolate(x, size, mode='bilinear', align_corners=False)



class SemsegCrossEntropy(nn.Module):
    def __init__(self, num_classes=19, ignore_id=19):
        super(SemsegCrossEntropy, self).__init__()
        self.num_classes = num_classes
        self.ignore_id = ignore_id
        self.step_counter = 0

    def loss(self, y, t):
        if y.shape[2:4] != t.shape[1:3]:
            y = upsample(y, t.shape[1:3])
        return F.cross_entropy(y, target=t, ignore_index=self.ignore_id)

    def forward(self, logits, labels, **kwargs):
        loss = self.loss(logits, labels)
        self.step_counter += 1
        return loss
