# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023

from efficientvit.apps.trainer.run_config import RunConfig

__all__ = ["SegRunConfig"]


class SegRunConfig(RunConfig):
    mixup_config: dict  # allow none to turn off mixup
    
    @property
    def none_allowed(self):
        return ["mixup_config"]+ super().none_allowed
