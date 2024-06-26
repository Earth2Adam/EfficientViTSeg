from efficientvit.models.efficientvit import (
    EfficientViTSeg,
    efficientvit_seg_b0,
)
from efficientvit.models.nn.norm import set_norm_eps
from efficientvit.models.utils import load_state_dict_from_file

__all__ = ["create_seg_model"]


REGISTERED_SEG_MODEL: dict[str, dict[str, str]] = {
    "cityscapes": {
        "b0": "ckpts/b0.pt",
    },
    "rellis": {
        "b0": "ckpts/b0.pt",
    },
}


def create_seg_model(name, dataset, pretrained=True, weight_url=None, **kwargs):
    model_dict = {
        "b0": efficientvit_seg_b0,
    }

    model_id = name.split("-")[0]
    if model_id not in model_dict:
        raise ValueError(f"Do not find {name} in the model zoo. List of models: {list(model_dict.keys())}")
    else:
        model = model_dict[model_id](dataset=dataset, **kwargs)


    if pretrained:
        weight_url = weight_url or REGISTERED_SEG_MODEL[dataset].get(name, None)
        if weight_url is None:
            raise ValueError(f"Do not find the pretrained weight of {name}.")
        else:
            weight = load_state_dict_from_file(weight_url)
            model.load_state_dict(weight)
    return model
