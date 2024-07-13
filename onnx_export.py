import argparse

import torch

from efficientvit.apps.utils import export_onnx
from efficientvit.seg_model_zoo import create_seg_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export_path", type=str)
    parser.add_argument("--dataset", type=str, default="rellis", choices=["rellis", "cityscapes"])
    parser.add_argument("--model", type=str, default="b0")
    parser.add_argument("--weight_url", type=str, default='ckpts/rellis_best.pt')
    parser.add_argument("--bs", help="batch size", type=int, default=1)
    parser.add_argument("--op_set", type=int, default=11)

    args = parser.parse_args()
    resolution = (640,512) 
    model = create_seg_model(
        name=args.model,
        dataset=args.dataset,
        weight_url=args.weight_url,
        #pretrained=False,
    )

    dummy_input = torch.rand((args.bs, 3,*resolution))
    export_onnx(model, args.export_path, dummy_input, simplify=True, opset=args.op_set)


if __name__ == "__main__":
    main()
