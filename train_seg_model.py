# for rellis!

import argparse
import os
import torch 
import torch.nn as nn
from efficientvit.apps import setup
from efficientvit.apps.utils import dump_config, parse_unknown_args
from efficientvit.seg_model_zoo import create_seg_model
from efficientvit.segcore.trainer import SegRunConfig, SegTrainer
from efficientvit.models.nn.drop import apply_drop_func
from efficientvit.segcore.data_provider import CityscapesDataProvider, RellisDataProvider

parser = argparse.ArgumentParser()
parser.add_argument("config", metavar="FILE", help="config file")

parser.add_argument("--path", type=str, metavar="DIR", help="run directory")
parser.add_argument("--gpu", type=str, default=None)  # used in single machine experiments
parser.add_argument("--manual_seed", type=int, default=0)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--fp16", action="store_true")

# initialization
parser.add_argument("--rand_init", type=str, default="trunc_normal@0.02")
parser.add_argument("--last_gamma", type=float, default=0)
parser.add_argument("--weight_url", type=str, help="ckpt directory")

parser.add_argument("--auto_restart_thresh", type=float, default=1.0)
parser.add_argument("--save_freq", type=int, default=25)


def main():
    # parse args
    args, opt = parser.parse_known_args()
    opt = parse_unknown_args(opt)


    # setup path, update args, and save args to path
    os.makedirs(args.path, exist_ok=True)
    dump_config(args.__dict__, os.path.join(args.path, "args.yaml"))

    # setup random seed
    setup.setup_seed(args.manual_seed, args.resume)

    # setup exp config
    config = setup.setup_exp_config(args.config, recursive=True, opt_args=opt)
    
    
    dump_config(config, os.path.join(args.path, "config.yaml")) ## i added

    # save exp config
    setup.save_exp_config(config, args.path)

        
    
    # setup data provider
    data_provider = setup.setup_data_provider(config, [RellisDataProvider], is_distributed=False)

    

    # setup run config
    run_config = setup.setup_run_config(config, SegRunConfig)
    dump_config(run_config, os.path.join(args.path, "run_config.yaml"))

        
    # setup 
    model = create_seg_model(config["net_config"]["name"], "cityscapes", dropout=config["net_config"]["dropout"], weight_url=args.weight_url)
    apply_drop_func(model.backbone.stages, config["backbone_drop"])

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        


    
    # setup trainer
    trainer = SegTrainer(
        path=args.path,
        model=model,
        data_provider=data_provider,
        #auto_restart_thresh=args.auto_restart_thresh,
    )
    # initialization
    setup.init_model(
        trainer.network,
        rand_init=args.rand_init,
        last_gamma=args.last_gamma,
    )

    # prep for training
    trainer.prep_for_training(run_config, ema_decay=None, fp16=args.fp16)


    # launch training
    trainer.train(save_freq=args.save_freq)


if __name__ == "__main__":
    main()
