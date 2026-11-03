import argparse
import os
import random
import numpy as np
import torch
from utils.config import _C
from utils.logger import setup_logger
from trainer import Trainer


def main(args):
    cfg = _C.clone()
    cfg.merge_from_file(f'configs/data/{args.data}.yaml')
    cfg.merge_from_file(f'configs/model/{args.model}.yaml')
    cfg.merge_from_list(args.opts)
    cfg.output_dir = os.path.join('output',cfg.output_dir or f'{args.data}_{args.model}')
    os.makedirs(cfg.output_dir,exist_ok=True)
    setup_logger(cfg.output_dir,cfg.dataset or 'sequence')
    print(cfg)
    random.seed(cfg.seed);np.random.seed(cfg.seed);torch.manual_seed(cfg.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic=cfg.deterministic
    torch.backends.cudnn.benchmark=not cfg.deterministic
    trainer=Trainer(cfg)
    if cfg.zero_shot: trainer.test_zero_shot()
    else: trainer.train()

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data','-d',default='TAIL')
    parser.add_argument('--model','-m',default='moram_vit_b16')
    parser.add_argument('opts',nargs=argparse.REMAINDER)
    main(parser.parse_args())
