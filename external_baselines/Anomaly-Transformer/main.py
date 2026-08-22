import os
import argparse
from pathlib import Path

from torch.backends import cudnn
from utils.utils import *

from solver import Solver


def str2bool(v):
    return v.lower() in ('true')


def main(config):
    cudnn.benchmark = True
    if (not os.path.exists(config.model_save_path)):
        mkdir(config.model_save_path)
    solver = Solver(vars(config))

    if config.mode == 'train':
        solver.train()
        # The original entry point stopped after optimization, leaving only
        # the checkpoint.  The unified external protocol requires the test
        # pass and its standardized metrics in the same output directory.
        solver.test()
    elif config.mode == 'test':
        solver.test()

    return solver


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--win_size', type=int, default=100)
    parser.add_argument('--input_c', type=int, default=38)
    parser.add_argument('--output_c', type=int, default=38)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='credit')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--data_path', type=str, default='./dataset/creditcard_ts.csv')
    parser.add_argument('--model_save_path', type=str, default='checkpoints')
    parser.add_argument('--anormly_ratio', type=float, default=4.00)
    parser.add_argument('--output_dir', type=str, default='')

    config = parser.parse_args()
    if config.output_dir:
        config.model_save_path = str(Path(config.output_dir) / "checkpoints")
        if config.pretrained_model and not os.path.isabs(config.pretrained_model):
            config.pretrained_model = str(Path(config.output_dir) / "checkpoints" / config.pretrained_model)

    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    main(config)
