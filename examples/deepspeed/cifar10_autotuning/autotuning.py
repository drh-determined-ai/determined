import argparse

import torch
import torch.nn as nn
import deepspeed

from model_def import Net


def parse_args():
    parser = argparse.ArgumentParser()
    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)
    # Absorb a possible `local_rank` arg from the launcher.
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="local rank passed from distributed launcher"
    )

    args = parser.parse_args()
    return args


def main():
    # GG: Why is this needed?
    deepspeed.init_distributed()

    net = Net()
    args = parse_args()
    model_engine, optimizer, trainloader, __ = deepspeed.initialize(
        args=args,
        model=net,
    )

    fp16 = model_engine.fp16_enabled()

if __name__ == "__main__":
    main()
