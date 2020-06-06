import yaml
import argparse

from trainer import BaseTrainer


def arg_parser():
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument("--config",
                        type=str,
                        default="config/train.yaml",
                        help="Specified the path of configuration file to be used.")

    return parser.parse_args()


if __name__ == '__main__':
    import torch

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    arg = arg_parser()
    config = arg.config

    with open(config) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    trainer = BaseTrainer(config, stage="Train")
    trainer.train()
