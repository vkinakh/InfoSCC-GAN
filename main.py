import argparse

from src.trainer import ConditionalGeneratorTrainer
from src.utils import get_config


def main(config_path):

    config = get_config(config_path)
    trainer = ConditionalGeneratorTrainer(config_path, config)
    trainer.train()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config',
                        type=str,
                        help='path to config file')
    args = parser.parse_args()
    main(args.config)
