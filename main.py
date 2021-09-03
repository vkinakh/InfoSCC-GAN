import argparse

from src.trainer import EpsilonConditionalGeneratorTrainer, ConditionalGeneratorTrainer, ClassificationTrainer
from src.utils import get_config


def run_generation(config_path):

    config = get_config(config_path)
    trainer = ConditionalGeneratorTrainer(config_path, config)
    trainer.train()


def run_classification(config_path):

    config = get_config(config_path)
    trainer = ClassificationTrainer(config_path, config)
    trainer.train()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--mode',
                        type=str,
                        help='mode to run',
                        choices=['classification', 'generation'])

    parser.add_argument('-c', '--config',
                        type=str,
                        help='path to config file')
    args = parser.parse_args()

    if args.mode == 'generation':
        run_generation(args.config)
    elif args.mode == 'classification':
        run_classification(args.config)
