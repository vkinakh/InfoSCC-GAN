import argparse

from src.trainer import ConditionalGeneratorTrainer, ClassificationTrainer
from src.utils import get_config


def train_generation(config_path):

    config = get_config(config_path)
    trainer = ConditionalGeneratorTrainer(config_path, config)
    trainer.train()


def evaluate_generation(config_path):

    config = get_config(config_path)
    trainer = ConditionalGeneratorTrainer(config_path, config)
    trainer.evaluate()


def run_classification(config_path):

    config = get_config(config_path)
    trainer = ClassificationTrainer(config_path, config)
    trainer.train()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--mode',
                        type=str,
                        help='mode to run',
                        choices=['train', 'evaluate'])

    parser.add_argument('-t', '--task',
                        type=str,
                        help='task to run',
                        choices=['generation', 'classification'])

    parser.add_argument('-c', '--config',
                        type=str,
                        help='path to config file')
    args = parser.parse_args()

    if args.mode == 'train':

        if args.task == 'generation':
            train_generation(args.config)
        elif args.task == 'classification':
            run_classification(args.config)

    elif args.mode == 'evaluate':
        evaluate_generation(args.config)
