import argparse

from src.trainer import SimCLRTrainer, ClassificationTrainer, ConditionalGeneratorTrainer
from src.utils import get_config


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--mode',
                        type=str,
                        help='mode to run',
                        choices=['train', 'evaluate'])

    parser.add_argument('-t', '--task',
                        type=str,
                        help='task to run',
                        choices=['encoder', 'generation', 'classification'])

    parser.add_argument('-c', '--config',
                        type=str,
                        help='path to config file')
    args = parser.parse_args()

    config_path = args.config
    config = get_config(config_path)
    mode = args.mode
    task = args.task

    if task == 'encoder':
        trainer = SimCLRTrainer(config_path, config)
    elif task == 'classification':
        trainer = ClassificationTrainer(config_path, config)
    elif task == 'generation':
        trainer = ConditionalGeneratorTrainer(config_path, config)

    if mode == 'train':
        trainer.train()
    elif mode == 'evaluate':

        if task == 'generation':
            trainer.evaluate()
