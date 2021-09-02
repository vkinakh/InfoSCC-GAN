from src.trainer import ConditionalGeneratorTrainer
from src.utils import get_config


if __name__ == '__main__':

    config_path = './configs/afhq_generation_multiclass_disc.yml'
    config = get_config(config_path)

    trainer = ConditionalGeneratorTrainer(config_path, config)
    trainer.train()
