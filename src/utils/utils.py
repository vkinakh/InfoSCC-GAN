from typing import NoReturn

import yaml
from functools import partial

import torch
import torch.nn as nn
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode


image_loader = partial(read_image, mode=ImageReadMode.RGB)


def get_device() -> str:
    """Returns available torch device

    Returns:
        str: available torch device
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return device


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


def accumulate(model1: nn.Module, model2: nn.Module, decay: float = 0.999) -> NoReturn:
    """Applies parameter accumulation to copy parameters from model2 into model1

    Args:
        model1: model to copy parameters to

        model2: model to copy parameters from

        decay: multiplier to use, when coping parameters
    """

    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)
