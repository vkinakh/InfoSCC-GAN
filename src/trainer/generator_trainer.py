from abc import abstractmethod
from typing import Dict

import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .base_trainer import BaseTrainer
from src.data import GenDataset, get_dataset, infinite_loader
from src.models.fid import get_fid_fn
from src.loss import get_adversarial_losses, get_regularizer
from src.utils import PathOrStr


class GeneratorTrainer(BaseTrainer):

    """Abstract class for generator trainers"""

    def __init__(self,
                 config_path: PathOrStr,
                 config: Dict):

        super().__init__(config_path, config)

        # mock object
        self._g_ema = None

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def _load_model(self):
        pass

    @abstractmethod
    def _save_model(self):
        pass

    def _get_data_transform(self):
        """Returns transform for the data, based on the config

        Returns:
            data tranform
        """

        name = self._config['dataset']['name']
        size = self._config['dataset']['size']

        if name in ['mnist', 'fashionmnist']:
            transform = transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5, inplace=True)
            ])
        elif name in ['afhq', 'celeba']:
            transform = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(0.5, 0.5),
            ])
        else:
            raise ValueError('Unsupported dataset')

        return transform

    def _compute_fid_score(self) -> float:
        """Computes FID score for the generator

        Returns:
            float: FID score
        """

        name = self._config['dataset']['name']
        path = self._config['dataset']['path']
        anno = None if 'anno' not in self._config['dataset'] else self._config['dataset']['anno']

        transform = self._get_data_transform()
        dataset = GenDataset(name, path, True, anno, transform=transform)

        fid_func = get_fid_fn(dataset, self._device, len(dataset))
        fid_score = fid_func(self._g_ema)
        return fid_score

    def _sample_label(self) -> torch.Tensor:
        """Samples y label for the dataset

        Returns:
            torch.Tensor: sampled random label
        """

        ds_name = self._config['dataset']['name']
        n_out = self._config['dataset']['n_out']  # either number of classes, or size of the out vector (celeba)
        batch_size = self._config['batch_size']

        if ds_name == 'celeba':
            label = torch.randint(2, (batch_size, n_out)).float().to(self._device)
        else:
            label = torch.randint(n_out, (batch_size,))
            label = F.one_hot(label, num_classes=n_out).float().to(self._device)
        return label

    def _get_loss(self):
        """Returns loss functions for GAN based on config

        Returns:
            loss functions
        """

        ds_name = self._config['dataset']['name']

        d_adv_loss, g_adv_loss = get_adversarial_losses(self._config['loss'])
        d_reg_loss = get_regularizer("r1")

        if ds_name == 'celeba':
            cls_loss = nn.BCEWithLogitsLoss()
        else:
            cls_loss = nn.CrossEntropyLoss()
        return d_adv_loss, g_adv_loss, d_reg_loss, cls_loss

    def _get_dl(self):

        name = self._config['dataset']['name']
        path = self._config['dataset']['path']
        anno = None if 'anno' not in self._config['dataset'] else self._config['dataset']['anno']
        batch_size = self._config['batch_size']
        n_workers = self._config['n_workers']

        transform = self._get_data_transform()
        dataset = get_dataset(name, path, anno_file=anno, transform=transform)
        loader = infinite_loader(
            DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=n_workers
            )
        )
        return loader
