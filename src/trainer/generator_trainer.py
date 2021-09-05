from abc import abstractmethod
from typing import Dict, NoReturn, Optional
from PIL import Image

from tqdm import tqdm
import numpy as np

from sklearn.manifold import TSNE

import torch
import torch.nn as nn
from torchvision import transforms, utils
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .base_trainer import BaseTrainer
from src.data import GenDataset, get_dataset, infinite_loader
from src.models.fid import get_fid_fn
from src.loss import get_adversarial_losses, get_regularizer
from src.utils import PathOrStr
from src.utils import tsne_display_tensorboard


class GeneratorTrainer(BaseTrainer):

    """Abstract class for generator trainers"""

    def __init__(self,
                 config_path: PathOrStr,
                 config: Dict):

        super().__init__(config_path, config)

        # mock objects
        self._g_ema = None
        self._encoder = None

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def _load_model(self):
        pass

    @abstractmethod
    def _save_model(self):
        pass

    def evaluate(self):
        ds_name = self._config['dataset']['name']

        # fid_score = self._compute_fid_score()
        # self._writer.add_scalar('FID', fid_score, 0)

        if ds_name != 'celeba':
            self._display_output_eps()
            self._explore_y()

        self._traverse_zk()
        self._explore_eps()
        self._explore_eps_zs()

    def _explore_eps_zs(self):
        traverse_samples = 8
        y = self._sample_label(traverse_samples)

        log_folder = self._writer.checkpoint_folder.parent / 'explore_eps_zs'
        log_folder.mkdir(exist_ok=True, parents=True)

        imgs = []

        for i in range(8):
            zs = self._g_ema.sample_zs(1)
            zs = torch.repeat_interleave(zs, traverse_samples, dim=0)

            eps = self._g_ema.sample_eps(1)
            eps = torch.repeat_interleave(eps, traverse_samples, dim=0)

            with torch.no_grad():
                img = self._g_ema(y, eps, zs).cpu()
                img = torch.cat([_img for _img in img], dim=1)

            imgs.append(img)

        imgs = torch.cat(imgs, dim=2)
        imgs = (imgs.permute(1, 2, 0).numpy() * 127.5 + 127.5).astype(np.uint8)

        Image.fromarray(imgs).save(
            log_folder / 'explore_eps_zs.png'
        )

    def _explore_eps(self):
        traverse_samples = 8
        y = self._sample_label(traverse_samples)

        log_folder = self._writer.checkpoint_folder.parent / 'explore_eps'
        log_folder.mkdir(exist_ok=True, parents=True)

        zs = self._g_ema.sample_zs(traverse_samples)
        imgs = []

        for i in range(traverse_samples):
            eps = self._g_ema.sample_eps(1)
            eps = torch.repeat_interleave(eps, traverse_samples, dim=0)

            with torch.no_grad():
                img = self._g_ema(y, eps, zs).cpu()
                img = torch.cat([_img for _img in img], dim=1)

            imgs.append(img)
        imgs = torch.cat(imgs, dim=2)
        imgs = (imgs.permute(1, 2, 0).numpy() * 127.5 + 127.5).astype(np.uint8)

        Image.fromarray(imgs).save(
            log_folder / 'traverse_eps.png'
        )

    def _traverse_zk(self):
        batch_size = self._config['batch_size']

        log_folder = self._writer.checkpoint_folder.parent / 'traverse_zk'
        log_folder.mkdir(exist_ok=True, parents=True)

        traverse_samples = 8
        y = self._sample_label(traverse_samples)

        # generate images
        with torch.no_grad():
            utils.save_image(
                self._g_ema(y),
                log_folder / 'sample.png',
                nrow=int(batch_size ** 0.5),
                normalize=True,
                value_range=(-1, 1),
            )

        traverse_range = 4.0
        intermediate_points = 9
        truncation = 0.7

        zs = self._g_ema.sample_zs(traverse_samples, truncation)
        es = self._g_ema.sample_eps(traverse_samples, truncation)
        _, n_layers, n_dim = zs.shape

        offsets = np.linspace(-traverse_range, traverse_range, intermediate_points)

        for i_layer in range(n_layers):
            for i_dim in range(n_dim):
                imgs = []
                for offset in offsets:
                    _zs = zs.clone()
                    _zs[:, i_layer, i_dim] = offset
                    with torch.no_grad():
                        img = self._g_ema(y, es, _zs).cpu()
                        img = torch.cat([_img for _img in img], dim=1)
                    imgs.append(img)
                imgs = torch.cat(imgs, dim=2)

                imgs = (imgs.permute(1, 2, 0).numpy() * 127.5 + 127.5).astype(np.uint8)
                Image.fromarray(imgs).save(
                    log_folder / f"traverse_L{i_layer}_D{i_dim}.png"
                )

    def _explore_y(self) -> NoReturn:
        n = 49
        y = self._sample_label(n)

        zs = self._g_ema.sample_zs(n)
        eps = self._g_ema.sample_eps(n)

        with torch.no_grad():
            imgs = self._g_ema(y, eps, zs).cpu()

        imgs = [imgs[i] for i in range(n)]
        imgs = torch.cat(imgs, dim=2)
        imgs = (imgs.permute(1, 2, 0).numpy() * 127.5 + 127.5).astype(np.uint8)

        log_folder = self._writer.checkpoint_folder.parent / 'explore_y'
        log_folder.mkdir(exist_ok=True, parents=True)
        Image.fromarray(imgs).save(log_folder / 'explore_y.png')

    def _display_output_eps(self) -> NoReturn:
        n_classes = self._config['dataset']['n_out']
        ds_name = self._config['dataset']['name']

        loader = self._get_dl()
        labels, embeddings = [], []

        # real data embeddings
        for _ in tqdm(range(200)):
            img, label = next(loader)
            img = img.to(self._device)

            with torch.no_grad():
                h, _ = self._encoder(img)

            labels.extend(label.cpu().numpy().tolist())
            embeddings.extend(h.cpu().numpy())

        # generated data embeddings
        for _ in tqdm(range(200)):
            label_oh = self._sample_label()

            with torch.no_grad():
                img = self._g_ema(label_oh)
                h, _ = self._encoder(img)

            label = torch.argmax(label_oh, dim=1) + n_classes

            labels.extend(label.cpu().numpy().tolist())
            embeddings.extend(h.cpu().numpy())

        labels = np.array(labels)
        embeddings = np.array(embeddings)

        tsne_emb = TSNE(n_components=2).fit_transform(embeddings)

        if ds_name != 'celeba':
            img_tsne = tsne_display_tensorboard(tsne_emb, labels, r'T-SNE of the model $\varepsilon$')
        else:
            img_tsne = tsne_display_tensorboard(tsne_emb, title=r'T-SNE of the model $\varepsilon$')

        self._writer.add_image('TSNE', img_tsne, 0)

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

    def _sample_label(self, n: Optional[int] = None) -> torch.Tensor:
        """Samples y label for the dataset

        Args:
            n: number of labels to sample

        Returns:
            torch.Tensor: sampled random label
        """

        ds_name = self._config['dataset']['name']
        n_out = self._config['dataset']['n_out']  # either number of classes, or size of the out vector (celeba)

        if n is None:
            batch_size = self._config['batch_size']
            n = batch_size

        if ds_name == 'celeba':
            label = torch.randint(2, (n, n_out)).float().to(self._device)
        else:
            label = torch.randint(n_out, (n,))
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
        columns = None if 'columns' not in self._config['dataset'] else self._config['dataset']['columns']
        batch_size = self._config['batch_size']
        n_workers = self._config['n_workers']

        transform = self._get_data_transform()
        dataset = get_dataset(name, path, anno_file=anno, transform=transform, columns=columns)
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
