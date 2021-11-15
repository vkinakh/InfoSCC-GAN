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
from torch.utils.data import DataLoader, Dataset
from chamferdist import ChamferDistance

from .base_trainer import BaseTrainer
from src.data import GenDataset, get_dataset, infinite_loader, GANDataset
from src.models.fid import get_fid_fn
from src.models import inception_score
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
        self._classifier = None

        # if y type real, load sample dataset
        if config['generator']['y_type'] == 'real':
            self._sample_ds = self._get_ds()

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def _load_model(self):
        pass

    @abstractmethod
    def _save_model(self, *args, **kwargs):
        pass

    def evaluate(self) -> NoReturn:
        """Runs evaluation:
        - FID score
        - Inception score
        - Displays embeddings
        - Explores labels
        - Chamfer distance
        - Attribute control accuracy
        - Traverses z1, ... zk variables
        - Explores epsilon
        - Explores epsilon and zs
        """

        ds_name = self._config['dataset']['name']

        fid_score = self._compute_fid_score()
        self._writer.add_scalar('FID', fid_score, 0)

        i_score = self._compute_inception_score()
        self._writer.add_scalar('IS', i_score, 0)

        if ds_name != 'celeba':
            self._display_output_eps()
            self._explore_y()

        chamfer_dist = self._chamfer_distance()
        self._writer.add_scalar('Chamfer', float(chamfer_dist), 0)

        self._attribute_control_accuracy()

        self._traverse_zk()
        self._explore_eps()
        self._explore_eps_zs()

    def _attribute_control_accuracy(self):
        """Runs attribute control accuracy"""

        ds_name = self._config['dataset']['name']

        if ds_name == 'celeba':
            res = self._attribute_control_accuracy_multi_label()
        else:
            res = self._attributes_control_accuracy_one_hot()

        for key, val in res.items():
            self._writer.add_scalar(f'acc/{key}', val, 0)
        return res

    def _attribute_control_accuracy_multi_label(self):
        """Runs attribute control accuracy on multi label dataset"""

        n_out = self._config['dataset']['n_out']
        bs = self._config['batch_size']
        dataset = self._get_ds()
        columns = dataset.columns

        self._g_ema.eval()
        self._classifier.eval()

        result = {}

        for cls in range(n_out):
            label = self._sample_label(bs)
            label[:, cls] = 1.
            label = label.to(self._device)

            inputs = []
            outputs = []

            for _ in tqdm(range(200)):
                img = self._g_ema(label)
                h, _ = self._encoder(img)
                logits = self._classifier(h)
                pred = (torch.sigmoid(logits) > 0.2).float()

                inputs.append(label[:, cls])
                outputs.append(pred[:, cls])

            inputs = torch.cat(inputs)
            outputs = torch.cat(outputs)
            result[columns[cls]] = (inputs == outputs).float().mean().item()
        return result

    def _attributes_control_accuracy_one_hot(self):
        """Runs attribute control accuracy on one-hot label dataset"""

        n_out = self._config['dataset']['n_out']
        bs = self._config['batch_size']

        self._g_ema.eval()
        self._encoder.eval()
        self._classifier.eval()

        result = {}

        for cls in range(n_out):
            label = torch.tensor(cls)
            label_one_hot = F.one_hot(label, num_classes=n_out).float()
            label_input = label_one_hot.unsqueeze(0).repeat(bs, 1).to(self._device)

            inputs = []
            outputs = []

            for _ in tqdm(range(200)):
                img = self._g_ema(label_input)
                h, _ = self._encoder(img)
                pred = self._classifier(h)

                inputs.append(torch.argmax(label_input, dim=1))
                outputs.append(torch.argmax(pred, dim=1))

            inputs = torch.cat(inputs)
            outputs = torch.cat(outputs)

            result[cls] = (inputs == outputs).float().mean().item()

        return result

    def _chamfer_distance(self):
        """Runs Chamfer distance to compare real and generated samples"""

        loader = self._get_dl()
        embeddings = []

        # real data embeddings
        for _ in tqdm(range(200)):
            img, _ = next(loader)
            img = img.to(self._device)

            with torch.no_grad():
                h, _ = self._encoder(img)

            embeddings.extend(h.cpu().numpy())

        # generated data embeddings
        for _ in tqdm(range(200)):
            label_oh = self._sample_label()

            with torch.no_grad():
                img = self._g_ema(label_oh)
                h, _ = self._encoder(img)

            embeddings.extend(h.cpu().numpy())

        tsne_emb = TSNE(n_components=3).fit_transform(embeddings)
        n = len(tsne_emb)

        tsne_real = tsne_emb[:n//2, ]
        tsne_fake = tsne_emb[n//2:, ]

        tsne_real = torch.from_numpy(tsne_real).unsqueeze(0)
        tsne_fake = torch.from_numpy(tsne_fake).unsqueeze(0)

        chamfer_dist = ChamferDistance()
        return chamfer_dist(tsne_real, tsne_fake).detach().item()

    def _explore_eps_zs(self):
        """Runs exploration of epsilon and z1, ... zk features

        Epsilon and z1, ..., zk features are fixed, and y are explored
        Images are saved in `explore_eps_zs` """

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
        """Runs exploration of epsilon features

        z1, ... zk features are fixed, epsilon features are randomly sampled
        Images are saved in `explore_eps`"""

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
        """Runs explorations of z1, ... zk features

        Epsilon and y values are set and z_i are changed gradually"""

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
        """Runs exploration of y labels"""

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
        """Displays 2D TSNE of epsilon values of generated and real images"""

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
        """Computes FID score for the dataset

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

    def _compute_inception_score(self) -> float:
        """Computes inception score for the dataset

        Returns:
            float: inception score
        """

        batch_size = self._config['batch_size']

        dataset = GANDataset(self._g_ema, n=100_000)
        score = inception_score(dataset, batch_size=batch_size, resize=True)[0]
        return score

    def _sample_label(self, n: Optional[int] = None) -> torch.Tensor:
        """Samples y label for the dataset

        Args:
            n: number of labels to sample

        Returns:
            torch.Tensor: sampled random label
        """

        n_out = self._config['dataset']['n_out']  # either number of classes, or size of the out vector (celeba)
        y_type = self._config['generator']['y_type']

        if n is None:
            batch_size = self._config['batch_size']
            n = batch_size

        if y_type == 'multi_label':
            label = torch.randint(2, (n, n_out)).float().to(self._device)
        elif y_type == 'one_hot':
            label = torch.randint(n_out, (n,))
            label = F.one_hot(label, num_classes=n_out).float().to(self._device)
        elif y_type == 'mixed':
            k = n_out // 2
            y_one_hot = torch.randint(k, (n,))
            y_one_hot = F.one_hot(y_one_hot, num_classes=k)
            y_mult = torch.randint(2, (n, k))

            label = torch.cat((y_one_hot, y_mult), dim=1).float().to(self._device)
        elif y_type == 'real':

            label = []
            for i, (_, l) in enumerate(self._sample_ds):
                if i >= n:
                    break

                label.append(torch.from_numpy(l))

            label = torch.stack(label).to(self._device)

        return label

    def _get_loss(self):
        """Returns loss functions for GAN based on config

        Returns:
            loss functions
        """

        ds_name = self._config['dataset']['name']

        d_adv_loss, g_adv_loss = get_adversarial_losses(self._config['loss'])
        d_reg_loss = get_regularizer("r1")

        if ds_name in ['celeba']:
            cls_loss = nn.BCEWithLogitsLoss()
        else:
            cls_loss = nn.CrossEntropyLoss()
        return d_adv_loss, g_adv_loss, d_reg_loss, cls_loss

    def _get_ds(self) -> Dataset:

        name = self._config['dataset']['name']
        path = self._config['dataset']['path']
        anno = None if 'anno' not in self._config['dataset'] else self._config['dataset']['anno']
        columns = None if 'columns' not in self._config['dataset'] else self._config['dataset']['columns']

        transform = self._get_data_transform()
        dataset = get_dataset(name, path, anno_file=anno, transform=transform, columns=columns)
        return dataset

    def _get_dl(self) -> DataLoader:
        batch_size = self._config['batch_size']
        n_workers = self._config['n_workers']

        dataset = self._get_ds()
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
