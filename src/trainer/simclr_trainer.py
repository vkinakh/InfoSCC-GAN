from typing import Dict, NoReturn
import os

from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, ConcatDataset

from .base_trainer import BaseTrainer
from src.models import ResNetSimCLR
from src.loss import NTXentLoss
from src.data import get_dataset, CelebADataset
from src.transform import ContrastiveAugmentor, ValidAugmentor
from src.utils import run_tsne, run_tsne_celeba
from src.utils import PathOrStr


torch.backends.cudnn.benchmark = True


class SimCLRTrainer(BaseTrainer):

    """Trainer for SimCLR"""

    def __init__(self,
                 config_path: PathOrStr,
                 config: Dict):

        super().__init__(config_path, config)

        self._nt_xent_criterion = NTXentLoss(
            self._device, config["batch_size"], **config["loss"]
        )

    def _step(self,
              model: nn.Module,
              xis: torch.Tensor,
              xjs: torch.Tensor) -> torch.Tensor:

        xis = xis.to(self._device)
        xjs = xjs.to(self._device)

        # get the representations and the projections
        ris, zis = model(xis)  # [N,C]

        # get the representations and the projections
        rjs, zjs = model(xjs)  # [N,C]

        # normalize projection feature vectors
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)

        loss = self._nt_xent_criterion(zis, zjs)

        return loss

    def train(self) -> NoReturn:

        epochs = self._config['epochs']
        log_every = self._config['log_every']
        val_every = self._config['val_every']
        eval_every = self._config['eval_every']
        wd = eval(self._config['wd'])
        lr = eval(self._config['lr'])

        ds_name = self._config['dataset']['name']
        train_dl, val_dl, test_dl = self._get_dataloaders()

        model = self._load_model()
        optimizer = optim.Adam(model.parameters(), lr, weight_decay=wd)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_dl), eta_min=0, last_epoch=-1)

        global_step = 0
        best_valid_loss = 0
        for epoch in range(1, epochs + 1):

            for (xis, xjs), c in tqdm(train_dl, desc=f'Epoch {epoch}/{epochs}'):
                optimizer.zero_grad()
                loss = self._step(model, xis, xjs)

                if global_step % log_every == 0:
                    self._writer.add_scalar('train/loss', loss, global_step)
                loss.backward()
                optimizer.step()

                global_step += 1

            if epoch == 1 or epoch % val_every == 0:
                valid_loss = self._validate(model, val_dl)
                self._writer.add_scalar('val/loss', valid_loss, epoch)

                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    self._save_model(model, f'{epoch:04}_best')

            if epoch == 1 or epoch % eval_every == 0:

                if ds_name == 'celeba':
                    outputs = run_tsne_celeba(model, test_dl, self._device)

                    for output in outputs:
                        col = output['col']
                        img_h = output['h']
                        img_z = output['z']

                        self._writer.add_image(f'{col} TSNE-h', img_h, epoch)
                        self._writer.add_image(f'{col} TSNE-z', img_z, epoch)
                else:
                    img_tsne_h, img_tnse_z = run_tsne(model, test_dl, self._device)

                    self._writer.add_image('TSNE-h', img_tsne_h, epoch)
                    self._writer.add_image('TSNE-z', img_tnse_z, epoch)

            self._save_model(model, f'{epoch:04}')

            if epoch >= 10:
                scheduler.step()

            self._writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)

    def _get_dataloaders(self):

        batch_size = self._config['batch_size']
        input_shape = eval(self._config['input_size'])
        dataset_name = self._config['dataset']['name']

        cpu_count = os.cpu_count()
        contrastive_trans = ContrastiveAugmentor(input_shape)

        if dataset_name == 'celeba':
            ds_path = self._config['dataset']['path']
            anno_file = self._config['dataset']['anno_path']
            ds = CelebADataset(ds_path, anno_file, transform=contrastive_trans)
        elif dataset_name == 'afhq':
            train_ds_path = self._config['dataset']['train_path']
            valid_ds_path = self._config['dataset']['valid_path']

            train_ds = get_dataset('afhq', train_ds_path, transform=contrastive_trans)
            val_ds = get_dataset('afhq', valid_ds_path, transform=contrastive_trans)
            ds = ConcatDataset([train_ds, val_ds])

        n = len(ds)
        n_train = int(0.8 * n)

        train_ds = Subset(ds, indices=range(0, n_train))
        train_dl = DataLoader(train_ds, batch_size, num_workers=cpu_count,
                              shuffle=True, drop_last=True, pin_memory=True)

        val_ds = Subset(ds, indices=range(n_train, n))
        val_dl = DataLoader(val_ds, batch_size, num_workers=cpu_count,
                            shuffle=True, drop_last=True, pin_memory=True)

        val_trans = ValidAugmentor(input_shape)

        if dataset_name == 'celeba':
            anno_file = self._config['dataset']['anno_path']
            ds_path = self._config['dataset']['path']
            test_ds = CelebADataset(ds_path, anno_file, return_anno=True, transform=val_trans)
        elif dataset_name == 'afhq':
            ds_path = self._config['dataset']['valid_path']
            test_ds = get_dataset('afhq', ds_path, transform=val_trans)

        n_test = int(0.2 * len(test_ds))
        test_ds = Subset(test_ds, np.random.randint(0, len(test_ds), n_test))
        test_dl = DataLoader(test_ds, batch_size, num_workers=cpu_count,
                             shuffle=True, drop_last=False, pin_memory=True)
        return train_dl, val_dl, test_dl

    def _load_model(self) -> nn.Module:
        model_path = self._config['fine_tune_from']
        base_model = self._config['model']['base_model']
        n_channels = self._config['model']['n_channels']
        out_dim = self._config['model']['out_dim']

        model = ResNetSimCLR(base_model, n_channels, out_dim)

        try:
            if model_path is not None:
                state_dict = torch.load(model_path)
                model.load_state_dict(state_dict)
                print(f'Loaded: {model_path}')
            else:
                print('Training model from scratch')
        except:
            print('Training model from scratch')

        if torch.cuda.device_count() > 1:
            print(f'Use {torch.cuda.device_count()} GPUs')
            model = nn.DataParallel(model)

        model.to(self._device)
        return model

    def _save_model(self, model: nn.Module, suffix: str) -> NoReturn:
        checkpoint_folder = self._writer.checkpoint_folder
        save_file = checkpoint_folder / f'model_{suffix}.pth'

        if isinstance(model, nn.DataParallel):
            torch.save(model.module.state_dict(), save_file)
        else:
            torch.save(model.state_dict(), save_file)

    def _validate(self, model: nn.Module, dl: DataLoader) -> float:
        # validation steps
        with torch.no_grad():
            model.eval()

            valid_loss = 0.
            counter = 0
            for (xis, xjs), c in dl:
                loss = self._step(model, xis, xjs)
                valid_loss += loss.item()
                counter += 1
            valid_loss /= counter
        model.train()
        return valid_loss
