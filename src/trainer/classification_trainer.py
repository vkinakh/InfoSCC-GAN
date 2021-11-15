from typing import Dict, NoReturn, Tuple
from collections import defaultdict

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import transforms

from .base_trainer import BaseTrainer
from src.models import LinearClassifier
from src.models import ResNetSimCLR
from src.data import get_dataset


class ClassificationTrainer(BaseTrainer):

    def __init__(self,
                 config_path: str,
                 config: Dict):

        super().__init__(config_path, config)

        self._model, self._optimizer = self._load_model()

    def train(self) -> NoReturn:

        epochs = self._config['epochs']
        ds_name = self._config['dataset']['name']

        if ds_name in ['celeba']:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()

        train_dl, test_dl = self._get_dls()
        for epoch in range(1, epochs + 1):
            total_loss = 0

            for img, label in train_dl:
                img, label = img.to(self._device), label.to(self._device)

                self._optimizer.zero_grad()
                logits = self._model(img)
                loss = criterion(logits, label)

                loss.backward()
                self._optimizer.step()
                total_loss += loss.item()

            self._writer.add_scalar('train/loss', total_loss, epoch)
        self._eval(test_dl)
        self._save_model('final')

    def evaluate(self):
        _, test_dl = self._get_dls()
        return self._eval(test_dl)

    def _eval(self, loader: DataLoader):
        ds_name = self._config['dataset']['name']

        if ds_name in ['celeba']:
            return self._eval_multi_label(loader)

        return self._eval_classification(loader)

    def _eval_classification(self, loader: DataLoader):
        self._model.eval()

        correct = 0
        total = 0

        for (img, label) in loader:
            img, label = img.to(self._device), label.to(self._device)

            with torch.no_grad():
                logits = self._model(img)

            predicted = torch.argmax(logits, dim=1)
            total += img.size(0)
            correct += (predicted == label).sum().item()

        self._model.train()
        final_acc = correct / total
        self._writer.add_scalar('eval/accuracy', final_acc, 1)
        return final_acc

    def _eval_multi_label(self, loader: DataLoader):

        ds_name = self._config['dataset']['name']
        data_path = self._config['dataset']['path']
        anno_path = self._config['dataset']['anno']
        columns = None if 'columns' not in self._config['dataset'] else self._config['dataset']['columns']
        dataset = get_dataset(ds_name, data_path, anno_file=anno_path, columns=columns)
        columns = dataset.columns

        if ds_name == 'celeba':
            columns = columns[:-1]

        self._model.eval()
        correct = defaultdict(int)
        total = 0

        for (img, label) in loader:
            img, label = img.to(self._device), label.to(self._device)
            with torch.no_grad():
                logits = self._model(img)
            predicted = (torch.sigmoid(logits) > 0.5).float()

            for i in range(len(columns)):
                c = label[:, i] == predicted[:, i]
                correct[i] += c.sum().item()

            total += img.size(0)

        class_acc = correct
        for i in range(len(columns)):
            class_acc[i] = class_acc[i] / total

        self._model.train()
        global_acc = np.mean(list(class_acc.values()))
        self._writer.add_scalar('eval/accuracy', global_acc, 0)

        for (col, acc) in zip(columns, class_acc.values()):
            self._writer.add_scalar(f'eval/{col} accuracy', acc, 0)

            print(col, acc)

        return global_acc, list(zip(columns, class_acc.values()))

    def _get_dls(self) -> Tuple[DataLoader, DataLoader]:

        name = self._config['dataset']['name']
        size = self._config['dataset']['size']
        batch_size = self._config['batch_size']

        # get transform
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

        # load dataset
        if name in ['mnist', 'fashionmnist']:
            train_ds = get_dataset(name, './data', True, transform=transform)
            test_ds = get_dataset(name, './data', False, transform=transform)
        elif name == 'afhq':
            train_path = self._config['dataset']['train_path']
            train_anno = None if 'train_anno' not in self._config['dataset'] else self._config['dataset']['train_anno']
            train_ds = get_dataset(name, train_path, anno_file=train_anno, transform=transform)

            test_path = self._config['dataset']['test_path']
            test_anno = None if 'test_anno' not in self._config['dataset'] else self._config['dataset']['test_anno']
            test_ds = get_dataset(name, test_path, anno_file=test_anno, transform=transform)
        elif name in ['celeba']:
            data_path = self._config['dataset']['path']
            anno_path = self._config['dataset']['anno']
            columns = None if 'columns' not in self._config['dataset'] else self._config['dataset']['columns']
            dataset = get_dataset(name, data_path, anno_file=anno_path, transform=transform, columns=columns)

            n = len(dataset)
            test_ratio = 0.2
            n_train = int(n * (1 - test_ratio))
            train_idx = range(0, n_train)
            test_idx = range(n_train, n)

            train_ds = Subset(dataset, train_idx)
            test_ds = Subset(dataset, test_idx)

        train_emb_dl = DataLoader(train_ds, batch_size=batch_size, drop_last=True, num_workers=8)
        test_emb_dl = DataLoader(test_ds, batch_size=batch_size, drop_last=True, num_workers=8)

        # load encoder
        encoder_path = self._config['encoder']['path']
        base_model = self._config['encoder']['base_model']
        n_channels = self._config['encoder']['n_channels']
        out_dim = self._config['encoder']['out_dim']
        encoder = ResNetSimCLR(base_model, n_channels, out_dim).to(self._device)
        ckpt = torch.load(encoder_path)
        encoder.load_state_dict(ckpt)
        encoder.eval()

        # compute embeddings
        train_emb = []
        train_labels = []
        test_emb = []
        test_labels = []

        # compute train ds
        for (img, label) in tqdm(train_emb_dl):
            img = img.to(self._device)

            with torch.no_grad():
                h, _ = encoder(img)

            embedding = h.detach().cpu().numpy()
            label = label.numpy()

            train_emb.extend(embedding)
            train_labels.extend(label)

        # compute test ds
        for (img, label) in tqdm(test_emb_dl):
            img = img.to(self._device)

            with torch.no_grad():
                h, _ = encoder(img)

            embedding = h.detach().cpu().numpy()
            label = label.numpy()

            test_emb.extend(embedding)
            test_labels.extend(label)

        train_emb = np.array(train_emb, dtype=np.float32)
        train_labels = np.array(train_labels, dtype=np.float32)

        test_emb = np.array(test_emb, dtype=np.float32)
        test_labels = np.array(test_labels, dtype=np.float32)

        train_emb_ds = TensorDataset(torch.from_numpy(train_emb), torch.from_numpy(train_labels))
        train_emb_dl = DataLoader(train_emb_ds, batch_size=batch_size, num_workers=8)

        test_emb_ds = TensorDataset(torch.from_numpy(test_emb), torch.from_numpy(test_labels))
        test_emb_dl = DataLoader(test_emb_ds, batch_size=batch_size, num_workers=8)

        return train_emb_dl, test_emb_dl

    def _load_model(self):

        n_features = self._config['model']['n_features']
        n_out = self._config['model']['n_out']
        model_path = self._config['fine_tune_from']
        lr = eval(self._config['lr'])
        wd = eval(self._config['wd'])

        model = LinearClassifier(n_features, n_out).to(self._device)
        if model_path is not None:
            state_dict = torch.load(model_path)
            model.load_state_dict(state_dict)
            print(f'Loaded model from: {model_path}')

        optimizer = optim.Adam(model.parameters(), lr, weight_decay=wd)
        return model, optimizer

    def _save_model(self, suffix: str) -> NoReturn:
        checkpoint_folder = self._writer.checkpoint_folder
        save_file = checkpoint_folder / f'model_{suffix}.pth'
        torch.save(self._model.state_dict(), save_file)
