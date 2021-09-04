from typing import Optional, Callable, List

from torchvision import datasets
from torch.utils.data import Dataset

from src.data import AFHQDataset
from src.data import CelebADataset
from src.utils import PathOrStr
from src.utils import image_loader


AVAILABLE_DATASETS = ['mnist', 'fashionmnist', 'afhq', 'celeba']


def get_dataset(name: str,
                data_path: PathOrStr,
                train: bool = True,
                anno_file: Optional[PathOrStr] = None,
                transform: Optional[Callable] = None,
                columns: Optional[List[str]] = None) -> Dataset:

    """Returns dataset based on conditions

    Args:
        name: dataset name

        data_path: path to dataset

        train: if True, them loads train split, False - test split (only for MNIST, FashionMNIST)

        anno_file: path to file for annotation (only for AFHQ, CelebA)

        transform: transform to apply to the data

        columns: list of columns to select (only for CelebA)

    Returns:
        Dataset: loaded dataset
    """

    if name not in AVAILABLE_DATASETS:
        raise ValueError('Unsupported dataset')

    if name == 'mnist':
        dataset = datasets.MNIST(data_path, train=train, transform=transform, download=True)
    elif name == 'fashionmnist':
        dataset = datasets.FashionMNIST(data_path, train=train, transform=transform, download=True)
    elif name == 'afhq':

        if anno_file is not None:
            dataset = AFHQDataset(data_path, anno_file, transform)
        else:
            dataset = datasets.ImageFolder(data_path, transform, loader=lambda x: image_loader(str(x)))
    elif name == 'celeba':
        dataset = CelebADataset(data_path, anno_file, True, columns, transform)
    else:
        raise ValueError('Unsupported dataset')

    return dataset


def infinite_loader(data_loader):
    while True:
        for batch in data_loader:
            yield batch
