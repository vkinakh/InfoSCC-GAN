from typing import Optional, Callable

from torch.utils.data import Dataset

from src.data import get_dataset
from src.utils import PathOrStr


class GenDataset(Dataset):

    """Dataset that returns only images, without labels"""

    def __init__(self, name: str,
                 data_path: PathOrStr,
                 train: bool = True,
                 anno_file: Optional[PathOrStr] = None,
                 transform: Optional[Callable] = None):

        self._dataset = get_dataset(name, data_path, train, anno_file, transform)

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, i: int):
        return self._dataset[i][0]
