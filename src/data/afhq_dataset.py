from typing import Optional, Callable
from pathlib import Path

import pandas as pd

from torch.utils.data import Dataset

from src.utils import PathOrStr
from src.utils import image_loader


class AFHQDataset(Dataset):

    """Animal Faces High-Quality dataset with custom annotations"""

    def __init__(self,
                 data_path: PathOrStr,
                 anno_path: PathOrStr,
                 transform: Optional[Callable] = None):
        """
        Args:
            data_path: path to folder with images

            anno_path: path to file with annotations

            transform: transform to apply to images
        """

        self._root = Path(data_path)
        self._annotations = pd.read_csv(anno_path)
        self._transform = transform

    def __len__(self) -> int:
        return len(self._annotations)

    def __getitem__(self, i: int):
        row = self._annotations.iloc[i]
        img_path = self._root / row['img']
        cluster = int(row['cluster'])

        img = image_loader(str(img_path))
        if self._transform is not None:
            img = self._transform(img)

        return img, cluster
