from typing import Optional, Callable, List
from pathlib import Path

import numpy as np
import pandas as pd

from torch.utils.data import Dataset

from src.utils import PathOrStr
from src.utils import image_loader


class FFHQDataset(Dataset):

    """Dataset that loads FFHQ images and attributes"""

    def __init__(self,
                 root: PathOrStr,
                 anno_file: PathOrStr,
                 transform: Optional[Callable] = None):

        """
        Args:
            root: path to folder with images

            anno_file: path to file with attribute annotations

            transform: transform to apply to images
        """

        root = Path(root)
        self._image_paths = [x for x in root.glob('*') if x.is_file()]
        self._transform = transform
        self._annotations = self._get_annotations(anno_file)
        self._columns = list(self._annotations.columns)[1:]

    def _get_annotations(self, anno_file: PathOrStr) -> pd.DataFrame:
        df = pd.read_csv(anno_file)
        columns = ['image_number', 'gender', 'age_group']
        df = df[columns]

        df['image_number'] = df['image_number'].apply(lambda x: f'{x:05}')

        for col in ['male', 'female']:
            df[col] = (df['gender'] == col).astype(np.uint8)

        for col in ['0-2', '30-39', '3-6', '20-29', '40-49', '50-69', '10-14', '15-19', '7-9', '70-120']:
            df[col] = (df['age_group'] == col).astype(np.uint8)

        df = df.drop(['gender', 'age_group'], axis=1)
        return df

    def __len__(self) -> int:
        return len(self._annotations)

    def __getitem__(self, i: int):
        image_path = self._image_paths[i]
        x = image_loader(str(image_path))

        image_id = image_path.stem
        annotation = self._annotations[self._annotations['image_number'] == image_id]
        annotation = annotation.drop(['image_number'], axis=1).values[0]
        annotation = annotation.astype(np.float32)

        if self._transform is not None:
            x = self._transform(x)
        return x, annotation

    @property
    def columns(self) -> List[str]:
        return self._columns


if __name__ == '__main__':

    p = '/home/kinakh/Datasets/FFHQ/imgs'
    a = '/home/kinakh/Datasets/FFHQ/ffhq_aging_labels.csv'

    d = FFHQDataset(p, a)

    print(d.columns, len(d.columns))
