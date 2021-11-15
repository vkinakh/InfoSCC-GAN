from typing import Optional, Callable, List
from pathlib import Path

import numpy as np
import pandas as pd

from torch.utils.data import Dataset

from src.utils import PathOrStr
from src.utils import image_loader


def get_annotation(fnmtxt: PathOrStr, verbose: bool = True) -> pd.DataFrame:
    """Opens CelebA dataset annotations and converts them into pandas data frame

    Args:
        fnmtxt: path to annotation file
        verbose: if True, prints progress

    Returns:
        pd.DataFrame: data frame with annotations
    """

    if verbose:
        print("_" * 70)
        print(fnmtxt)

    rfile = open(fnmtxt, 'r')
    texts = rfile.read().split("\n")
    rfile.close()

    columns = np.array(texts[1].split(" "))
    columns = columns[columns != ""]
    df = []
    for txt in texts[2:]:
        txt = np.array(txt.split(" "))
        txt = txt[txt != ""]

        df.append(txt)

    df = pd.DataFrame(df)

    if df.shape[1] == len(columns) + 1:
        columns = ["image_id"] + list(columns)
    df.columns = columns
    df = df.dropna()
    if verbose:
        print(" Total number of annotations {}\n".format(df.shape))
    # cast to integer
    for nm in df.columns:
        if nm != "image_id":
            df[nm] = pd.to_numeric(df[nm], downcast="integer")
    return df


class CelebADataset(Dataset):

    """Dataset that loads CelebA images and attributes"""

    def __init__(self,
                 root: PathOrStr,
                 anno_file: PathOrStr,
                 return_anno: bool = False,
                 columns: Optional[List[str]] = None,
                 transform: Optional[Callable] = None):
        """
        Args:
            root: path to folder with images

            anno_file: path to file with attribute annotations

            return_anno: if True, will return 40 attributes, else will return mock value

            columns: columns to select

            transform: transform to apply to images
        """

        root = Path(root)
        self._transform = transform
        self._image_paths = [x for x in root.glob('*') if x.is_file()]
        self._annotations = get_annotation(anno_file)

        if columns is not None:

            if 'image_id' not in columns:
                columns.append('image_id')
            self._annotations = self._annotations[columns]

        self._return_anno = return_anno
        self._columns = list(self._annotations.columns)[:-1]

    def __len__(self) -> int:
        return len(self._image_paths)

    def __getitem__(self, i: int):
        image_path = self._image_paths[i]
        x = image_loader(str(image_path))

        if self._return_anno:
            image_id = image_path.name
            annotation = self._annotations[self._annotations['image_id'] == image_id]
            annotation = annotation.drop(['image_id'], axis=1)
            annotation = (annotation.values[0] + 1) / 2
            annotation = annotation.astype(np.float32)
        else:
            annotation = 0

        if self._transform is not None:
            x = self._transform(x)
        return x, annotation

    @property
    def columns(self):
        return self._columns
