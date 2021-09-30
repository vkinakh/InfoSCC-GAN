from torch.utils.data import Dataset

from src.models import ConditionalGenerator


class GANDataset(Dataset):

    def __init__(self, model: ConditionalGenerator,
                 n: int):
        self._model = model
        self._n = n

    def __len__(self):
        return self._n

    def __getitem__(self, i: int):
        return self._model.sample(1).squeeze()
