from abc import ABC, abstractmethod
from typing import Dict

from src.utils import PathOrStr
from src.utils import get_device
from src.utils import SummaryWriterWithSources


class BaseTrainer(ABC):

    def __init__(self,
                 config_path: PathOrStr,
                 config: Dict):

        self._config = config
        self._device = get_device()
        self._writer = SummaryWriterWithSources(
            files_to_copy=[config_path],
            experiment_name=config['comment']
        )

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def _load_model(self):
        pass

    @abstractmethod
    def _save_model(self, *args, **kwargs):
        pass
