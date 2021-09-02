from typing import List, Optional, NoReturn
import shutil
from distutils.dir_util import copy_tree
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


class SummaryWriterWithSources:

    """Summary writer, that saves source files and dirs to config files"""

    def __init__(self, files_to_copy: Optional[List[str]] = None,
                 dirs_to_copy: Optional[List[str]] = None,
                 experiment_name: str = ''):
        """
        Args:
            files_to_copy: list of files to copy into config folder. Only names

            dirs_to_copy: list of files to copy into config folder. Only names

            experiment_name: experiment name to add to config dir name
        """

        self._files_to_copy = files_to_copy if files_to_copy is not None else []
        self._dirs_to_copy = dirs_to_copy if dirs_to_copy is not None else []
        self._experiment_name = experiment_name
        self._writer = SummaryWriter(comment='_' + experiment_name)

        root_dir = Path(__file__).parent.parent.parent
        log_dir = Path(self._writer.log_dir)

        # copy files
        for file in self._files_to_copy:
            file_path = root_dir / file

            shutil.copy(str(file_path), str(log_dir / file_path.name))

        # copy folders
        for d in self._dirs_to_copy:
            copy_tree(str(root_dir / d), str(log_dir / d))

        # create checkpoint folder
        self._checkpoint_folder = log_dir / 'checkpoint'
        self._checkpoint_folder.mkdir(parents=True, exist_ok=True)

    def add_scalar(self,
                   tag: str,
                   scalar_value,
                   global_step: Optional[int] = None,
                   walltime: Optional = None) -> NoReturn:

        """Adds scalar value to Tensorboard

        Args:
            tag: scalar tag to use
            scalar_value: scalar value to save
            global_step: step
            walltime:
        """

        self._writer.add_scalar(tag, scalar_value, global_step, walltime)

    def add_image(self,
                  tag: str,
                  img_tensor,
                  global_step: Optional[int] = None,
                  walltime: Optional = None) -> NoReturn:

        self._writer.add_image(tag, img_tensor, global_step, walltime)

    @property
    def writer(self) -> SummaryWriter:
        """
        Returns:
            SummaryWriter: summary writer object
        """

        return self._writer

    @property
    def checkpoint_folder(self) -> Path:
        """
        Returns:
            Path: path to checkpoint folder, where to save models
        """

        return self._checkpoint_folder

    @property
    def experiment_name(self) -> str:
        """
        Returns:
            str: experiment name
        """

        return self._experiment_name

    @property
    def files_to_copy(self) -> List[str]:
        """
        Returns:
            List[str]: list of files to save to config folder
        """

        return self._files_to_copy

    @property
    def dirs_to_copy(self) -> List[str]:
        """
        Returns:
            List[str]: list of dirs to save to config folder
        """

        return self._dirs_to_copy
