from typing import Tuple

import torch
from torchvision import transforms


class SimCLRDataTransform:
    """Applies augmentations to sample two times, as described in SimCLR paper"""

    def __init__(self, transform: transforms.Compose):
        self.transform = transform

    def __call__(self, sample: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xi = self.transform(sample)
        xj = self.transform(sample)
        return xi, xj


class ContrastiveAugmentor:

    """Applies augmentation for contrastive learning, as in SimCLR paper"""

    def __init__(self, input_size: Tuple[int, int, int]):
        """
        Args:
            input_size: input image size

        Raises:
            ValueError: if specified dataset is unsupported
        """

        h, w = input_size[:2]
        size = (h, w)
        blur_kernel_size = 2 * int(.05 * h) + 1
        gaussian = transforms.GaussianBlur(kernel_size=blur_kernel_size)

        color = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)

        augmentations = [
            transforms.RandomResizedCrop(size=size),
            transforms.ConvertImageDtype(torch.float),
            transforms.RandomApply([color], p=0.8),
            transforms.RandomGrayscale(p=0.2),

            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([gaussian]),

            transforms.Normalize(0.5, 0.5)
        ]

        self._augmentations = SimCLRDataTransform(transforms.Compose(augmentations))

    def __call__(self, sample: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._augmentations(sample)


class ValidAugmentor:

    """Applies augmentation for validation and testing"""

    def __init__(self, input_size: Tuple[int, int, int]):
        """
        Args:
            input_size: input image size

        Raises:
            ValueError: if specified dataset is unsupported
        """

        h, w = input_size[:2]
        size = (h, w)
        augmentations = [
            transforms.Resize(size=size),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(0.5, 0.5),
        ]

        self._augmentations = transforms.Compose(augmentations)

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        return self._augmentations(sample)
