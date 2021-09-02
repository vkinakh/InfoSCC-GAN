import torch
import torch.nn as nn


class LinearClassifier(nn.Module):

    """One layer linear classifier"""

    def __init__(self, n_features: int, n_classes: int):
        """
        Args:
            n_features: number of input features
            n_classes: number of output classes
        """

        super(LinearClassifier, self).__init__()
        self.model = nn.Linear(n_features, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
