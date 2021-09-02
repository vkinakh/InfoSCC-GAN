import torch.nn as nn
import torch.nn.functional as F


class EpsilonDiscriminator(nn.Module):
    def __init__(self, N: int, z_dim: int):
        super(EpsilonDiscriminator, self).__init__()
        self.lin1 = nn.Linear(z_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3 = nn.Linear(N, 1)

    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.2, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=0.2, training=self.training)
        x = F.relu(x)
        return self.lin3(x)
