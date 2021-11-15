from typing import Tuple

import numpy as np
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .plot import tsne_display_tensorboard


def run_tsne(model: nn.Module,
             loader: DataLoader,
             device: str) -> Tuple[np.ndarray, np.ndarray]:
    """Runs 2D TSNE from the features of model, creates image compatible with Tensorboard

    Args:
        model: model to compute features
        loader:
        device:

    Returns:
        Tuple[np.ndarray, np.ndarray]: 2D TSNE images for Tensorboard
    """

    model.eval()
    h_vector = []
    z_vector = []
    c_vector = []

    for img, c in loader:
        img = img.to(device)

        with torch.no_grad():
            h, z = model(img)
            h = F.normalize(h, dim=1)
            z = F.normalize(z, dim=1)

        h_vector.extend(h.cpu().detach().numpy())
        z_vector.extend(z.cpu().detach().numpy())
        c_vector.extend(c.cpu().detach().numpy())

    model.train()

    h_vector = np.array(h_vector)
    z_vector = np.array(z_vector)
    c_vector = np.array(c_vector)
    embeddings_h = TSNE(n_components=2).fit_transform(h_vector)
    embeddings_z = TSNE(n_components=2).fit_transform(z_vector)

    return tsne_display_tensorboard(embeddings_h, c_vector), \
        tsne_display_tensorboard(embeddings_z, c_vector)


def run_tsne_celeba(model: nn.Module,
                    loader: DataLoader,
                    device: str):
    """Runs 2D TSNE from the features of model, creates image compatible with Tensorboard
    for each of the CelebA labels

    Args:
        model: model to compute features
        loader:
        device:

    Returns:
        2D TSNE images for each labels
    """

    model.eval()
    columns = [
        '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose',
        'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses',
        'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache',
        'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
        'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
        'Wearing_Necklace', 'Wearing_Necktie', 'Young']

    h_vector = []
    z_vector = []
    c_vector = []

    for img, c in loader:
        img = img.to(device)

        with torch.no_grad():
            h, z = model(img)
            h = F.normalize(h, dim=1)
            z = F.normalize(z, dim=1)

        h_vector.extend(h.cpu().detach().numpy())
        z_vector.extend(z.cpu().detach().numpy())
        c_vector.extend(c.cpu().detach().numpy())

    model.train()

    h_vector = np.array(h_vector)
    z_vector = np.array(z_vector)

    c_vector = np.array(c_vector)
    c_vector = (c_vector + 1) / 2

    embeddings_h = TSNE(n_components=2).fit_transform(h_vector)
    embeddings_z = TSNE(n_components=2).fit_transform(z_vector)

    outputs = []

    for i in range(len(columns)):
        c = c_vector[:, i]
        img_h = tsne_display_tensorboard(embeddings_h, c)
        img_z = tsne_display_tensorboard(embeddings_z, c)
        outputs.append({
            'h': img_h,
            'z': img_z,
            'col': columns[i]
        })
    return outputs
