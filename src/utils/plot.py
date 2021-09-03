from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import cv2


def tsne_display_tensorboard(embeddings, c_vector:Optional = None, title: Optional[str] = None) -> np.ndarray:
    fig = plt.figure()

    if title is not None:
        plt.title(title)

    if c_vector is not None:
        plt.scatter(embeddings[:, 0], embeddings[:, 1], c=c_vector)
    else:
        plt.scatter(embeddings[:, 0], embeddings[:, 1])
    plt.colorbar()
    fig.canvas.draw()

    img_plot = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    img_plot = img_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,)) / 255
    img_plot = cv2.flip(img_plot, 0)
    img_plot = cv2.rotate(img_plot, cv2.ROTATE_90_CLOCKWISE)
    img_plot = np.swapaxes(img_plot, 0, 2)

    return img_plot
