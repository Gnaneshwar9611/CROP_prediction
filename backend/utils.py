"""
Image preprocessing utilities for crop disease prediction.
"""

from PIL import Image
import numpy as np


def preprocess_image(image: Image.Image, target_size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Resize an RGB PIL image to `target_size`, normalise pixel values
    to [0, 1], and add a batch dimension.

    Parameters
    ----------
    image : PIL.Image.Image
        Input image (must already be RGB).
    target_size : tuple[int, int]
        (width, height) the model expects.

    Returns
    -------
    np.ndarray
        Array of shape (1, height, width, 3) with float values in [0, 1].
    """
    image = image.resize(target_size)
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array