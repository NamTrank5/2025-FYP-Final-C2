
import numpy as np
from skimage import morphology
from scipy.spatial import ConvexHull
from math import pi

def compactness_score(mask):
    """
    Compactness: measures how close the shape is to a circle.
    Lower values → more irregular border.
    """
    area = np.sum(mask)

    struct_el = morphology.disk(2)
    eroded = morphology.binary_erosion(mask, struct_el)
    perimeter = np.sum(mask) - np.sum(eroded)

    compactness = (4 * pi * area) / (perimeter ** 2 + 1e-6)  # +1e-6 to avoid division by zero
    return round(1 - compactness, 3)  # higher = less compact
