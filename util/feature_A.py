import cv2
import numpy as np
from skimage.transform import rotate


def find_midpoint_v4(mask):
    summed = np.sum(mask, axis=0)
    half_sum = np.sum(summed) / 2
    for i, n in enumerate(np.add.accumulate(summed)):
        if n > half_sum:
            return i
        
def crop(mask):
    mid = find_midpoint_v4(mask)
    y_nonzero, x_nonzero = np.nonzero(mask)
    y_lims = [np.min(y_nonzero), np.max(y_nonzero)]
    x_lims = np.array([np.min(x_nonzero), np.max(x_nonzero)])
    x_dist = max(np.abs(x_lims - mid))
    x_lims = [mid - x_dist, mid + x_dist]
    return mask[y_lims[0]:y_lims[1], x_lims[0]:x_lims[1]]

def get_asymmetry(mask):
    # mask = color.rgb2gray(mask)
    scores = []
    for _ in range(6):
        segment = crop(mask)
        (np.sum(segment))
        scores.append(np.sum(np.logical_xor(segment, np.flip(segment))) / (np.sum(segment)))
        mask = rotate(mask, 30)
    return sum(scores) / len(scores)

