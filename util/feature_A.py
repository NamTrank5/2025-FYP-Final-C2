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
    scores = []

    for _ in range(6):
        # Rotate the mask and binarize it again
        rotated = rotate(mask, 30, preserve_range=True)
        rotated = (rotated > 0.5).astype(np.uint8)

        segment = crop(rotated)
        segment_sum = np.sum(segment)

        if segment_sum == 0:
            scores.append(0)
        else:
            # Flip horizontally for asymmetry comparison
            flipped = np.fliplr(segment)
            diff = np.logical_xor(segment, flipped)
            score = np.sum(diff) / segment_sum
            scores.append(score)

        # Prepare for next rotation
        mask = rotated

    raw_score = np.mean(scores)
    return np.clip(raw_score, 0, 1)
