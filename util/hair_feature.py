import cv2
import numpy as np
from util.inpaint_util import removeHair

def detect_hair(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, thresh, _ = removeHair(img_rgb, gray)
    hair_pixels = np.sum(thresh == 255)
    total_pixels = thresh.shape[0] * thresh.shape[1]
    return hair_pixels / total_pixels