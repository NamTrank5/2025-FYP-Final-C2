import numpy as np
from sklearn.cluster import KMeans
from skimage.transform import resize

def get_multicolor_rate(image, mask, n=5):
    """
    Estimates color variation by clustering lesion pixels and computing
    the max distance between dominant cluster centers (in RGB).
    """
    # Downscale for speed
    image_small = resize(image, (image.shape[0] // 4, image.shape[1] // 4), anti_aliasing=True)
    mask_small = resize(mask, (mask.shape[0] // 4, mask.shape[1] // 4), anti_aliasing=True)

    # Apply mask
    lesion_pixels = image_small[mask_small > 0]

    if len(lesion_pixels) < n:
        return 0  # not enough pixels to cluster

    # Cluster colors in lesion area
    kmeans = KMeans(n_clusters=n, n_init=10)
    kmeans.fit(lesion_pixels)

    # Get cluster centers
    centers = kmeans.cluster_centers_

    # Compute max pairwise distance
    max_dist = 0
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            max_dist = max(max_dist, dist)

    return round(max_dist, 2)