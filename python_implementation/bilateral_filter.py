import cv2
import numpy as np


def bilateral_filter(image, guidance, window_size, sigma_space, sigma_range):
    """
    Apply bilateral filtering to an image.

    Args:
        image: Input image
        guidance: Guidance image
        window_size: Window radius
        sigma_space: Spatial standard deviation
        sigma_range: Range standard deviation

    Returns:
        Filtered image
    """
    # Convert to float32 for better precision
    image_ = cv2.copyMakeBorder(image, window_size, window_size, window_size, window_size, cv2.BORDER_REFLECT)
    guidance_ = cv2.copyMakeBorder(guidance, window_size, window_size, window_size, window_size, cv2.BORDER_REFLECT)

    gaussian_weight = np.zeros((2 * window_size + 1, 2 * window_size + 1))
    for w in range(-window_size, window_size + 1):
        for h in range(-window_size, window_size + 1):
            gaussian_weight[w + window_size, h + window_size] = np.exp(-(w**2 + h**2) / (2 * sigma_space**2))

    # Apply joint bilateral filter using OpenCV
    width = image_.shape[0]
    height = image_.shape[1]
    channels = image.shape[2] if image.ndim == 3 else 1
    filtered = np.zeros_like(image)
    for w in range(window_size, width - window_size):
        for h in range(window_size, height - window_size):
            image_patch = image_[w - window_size : w + window_size + 1, h - window_size : h + window_size + 1]
            guidance_patch = guidance_[w - window_size : w + window_size + 1, h - window_size : h + window_size + 1]

            guidance_center = guidance_[w, h]
            diff_range = guidance_patch - guidance_center
            if channels == 1:
                range_weight = np.exp(-(diff_range**2) / (2 * sigma_range**2))
                weight = range_weight * gaussian_weight
                filtered[w - window_size, h - window_size] = np.sum(image_patch * weight) / np.sum(weight)
            if channels == 3:
                range_weight = np.exp(-(diff_range**2).sum(axis=2) / (2 * sigma_range**2))
                weight = (range_weight * gaussian_weight)[:, :, np.newaxis]
                filtered[w - window_size, h - window_size, :] = np.sum(image_patch * weight, axis=(0, 1)) / np.sum(
                    weight
                )

    filtered = np.clip(filtered, 0, 1)
    return filtered
