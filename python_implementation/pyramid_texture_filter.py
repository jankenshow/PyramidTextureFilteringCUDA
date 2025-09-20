import cv2
import numpy as np
from bilateral_filter import bilateral_filter
from pyramid import pyramid


def pyramid_texture_filter(image, sigma_s=5.0, sigma_r=0.05, nlev=11, scale=0.8):
    """
    Apply pyramid texture filtering to an image.

    Args:
        image: Input image
        sigma_s: Spatial standard deviation
        sigma_r: Range standard deviation
        nlev: Number of pyramid levels
        scale: Scale factor for pyramid construction

    Returns:
        Filtered image
    """
    # Ensure image is in float32 format
    image = image.astype(np.float32)

    # Build pyramids
    G, L = pyramid(image, nlev, scale)
    print(f"Gaussian pyramid: {len(G)}, Laplacian pyramid: {len(L)}")

    # Start from the coarsest level
    result = G[-1]

    # Process each level from coarse to fine
    for level in range(len(G) - 2, -1, -1):
        print(f"Processing level {level}")
        # Calculate adaptive parameters
        adaptive_sigma_s = sigma_s * (scale**level)
        w1 = int(np.ceil(adaptive_sigma_s * 0.5 + 1))
        w2 = int(np.ceil(adaptive_sigma_s * 2.0 + 1))

        # Upsample current result
        print(f"Upsampling from {result.shape} to {G[level].shape}")
        result_up = cv2.resize(result, (G[level].shape[1], G[level].shape[0]), interpolation=cv2.INTER_LINEAR)

        # First bilateral filtering
        result_hat = bilateral_filter(result_up, G[level], w1, adaptive_sigma_s, sigma_r)

        # Add Laplacian detail
        result_lap = result_hat + L[level]

        # Second bilateral filtering
        result_out = bilateral_filter(result_lap, result_hat, w2, adaptive_sigma_s, sigma_r)

        # Final enhancement
        result_refine = bilateral_filter(result_out, result_out, w2, adaptive_sigma_s, sigma_r)

        result = result_refine

    return result
