import cv2


def downsample(image, filter_kernel, scale):
    """Downsample an image using the given filter kernel and scale factor."""
    # Apply Gaussian filter
    filtered = cv2.filter2D(image, -1, filter_kernel, borderType=cv2.BORDER_REFLECT)

    # Resize
    new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
    output = cv2.resize(filtered, new_size, interpolation=cv2.INTER_LINEAR)

    return output


def upsample(image, filter_kernel, ref_image):
    """Upsample an image to match the reference size."""
    # Resize
    output = cv2.resize(image, (ref_image.shape[1], ref_image.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Apply Gaussian filter
    output = cv2.filter2D(output, -1, filter_kernel, borderType=cv2.BORDER_REFLECT)

    return output


def pyramid(image, nlev=5, scale=0.5, sigma=1.0):
    """Build Gaussian and Laplacian pyramids."""
    # Create Gaussian kernel
    filter_size = 5
    filter_kernel = cv2.getGaussianKernel(filter_size, sigma)
    filter_kernel = filter_kernel * filter_kernel.T

    # Initialize pyramids
    G_pyr = [None] * nlev
    L_pyr = [None] * (nlev - 1)

    G_pyr[0] = image.copy()

    for level in range(nlev - 1):
        # Downsample
        image_down = downsample(image, filter_kernel, scale)

        # Compute Laplacian
        image_up = upsample(image_down, filter_kernel, image)
        L_pyr[level] = image - image_up

        # Store Gaussian level
        G_pyr[level + 1] = image_down

        # Update for next iteration
        image = image_down

    return G_pyr, L_pyr
