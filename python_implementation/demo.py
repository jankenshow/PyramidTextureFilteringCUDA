import os
import time

import cv2
import numpy as np
from pyramid_texture_filter import pyramid_texture_filter


def main():
    # I/O settings
    input_path = "data/"
    output_path = "output/"
    os.makedirs(output_path, exist_ok=True)

    # Parameters
    # img_name = "01.png"
    # sigma_s = 5.0
    # sigma_r = 0.07

    img_name = "08.png"
    sigma_s = 7.0
    sigma_r = 0.08

    # Read image
    image = cv2.imread(os.path.join(input_path, img_name))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0

    # Apply filtering
    start_time = time.time()
    result = pyramid_texture_filter(image, sigma_s, sigma_r)
    end_time = time.time()
    print(f"Processing time: {end_time - start_time:.2f} seconds")

    # Save result
    result = (result * 255).astype(np.uint8)
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(output_path, f"result_{img_name}"), result)


if __name__ == "__main__":
    main()
