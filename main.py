import os
import cv2
import numpy as np


def add_gaussian_noise(input_folder, output_folder):
    # Check if output folder exists, if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".bmp"):
            img = cv2.imread(os.path.join(input_folder, filename))

            # Add Gaussian noise
            mean = 0
            var = 0.2
            sigma = var ** 0.5
            gaussian = np.random.normal(mean, sigma, (img.shape[0], img.shape[1])) #  np.zeros((224, 224), np.float32)

            noisy_image = np.zeros(img.shape, np.float32)

            if len(img.shape) == 2:
                noisy_image = img + gaussian
            else:
                noisy_image[:, :, 0] = img[:, :, 0] + gaussian
                noisy_image[:, :, 1] = img[:, :, 1] + gaussian
                noisy_image[:, :, 2] = img[:, :, 2] + gaussian

            # The new values can be out of range [0, 255]. We need to clip the values
            cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)
            noisy_image = noisy_image.astype(np.uint8)

            # Save the image to the output folder
            cv2.imwrite(os.path.join(output_folder, filename), noisy_image)



add_gaussian_noise("input_folder", "output_folder")

print("Images in the input folder were successfully processed and saved to the output folder.")
