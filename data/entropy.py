import cv2
import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
import os

def compute_entropy(img, window_size=5):
    """
    Compute entropy map of an image using a sliding window approach.

    Parameters:
        img (numpy.ndarray): Input grayscale image.
        window_size (int): Size of the sliding window.

    Returns:
        numpy.ndarray: Entropy map of the image.
    """
    height, width = img.shape
    entropy_map = np.zeros((height, width))

    pad = window_size // 2
    padded_img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)



    for i in range(height):
        for j in range(width):
            window = padded_img[i:i + window_size, j:j + window_size].flatten()
            hist, _ = np.histogram(window, bins=256, range=(0, 256), density=True)
            entropy_map[i, j] = entropy(hist, base=2)

    return entropy_map


def main(image_path):
    """
    Load an image, compute its entropy map, and display the result.

    Parameters:
        image_path (str): Path to the input image.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Unable to load image.")
        return

    entropy_map = compute_entropy(img, window_size=5)

    plt.figure(figsize=(5, 5))
    plt.subplot(1, 2, 1)
    # plt.title("Original Image")
    plt.imshow(img, cmap='gray')
    plt.axis("off")

    plt.subplot(1, 2, 1)
    plt.title("Entropy Map")
    plt.imshow(entropy_map, cmap='jet')
    plt.axis("off")

    destination_folder = r"C:\Users\yvona\PycharmProjects\DeepSeek/"

    output_path = os.path.join(destination_folder, f"2222.jpg")
    cv2.imwrite(output_path, entropy_map)

    plt.show()


if __name__ == "__main__":
    # image_path = r"C:\Users\yvona\Downloads\TestDataset\TestDataset\COD10K\Imgs\COD10K-CAM-1-Aquatic-6-Fish-203.jpg"  # Change this to your image path
    image_path = r"..\images\001_crop_0_1.jpg"  # Change this to your image path
    image_path = r"..\data\retina\images\6192.jpg"  # Change this to your image path
    image_path = r"..\train\image\51.png"  # Change this to your image path

    main(image_path)
