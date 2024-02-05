import torch
from PIL import Image
from ZoeDepth.zoedepth.utils.misc import save_raw_16bit, colorize
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


class DepthEstimator:
    def __init__(self):
        repo = "isl-org/ZoeDepth"
        model_zoe_n = torch.hub.load(repo, "ZoeD_N", pretrained=True)
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.zoe = model_zoe_n.to(self.DEVICE)

    def predict(self, image_path: str):
        digits = ''.join(filter(lambda x: x.isdigit(), image_path))
        depth_path = f"tmp/depth{digits}.npy"
        image_path_np = f"tmp/image{digits}.npy"
        
        if os.path.exists(depth_path) and os.path.exists(image_path_np):
            depth = np.load(depth_path)
            image = np.load(image_path_np)
        else:
            image = Image.open(image_path).convert("RGB")
            depth = self.zoe.infer_pil(image)
            np.save(depth_path, depth)
            np.save(image_path_np, np.array(image))
            plot(depth)
        return np.array(image), depth

    def save_image(self, depth: torch.Tensor, fpath: str):
        save_raw_16bit(depth, f"{fpath}_raw.png")
        colored = colorize(depth)
        fpath_colored = f"{fpath}_colored.png"
        Image.fromarray(colored).save(fpath_colored)


def plot(depth):
    x, y = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, depth, cmap='viridis')
    plt.show()


def transform(depth):
    depth_filtered = cv2.GaussianBlur(depth, (5, 5), 0)
    x, y = np.meshgrid(np.arange(depth_filtered.shape[1]), np.arange(depth_filtered.shape[0]))
    A, B = np.c_[x.ravel(), y.ravel(), np.ones(x.size)], depth_filtered.ravel()
    C, _, _, _ = np.linalg.lstsq(A, B)
    z_fit = C[0]*x + C[1]*y + C[2]
    return depth - z_fit


def sharpen_image(image):
    sharpening_kernel = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ])
    sharpened_image = cv2.filter2D(image, -1, sharpening_kernel)
    return sharpened_image


def sobel_filter(depth):
    sobelx = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=5)
    sobel = cv2.magnitude(sobelx, sobely).astype(np.uint8)
    _, sobel = cv2.threshold(sobel, 50, 255, cv2.THRESH_BINARY)
    return sobel


def map_to_gray(image):
    return np.interp(image, (image.min(), image.max()), (0, 255)).astype(np.uint8)


def adaptive_thresholding(image):
    edges = cv2.Canny(map_to_gray(image), 50, 200)
    kernel = np.ones((5, 5), np.uint8)
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(closed_edges, kernel, iterations=6)
    cv2.imwrite("edges.png", dilated_edges)
    dilated_edges = dilated_edges.astype(np.float32) / 255.0
    size = dilated_edges.shape[0] * dilated_edges.shape[1]

    def cost(threshold):
        _, mask = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        mask = mask.astype(np.float32) / 255.0
        # Calculate the cost considering the difference and penalizing thresholds far from 0.5
        return np.sum((mask - dilated_edges) ** 2) + 2 * size * (threshold / 255 - 0.5) ** 2

    thresholds = np.arange(256)
    costs = np.array([cost(threshold) for threshold in thresholds])
    
    plt.plot(thresholds, costs)
    plt.show()

    optimal_threshold = thresholds[np.argmin(costs)]
    print(f"Optimal Threshold: {optimal_threshold}")
    _, optimal_mask = cv2.threshold(image, optimal_threshold, 255, cv2.THRESH_BINARY)
    print('Optimization done')

    return optimal_mask


def segment(depth):
    depth = sharpen_image(depth)
    depth_t = transform(depth)
    depth_mapped = map_to_gray(depth_t)
    _, mask = cv2.threshold(255 - depth_mapped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    # cv2.imwrite("mask.png", mask)
    # mask = adaptive_thresholding(255 - depth_mapped)
    # cv2.imwrite("opt_mask.png", mask)
    return mask


def main():
    processor = DepthEstimator()
    color, depth = processor.predict("data/00.jpg")
    cv2.imwrite("color.png", color)
    mask = segment(depth)
    depth = transform(depth)


if __name__ == "__main__":
    main()