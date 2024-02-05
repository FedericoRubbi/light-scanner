import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import cv2
from scipy.optimize import minimize


def get_profile(A, sigma_x, sigma_y, w, h):
    x, y = np.arange(w), np.arange(h)
    xv, yv = np.meshgrid(x, y)
    return A * (1 - np.exp(-((xv - w/2)**2 / (2*sigma_x**2) + (yv - h/2)**2 / (2*sigma_y**2))))
    

def cost_function(A, sigma_x, sigma_y, image):
    corrected_image = image + get_profile(A, sigma_x, sigma_y, image.shape[1], image.shape[0])
    variances = np.var(corrected_image, axis=1)
    return np.sum(variances)


def optimize_lsc_params(image, w, h):
    initial_guess = [40, image.shape[1] / 4, image.shape[0] / 4]
    cons = (
    {'type': 'eq', 'fun': lambda x: x[1] / x[2] - w / h},
    )

    def objective(params):
        return cost_function(params[0], params[1], params[2], image)

    result = minimize(objective, initial_guess, constraints=cons)
    return result.x  # returns optimized parameters


def compute_profile(photo):
    image_l = cv2.cvtColor(photo, cv2.COLOR_BGR2Lab)[:, :, 0]
    plot(image_l)
    filtered_image_l = cv2.GaussianBlur(image_l, (55, 55), -1)
    h, w, = photo.shape[1], photo.shape[0]
    A, sigma_x, sigma_y = optimize_lsc_params(filtered_image_l, w, h)
    lsc_profile = get_profile(A, sigma_x, sigma_y, w, h)
    lsc_profile /= np.mean(filtered_image_l)
    return lsc_profile

def apply_profile(photo, profile, weight: float=0.2):
    h, w = photo.shape[1], photo.shape[0]
    profile = profile[:h, :w]
    image = cv2.cvtColor(photo, cv2.COLOR_BGR2Lab).astype(np.float64)
    scale = np.mean(image)
    offset = np.mean(profile * scale)

    image[:, :, 0] += (profile * scale * weight - offset)
    image[:, :, 0] = np.clip(image[:, :, 0], 0, 255)  # Clip right after adding the profile

    image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_LAB2RGB)
    return image


def mean_kernel(size: int) -> np.ndarray:
    return np.ones([size] * 2) / size ** 2


def gaussian_kernel(size, sigma_x: float=-1, sigma_y: float=-1) -> np.ndarray:
    sigma_x = sigma_x if sigma_x > 0 else 0.3 * ((size - 1) * 0.5 - 1) + 0.8
    sigma_y = sigma_y if sigma_y > 0 else 0.3 * ((size - 1) * 0.5 - 1) + 0.8
    ax = np.linspace(-0.5, 0.5, size)
    gauss_x = np.exp(-0.5 * np.square(ax) / np.square(sigma_x))
    gauss_y = np.exp(-0.5 * np.square(ax) / np.square(sigma_y))
    kernel = np.outer(gauss_x, gauss_y)
    return kernel / np.sum(kernel)


def conv_correction(image: np.ndarray, k_size: int=20) -> np.ndarray:
    #lightness_mask = cv2.filter2D(image, -1, gaussian_kernel(k_size))
    lightness_mask = cv2.filter2D(image, -1, mean_kernel(k_size))
    return lightness_mask


def get_ref_lightness(photo, lightness_factor: float=0.7, area: int=0.3) -> float:
    image = cv2.cvtColor(photo, cv2.COLOR_RGB2Lab)
    area = int(area * min(photo.shape[0], photo.shape[1]))
    o_y, o_x = photo.shape[1] // 2, photo.shape[0] // 2
    return np.mean(image[o_y-area:o_y+area, o_x-area:o_x+area, 0]) * lightness_factor


def correct_light(photo, w_conv: float=0.5, ref_lightness: float=None) -> np.ndarray:
    image = photo
    image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

    ref_lightness = get_ref_lightness(photo) if ref_lightness is None else ref_lightness
    delta = 0
    if w_conv:
        conv = conv_correction(image[:, :, 0])
        delta += w_conv * (ref_lightness - conv)
    corrected_lightness = image[:, :, 0] + delta
    corrected_lightness = np.clip(corrected_lightness, 0, 255).astype(np.uint8)
    image[:, :, 0] = corrected_lightness
    return cv2.cvtColor(image, cv2.COLOR_Lab2RGB)


def plot(depth):
    x, y = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, depth, cmap='viridis')
    plt.show()


def main():
    image = Image.open("data/00.jpg")
    image = np.array(image)
    profile = compute_profile(image)
    cv2.imwrite("original.png", -profile*255)
    image = apply_profile(image, profile)
    cv2.imwrite("corrected.png", image)
    breakpoint()


def test():
    image = Image.open("data/00.jpg")
    image = np.array(image)
    image = correct_light(image)
    cv2.imwrite("corrected.png", image)


if __name__ == "__main__":
    test()