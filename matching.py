from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def match_images(image0_path, image1_path, extractor_type='superpoint'):
    # Load the extractor and matcher based on the extractor_type
    if extractor_type.lower() == 'superpoint':
        extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
        matcher = LightGlue(features='superpoint').eval().to(device)
    elif extractor_type.lower() == 'disk':
        extractor = DISK(max_num_keypoints=2048).eval().to(device)
        matcher = LightGlue(features='disk').eval().to(device)
    else:
        raise ValueError("Invalid extractor_type. Choose either 'superpoint' or 'disk'.")

    # Load images
    image0 = load_image(image0_path).to(device)
    image1 = load_image(image1_path).to(device)

    # Extract local features
    feats0 = extractor.extract(image0)
    feats1 = extractor.extract(image1)

    matches01 = matcher({"image0": feats0, "image1": feats1})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension

    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

    # Convert tensors to numpy arrays for plotting
    image0_np = image0.cpu().numpy().transpose(1, 2, 0)
    image1_np = image1.cpu().numpy().transpose(1, 2, 0)

    m_kpts0_np = m_kpts0.cpu().numpy()
    m_kpts1_np = m_kpts1.cpu().numpy()

    return image0_np, image1_np, m_kpts0_np, m_kpts1_np

def display_images_and_matches(image0_np, image1_np, m_kpts0_np, m_kpts1_np):
    # Create a new figure and two subplots, sharing both axes
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex='col', sharey='row', figsize=(10,5))

    # Display the images
    ax1.imshow(image0_np)
    ax2.imshow(image1_np)

    # Plot the keypoints
    ax1.plot(m_kpts0_np[:, 0], m_kpts0_np[:, 1], 'b.')
    ax2.plot(m_kpts1_np[:, 0], m_kpts1_np[:, 1], 'b.')

    # Draw lines between matching keypoints
    for i in range(m_kpts0_np.shape[0]):
        xyA = m_kpts0_np[i, 0], m_kpts0_np[i, 1]
        xyB = m_kpts1_np[i, 0], m_kpts1_np[i, 1]
        coordsA = "data"
        coordsB = "data"
        con = mpatches.ConnectionPatch(xyA=xyB, xyB=xyA, coordsA=coordsB, coordsB=coordsA,
                                      axesA=ax2, axesB=ax1, color='green', linewidth=0.1)
        ax2.add_artist(con)

    # Show the plot
    plt.show()


def compute_transform(m_kpts0, m_kpts1):
    H, mask = cv2.findHomography(m_kpts0, m_kpts1, cv2.RANSAC, 5.0)
    return H


def main():
    # Usage
    image0_path, image1_path = "data/00.jpg", "data/01.jpg"
    image0_np, image1_np, m_kpts0_np, m_kpts1_np = match_images(image0_path, image1_path)
    H = compute_transform(m_kpts0_np, m_kpts1_np)
    display_images_and_matches(image0_np, image1_np, m_kpts0_np, m_kpts1_np)


if __name__ == "__main__":
    main()