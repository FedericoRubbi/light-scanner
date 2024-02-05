import cv2
import numpy as np
import open3d as o3d
import os

from depth_estimation import DepthEstimator, segment
from lighting import correct_light
from viewer import draw_geometries


def load_point_clouds():
    path = "tmp"
    pcds = [x for x in os.listdir(path) if x.endswith(".ply")]
    data = []
    for i in pcds:
        pcd = o3d.io.read_point_cloud(f"tmp/{i}")
        data.append(pcd)
    return data


def process_data(load_pcd=True):
    try:
        if load_pcd:
            return load_point_clouds()
    except:
        pass

    processor = DepthEstimator()
    path = "data"

    data = []
    for i in os.listdir(path):
        color, depth = processor.predict(os.path.join(path, i))
        mask = segment(depth)
        cv2.imwrite(f"mask/mask{i.replace('.jpg', '')}.png", mask)
        color = correct_light(color)
        color = cv2.bitwise_and(color, color, mask=mask)
        depth = cv2.bitwise_and(depth, depth, mask=mask)
        color_o3d = o3d.geometry.Image(color.astype(np.uint8))
        depth_o3d = o3d.geometry.Image((depth*100).astype(np.uint16))
        pcd = compute_cloud(color_o3d, depth_o3d)
        data.append(pcd)
        o3d.io.write_point_cloud(f"tmp/pcd{i.replace('.jpg', '')}.ply", pcd)
    return data


def compute_cloud(color, depth):
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, convert_rgb_to_intensity = False)
    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_camera_intrinsic)
    pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
    pcd = pcd.voxel_down_sample(voxel_size=0.001)
    return pcd


def merge_clouds(pcd0, pcd1, filter=False):
    # Precise alignment using ICP
    threshold = 0.05  # Set a threshold for ICP (depends on the scale of your model)
    trans_init = np.eye(4)  # Initial transformation can be the identity matrix, or use a rough transformation if available
    registration_result = o3d.pipelines.registration.registration_icp(
        pcd0, pcd1, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    # Use the transformation from ICP to transform the second point cloud
    pcd1_transformed = pcd0.transform(registration_result.transformation)
    merged_pcd = pcd0 + pcd1_transformed
    if filter:
        merged_pcd, ind = merged_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2)
    #draw_geometries(merged_pcd)
    return merged_pcd


def main():
    data = process_data(load_pcd=True)
    while len(data) > 1:
        new_data = []
        for i in (np.arange(0, len(data), 2) + 5) % len(data):
            if i + 1 < len(data):
                merged_cloud = merge_clouds(data[i], data[(i + 2) % len(data)], filter=True)
                new_data.append(merged_cloud)
            else:
                new_data.append(data[i])
        data = new_data
    print(data)
    draw_geometries(data[0])


if __name__ == "__main__":
    main()
