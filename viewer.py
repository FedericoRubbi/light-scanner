import open3d as o3d
import numpy as np

def draw_geometries(pcd):

    def rotate_view(vis):
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        ctr = vis.get_view_control()
        ctr.rotate(10.0, 0.0)
        return False

    o3d.visualization.draw_geometries_with_animation_callback([pcd], rotate_view)
