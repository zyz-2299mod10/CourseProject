import open3d as o3d
import os

# show the result of saved ply file
def show_ply(filepath):
    pcd = o3d.io.read_point_cloud(filepath)
    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    cases = ["bunny", "star", "venus", "noisy_venus"]
    # result_file = 'result'

    for case in cases:
        # result_path = os.path.join(result_file, case)

        ply_file = '/' + case + '.ply'
        show_ply('.' + ply_file)