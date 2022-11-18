import numpy as np
import torch
from home_robot.utils.image import show_point_cloud
from torch_geometric.nn import PointConv, fps, radius, global_max_pool, MLP


def color_points(color, points, color_indices, index_color, show):
    # If a point has already been colored in, it will leave it as-is
    color = color.copy()

    for point_id in color_indices:
        if np.all(color[point_id] == 0):
            color[point_id] = index_color

    if show:
        show_point_cloud(points, color, orig=np.zeros(3, ))

    return color


def radius_test(r):
    print(f"Running with r={r}")
    x_range = 100
    y_range = 100
    z_range = 10
    scale = 1
    r = r / scale
    idx_offset = 122  # Used to find the centers
    points = np.array([[x, y, z] for x in range(x_range) for y in range(y_range) for z in range(z_range)]) / scale
    np.random.shuffle(points)
    batch = np.array([0 for _ in range(len(points))])
    idx = np.array([i for i in range(0, len(points), idx_offset)])

    # Display the original grid of selected points
    color = np.array([[0, 0, 0] for _ in range(len(points))])
    original_point_color = color_points(color, points * scale, idx, np.array([255, 0, 0]), show=True)  # Scaling back up just to make it a little nicer to view

    # Using the CPU, find the points in the given radius (r) and display them
    points = torch.tensor(points, dtype=torch.float)
    batch = torch.tensor(batch, dtype=torch.float)
    idx = torch.tensor(idx, dtype=torch.int64)
    radius_result = radius(points, points[idx], r, batch, batch[idx])
    print(f"CPU: {radius_result}")
    #display_points(points, radius_result[0], np.array([255, 0, 0]))  # Not exactly sure what radius_result[0] is - doesn't seem to be indices into points
    color_points(original_point_color, points * scale, radius_result[1], np.array([0, 255, 0]), show=True)

    # Using CUDA, run the same experiment and display the points in the given radius
    points = points.cuda()
    batch = batch.cuda()
    idx = idx.cuda()
    radius_result = radius(points, points[idx], r, batch, batch[idx])
    print(f"CUDA: {radius_result}")
    #display_points(points.cpu().numpy(), radius_result[0].cpu().numpy(), np.array([255, 0, 0]))
    color_points(original_point_color, points.cpu().numpy() * scale, radius_result[1].cpu().numpy(), np.array([0, 0, 255]), show=True)


if __name__ == "__main__":
    radius_test(r=2)
    radius_test(r=10)
    radius_test(r=30)
    radius_test(r=100)
