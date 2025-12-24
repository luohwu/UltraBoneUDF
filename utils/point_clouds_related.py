import numpy as np
import open3d as o3d

def filter_source_by_target_distance(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    dist_threshold: float,
) -> o3d.geometry.PointCloud:
    """
    Keep only points in `source` whose nearest-neighbor distance to `target`
    is <= dist_threshold. (Points farther than threshold are removed.)

    Returns a NEW point cloud (does not modify input).
    """
    if len(source.points) == 0:
        return o3d.geometry.PointCloud()
    if len(target.points) == 0:
        # No target points -> all source points are "far"
        return o3d.geometry.PointCloud()

    # Distance from each source point to its nearest point in target
    dists = np.asarray(source.compute_point_cloud_distance(target), dtype=np.float64)

    keep_idx = np.where(dists <= dist_threshold)[0]
    return source.select_by_index(keep_idx)
