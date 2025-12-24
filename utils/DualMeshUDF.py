import torch
import numpy as np
from DualMeshUDF_core import OctreeNode, Octree, triangulate_faces
import igl


def extract_mesh(
        udf_func,
        udf_grad_func,
        batch_size=150000,
        max_depth=7,
        max_UDF_threshold=0.01,
        min_UDF_threshold=0.00001,
        singular_value_threshold=0.5,
):
    """
    Extract the mesh from a UDF
    Parameters
    ------------
    udf_func : udf function
    udf_grad_func : udf with gradient
    batch_size: batch size for inferring the UDF network
    max_depth: the max depth of the octree, e.g., max_depth=7 stands for resolution of 128^3
    """

    octree = Octree(
                    max_depth=max_depth,
                    min_corner=np.array([[-1.], [-1.], [-1.]]),
                    max_corner=np.array([[1.], [1.], [1.]]),
                    sampling_depth=1)

    cur_depth = 0

    while cur_depth <= max_depth:
        # get centroids of the nodes in the current depth
        centroids_coords = octree.centroids_of_new_nodes().astype(np.float32)

        # query udf values and gradients for the centroids
        centroid_udfs, centroid_grads = query_udf_and_grad(udf_grad_func, centroids_coords, batch_size)

        # adaptively subdivide the cells
        octree.adaptive_subdivide(centroid_udfs, centroid_grads, 0.002)

        cur_depth += 1

    new_grid_indices, new_grid_coords = octree.get_samples_of_new_nodes()

    # query udf values and gradients for the samples
    new_grid_udfs, new_grid_grads = query_udf_and_grad(udf_grad_func, new_grid_coords.astype(np.float32), batch_size)

    octree.set_new_grid_data(new_grid_indices, new_grid_udfs, new_grid_grads)

    # check if projections are reliable
    indices, projections = octree.get_projections_for_checking_validity()
    projection_udfs = query_udf(udf_func, projections, batch_size)
    octree.set_grid_validity(indices, projection_udfs <max_UDF_threshold)

    # (reliable udf threshold, corner factor, edge factor, face factor, singular value threshold)
    octree.batch_solve(min_UDF_threshold, 1.0, 1.0, 1.0, singular_value_threshold)

    octree.generate_mesh()

    tri_faces = triangulate_faces(octree.mesh_v, octree.mesh_f, octree.v_type, octree.mesh_v_dir)
    if not len(np.array(octree.mesh_v))>0:
        return None,None

    v, _, _, f = igl.remove_duplicate_vertices(np.array(octree.mesh_v), tri_faces, 1e-15)

    v, f, _, _ = igl.remove_unreferenced(v, f)

    return v, f


def query_udf(udf_func, coords, max_batch_size=-1):
    '''
    coords should be N*M*3, where N is the batch size
    '''
    input_shape = list(coords.shape)
    query_points = coords.reshape(-1, 3)
    if max_batch_size > 0 and query_points.shape[0] > max_batch_size:
        batch_num = query_points.shape[0] / max_batch_size + 1
        pts_list = np.array_split(query_points, batch_num)
        d = []
        for q_per_batch in pts_list:
            temp_d = udf_func(q_per_batch)
            d.append(temp_d)
        d = np.vstack(d)
    else:
        d = udf_func(query_points)

    input_shape[-1] = 1
    d = d.reshape(input_shape)
    return d


def query_udf_and_grad(udf_grad_func, coords, batch_size=-1):
    '''
    coords should be N*M*3 or N*3, where N is the batch size
    '''
    input_shape = list(coords.shape)
    query_points = coords.reshape(-1, 3)
    if batch_size > 0 and query_points.shape[0] > batch_size:
        batch_num = query_points.shape[0] / batch_size + 1
        pts_list = np.array_split(query_points, batch_num)
        d = []
        g = []
        for q_per_batch in pts_list:
            temp_d, temp_g = udf_grad_func(q_per_batch)
            d.append(temp_d)
            g.append(temp_g)
        d = np.vstack(d)
        g = np.vstack(g)
    else:
        d, g = udf_grad_func(query_points)

    g = g.reshape(input_shape)
    input_shape[-1] = 1
    d = d.reshape(input_shape)
    return d, g



def extract_mesh_from_udf(
        net,
        device,
        batch_size=150000,
        max_depth=7,
        max_UDF_threshold=0.01,
        min_UDF_threshold=0.00001,
        singular_value_threshold=0.5,
):
    """
    Extract the mesh from a UDF network
    Parameters
    ------------
    net : the udf network
    device : the device of the net parameters
    batch_size: batch size for inferring the UDF network
    max_depth: the max depth of the octree, e.g., max_depth=7 stands for resolution of 128^3
    """

    # compose functions
    udf_func = udf_from_net(net, device)
    udf_grad_func = udf_grad_from_net(net, device)

    # get mesh
    mesh_v, mesh_f = extract_mesh(udf_func, udf_grad_func, batch_size, max_depth,max_UDF_threshold=max_UDF_threshold,min_UDF_threshold=min_UDF_threshold,singular_value_threshold=singular_value_threshold)

    return mesh_v, mesh_f


def normalize(v, dim=-1):
    norm = torch.linalg.norm(v, axis=dim, keepdims=True)
    norm[norm == 0] = 1
    return v / norm


def udf_from_net(net, device):
    def udf(pts, net=net, device=device):
        net.eval()

        target_shape = list(pts.shape)

        pts = pts.reshape(-1, 3)
        pts = torch.from_numpy(pts).to(device)

        input = pts.reshape(-1, pts.shape[-1]).float()
        udf_p = net(input)

        target_shape[-1] = 1
        udf_p = udf_p.reshape(target_shape).detach().cpu().numpy()

        return udf_p

    return udf


def udf_grad_from_net(net, device):
    def grad(pts, net=net, device=device):
        net.eval()

        target_shape = list(pts.shape)

        pts = pts.reshape(-1, 3)
        pts = torch.from_numpy(pts).to(device)
        pts.requires_grad = True

        input = pts.reshape(-1, pts.shape[-1]).float()
        udf_p = net(input)

        udf_p.sum().backward()
        grad_p = pts.grad.detach()
        grad_p = normalize(grad_p)

        grad_p = grad_p.reshape(target_shape).detach().cpu().numpy()
        target_shape[-1] = 1
        udf_p = udf_p.reshape(target_shape).detach().cpu().numpy()

        return udf_p, grad_p

    return grad