

import torch
import numpy as np
import os
from scipy.spatial import cKDTree
import open3d as o3d






######Load data############
class Dataset:
    def __init__(self, conf):
        super(Dataset, self).__init__()
        self.device = torch.device('cuda')
        self.conf = conf
        self.processed_data_file_name = conf.dataset.pcd_file.replace(".xyz",".pt")
        if os.path.exists(self.processed_data_file_name):
            print('Data existing. Loading data...')
        else:
            print('Data not found. Processing data...')
            self.process_data(conf.dataset.pcd_file)

        print("Loading from saved data...")
        prep_data = torch.load(self.processed_data_file_name,map_location=self.device,weights_only=False)
        self.sample_near = prep_data["sample_near"]
        self.sample_near_normal=prep_data["sample_near_normal"]
        self.sample = prep_data["query_points"]
        self.pcd_gt = prep_data["pointcloud"]
        self.pcd_gt_normals=prep_data["pointcloud_normals"]

        self.shape_scale=prep_data["shape_scale"]
        self.shape_center = prep_data["shape_center"]
        self.grid_sparse_udf_gt=prep_data["grid_sparse_udf_gt"]
        self.grid_sparse=prep_data["grid_sparse"]


        self.sample_points_num = self.sample.shape[0]-1
        self.object_bbox_min, _  = torch.min(self.sample_near, dim = 0)
        self.object_bbox_min =  self.object_bbox_min - 0.05
        self.object_bbox_max,_  = torch.max(self.sample_near, dim = 0)
        self.object_bbox_max = self.object_bbox_max + 0.05
        print('Data bounding box:',self.object_bbox_min,self.object_bbox_max)


        
        print('NP Load data: End')

    def np_train_data(self, batch_size):
        index_coarse = np.random.choice(10, 1)
        index_fine = np.random.choice(self.sample_points_num//10, batch_size, replace = False)
        index = index_fine * 10 + index_coarse
        samples = self.sample[index]
        samples_near = self.sample_near[index]
        samples_near_normal=self.sample_near_normal[index]


        return samples,samples_near, samples_near_normal, self.pcd_gt,self.pcd_gt_normals




    ########Convert the .ply file into .npz ############
    def process_data(self,raw_data_file):
        pointcloud = np.asarray(
            o3d.io.read_point_cloud(raw_data_file).voxel_down_sample(0.1).points)
        shape_scale = np.max([np.max(pointcloud[:,0])-np.min(pointcloud[:,0]),np.max(pointcloud[:,1])-np.min(pointcloud[:,1]),np.max(pointcloud[:,2])-np.min(pointcloud[:,2])])
        shape_center = [(np.max(pointcloud[:,0])+np.min(pointcloud[:,0]))/2, (np.max(pointcloud[:,1])+np.min(pointcloud[:,1]))/2, (np.max(pointcloud[:,2])+np.min(pointcloud[:,2]))/2]
        self.shape_scale=shape_scale
        self.shape_center=shape_center
        pointcloud = pointcloud - shape_center
        pointcloud = pointcloud / shape_scale


        pcd_np,normals_np = self.FPS_sampling(pointcloud)


        pointcloud=torch.from_numpy(pcd_np).to(self.device).float()
        pointcloud_normals=torch.from_numpy(normals_np).to(self.device).float()

        grid_samp = 30000
        def gen_grid(start, end, num):
            x = np.linspace(start,end,num=num)
            y = np.linspace(start,end,num=num)
            z = np.linspace(start,end,num=num)
            g = np.meshgrid(x,y,z)
            positions = np.vstack([np.ravel(arr) for arr in g])
            return positions.swapaxes(0,1)



        dot5 = gen_grid(-0.5,0.5, 70)
        dot10 = gen_grid(-1.0, 1.0, 50)
        grid_sparse=gen_grid(-1.0, 1.0, 10)
        grid_sparse_udf_gt = self.chamfer_distance(grid_sparse, pcd_np)


        grid_sparse=torch.from_numpy(grid_sparse).to(self.device).float()
        grid_sparse_udf_gt = torch.from_numpy(grid_sparse_udf_gt).to(self.device).float()

        grid = np.concatenate((dot5,dot10))
        # grid = dot5
        grid = torch.from_numpy(grid).to(self.device).float()
        grid_f = grid[ torch.randperm(grid.shape[0])[0:grid_samp] ]



        query_per_point = 20
        query_points = self.sample_query2(query_per_point,pointcloud,pcd_np)

        # concat sampled points with grid points 
        query_points = torch.cat([query_points, grid_f]).float()

        ## find nearest neiboring point cloud for each query point
        POINT_NUM = 1000  # divide by batch to avoid out-of-memory
        if query_points.shape[0]%POINT_NUM>0:
            query_points=query_points[:-(query_points.shape[0]%POINT_NUM),:]
        query_points_nn = torch.reshape(query_points, (-1, POINT_NUM, 3))
        sample_near_tmp = []
        sample_near_normal_temp = []
        for j in range(query_points_nn.shape[0]):
            nearest_idx = self.search_nearest_point(torch.tensor(query_points_nn[j]).float().cuda(), torch.tensor(pointcloud).float().cuda())
            nearest_points = pointcloud[nearest_idx]
            nearest_points = nearest_points.reshape(-1,3)
            sample_near_tmp.append(nearest_points)

            nearest_normals=pointcloud_normals[nearest_idx]
            nearest_normals = nearest_normals.reshape(-1, 3)
            sample_near_normal_temp.append(nearest_normals)

        sample_near_tmp=torch.stack(sample_near_tmp,0)  
        sample_near_tmp = sample_near_tmp.reshape(-1,3)
        sample_near = sample_near_tmp

        sample_near_normal_temp=torch.stack(sample_near_normal_temp,0)
        sample_near_normal_temp=sample_near_normal_temp.reshape(-1,3)
        sample_near_normal=sample_near_normal_temp



        print("Saving files...")
        torch.save( {
                    "shape_scale":shape_scale,
                    "shape_center":shape_center,
                    "pointcloud":pointcloud,
                    "pointcloud_normals":pointcloud_normals,
                    "query_points":query_points,
                    "sample_near":sample_near,
                    "sample_near_normal":sample_near_normal,
                    "grid_sparse_udf_gt":grid_sparse_udf_gt,
                    "grid_sparse":grid_sparse,
                    },
                    raw_data_file.replace(".xyz",".pt"))

    def chamfer_distance(self,A, B):
        # Calculate the squared Euclidean distance between each pair of points
        # Result is an N x M matrix where entry (i, j) is the squared distance between A[i] and B[j]
        distances = np.sum((A[:, np.newaxis, :] - B[np.newaxis, :, :]) ** 2, axis=2)

        # Find the minimum distance for each point in A to any point in B
        min_distances_A_to_B = np.min(distances, axis=1)

        # Return the array of Chamfer distances for each point in A to B
        return min_distances_A_to_B

    def FPS_sampling(self,point_cloud):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud.reshape(-1,3))
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=1000))


        pcd_concat = o3d.geometry.PointCloud()

        if len(pcd.points) > 60000:
            # Perform farthest point sampling on the original point cloud
            pcd_down_1 = pcd.farthest_point_down_sample(5000 )  # First sampling
            pcd_down_2 = pcd.farthest_point_down_sample( 15000)  # Second sampling

            # Concatenate sampled points and their normals
            points_concat = np.concatenate(
                (np.asarray(pcd_down_1.points), np.asarray(pcd_down_2.points)), axis=0
            )
            normals_concat = np.concatenate(
                (np.asarray(pcd_down_1.normals), np.asarray(pcd_down_2.normals)), axis=0
            )
        else:
            # Use all points and normals directly if the cloud is small
            points_concat = np.asarray(pcd.points)
            normals_concat = np.asarray(pcd.normals)

            # Set points and normals to the concatenated point cloud
        pcd_concat.points = o3d.utility.Vector3dVector(points_concat.reshape(-1, 3))
        pcd_concat.normals = o3d.utility.Vector3dVector(normals_concat.reshape(-1, 3))
        pcd_concat.orient_normals_consistent_tangent_plane(15)

        # (Optional) Visualize the concatenated point cloud
        # o3d.visualization.draw_geometries([pcd_concat])

        print("Number of points:", len(pcd_concat.points))
        point_cloud_concat = np.asarray(pcd_concat.points)
        normals_concat = np.asarray(pcd_concat.normals)

        return point_cloud_concat, normals_concat

    # find the 50th nearest neighbor for each point in pc 
    # this will be the std for the gaussian for generating query 
    def sample_query(self,query_per_point, pc): 

        # scale = 0.25

        dists = torch.cdist(pc, pc)

        std, _ = torch.topk(dists, 50, dim=-1, largest=False) # shape: 1024, 50

        std = std[:,-1].unsqueeze(-1) # new shape is 1024, 1

        query_points = torch.empty(size=(pc.shape[0]*query_per_point, 3)).to(self.device).float()
        count = 0

        for idx, p in enumerate(pc):

            # query locations from p
            q_loc = torch.normal(mean=0.0, std=std[idx].item(),
                                    size=(query_per_point, 3)).to(self.device).float()

            # query locations in space
            q = p + q_loc

            query_points[count:count+query_per_point] = q

            count += query_per_point

        return query_points

    def sample_query2(self,query_per_point, pc,pointcloud):
        # divide point cloud by bacth to avoid out-of-memory

        ptree = cKDTree(pointcloud)
        std = []
        for idx_progress,p in enumerate(np.array_split(pointcloud,100,axis=0)):
            print(idx_progress)
            d = ptree.query(p,51)
            std.append(d[0][:,-1])
        
        std = np.concatenate(std)


        std=torch.from_numpy(std).to(self.device).float()

        std = std.unsqueeze(-1) # new shape is 1024, 1

        query_points = torch.empty(size=(pc.shape[0]*query_per_point, 3)).to(self.device).float()
        count = 0

        for idx, p in enumerate(pc):
            print(f"working on {idx}/{len(pc)}")

            # query locations from p
            q_loc = torch.normal(mean=0.0, std=std[idx].item(),
                                    size=(query_per_point, 3)).to(self.device).float()

            # query locations in space
            q = p + q_loc

            query_points[count:count+query_per_point] = q

            count += query_per_point

        return query_points

    # the closest point in the pc for all query points 
    def find_nearest_query_neighbor(self,pc, query_points):

        dists = torch.cdist(query_points, pc).detach().cpu().numpy()
        min_dist, min_idx = torch.min(dists, dim=-1).detach().cpu().numpy()  
        nearest_neighbors = pc[min_idx]

        return nearest_neighbors, min_dist.unsqueeze(-1)

    ######Find the nearest neighbour points ##########
    def search_nearest_point(self,point_batch, point_gt):
        num_point_batch, num_point_gt = point_batch.shape[0], point_gt.shape[0]
        point_batch = point_batch.unsqueeze(1).repeat(1, num_point_gt, 1)
        point_gt = point_gt.unsqueeze(0).repeat(num_point_batch, 1, 1)

        distances = torch.sqrt(torch.sum((point_batch-point_gt) ** 2, axis=-1) + 1e-12) 
        dis_idx = torch.argmin(distances, axis=1).detach().cpu().numpy()

        return dis_idx



