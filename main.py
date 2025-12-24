# -*- coding: utf-8 -*-

import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
from models.dataset import Dataset
from models.networks import UltraBoneUDF
import argparse
import os
from shutil import copyfile
import numpy as np
from utils.logger import get_root_logger, print_log
import math
import warnings
warnings.filterwarnings("ignore")
import torch.nn as nn
from datetime import datetime
import random
from DualMeshUDF import write_obj
from utils.DualMeshUDF import extract_mesh_from_udf
from extensions.chamfer_dist import ChamferDistance
import open3d as o3d
from utils.read_conf import *
from utils.point_clouds_related import filter_source_by_target_distance
device="cuda" if torch.cuda.is_available() else "cpu"

import cpuinfo
print("CPU:", cpuinfo.get_cpu_info()['brand_raw'])
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name())





def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)  # Python random module
    np.random.seed(seed_value)  # Numpy library
    torch.manual_seed(seed_value)  # Torch

    # if using CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # if using multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

################### Bone_reconstruction_UDF Implementation ##################################3
class Runner:
    def __init__(self, conf,congif_file):
        self.device = torch.device('cuda')

        self.conf = conf
        self.congif_file=congif_file
        self.base_exp_dir = os.path.join(conf['general.base_exp_dir'],f"specimen{conf.dataset.specimen_id:02d}",f"{conf.dataset.anatomy}",
                                         f"{conf.dataset.record_id}")
        os.makedirs(self.base_exp_dir, exist_ok=True)

        self.dataset_processed = Dataset(conf) # The processed dataset: normalization, query points sampling etc. All coordinates are in [-1,1]
        self.input_pcd_raw = o3d.io.read_point_cloud(conf.dataset.pcd_file).paint_uniform_color((1,0,0))  # raw input pcd file, in mm
        self.CT_bone_segmentation_mesh=o3d.io.read_triangle_mesh(conf.dataset.CT_bone_segmentation_file) # raw CT bone mesh, in mm

        # as the input data does not cover the whole GT mesh, we apply distance-based filter to keep regions that are actually captured.
        self.CT_pcd_filter=filter_source_by_target_distance(self.CT_bone_segmentation_mesh.sample_points_uniformly(int(5e6)),
                                                            self.input_pcd_raw,1).paint_uniform_color((0,1,0))
        # o3d.visualization.draw_geometries([self.input_pcd_raw,self.CT_pcd_filter])
        self.iter_step = 0
        # momentum
        # Training parameters
        self.lambda_NC=self.conf.get_float('train.lambda_NC')

        self.lambda_GS=self.conf.get_float('train.lambda_GS')

        self.maxiter = self.conf.get_int('train.maxiter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.ChamferDis = ChamferDistance().cuda()
        self.L2_loss=nn.MSELoss()

        self.mode = conf.general.mode
        self.udf_network = UltraBoneUDF(**self.conf['model.udf_network']).to(self.device)
        self.udf_optimizer = torch.optim.Adam(self.udf_network.parameters(), lr=self.learning_rate)

        # Backup codes and configs for debug

        if self.mode == 'train':
            self.file_backup()

    def train(self):
        timestamp_start = time.strftime('%Y%m%d_%H%M%S', time.localtime())

        print(f"Start time:{timestamp_start}")
        log_file = os.path.join(os.path.join(self.base_exp_dir), 'logger.log')

        logger = get_root_logger(log_file=log_file, name='outs')
        self.logger = logger

        res_step = self.maxiter - self.iter_step


        grid_sparse=self.dataset_processed.grid_sparse
        grid_sparse_udf_gt=self.dataset_processed.grid_sparse_udf_gt
        idx=grid_sparse_udf_gt>0.1
        grid_sparse_udf_gt=grid_sparse_udf_gt[idx]
        grid_sparse=grid_sparse[idx]

        for iter_i in tqdm(range(res_step)):
            self.update_learning_rate_np(iter_i)

            samples,samples_near, samples_near_normal,pcd_gt,pcd_gt_normals = self.dataset_processed.np_train_data(self.batch_size)

            ###########Train Bone_reconstruction_UDF Network################
            self.udf_optimizer.zero_grad()
            samples.requires_grad = True
            gradients_sample = self.udf_network.gradient(samples).squeeze()  # 5000x3

            udf_sample = self.udf_network.udf(samples)  # 5000x1
            grad_norm = F.normalize(gradients_sample, dim=1)
            samples_moved = samples - grad_norm * udf_sample
            _, idx1, idx2 = self.ChamferDis(samples_moved.unsqueeze(0), pcd_gt.unsqueeze(0))
            loss_NC = torch.abs(
                (grad_norm * (samples - pcd_gt[idx1[0]])).sum(dim=1,
                                                              keepdim=True) - udf_sample).mean()
            grid_sparse_udf_pred = self.udf_network.udf(grid_sparse)
            grid_sprase_loss = self.L2_loss(grid_sparse_udf_pred, grid_sparse_udf_gt)
            total_loss = self.lambda_NC * loss_NC + self.lambda_GS * grid_sprase_loss

            total_loss.backward()
            self.udf_optimizer.step()


            ############# Saving #################
            self.iter_step += 1
            if self.iter_step % self.report_freq == 0:
                print_log('iter: {:8>d} total_loss = {} lr = {}'.format(self.iter_step, total_loss,
                                                                        self.udf_optimizer.param_groups[0]['lr']),
                          logger=logger)

            if self.iter_step %self.save_freq==0 and self.iter_step>=15000:
                self.validate_mesh()
                self.save_checkpoint()

        timestamp_end = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        start_datetime = datetime.strptime(timestamp_start, '%Y%m%d_%H%M%S')
        end_datetime = datetime.strptime(timestamp_end, '%Y%m%d_%H%M%S')
        duration = end_datetime - start_datetime

    def validate_mesh(self):
        print(f"start extracting meshes")
        try:
            mesh_v, mesh_f = extract_mesh_from_udf(self.udf_network, "cuda", max_depth=9,
                                                   min_UDF_threshold=0.000001, max_UDF_threshold=0.01,
                                                   singular_value_threshold=0.2)
        except Exception as e:
            print(e)
            return
        if mesh_v is None:
            return
        if not os.path.isdir(os.path.join(self.base_exp_dir, 'outputs')):
            os.mkdir(os.path.join(self.base_exp_dir, 'outputs'))
        mesh_file = os.path.join(self.base_exp_dir, 'outputs',
                                 f'{self.iter_step:0>8d}_UltraBoneUDF.obj')
        write_obj(mesh_file, mesh_v, mesh_f)

        mesh = o3d.io.read_triangle_mesh(mesh_file)
        vertices = np.asarray(mesh.vertices)

        vertices = (vertices * self.dataset_processed.shape_scale) + self.dataset_processed.shape_center

        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(mesh_file, mesh)

        pcd_reconstruction=mesh.sample_points_uniformly(int(5e6))
        CD_reconstruction_to_GT=np.asarray(pcd_reconstruction.compute_point_cloud_distance(self.CT_pcd_filter)).mean()
        CD_GT_to_reconstruction=np.asarray(self.CT_pcd_filter.compute_point_cloud_distance(pcd_reconstruction)).mean()
        CD_two_side=0.5*(CD_reconstruction_to_GT+CD_GT_to_reconstruction)
        print(f"specimen_id:{self.conf.dataset.specimen_id}, anatomy:{self.conf.dataset.anatomy}, record_id:{self.conf.dataset.record_id}, "
              f"CD-recon-to-GT:{CD_reconstruction_to_GT:.3f}, CD-GT-to-recon:{CD_GT_to_reconstruction:.3f}, CD-double-side:{CD_two_side:.3f}")



    def update_learning_rate_np(self, iter_step):
        warn_up = self.warm_up_end
        max_iter = self.maxiter
        init_lr = self.learning_rate
        lr = (iter_step / warn_up) if iter_step < warn_up else 0.5 * (
                    math.cos((iter_step - warn_up) / (max_iter - warn_up) * math.pi) + 1)
        lr = lr * init_lr
        for g in self.udf_optimizer.param_groups:
            g['lr'] = lr



    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.congif_file, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

    def load_checkpoint(self, checkpoint_name):
        checkpoint_file=os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name)
        print(checkpoint_file)
        assert os.path.isfile(checkpoint_file),f"file does not exist:{checkpoint_file}"

        checkpoint = torch.load(checkpoint_file, map_location=self.device)

        self.udf_network.load_state_dict(checkpoint['udf_network_fine'])

        self.iter_step = checkpoint['iter_step']

    def save_checkpoint(self):
        checkpoint = {
            'udf_network_fine': self.udf_network.state_dict(),
            'iter_step': self.iter_step,
        }
        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint,
                   os.path.join(self.base_exp_dir, 'checkpoints', f"ckpt_{self.iter_step:0>6d}_UltraBoneUDF.pth"))










def UltraBoneUDF_UltraBones100k():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--target_specimen_id', type=int, default=-1)
    args = parser.parse_args()


    torch.cuda.set_device(args.gpu)
    congif_file="conf/conf_UltraBones100k.conf"
    conf=read_confs(congif_file)
    conf.put("general.mode",args.mode)
    for specimen_id in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14][:]:
        if args.target_specimen_id>0 and not specimen_id==args.target_specimen_id:
            continue
        specimen_folder=os.path.join(conf.dataset.data_dir,f"specimen{specimen_id:02d}")
        ultrasound_records_folder=os.path.join(specimen_folder,"ultrasound_records")
        CT_bone_segmentation_file=os.path.join(specimen_folder,"CT_bone_segmentations","CT_bone_model_merged.stl")
        conf.put("dataset.specimen_id", specimen_id)
        conf.put("dataset.ultrasound_records_folder", ultrasound_records_folder)
        conf.put("dataset.CT_bone_segmentation_file", CT_bone_segmentation_file)

        assert os.path.isdir(ultrasound_records_folder)

        for anatomy in ["fibula", "tibia", "foot"][:]:
            anatomy_folder=os.path.join(ultrasound_records_folder,anatomy)
            conf.put("dataset.anatomy", anatomy)
            conf.put("dataset.anatomy_folder", anatomy_folder)
            for record_folder_name in os.listdir(anatomy_folder):
                record_folder=os.path.join(anatomy_folder,record_folder_name)
                conf.put("dataset.record_id", record_folder_name)
                conf.put("dataset.record_folder", record_folder)
                pcd_file=os.path.join(record_folder,"3D_reconstructions","with_pred_labels","reconstruction_pcd.xyz")
                assert os.path.isfile(pcd_file),f"file does not exist:{pcd_file}"
                conf.put("dataset.pcd_file",pcd_file)
                runner = Runner(conf, congif_file)
                if args.mode == 'train':
                    runner.train()
                else:
                    iter=30000
                    runner.load_checkpoint(f"ckpt_{iter:0>6d}_UltraBoneUDF.pth")
                    runner.validate_mesh()


if __name__ == '__main__':
    UltraBoneUDF_UltraBones100k()





