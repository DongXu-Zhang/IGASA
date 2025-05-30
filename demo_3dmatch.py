import os
import json
import torch
import argparse
import numpy as np
import open3d as o3d
from munch import munchify
from engine.trainer import EpochBasedTrainer
from data.indoor_data import IndoorDataset, IndoorTestDataset
from models.models.igasa import IGASA



import argparse
import json
import numpy as np
import torch
import open3d as o3d
from munch import munchify

parser = argparse.ArgumentParser()
parser.add_argument("--split", default='train', choices=['train', 'val', 'test'])
parser.add_argument("--benchmark", default='3DMatch', choices=['3DMatch', '3DLoMatch'])
parser.add_argument("--load_pretrained", default='cast-epoch-15', type=str)
parser.add_argument("--id", default=0, type=int)

_args = parser.parse_args()


class Engine(EpochBasedTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.train_dataset = IndoorDataset(cfg.data.root, 'train', cfg.data.npoints, cfg.data.voxel_size, cfg.data_list, 0.0)
        self.val_dataset = IndoorDataset(cfg.data.root, 'val', cfg.data.npoints, cfg.data.voxel_size, cfg.data_list, 0.0)
        self.test_dataset = IndoorTestDataset(cfg.data.root, _args.benchmark, cfg.data.npoints, cfg.data.voxel_size, cfg.data_list)
        
        self.model = IGASA(cfg.model).cuda()


if __name__ == "__main__":
    # 加载配置文件
    with open('./config/3dmatch.json', 'r') as cfg:
        args = json.load(cfg)
        args = munchify(args)
    
    tester = Engine(args)
    tester.set_eval_mode()
    tester.load_snapshot(_args.load_pretrained)

    # 根据 split 参数选择数据集
    if _args.split == 'train':
        data = tester.train_dataset[_args.id]
        data_dict = tester.train_dataset.dataset[_args.id]
    elif _args.split == 'val':
        data = tester.val_dataset[_args.id]
        data_dict = tester.val_dataset.dataset[_args.id]
    else:
        data = tester.test_dataset[_args.id]
        data_dict = tester.test_dataset.dataset[_args.id]
    

    # 打印变换矩阵（仅用于参考）
    print(data_dict)
    
    gt_trans = data[2].numpy()
    print("gttrans")
    print(gt_trans)
    # 读取参考点云和源点云
    ref_cloud = o3d.io.read_point_cloud(data_dict['points1'])
    src_cloud = o3d.io.read_point_cloud(data_dict['points2'])

    # 设置自定义颜色（蓝色：参考点云；黄色：源点云）
    custom_yellow = np.asarray([[221., 184., 34.]]) / 255.0
    custom_blue = np.asarray([[9., 151., 247.]]) / 255.0
    ref_cloud.paint_uniform_color(custom_blue.T)
    src_cloud.paint_uniform_color(custom_yellow.T)
    
    # 保存初始位置的点云（参考点云和源点云）
    initial_combined_cloud = o3d.geometry.PointCloud()
    points_ref = np.asarray(ref_cloud.points)
    points_src = np.asarray(src_cloud.points)
    colors_ref = np.asarray(ref_cloud.colors)
    colors_src = np.asarray(src_cloud.colors)
    initial_combined_points = np.vstack((points_ref, points_src))
    initial_combined_colors = np.vstack((colors_ref, colors_src))
    initial_combined_cloud.points = o3d.utility.Vector3dVector(initial_combined_points)
    initial_combined_cloud.colors = o3d.utility.Vector3dVector(initial_combined_colors)
    initial_output_filename = "initial_position_5.ply"
    o3d.io.write_point_cloud(initial_output_filename, initial_combined_cloud)
    print(f"Initial 3D object saved as {initial_output_filename}")
    
    # 将数据转移到 GPU，并增加 batch 维度
    data = [v.cuda().unsqueeze(0) for v in data]
    with torch.no_grad():
        output_dict = tester.model(*data)
        trans = output_dict['refined_transform'].cpu().numpy()
    print(trans)
    # 估计法向量
    ref_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
    src_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))

    # 对源点云应用模型预测的变换
    src_cloud.transform(trans)

    # 保存变换后的点云（参考点云和变换后的源点云）
    transformed_combined_cloud = o3d.geometry.PointCloud()
    points_ref = np.asarray(ref_cloud.points)
    points_src = np.asarray(src_cloud.points)
    colors_ref = np.asarray(ref_cloud.colors)
    colors_src = np.asarray(src_cloud.colors)
    transformed_combined_points = np.vstack((points_ref, points_src))
    transformed_combined_colors = np.vstack((colors_ref, colors_src))
    transformed_combined_cloud.points = o3d.utility.Vector3dVector(transformed_combined_points)
    transformed_combined_cloud.colors = o3d.utility.Vector3dVector(transformed_combined_colors)
    transformed_output_filename = "transformed_position_5.ply"
    o3d.io.write_point_cloud(transformed_output_filename, transformed_combined_cloud)
    print(f"Transformed 3D object saved as {transformed_output_filename}")