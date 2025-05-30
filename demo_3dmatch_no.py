import json
import numpy as np
import torch
import open3d as o3d
from munch import munchify
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


def points_to_spheres(points, point_size=0.015, color=[1.0, 1.0, 1.0]):
    """
    将 Nx3 点云数据的每个点转换为一个小球，并合并成一个 TriangleMesh 对象返回。
    """
    # 检查点云格式
    if len(points.shape) != 2 or points.shape[1] != 3:
        print(f"点云数据的形状不正确: {points.shape}，应为 Nx3 格式。")
        return None

    sphere_list = o3d.geometry.TriangleMesh()
    # 创建基础小球
    base_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=point_size)
    base_sphere.compute_vertex_normals()

    for p in points:
        s = o3d.geometry.TriangleMesh(base_sphere)  # 复制基础球体
        trans = np.identity(4)
        trans[:3, 3] = p  # 平移到对应位置
        s.transform(trans)
        s.paint_uniform_color(color)
        sphere_list += s

    return sphere_list


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
    print(data)
    gt_trans = data[2].numpy()

    # 读取参考点云和源点云
    ref_cloud = o3d.io.read_point_cloud(data_dict['points1'])
    src_cloud = o3d.io.read_point_cloud(data_dict['points2'])

    # 设置自定义颜色（蓝色：参考点云；黄色：源点云）
    custom_yellow = np.asarray([[221., 184., 34.]]) / 255.0
    custom_blue = np.asarray([[9., 151., 247.]]) / 255.0
    ref_cloud.paint_uniform_color(custom_blue.T)
    src_cloud.paint_uniform_color(custom_yellow.T)
    
    # 将点云转换为小球形式（使用 points_to_spheres 函数）
    ref_points = np.asarray(ref_cloud.points)
    src_points = np.asarray(src_cloud.points)
    
    ref_cloud_spheres = points_to_spheres(ref_points, point_size=0.015, color=custom_blue[0])
    src_cloud_spheres = points_to_spheres(src_points, point_size=0.015, color=custom_yellow[0])
    
    # 保存初始位置的点云（参考点云和源点云）
    initial_combined_cloud = ref_cloud_spheres + src_cloud_spheres
    initial_output_filename = "initial_position_5.ply"
    o3d.io.write_triangle_mesh(initial_output_filename, initial_combined_cloud)
    print(f"Initial 3D object saved as {initial_output_filename}")
    
    # 将数据转移到 GPU，并增加 batch 维度
    data = [v.cuda().unsqueeze(0) for v in data]
    with torch.no_grad():
        output_dict = tester.model(*data)
        trans = output_dict['refined_transform'].cpu().numpy()

    # 估计法向量
    ref_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
    src_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))

    # 对源点云应用模型预测的变换
    src_cloud.transform(trans)

    # 将变换后的源点云转换为小球形式
    src_points_transformed = np.asarray(src_cloud.points)
    src_cloud_transformed_spheres = points_to_spheres(src_points_transformed, point_size=0.015, color=custom_yellow[0])

    # 保存变换后的点云（参考点云和变换后的源点云）
    transformed_combined_cloud = ref_cloud_spheres + src_cloud_transformed_spheres
    transformed_output_filename = "transformed_position_5.ply"
    o3d.io.write_triangle_mesh(transformed_output_filename, transformed_combined_cloud)
    print(f"Transformed 3D object saved as {transformed_output_filename}")
