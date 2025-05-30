import open3d as o3d

# 创建一个测试点云
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector([[1, 1, 1], [2, 2, 2]])

# 使用 Open3D 渲染
o3d.visualization.draw_geometries([pcd])
