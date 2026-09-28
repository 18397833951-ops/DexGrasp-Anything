import sys
import os
# 把项目根目录加入搜索路径(脚本在子目录里运行时能 import utils)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pickle
from utils.handmodel import HandModel

d = torch.load('/home/tjw/DexGrasp-Anything/data/Grasp_anything/Grasp_anyting_shadowhand.pt', map_location='cpu')
m = d['metadata'][0]

q = torch.cat([m['translations'], m['joint_positions']]).unsqueeze(0).cuda()
hm = HandModel('shadowhand', 'assets/urdf/sr_grasp_description/urdf/shadowhand.urdf',
               'assets/urdf/sr_grasp_description/meshes', batch_size=1, device='cuda')
hm.update_kinematics(q)
hand_pts = hm.get_surface_points(q).squeeze(0)

obj = torch.tensor(
    pickle.load(open('/home/tjw/DexGrasp-Anything/data/Grasp_anything/object_pcds_nors.pkl', 'rb'))[m['object_name']][:, :3]
).unsqueeze(0).cuda().float()

# 关键: 项目约定加载器会把物体点云乘 rotations.T (a^T) 再和手比较
# 验证也必须这么做, 否则手相对的是"未旋转物体"
R = m['rotations'].cuda().float()
obj_rot = torch.matmul(R.transpose(0, 1), obj.transpose(1, 2)).transpose(1, 2)

from pytorch3d.ops import knn_points
dd = knn_points(hand_pts.unsqueeze(0), obj_rot).dists[0, :, 0].sqrt()

print('物体:', m['object_name'])
print('手表面点数:', len(hand_pts))
print('手->物体最近距离 中位数: {:.4f} 米'.format(dd.median().item()))
print('手->物体最近距离 最小值: {:.4f} 米'.format(dd.min().item()))
print('判断: 中位数 < 0.01米 ≈ 手贴住物体(正确); > 0.05米 = 手没贴上(参考点/坐标有问题)')
