import sys, os
sys.path.insert(0, '/home/tjw/DexGrasp-Anything')
import torch, numpy as np, pickle
from utils.handmodel import HandModel
from pytorch3d.ops import knn_points

d = torch.load('/home/tjw/DexGrasp-Anything/data/Grasp_anything/Grasp_anyting_shadowhand.pt', map_location='cpu')
scene = pickle.load(open('/home/tjw/DexGrasp-Anything/data/Grasp_anything/object_pcds_nors.pkl', 'rb'))

hm = HandModel('shadowhand', 'assets/urdf/sr_grasp_description/urdf/shadowhand.urdf',
               'assets/urdf/sr_grasp_description/meshes', batch_size=1, device='cuda')

# 每个物体取 2 条, 共验证最多 30 个物体
objs = sorted({m['object_name'] for m in d['metadata']})[:30]
mins, medians = [], []
per_obj_min = {}

from collections import defaultdict
by_obj = defaultdict(list)
for m in d['metadata']:
    by_obj[m['object_name']].append(m)

for obj in objs:
    recs = by_obj[obj][:2]
    for m in recs:
        q = torch.cat([m['translations'], m['joint_positions']]).unsqueeze(0).cuda().float()
        hm.update_kinematics(q)
        hand_pts = hm.get_surface_points(q).squeeze(0)
        obj_raw = torch.tensor(scene[obj][:, :3]).cuda().float()
        R = m['rotations'].cuda().float()
        obj_rot = torch.matmul(R.transpose(0, 1), obj_raw.transpose(0, 1)).transpose(0, 1)
        dd = knn_points(hand_pts.unsqueeze(0), obj_rot.unsqueeze(0)).dists[0, :, 0].sqrt()
        mins.append(dd.min().item()); medians.append(dd.median().item())
        per_obj_min.setdefault(obj, min(per_obj_min.get(obj, 1e9), dd.min().item()))

mins = np.array(mins); medians = np.array(medians)
print(f'验证样本数: {len(mins)}, 物体数: {len(objs)}')
print(f'最近距离 min : 中位数={np.median(mins):.4f}  平均={mins.mean():.4f}  最大={mins.max():.4f}')
print(f'距离中位数    : 中位数={np.median(medians):.4f}  平均={medians.mean():.4f}')
print(f'最近距离 < 2cm 的样本占比: {(mins < 0.02).mean()*100:.0f}%')
print(f'最近距离 < 5cm 的样本占比: {(mins < 0.05).mean()*100:.0f}%')
print(f'最近距离 > 10cm 的样本(可疑): {[(o, round(v,3)) for o, v in per_obj_min.items() if v > 0.10][:10]}')
