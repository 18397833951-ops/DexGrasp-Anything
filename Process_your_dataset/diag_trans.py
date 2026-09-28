import sys, os
sys.path.insert(0, '/home/tjw/DexGrasp-Anything')
import torch, numpy as np, pickle, glob
from utils.handmodel import HandModel
from pytorch3d.ops import knn_points

obj_dir = '/home/tjw/DexGrasp-Anything/graspdata'
objs = sorted([d for d in os.listdir(obj_dir) if os.path.isdir(os.path.join(obj_dir, d))])
obj_name = objs[0]
f = sorted(glob.glob(os.path.join(obj_dir, obj_name, '*.npy')))[0]
a = np.load(f, allow_pickle=True).item()
q = a['grasp_qpos']
xyz = q[0:3].astype(np.float64)
quat = q[3:7].astype(np.float64)
joints22 = q[7:29]
joints24 = np.concatenate([[0.0, 0.0], joints22]).astype(np.float32)

def quat_to_mat(qq):
    w, x, y, z = qq
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])

R = quat_to_mat(quat)
p = np.array([0.0, -0.010, 0.213])
cands = {
    'xyz          ': xyz,
    'xyz - p      ': xyz - p,
    'xyz + p      ': xyz + p,
    'aT*xyz       ': R.T @ xyz,
    'aT*xyz - p   ': R.T @ xyz - p,
    'aT*xyz + p   ': R.T @ xyz + p,
}

hm = HandModel('shadowhand', 'assets/urdf/sr_grasp_description/urdf/shadowhand.urdf',
               'assets/urdf/sr_grasp_description/meshes', batch_size=1, device='cuda')
obj_raw = torch.tensor(pickle.load(open('data/Grasp_anything/object_pcds_nors.pkl', 'rb'))[obj_name][:, :3]).cuda().float()

print(f'物体: {obj_name} | 原始 xyz={np.round(xyz,4)} | quat={np.round(quat,4)}')
for name, trans in cands.items():
    qq = torch.tensor(np.concatenate([trans, joints24])).unsqueeze(0).cuda().float()
    hm.update_kinematics(qq)
    hand_pts = hm.get_surface_points(qq).squeeze(0)
    obj_rot = torch.matmul(torch.tensor(R).cuda().float().T, obj_raw.transpose(0, 1)).transpose(0, 1)
    dd = knn_points(hand_pts.unsqueeze(0), obj_rot.unsqueeze(0)).dists[0, :, 0].sqrt()
    print(f'{name}: median={dd.median().item():.4f} m  min={dd.min().item():.4f} m')
