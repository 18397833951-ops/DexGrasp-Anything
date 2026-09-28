"""
把 res_diffuser.pkl 转成测评要的 .npy 格式。
- 一个物体 = 一个文件夹
- 一个抓取 = 一个 .npy（dict，三个 qpos 设成一样）
- 用 FK 把腕关节(WRJ)烤进朝向 + 按手掌位姿调 xyz
  (测评没腕关节 wrist=0，所以要把 WRJ 的影响算进朝向/位置)
- obj_scale 自动逐物体计算 = DGA抓取帧对角线 / DGBench mesh对角线
  (DexGraspNet 的抓取帧 = DGA点云 × scales.pkl；其他数据集 = DGA点云)
"""
import pickle
import numpy as np
import os
import torch
import trimesh
from scipy.spatial.transform import Rotation
from utils.handmodel import get_handmodel

# ================= 每次转换只改这 3 个 =================
# DATASET  = 'DexGraspNet'  # Realdex / MultiDex / DexGRAB / DexGraspNet / Grasp_anyting —— 这个 pkl 是哪个数据集的物体
# PKL_PATH = '/home/tjw/DexGrasp-Anything/outputs/DexGraspNet/eval/final/official_model_obj/res_diffuser.pkl'
# OUT_DIR  = '/home/tjw/DexGrasp-Anything/outputs/DexGraspNet/eval/final/official_model_obj/graspdata'
# DATASET  = 'DexGraspNet'
# PKL_PATH = '/home/tjw/DexGrasp-Anything/outputs/DexGraspNet/eval/final/official_model_obj/res_diffuser.pkl'
# OUT_DIR  = '/home/tjw/DexGrasp-Anything/outputs/DexGraspNet/eval/final/official_model_obj/graspdata'

# DATASET  = 'Grasp_anything'  
# PKL_PATH = '/home/tjw/DexGrasp-Anything/outputs/MultiDex/eval/final/myobj/res_diffuser.pkl'
# OUT_DIR  = '/home/tjw/DexGrasp-Anything/outputs/MultiDex/eval/final/myobj/graspdata'
DATASET = 'Grasp_anything'
PKL_PATH = '/home/tjw/DexGrasp-Anything/res_diffuser_quat.pkl'
OUT_DIR  = '/home/tjw/DexGrasp-Anything/zgraspdata'


# ========================================================

# ===== 数据集配置（obj_scale 自动算用，一般不用改） =====
DGBENCH_DIR = '/home/tjw/DexGraspBench'   # DGBench 根目录
DATASET_CONFIG = {
    'Realdex':       {'pcd': '/home/tjw/DexGrasp-Anything/data/Realdex/object_pcds_nors.pkl',
                      'dgb': 'assets/Realdex', 'scales': None},
    'MultiDex':      {'pcd': '/home/tjw/DexGrasp-Anything/data/MultiDex_UR/object_pcds_nors.pkl',
                      'dgb': 'assets/MultiDex_UR', 'scales': None},
    'DexGRAB':       {'pcd': '/home/tjw/DexGrasp-Anything/data/DexGRAB/object_pcds_nors.pkl',
                      'dgb': 'assets/DexGRAB', 'scales': None},
    'DexGraspNet':   {'pcd': '/home/tjw/DexGrasp-Anything/data/DexGraspNet/object_pcds_nors.pkl',
                      'dgb': 'assets/DexGraspNet',
                      'scales': '/home/tjw/DexGrasp-Anything/data/DexGraspNet/scales.pkl'},
    'Grasp_anything': {'pcd': '/home/tjw/DexGrasp-Anything/data/Grasp_anything/object_pcds_nors.pkl',
                      'dgb': 'assets/processedOriginalScale', 'scales': None},
}
cfg = DATASET_CONFIG[DATASET]

# ===== 物体位姿：原点 + 单位四元数 (固定) =====
OBJ_POSE = np.array([0., 0., 0., 1., 0., 0., 0.], dtype=np.float64)

# 测评手模型里「手掌根→手腕」的偏移(米)，+z 指向指尖方向。
# eval_xyz = palm_pos - palm_R @ D_PALM_TO_WRIST  (手腕在手掌后方)
D_PALM_TO_WRIST = np.array([0., 0., 0.034])

# ===== 加载手模型(FK 用，只加载一次) =====
hand = get_handmodel(batch_size=1, device='cuda')

def wxyz_to_matrix(q):
    """(w,x,y,z) 四元数 -> 3x3 矩阵"""
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()  # scipy 要 (x,y,z,w)

def matrix_to_wxyz(R):
    """3x3 矩阵 -> (w,x,y,z) 四元数"""
    xyzw = Rotation.from_matrix(R).as_quat()                          # scipy 给 (x,y,z,w)
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])             # 转成 (w,x,y,z)

def palm_world_pose(xyz, R_forearm, joints24):
    """FK(含WRJ) 算手掌的世界位姿。返回 (palm_pos, palm_R) 都是 world 系。"""
    hand.global_translation = torch.tensor(xyz, device='cuda').float().unsqueeze(0)
    hand.global_rotation = torch.tensor(R_forearm, device='cuda').float().unsqueeze(0)
    hand.current_status = hand.robot.forward_kinematics(
        torch.tensor(joints24, device='cuda').float().unsqueeze(0))
    tm = hand.current_status['palm'].get_matrix()[0].detach().cpu().numpy()  # palm 在前臂根坐标系
    palm_pos = R_forearm @ tm[:3, 3] + xyz        # 世界位置(含WRJ)
    palm_R   = R_forearm @ tm[:3, :3]             # 世界朝向(含WRJ)
    return palm_pos, palm_R

# ===== 预加载 DGA 点云 + (可选) scales.pkl，用于逐物体算 obj_scale =====
DGA_PCD = pickle.load(open(cfg['pcd'], 'rb'))
SCALES = pickle.load(open(cfg['scales'], 'rb')) if cfg['scales'] else None

def obj_scale_for(obj):
    """obj_scale = DGA抓取帧对角线 / DGBench mesh对角线"""
    if obj not in DGA_PCD:
        print(f'[WARN] {obj} 不在 DGA 点云里, obj_scale=1.0')
        return 1.0
    pcd = DGA_PCD[obj][:, :3]
    pcd_diag = float(np.sqrt(((pcd.max(0) - pcd.min(0)) ** 2).sum()))
    s = float(SCALES[obj]) if SCALES else 1.0            # DexGraspNet 采样时乘 scales.pkl
    grasp_frame_diag = pcd_diag * s
    mesh_path = os.path.join(DGBENCH_DIR, cfg['dgb'], obj, 'mesh', 'simplified.obj')
    try:
        m = trimesh.load(mesh_path, force='mesh', process=False)
        dgb_diag = float(np.sqrt(((m.vertices.max(0) - m.vertices.min(0)) ** 2).sum()))
        return grasp_frame_diag / dgb_diag
    except Exception as e:
        print(f'[WARN] 读不到 DGBench mesh {mesh_path}: {e}, obj_scale=1.0')
        return 1.0

# ===== 加载 =====
res = pickle.load(open(PKL_PATH, 'rb'))

# ===== 转换 =====
n_obj, n_grasp = 0, 0
for obj, grasps in res['sample_qpos'].items():
    obj_dir = os.path.join(OUT_DIR, obj)
    os.makedirs(obj_dir, exist_ok=True)

    obj_scale = obj_scale_for(obj)
    obj_path = f'{cfg["dgb"]}/{obj}'

    for i, qpos in enumerate(grasps):                 # qpos: 31 维 [xyz3 + quat4(w,x,y,z) + 关节24]
        xyz       = qpos[:3]                           # 前臂根位置
        quat      = qpos[3:7]                          # 前臂朝向 (w,x,y,z)
        joints24  = qpos[7:31]                         # 24关节(含 WRJ: idx0,1 = WRJ2,WRJ1)
        R_forearm = wxyz_to_matrix(quat)

        # 1) FK 算手掌世界位姿(含 WRJ 弯折)
        palm_pos, palm_R = palm_world_pose(xyz, R_forearm, joints24)

        # 2) eval: WRJ 烤进朝向；xyz 按手掌位姿调(手腕在手掌后方)
        eval_quat = matrix_to_wxyz(palm_R)                         # 手掌朝向(含WRJ)
        eval_xyz  = palm_pos - palm_R @ D_PALM_TO_WRIST            # 手腕 = 手掌 - 退一个偏移
        joints22  = qpos[9:31]                                     # 丢掉 WRJ(idx7,8)

        grasp_qpos = np.concatenate([eval_xyz, eval_quat, joints22]).astype(np.float32)

        data = {
            'obj_pose': OBJ_POSE,
            'obj_scale': obj_scale,
            'obj_path': obj_path,
            'pregrasp_qpos': grasp_qpos,
            'grasp_qpos': grasp_qpos,
            'squeeze_qpos': grasp_qpos,
        }
        np.save(os.path.join(obj_dir, f'{i}.npy'), data)
        n_grasp += 1
    n_obj += 1

print(f'done: {n_obj} 个物体, 共 {n_grasp} 个抓取 -> {OUT_DIR}')
