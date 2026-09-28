"""
把 res_diffuser.pkl 转成测评要的 .npy 格式。
- 一个物体 = 一个文件夹
- 一个抓取 = 一个 .npy（dict，三个 qpos 设成一样）
- 用 FK 把腕关节(WRJ)烤进朝向 + 按手掌位姿调 xyz
  (测评没腕关节 wrist=0，所以要把 WRJ 的影响算进朝向/位置)
"""
import pickle
import numpy as np
import os
import torch
from scipy.spatial.transform import Rotation
from utils.handmodel import get_handmodel

# ===== 写死的路径 =====
# PKL_PATH = '/home/zhaneven/DexGrasp-Anything/outputs/DexGraspNet/eval/final/resume/res_diffuser.pkl'
# SCALES_PATH = '/home/zhaneven/DexGrasp-Anything/data/DexGraspNet/scales.pkl'
# OUT_DIR = '/home/zhaneven/DexGrasp-Anything/outputs/DexGraspNet/eval/final/resume/graspdata'
# OBJ_PATH_BASE = 'assets/processedOriginalScale'   # obj_path 的前缀，按测评要求改



# PKL_PATH = '/home/tjw/DexGrasp-Anything/outputs/Grasp_anything/eval/final/my_model_obj/res_diffuser.pkl'
# OUT_DIR  = '/home/tjw/DexGrasp-Anything/outputs/Grasp_anything/eval/final/my_model_obj/graspdata'
# OBJ_PATH_BASE = 'assets/processedOriginalScale'

PKL_PATH = '/home/tjw/DexGrasp-Anything/outputs/DexGraspNet/eval/final/official_model_obj/res_diffuser.pkl'
OUT_DIR  = '/home/tjw/DexGrasp-Anything/outputs/DexGraspNet/eval/final/official_model_obj/graspdata'
OBJ_PATH_BASE = 'assets/DexGraspNet'














# ===== 物体位姿：原点 + 单位四元数 (固定) =====
OBJ_POSE = np.array([0., 0., 0., 1., 0., 0., 0.], dtype=np.float64)

# 测评手模型里「手掌根→手腕」的偏移(米)，+z 指向指尖方向。
# eval_xyz = palm_pos - palm_R @ D_PALM_TO_WRIST  (手腕在手掌后方)
# 若测评里方向反了(手飘走)，就把 D_PALM_TO_WRIST 改成 [0,0,-0.034] 重测
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

# ===== 加载 =====
res = pickle.load(open(PKL_PATH, 'rb'))

# ===== 转换 =====
n_obj, n_grasp = 0, 0
for obj, grasps in res['sample_qpos'].items():
    obj_dir = os.path.join(OUT_DIR, obj)
    os.makedirs(obj_dir, exist_ok=True)

    obj_scale = 1.0
    obj_path = f'{OBJ_PATH_BASE}/{obj}'

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
