import os
import glob
import json
import numpy as np
import torch

# ====================== 修改这里 ======================
GRASPDATA  = "/home/tjw/DexGrasp-Anything/graspdata"                    # 抓取数据目录
OUTPUT     = "/home/tjw/DexGrasp-Anything/data/Grasp_anything/Grasp_anyting_shadowhand.pt"  # 输出
JSON_LIST  = "/home/tjw/DexGrasp-Anything/物体抽样清单v4.json"            # 物体清单(可选)
# ======================================================

def quat_wxyz_to_mat(q):
    """ 四元数(wxyz) -> 旋转矩阵 """
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])

def main():
    if os.path.exists(JSON_LIST):
        obj_names = json.load(open(JSON_LIST))["all"]
        print(f"[1/3] 使用清单中的 {len(obj_names)} 个物体")
    else:
        obj_names = [d for d in os.listdir(GRASPDATA) if os.path.isdir(os.path.join(GRASPDATA, d))]
        print(f"[1/3] 未找到清单, 使用全部 {len(obj_names)} 个物体")

    metadata = []
    num_per_object = {}
    total = 0
    skipped = []

    for obj in obj_names:
        npy_files = glob.glob(os.path.join(GRASPDATA, obj, "*.npy"))
        npy_files.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
        if not npy_files:
            skipped.append((obj, "无 npy"))
            continue
        num_per_object[obj] = len(npy_files)
        for f in npy_files:
            a = np.load(f, allow_pickle=True).item()
            q = np.asarray(a["grasp_qpos"], dtype=np.float32)   # (29,) = xyz + quat(wxyz) + 22关节
            if q.shape[0] != 29:
                skipped.append((obj, f"qpos 维度 {q.shape[0]} != 29"))
                continue
            xyz   = q[0:3]                                        # 手(手腕参考)在物体系的位置
            quat  = q[3:7]                                        # 手朝向 四元数(wxyz)
            joints22 = q[7:29]                                    # 22 个手指关节
            a_mat = quat_wxyz_to_mat(quat)                        # rotations = 手的旋转矩阵 a

            # 诊断脚本实测(6种组合对比): translations = a^T @ xyz - p 最优(手贴物体)
            # - 乘 a^T: 项目约定 Y=a(x+b), 加载器把物体转 a^T, 手在旋转坐标系里的位置 = a^T @ xyz
            # - 减 p:   参考点从前臂改到手腕, WRJ2 origin=(0,-0.010,0.213), 前臂 = 手腕 - p
            p_w = np.array([0.0, -0.010, 0.213], dtype=np.float32)
            trans = a_mat.T @ xyz - p_w
            # 24 关节 = 2 个腕关节(WRJ1/WRJ2 置 0) + 22 个手指关节
            joints24 = np.concatenate([np.zeros(2, dtype=np.float32), joints22])

            metadata.append({
                "rotations":        torch.from_numpy(a_mat.astype(np.float32)),
                "translations":     torch.from_numpy(trans.astype(np.float32)),
                "joint_positions":  torch.from_numpy(joints24.astype(np.float32)),
                "object_name":      obj,
                "scale":            float(a.get("obj_scale", 1.0)),
            })
            total += 1

    # 2. 组装并保存
    grasp_dataset = {
        "info": {"num_per_object": num_per_object},
        "metadata": metadata,
    }
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    torch.save(grasp_dataset, OUTPUT)

    print(f"[2/3] 共 {total} 条抓取记录 -> {OUTPUT}")
    print(f"[3/3] 物体数: {len(num_per_object)}")
    if skipped:
        print("\n跳过的问题样本:")
        for n, e in skipped[:20]:
            print(f"   {n}: {e}")
        print(f"   共 {len(skipped)} 个")

    # 3. 验证第一条
    if metadata:
        m = metadata[0]
        print("\n第一条记录:")
        print(f"  object_name: {m['object_name']}")
        print(f"  rotations:   {m['rotations'].shape}")
        print(f"  translations:{m['translations'].tolist()}")
        print(f"  joint_positions[:4]: {m['joint_positions'][:4].tolist()}")

if __name__ == "__main__":
    main()
