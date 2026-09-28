import os
import json
import pickle
import numpy as np
import trimesh

# ====================== 修改这里 ======================
ROOT        = "/home/tjw/DexGrasp-Anything/data/Grasp_anything/processedOriginalScale"   # 物体根目录
JSON_LIST   = "/home/tjw/DexGrasp-Anything/物体抽样清单v4.json"       # 只处理清单里的物体(all)
OUTPUT      = "/home/tjw/DexGrasp-Anything/data/Grasp_anything/object_pcds_nors.pkl"      # 输出文件
MESH_FILE   = "simplified.obj"   # 用哪个网格: simplified.obj(原始尺度) / normalized.obj(归一化) / coacd.obj(凸分解)
NUM_POINTS  = 10240              # 每个物体采样点数(>2048即可)
SEED        = 42                 # 固定随机种子, 保证可复现; 设 None 则不固定
# ======================================================

def main():
    # 1. 确定要处理的物体列表
    if os.path.exists(JSON_LIST):
        obj_names = json.load(open(JSON_LIST))["all"]
        print(f"[1/3] 使用清单中的 {len(obj_names)} 个物体: {JSON_LIST}")
    else:
        obj_names = [d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))]
        print(f"[1/3] 未找到清单, 使用文件夹中全部 {len(obj_names)} 个物体")

    # 2. 逐个物体采样点云 + 法线
    data, failed = {}, []
    for i, name in enumerate(obj_names):
        mesh_path = os.path.join(ROOT, name, "mesh", MESH_FILE)
        if not os.path.exists(mesh_path):
            failed.append((name, f"缺少 mesh/{MESH_FILE}"))
            continue
        mesh = trimesh.load(mesh_path, force="mesh", process=False)
        if SEED is not None:
            np.random.seed(SEED)
        points, face_idx = mesh.sample(NUM_POINTS, return_index=True)
        normals = mesh.face_normals[face_idx]
        # 每行 = [xyz | normal], 共6列, 与 DGA/Realdex 的 object_pcds_nors.pkl 格式一致
        # 注意: 作者数据的 pkl 是 float64, 保持一致避免采样时类型不匹配
        data[name] = np.hstack((points, normals)).astype(np.float64)
        if (i + 1) % 50 == 0 or i == len(obj_names) - 1:
            print(f"    ...已完成 {i + 1}/{len(obj_names)}")

    # 3. 保存
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "wb") as f:
        pickle.dump(data, f)
    print(f"[2/3] 已保存 {len(data)} 个物体 -> {OUTPUT}")
    print(f"[3/3] 每个物体点数: {NUM_POINTS}, 维度: 6 (xyz|normal)")

    if failed:
        print("\n以下物体缺失网格文件(未处理):")
        for n, e in failed[:20]:
            print(f"   {n}: {e}")
        print(f"   共 {len(failed)} 个")

if __name__ == "__main__":
    main()
