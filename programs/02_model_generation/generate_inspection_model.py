"""
单模型验证：20mm 单胞, 2x2x2 = 40mm, rho=0.30, w=0.6

隐式曲面带方法：从 TPMS 隐式场提取薄壳结构，可清晰看到晶格。
壁厚通过二分法校准以匹配目标密度。

隐式函数参考 python_voxel_fem_ipc_compression.py
"""

import sys
sys.path.insert(0, r"E:\code\new\01_Programs")
sys.path.insert(0, r"E:\code\new\01_Programs\02_voxel-based FEM")

import os
import numpy as np
import trimesh
from skimage import measure
from python_voxel_fem_ipc_compression import F_BCC, F_IWP_modified

# ============================================================
# 参数
# ============================================================
CELL_SIZE = 20.0
CELLS = (2, 2, 2)
N_PER_CELL = 50
ALPHA = 5.0
TARGET_RHO = 0.30
TARGET_W = 0.6
OUTPUT_DIR = r"E:\code\new\02_Models"

# 壁厚校准
THICK_MIN = 0.3
THICK_MAX = 3.0
CALIB_ITER = 14


def sdf_to_solid(phi, dx, thickness_mm, eps=1e-9):
    """隐式面扩展为薄壳实体：|phi/|grad(phi)|| <= thickness/2"""
    gx, gy, gz = np.gradient(phi, dx, dx, dx, edge_order=2)
    grad_norm = np.sqrt(gx**2 + gy**2 + gz**2) + eps
    signed_distance = phi / grad_norm
    return np.abs(signed_distance) <= thickness_mm / 2.0


def calibrate_thickness(phi_bcc, phi_iwp, dx, target_rho, w):
    """二分法校准壁厚以匹配目标密度"""
    lo, hi = THICK_MIN, THICK_MAX

    for _ in range(CALIB_ITER):
        mid = (lo + hi) / 2.0
        bcc = sdf_to_solid(phi_bcc, dx, mid)
        iwp = sdf_to_solid(phi_iwp, dx, mid)

        if w <= 0:
            rho = float(np.mean(iwp))
        elif w >= 1:
            rho = float(np.mean(bcc))
        else:
            rho = float(np.mean(bcc | iwp))

        if rho < target_rho:
            lo = mid
        else:
            hi = mid

    return (lo + hi) / 2.0


def generate_single_model():
    Lx = CELL_SIZE * CELLS[0]
    nx = N_PER_CELL * CELLS[0]
    dx = Lx / nx

    # 全局坐标
    x = np.linspace(dx / 2.0, Lx - dx / 2.0, nx)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    Xc, Yc, Zc = X - Lx / 2, Y - Lx / 2, Z - Lx / 2
    kx = ky = kz = 2.0 * np.pi / CELL_SIZE

    # TPMS 隐式场（phi=0 为表面）
    # w 控制 BCC/IWP 比例：通过 threshold 偏移
    # 用 t2=0 求场，再按 w 调整各相贡献
    # 这里简化：直接用 threshold=0 的场，壁厚控制密度
    phi_bcc = F_BCC(Xc, Yc, Zc, kx, ky, kz)
    phi_iwp = F_IWP_modified(Xc, Yc, Zc, kx, ky, kz, t2=0.0, alpha=ALPHA)

    # 校准壁厚
    wall_thickness = calibrate_thickness(phi_bcc, phi_iwp, dx, TARGET_RHO, TARGET_W)
    print(f"校准壁厚: {wall_thickness:.4f} mm")

    # 生成实体
    solid_bcc = sdf_to_solid(phi_bcc, dx, wall_thickness)
    solid_iwp = sdf_to_solid(phi_iwp, dx, wall_thickness)

    # 按 w 混合
    if TARGET_W <= 0:
        solid = solid_iwp
    elif TARGET_W >= 1:
        solid = solid_bcc
    else:
        solid = solid_bcc | solid_iwp

    rho = float(np.mean(solid))
    rho_bcc = float(np.mean(solid_bcc))
    rho_iwp = float(np.mean(solid_iwp))
    w_actual = rho_bcc / rho if rho > 1e-12 else 0.0

    print(f"\nrho={rho:.4f} (target {TARGET_RHO})")
    print(f"w={w_actual:.4f} (target {TARGET_W})")
    print(f"rho_bcc={rho_bcc:.4f}, rho_iwp={rho_iwp:.4f}")

    # STL
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pad = np.pad(solid, 1, constant_values=False)
    v, f, _, _ = measure.marching_cubes(pad.astype(np.float32), 0.5, spacing=(dx, dx, dx))
    v -= dx
    mesh = trimesh.Trimesh(vertices=v, faces=f, process=True)
    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    trimesh.repair.fill_holes(mesh)

    stl_path = os.path.join(OUTPUT_DIR, f"IPC_rho{TARGET_RHO:.2f}_w{TARGET_W:.1f}_{Lx:.0f}mm_{CELLS[0]}x{CELLS[1]}x{CELLS[2]}.stl")
    mesh.export(stl_path)

    size_mb = os.path.getsize(stl_path) / (1024 * 1024)
    print(f"\nSTL: {stl_path}")
    print(f"vertices={len(mesh.vertices)}, faces={len(mesh.faces)}")
    print(f"watertight={mesh.is_watertight}")
    print(f"volume={abs(mesh.volume):.1f} mm3")
    print(f"size={size_mb:.1f} MB")


if __name__ == "__main__":
    generate_single_model()
