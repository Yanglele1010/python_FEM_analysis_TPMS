"""
批量生成打印检测 STL 模型

隐式曲面带方法，20mm 单胞，2x2x2 = 40mm，50 samples/cell
rho=0.30, w=0.0/0.2/0.4/0.6/0.8/1.0
"""

import sys
sys.path.insert(0, r"E:\code\new\01_Programs\02_voxel-based FEM")

import os
import csv
import numpy as np
import trimesh
from skimage import measure
from python_voxel_fem_ipc_compression import F_BCC, F_IWP_modified

# ============================================================
# 参数
# ============================================================
CELL_SIZE = 10.0
CELLS = (4, 4, 4)
N_PER_CELL = 50
ALPHA = 5.0
TARGET_RHO = 0.30
W_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
OUTPUT_DIR = r"E:\code\new\02_Models"

THICK_MIN = 0.3
THICK_MAX = 3.0
CALIB_ITER = 14


def sdf_to_solid(phi, dx, thickness_mm, eps=1e-9):
    gx, gy, gz = np.gradient(phi, dx, dx, dx, edge_order=2)
    grad_norm = np.sqrt(gx**2 + gy**2 + gz**2) + eps
    signed_distance = phi / grad_norm
    return np.abs(signed_distance) <= thickness_mm / 2.0


def calibrate_thickness(phi_bcc, phi_iwp, dx, target_rho, w):
    """校准壁厚：BCC 壁厚 = w*T, IWP 壁厚 = (1-w)*T，控制密度比"""
    lo, hi = THICK_MIN, THICK_MAX
    for _ in range(CALIB_ITER):
        mid = (lo + hi) / 2.0
        thick_bcc = w * mid if w > 0 else 0.0
        thick_iwp = (1.0 - w) * mid if w < 1 else 0.0

        if w <= 0:
            rho = float(np.mean(sdf_to_solid(phi_iwp, dx, thick_iwp)))
        elif w >= 1:
            rho = float(np.mean(sdf_to_solid(phi_bcc, dx, thick_bcc)))
        else:
            bcc = sdf_to_solid(phi_bcc, dx, thick_bcc)
            iwp = sdf_to_solid(phi_iwp, dx, thick_iwp)
            rho = float(np.mean(bcc | iwp))
        if rho < target_rho:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def voxel_to_stl(solid, dx, filename):
    pad = np.pad(solid, 1, constant_values=False)
    v, f, _, _ = measure.marching_cubes(pad.astype(np.float32), 0.5, spacing=(dx, dx, dx))
    v -= dx
    m = trimesh.Trimesh(vertices=v, faces=f, process=True)
    m.update_faces(m.unique_faces())
    m.update_faces(m.nondegenerate_faces())
    m.remove_unreferenced_vertices()
    m.fix_normals()
    trimesh.repair.fill_holes(m)
    m.export(filename)
    return m


def generate_one(w, output_dir):
    Lx = CELL_SIZE * CELLS[0]
    nx = N_PER_CELL * CELLS[0]
    dx = Lx / nx

    x = np.linspace(dx / 2.0, Lx - dx / 2.0, nx)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    Xc, Yc, Zc = X - Lx / 2, Y - Lx / 2, Z - Lx / 2
    kx = ky = kz = 2.0 * np.pi / CELL_SIZE

    phi_bcc = F_BCC(Xc, Yc, Zc, kx, ky, kz)
    phi_iwp = F_IWP_modified(Xc, Yc, Zc, kx, ky, kz, t2=0.0, alpha=ALPHA)

    T = calibrate_thickness(phi_bcc, phi_iwp, dx, TARGET_RHO, w)
    thick_bcc = w * T if w > 0 else 0.0
    thick_iwp = (1.0 - w) * T if w < 1 else 0.0

    solid_bcc = sdf_to_solid(phi_bcc, dx, thick_bcc) if w > 0 else np.zeros((nx, nx, nx), dtype=bool)
    solid_iwp = sdf_to_solid(phi_iwp, dx, thick_iwp) if w < 1 else np.zeros((nx, nx, nx), dtype=bool)

    solid = solid_bcc | solid_iwp

    rho = float(np.mean(solid))
    rho_bcc = float(np.mean(solid_bcc))
    rho_iwp = float(np.mean(solid_iwp))
    overlap = float(np.mean(solid_bcc & solid_iwp))
    w_actual = rho_bcc / rho if rho > 1e-12 else 0.0

    prefix = f"IPC_rho{TARGET_RHO:.2f}_w{w:.1f}_{Lx:.0f}mm_{CELLS[0]}x{CELLS[1]}x{CELLS[2]}"
    stl_path = os.path.join(output_dir, prefix + ".stl")

    mesh = voxel_to_stl(solid, dx, stl_path)
    size_mb = os.path.getsize(stl_path) / (1024 * 1024)

    info = {
        "cell_size_mm": CELL_SIZE,
        "cells": f"{CELLS[0]}x{CELLS[1]}x{CELLS[2]}",
        "total_size_mm": Lx,
        "n_per_cell": N_PER_CELL,
        "dx_mm": round(dx, 4),
        "target_rho": TARGET_RHO,
        "target_w": w,
        "T_base": round(T, 4),
        "wall_bcc_mm": round(thick_bcc, 4),
        "wall_iwp_mm": round(thick_iwp, 4),
        "alpha": ALPHA,
        "actual_rho": round(rho, 4),
        "actual_w": round(w_actual, 4),
        "rho_bcc": round(rho_bcc, 4),
        "rho_iwp": round(rho_iwp, 4),
        "overlap": round(overlap, 4),
        "n_vertices": len(mesh.vertices),
        "n_faces": len(mesh.faces),
        "watertight": mesh.is_watertight,
        "volume_mm3": round(abs(mesh.volume), 2),
        "file_size_mb": round(size_mb, 1),
        "stl_file": os.path.basename(stl_path),
    }

    return info


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results = []

    Lx = CELL_SIZE * CELLS[0]
    print(f"配置: {CELL_SIZE}mm 单胞, {CELLS[0]}x{CELLS[1]}x{CELLS[2]} = {Lx}mm")
    print(f"采样: {N_PER_CELL}/cell, alpha={ALPHA}, target_rho={TARGET_RHO}")
    print(f"{'='*60}")

    for w in W_VALUES:
        print(f"\nw={w:.1f} ...", end=" ", flush=True)
        info = generate_one(w, OUTPUT_DIR)
        all_results.append(info)
        print(f"rho={info['actual_rho']:.4f}, w={info['actual_w']:.4f}, "
              f"bcc={info['wall_bcc_mm']:.3f}mm, iwp={info['wall_iwp_mm']:.3f}mm, "
              f"faces={info['n_faces']}, {info['file_size_mb']:.1f}MB")

    csv_path = os.path.join(OUTPUT_DIR, "models_record.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n{'='*60}")
    print(f"完成! {len(all_results)} 个模型")
    print(f"参数记录: {csv_path}")
    print(f"STL 目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
