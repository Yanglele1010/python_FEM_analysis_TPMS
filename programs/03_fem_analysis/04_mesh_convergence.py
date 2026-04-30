#!/usr/bin/env python3
"""
网格无关性验证脚本

功能：
1. 测试不同网格分辨率下的有限元分析结果
2. 验证网格收敛性
3. 生成网格收敛性报告

测试模型：
- rho*=0.30, w=0.2
- rho*=0.30, w=0.5
- rho*=0.30, w=0.8

测试分辨率：
- N_PER_CELL_FE = 12
- N_PER_CELL_FE = 16
- N_PER_CELL_FE = 20
"""

import os
import csv
import time
import numpy as np

from python_voxel_fem_ipc_compression import (
    F_BCC,
    F_IWP_modified,
    generate_ipc_voxels_threshold,
    solve_compression,
)

# 基础参数
CELL_SIZE = 20.0
CELLS = (2, 2, 2)
ALPHA = 5.0

# 查找表分辨率，越大越准，但越慢
N_PER_CELL_LOOKUP = 16

# 材料参数
E_SOLID = 2200.0
NU_SOLID = 0.35

# 40 mm 高度，-0.4 mm 位移约为 1% 压缩应变
DISP = -0.4

# ============================================================# 读取查找表# ============================================================
def read_lookup_table(csv_file):
    rows = []

    with open(csv_file, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows.append({
                "t1": float(row["t1"]),
                "t2": float(row["t2"]),
                "rho_total": float(row["rho_total"]),
                "rho_bcc": float(row["rho_bcc"]),
                "rho_iwp": float(row["rho_iwp"]),
                "w_union": float(row["w_union"]),
                "w_phase": float(row["w_phase"]),
                "overlap": float(row["overlap"]),
                "active_elements": int(float(row["active_elements"])),
            })

    return rows

# ============================================================# 按目标 rho,w 从查找表中找最近的 t1,t2# ============================================================
def find_best_t1_t2(
    lookup_rows,
    target_rho,
    target_w,
    rho_scale=0.01,
    w_scale=0.05,
    overlap_weight=0.0,
):
    best = None
    best_score = 1e99

    for row in lookup_rows:
        rho = row["rho_total"]
        w = row["w_union"]
        overlap = row["overlap"]

        score_rho = ((rho - target_rho) / rho_scale) ** 2
        score_w = ((w - target_w) / w_scale) ** 2
        score_overlap = overlap_weight * overlap ** 2

        score = score_rho + score_w + score_overlap

        if score < best_score:
            best_score = score
            best = row.copy()
            best["score"] = float(score)

    return best

# ============================================================# 运行单个模型的有限元分析# ============================================================
def run_fem_for_model(target_rho, target_w, n_per_cell_fe, lookup_rows):
    print(f"\n================================")
    print(f"Model: rho*={target_rho:.2f}, w={target_w:.1f}")
    print(f"Resolution: N_PER_CELL_FE={n_per_cell_fe}")
    
    # 查找最佳t1,t2参数
    best = find_best_t1_t2(
        lookup_rows,
        target_rho=target_rho,
        target_w=target_w,
        rho_scale=0.01,
        w_scale=0.05,
        overlap_weight=0.0,
    )

    t1 = best["t1"]
    t2 = best["t2"]

    print(f"查表结果：")
    print(f"t1 = {t1:.4f}, t2 = {t2:.4f}")
    print(f"actual rho* = {best['rho_total']:.4f}")
    print(f"actual w    = {best['w_union']:.4f}")
    print(f"overlap     = {best['overlap']:.4f}")
    print("开始 FEM...")

    start = time.time()

    # 生成体素模型
    solid, size, elem_size = generate_ipc_voxels_threshold(
        cell_size=CELL_SIZE,
        cells=CELLS,
        n_per_cell_fe=n_per_cell_fe,
        t1=t1,
        t2=t2,
        alpha=ALPHA,
        keep_largest=True,
    )

    # 计算活跃单元数
    active_elements = np.sum(solid)

    # 进行有限元分析
    result = solve_compression(
        solid=solid,
        size=size,
        elem_size=elem_size,
        E=E_SOLID,
        nu=NU_SOLID,
        prescribed_disp=DISP,
    )

    end = time.time()
    runtime = end - start

    print(f"结果：")
    print(f"E_eff = {result['E_eff']:.4f} MPa")
    print(f"reaction_force = {result['reaction_force']:.4f} N")
    print(f"active_elements = {active_elements}")
    print(f"runtime = {runtime:.2f} s")
    print("================================")

    return {
        "target_rho": target_rho,
        "target_w": target_w,
        "N_PER_CELL_FE": n_per_cell_fe,
        "t1": t1,
        "t2": t2,
        "actual_rho": best["rho_total"],
        "actual_w": best["w_union"],
        "active_elements": active_elements,
        "E_eff": result["E_eff"],
        "reaction_force": result["reaction_force"],
        "runtime": runtime
    }

# ============================================================# 主函数# ============================================================
def main():
    # 输入文件
    lookup_csv = "lookup_table_cell20_fine.csv"
    output_csv = "mesh_convergence_results.csv"

    # 检查文件是否存在
    if not os.path.exists(lookup_csv):
        print(f"错误: 找不到文件 {lookup_csv}")
        return

    print(f"读取查找表: {lookup_csv}")
    lookup_rows = read_lookup_table(lookup_csv)
    print(f"读取到 {len(lookup_rows)} 条记录")

    # 测试模型
    test_models = [
        (0.30, 0.2),
        (0.30, 0.5),
        (0.30, 0.8)
    ]

    # 测试分辨率
    resolutions = [12, 16, 20]

    # 记录结果
    results = []

    # 运行所有测试
    for target_rho, target_w in test_models:
        for n_per_cell_fe in resolutions:
            result = run_fem_for_model(target_rho, target_w, n_per_cell_fe, lookup_rows)
            results.append(result)

    # 保存结果
    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "target_rho",
            "target_w",
            "N_PER_CELL_FE",
            "t1",
            "t2",
            "actual_rho",
            "actual_w",
            "active_elements",
            "E_eff",
            "reaction_force",
            "runtime"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n网格无关性验证完成！")
    print(f"结果保存到: {output_csv}")

    # 分析结果
    print("\n================================")
    print("网格收敛性分析")
    print("================================")

    for target_rho, target_w in test_models:
        print(f"\nModel: rho*={target_rho:.2f}, w={target_w:.1f}")
        
        # 提取不同分辨率的结果
        model_results = [r for r in results if abs(r['target_rho'] - target_rho) < 1e-6 and abs(r['target_w'] - target_w) < 1e-6]
        model_results.sort(key=lambda x: x['N_PER_CELL_FE'])
        
        # 计算E_eff变化率
        for i in range(1, len(model_results)):
            prev = model_results[i-1]
            curr = model_results[i]
            change = abs(curr['E_eff'] - prev['E_eff']) / prev['E_eff'] * 100
            print(f"N={prev['N_PER_CELL_FE']} → N={curr['N_PER_CELL_FE']}: E_eff变化率 = {change:.2f}%")
        
        # 检查16→20的变化率
        if len(model_results) >= 3:
            n16 = [r for r in model_results if r['N_PER_CELL_FE'] == 16][0]
            n20 = [r for r in model_results if r['N_PER_CELL_FE'] == 20][0]
            change_16_20 = abs(n20['E_eff'] - n16['E_eff']) / n16['E_eff'] * 100
            print(f"16→20变化率: {change_16_20:.2f}%")
            
            if change_16_20 < 5:
                print("✓ 网格收敛性良好（变化率 < 5%）")
            elif change_16_20 < 10:
                print("✓ 网格收敛性可接受（变化率 < 10%）")
            else:
                print("⚠ 网格收敛性较差（变化率 ≥ 10%），建议使用更高分辨率")

if __name__ == "__main__":
    main()
