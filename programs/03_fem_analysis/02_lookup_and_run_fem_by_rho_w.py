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


# ============================================================
# 1. 基础参数
# ============================================================

CELL_SIZE = 20.0
CELLS = (2, 2, 2)
ALPHA = 5.0

# 查找表分辨率，越大越准，但越慢
N_PER_CELL_LOOKUP = 16

# FEM 分辨率，先用 16，跑通后可改 20 或 24
N_PER_CELL_FE = 16

# 材料参数
# PLA 可先用 2200 MPa, 0.35
# 如果用论文树脂，可改为 E=1993.4, nu=0.3
E_SOLID = 2200.0
NU_SOLID = 0.35

# 40 mm 高度，-0.4 mm 位移约为 1% 压缩应变
DISP = -0.4


# ============================================================
# 2. 计算某个 t1,t2 对应的 rho* 和 w
# ============================================================

def compute_rho_w_for_t1_t2(
    t1,
    t2,
    cell_size=20.0,
    cells=(2, 2, 2),
    n_per_cell=16,
    alpha=5.0,
):
    Lx = cell_size * cells[0]
    Ly = cell_size * cells[1]
    Lz = cell_size * cells[2]

    nx = n_per_cell * cells[0]
    ny = n_per_cell * cells[1]
    nz = n_per_cell * cells[2]

    dx = Lx / nx
    dy = Ly / ny
    dz = Lz / nz

    x = np.linspace(dx / 2.0, Lx - dx / 2.0, nx)
    y = np.linspace(dy / 2.0, Ly - dy / 2.0, ny)
    z = np.linspace(dz / 2.0, Lz - dz / 2.0, nz)

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    Xc = X - Lx / 2.0
    Yc = Y - Ly / 2.0
    Zc = Z - Lz / 2.0

    kx = ky = kz = 2.0 * np.pi / cell_size

    FBCC = F_BCC(Xc, Yc, Zc, kx, ky, kz)
    FIWP = F_IWP_modified(
        Xc, Yc, Zc,
        kx, ky, kz,
        t2=t2,
        alpha=alpha
    )

    solid_bcc = FBCC <= t1
    solid_iwp = FIWP <= t2
    solid_total = solid_bcc | solid_iwp

    rho_total = float(np.mean(solid_total))
    rho_bcc = float(np.mean(solid_bcc))
    rho_iwp = float(np.mean(solid_iwp))
    overlap = float(np.mean(solid_bcc & solid_iwp))

    if rho_total > 1e-12:
        w_union = rho_bcc / rho_total
    else:
        w_union = 0.0

    if rho_bcc + rho_iwp > 1e-12:
        w_phase = rho_bcc / (rho_bcc + rho_iwp)
    else:
        w_phase = 0.0

    return {
        "t1": float(t1),
        "t2": float(t2),
        "rho_total": rho_total,
        "rho_bcc": rho_bcc,
        "rho_iwp": rho_iwp,
        "w_union": float(w_union),
        "w_phase": float(w_phase),
        "overlap": overlap,
        "active_elements": int(np.sum(solid_total)),
    }


# ============================================================
# 3. 建立 t1,t2 -> rho,w 查找表
# ============================================================

def build_lookup_table(
    output_csv="lookup_table_cell20.csv",
    t1_min=-3.0,
    t1_max=1.0,
    t2_min=-3.0,
    t2_max=1.0,
    num_t1=41,
    num_t2=41,
):
    t1_list = np.linspace(t1_min, t1_max, num_t1)
    t2_list = np.linspace(t2_min, t2_max, num_t2)

    rows = []
    total = len(t1_list) * len(t2_list)
    count = 0

    print("--------------------------------")
    print("开始建立查找表")
    print(f"t1 range: {t1_min} ~ {t1_max}, num={num_t1}")
    print(f"t2 range: {t2_min} ~ {t2_max}, num={num_t2}")
    print(f"total cases: {total}")
    print("--------------------------------")

    start = time.time()

    for t1 in t1_list:
        for t2 in t2_list:
            count += 1

            data = compute_rho_w_for_t1_t2(
                t1=t1,
                t2=t2,
                cell_size=CELL_SIZE,
                cells=CELLS,
                n_per_cell=N_PER_CELL_LOOKUP,
                alpha=ALPHA,
            )

            rows.append(data)

            if count % 100 == 0:
                print(f"lookup progress: {count}/{total}")

    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "t1",
                "t2",
                "rho_total",
                "rho_bcc",
                "rho_iwp",
                "w_union",
                "w_phase",
                "overlap",
                "active_elements",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    end = time.time()

    print("--------------------------------")
    print("查找表完成")
    print(f"file: {output_csv}")
    print(f"time: {end - start:.2f} s")
    print("--------------------------------")

    return rows


# ============================================================
# 4. 读取查找表
# ============================================================

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


# ============================================================
# 5. 按目标 rho,w 从查找表中找最近的 t1,t2
# ============================================================

def find_best_t1_t2(
    lookup_rows,
    target_rho,
    target_w,
    rho_scale=0.02,
    w_scale=0.05,
    overlap_weight=0.0,
):
    """
    target_w 使用 w_union，也就是论文定义近似：
        w = rho_BCC / rho_total

    如果你希望减少两相重叠，可以把 overlap_weight 设为 0.2 或 0.5。
    """

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


# ============================================================
# 6. 对目标 rho,w 批量运行 FEM
# ============================================================

def run_fem_for_target_grid(
    lookup_csv="lookup_table_cell20.csv",
    output_csv="fem_results_by_rho_w.csv",
    target_rho_list=None,
    target_w_list=None,
):
    if target_rho_list is None:
        target_rho_list = [0.20, 0.25, 0.30, 0.35, 0.40]

    if target_w_list is None:
        target_w_list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    lookup_rows = read_lookup_table(lookup_csv)

    fieldnames = [
        "target_rho",
        "target_w",
        "actual_rho_lookup",
        "actual_w_lookup",
        "actual_w_phase_lookup",
        "rho_bcc_lookup",
        "rho_iwp_lookup",
        "overlap_lookup",
        "t1",
        "t2",
        "lookup_score",
        "active_elements_lookup",
        "reaction_force_N",
        "stress_MPa",
        "strain",
        "E_eff_MPa",
        "compliance",
        "fem_time_s",
    ]

    write_header = not os.path.exists(output_csv)

    with open(output_csv, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()

        total = len(target_rho_list) * len(target_w_list)
        case_id = 0

        for target_rho in target_rho_list:
            for target_w in target_w_list:
                case_id += 1

                print("\n================================")
                print(f"FEM Case {case_id}/{total}")
                print(f"target rho*: {target_rho}")
                print(f"target w:    {target_w}")

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

                print("查表结果：")
                print(f"t1 = {t1:.4f}, t2 = {t2:.4f}")
                print(f"actual rho* = {best['rho_total']:.4f}")
                print(f"actual w    = {best['w_union']:.4f}")
                print(f"overlap     = {best['overlap']:.4f}")
                print("开始 FEM...")

                start = time.time()

                solid, size, elem_size = generate_ipc_voxels_threshold(
                    cell_size=CELL_SIZE,
                    cells=CELLS,
                    n_per_cell_fe=N_PER_CELL_FE,
                    t1=t1,
                    t2=t2,
                    alpha=ALPHA,
                    keep_largest=True,
                )

                result = solve_compression(
                    solid=solid,
                    size=size,
                    elem_size=elem_size,
                    E=E_SOLID,
                    nu=NU_SOLID,
                    prescribed_disp=DISP,
                )

                end = time.time()

                row = {
                    "target_rho": target_rho,
                    "target_w": target_w,
                    "actual_rho_lookup": best["rho_total"],
                    "actual_w_lookup": best["w_union"],
                    "actual_w_phase_lookup": best["w_phase"],
                    "rho_bcc_lookup": best["rho_bcc"],
                    "rho_iwp_lookup": best["rho_iwp"],
                    "overlap_lookup": best["overlap"],
                    "t1": t1,
                    "t2": t2,
                    "lookup_score": best["score"],
                    "active_elements_lookup": best["active_elements"],
                    "reaction_force_N": result["reaction_force"],
                    "stress_MPa": result["stress"],
                    "strain": result["strain"],
                    "E_eff_MPa": result["E_eff"],
                    "compliance": result["compliance"],
                    "fem_time_s": end - start,
                }

                writer.writerow(row)
                f.flush()

                print("结果：")
                print(f"E_eff = {result['E_eff']:.4f} MPa")
                print(f"compliance = {result['compliance']:.4f}")
                print(f"time = {end - start:.2f} s")

    print("--------------------------------")
    print("全部 FEM 完成")
    print(f"结果文件: {output_csv}")
    print("--------------------------------")


# ============================================================
# 7. 主程序
# ============================================================

if __name__ == "__main__":

    LOOKUP_CSV = "lookup_table_cell20_fine.csv"
    FEM_RESULTS_CSV = "fem_results_by_rho_w_full.csv"

    BUILD_LOOKUP = True
    RUN_FEM = True

    if BUILD_LOOKUP:
        build_lookup_table(
            output_csv=LOOKUP_CSV,
            t1_min=-3.0,
            t1_max=1.0,
            t2_min=-6.0,   # 扩展到 -6.0 以支持 rho=0.20
            t2_max=1.0,
            num_t1=81,
            num_t2=121,    # 增加 t2 分辨率以覆盖更大范围
        )

    if RUN_FEM:
        # 只运行 rho=0.20，其他已完成
        target_rho_list = [0.20]
        target_w_list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        run_fem_for_target_grid(
            lookup_csv=LOOKUP_CSV,
            output_csv=FEM_RESULTS_CSV,
            target_rho_list=target_rho_list,
            target_w_list=target_w_list,
        )