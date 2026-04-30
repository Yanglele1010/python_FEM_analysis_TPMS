import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.ndimage import label


# ============================================================
# 1. 文档中的 BCC 与修正 I-WP 函数
# ============================================================

def F_BCC(x, y, z, kx, ky, kz):
    return (
        np.cos(2.0 * kx * x)
        + np.cos(2.0 * ky * y)
        + np.cos(2.0 * kz * z)
        - 2.0 * (
            np.cos(kx * x) * np.cos(ky * y)
            + np.cos(ky * y) * np.cos(kz * z)
            + np.cos(kz * z) * np.cos(kx * x)
        )
    )


def F_IWP_modified(x, y, z, kx, ky, kz, t2, alpha=5.0):
    return (
        -np.cos(2.0 * kx * x)
        + np.cos(2.0 * ky * y)
        + np.cos(2.0 * kz * z)
        + (t2 - alpha) * (
            np.cos(kx * x) * np.cos(ky * y)
            + np.cos(ky * y) * np.cos(kz * z)
            + np.cos(kz * z) * np.cos(kx * x)
        )
    )


# ============================================================
# 2. 生成体素实体
# ============================================================

def keep_largest_component(solid):
    labeled, num = label(solid)

    if num <= 1:
        return solid

    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    largest = np.argmax(counts)

    return labeled == largest


def generate_ipc_voxels_threshold(
    cell_size=20.0,
    cells=(2, 2, 2),
    n_per_cell_fe=16,
    t1=-1.5,
    t2=-1.5,
    alpha=5.0,
    keep_largest=True
):
    """
    使用文档阈值法生成体素实体。

    注意：
    n_per_cell_fe 是有限元分辨率，不是 STL 分辨率。
    建议先用 12、16、20 测试。
    """

    Lx = cell_size * cells[0]
    Ly = cell_size * cells[1]
    Lz = cell_size * cells[2]

    nx = n_per_cell_fe * cells[0]
    ny = n_per_cell_fe * cells[1]
    nz = n_per_cell_fe * cells[2]

    dx = Lx / nx
    dy = Ly / ny
    dz = Lz / nz

    # 单元中心点
    x = np.linspace(dx / 2.0, Lx - dx / 2.0, nx)
    y = np.linspace(dy / 2.0, Ly - dy / 2.0, ny)
    z = np.linspace(dz / 2.0, Lz - dz / 2.0, nz)

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    # 转换为以中心为原点的坐标，匹配 TPMS 周期
    Xc = X - Lx / 2.0
    Yc = Y - Ly / 2.0
    Zc = Z - Lz / 2.0

    kx = ky = kz = 2.0 * np.pi / cell_size

    FBCC = F_BCC(Xc, Yc, Zc, kx, ky, kz)
    FIWP = F_IWP_modified(Xc, Yc, Zc, kx, ky, kz, t2=t2, alpha=alpha)

    solid_bcc = FBCC <= t1
    solid_iwp = FIWP <= t2
    solid = solid_bcc | solid_iwp

    if keep_largest:
        solid = keep_largest_component(solid)

    rho = np.mean(solid)

    print("--------------------------------")
    print("体素模型生成完成")
    print(f"overall size: {Lx:.2f} × {Ly:.2f} × {Lz:.2f} mm")
    print(f"FE voxels: {nx} × {ny} × {nz}")
    print(f"element size: {dx:.3f} × {dy:.3f} × {dz:.3f} mm")
    print(f"rho*: {rho:.4f}")
    print(f"active elements: {np.sum(solid)}")
    print("--------------------------------")

    return solid, (Lx, Ly, Lz), (dx, dy, dz)


# ============================================================
# 3. Hex8 单元刚度矩阵
# ============================================================

def elasticity_matrix(E, nu):
    """
    3D 各向同性弹性矩阵。
    单位：
        E: MPa = N/mm^2
    """

    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))

    D = np.array([
        [lam + 2 * mu, lam, lam, 0, 0, 0],
        [lam, lam + 2 * mu, lam, 0, 0, 0],
        [lam, lam, lam + 2 * mu, 0, 0, 0],
        [0, 0, 0, mu, 0, 0],
        [0, 0, 0, 0, mu, 0],
        [0, 0, 0, 0, 0, mu],
    ])

    return D


def hex8_element_stiffness(dx, dy, dz, E, nu):
    """
    8 节点六面体单元刚度矩阵。
    节点顺序：
        0: (-,-,-)
        1: (+,-,-)
        2: (+,+,-)
        3: (-,+,-)
        4: (-,-,+)
        5: (+,-,+)
        6: (+,+,+)
        7: (-,+,+)
    """

    D = elasticity_matrix(E, nu)

    # 自然坐标下节点位置
    node_nat = np.array([
        [-1, -1, -1],
        [ 1, -1, -1],
        [ 1,  1, -1],
        [-1,  1, -1],
        [-1, -1,  1],
        [ 1, -1,  1],
        [ 1,  1,  1],
        [-1,  1,  1],
    ], dtype=float)

    gauss = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]

    Ke = np.zeros((24, 24), dtype=float)

    # 规则六面体雅可比
    J = np.diag([dx / 2.0, dy / 2.0, dz / 2.0])
    invJ = np.linalg.inv(J)
    detJ = np.linalg.det(J)

    for xi in gauss:
        for eta in gauss:
            for zeta in gauss:

                dN_dnat = np.zeros((8, 3), dtype=float)

                for a in range(8):
                    xi_a, eta_a, zeta_a = node_nat[a]

                    dN_dxi = 0.125 * xi_a * (1 + eta_a * eta) * (1 + zeta_a * zeta)
                    dN_deta = 0.125 * eta_a * (1 + xi_a * xi) * (1 + zeta_a * zeta)
                    dN_dzeta = 0.125 * zeta_a * (1 + xi_a * xi) * (1 + eta_a * eta)

                    dN_dnat[a, :] = [dN_dxi, dN_deta, dN_dzeta]

                dN_dxyz = dN_dnat @ invJ.T

                B = np.zeros((6, 24), dtype=float)

                for a in range(8):
                    dNx, dNy, dNz = dN_dxyz[a]

                    col = 3 * a

                    B[0, col + 0] = dNx
                    B[1, col + 1] = dNy
                    B[2, col + 2] = dNz

                    B[3, col + 0] = dNy
                    B[3, col + 1] = dNx

                    B[4, col + 1] = dNz
                    B[4, col + 2] = dNy

                    B[5, col + 0] = dNz
                    B[5, col + 2] = dNx

                Ke += B.T @ D @ B * detJ

    return Ke


# ============================================================
# 4. 组装全局刚度矩阵
# ============================================================

def node_id(i, j, k, ny, nz):
    return i * (ny + 1) * (nz + 1) + j * (nz + 1) + k


def element_nodes(i, j, k, ny, nz):
    n000 = node_id(i,     j,     k,     ny, nz)
    n100 = node_id(i + 1, j,     k,     ny, nz)
    n110 = node_id(i + 1, j + 1, k,     ny, nz)
    n010 = node_id(i,     j + 1, k,     ny, nz)

    n001 = node_id(i,     j,     k + 1, ny, nz)
    n101 = node_id(i + 1, j,     k + 1, ny, nz)
    n111 = node_id(i + 1, j + 1, k + 1, ny, nz)
    n011 = node_id(i,     j + 1, k + 1, ny, nz)

    return np.array([n000, n100, n110, n010, n001, n101, n111, n011], dtype=int)


def assemble_global_stiffness(solid, dx, dy, dz, E=2200.0, nu=0.35):
    """
    组装体素有限元刚度矩阵。
    """

    nx, ny, nz = solid.shape

    num_nodes = (nx + 1) * (ny + 1) * (nz + 1)
    num_dofs = num_nodes * 3

    Ke = hex8_element_stiffness(dx, dy, dz, E, nu)

    rows = []
    cols = []
    data = []

    used_nodes = set()

    active_indices = np.argwhere(solid)

    print("开始组装全局刚度矩阵...")
    print(f"active elements: {len(active_indices)}")

    for e_id, (i, j, k) in enumerate(active_indices):
        enodes = element_nodes(i, j, k, ny, nz)

        for n in enodes:
            used_nodes.add(int(n))

        edofs = np.zeros(24, dtype=int)

        for a, n in enumerate(enodes):
            edofs[3 * a + 0] = 3 * n + 0
            edofs[3 * a + 1] = 3 * n + 1
            edofs[3 * a + 2] = 3 * n + 2

        rr = np.repeat(edofs, 24)
        cc = np.tile(edofs, 24)

        rows.append(rr)
        cols.append(cc)
        data.append(Ke.ravel())

        if (e_id + 1) % 5000 == 0:
            print(f"assembled {e_id + 1} / {len(active_indices)} elements")

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    data = np.concatenate(data)

    K = coo_matrix((data, (rows, cols)), shape=(num_dofs, num_dofs)).tocsr()

    used_nodes = np.array(sorted(list(used_nodes)), dtype=int)
    used_dofs = np.concatenate([
        3 * used_nodes + 0,
        3 * used_nodes + 1,
        3 * used_nodes + 2
    ])
    used_dofs = np.unique(used_dofs)

    print("刚度矩阵组装完成")
    print(f"num nodes: {num_nodes}")
    print(f"num dofs: {num_dofs}")
    print(f"used nodes: {len(used_nodes)}")
    print(f"used dofs: {len(used_dofs)}")
    print("--------------------------------")

    return K, used_nodes, used_dofs


# ============================================================
# 5. 压缩边界条件与求解
# ============================================================

def node_ijk_from_id(n, ny, nz):
    i = n // ((ny + 1) * (nz + 1))
    r = n % ((ny + 1) * (nz + 1))
    j = r // (nz + 1)
    k = r % (nz + 1)
    return i, j, k


def solve_compression(
    solid,
    size,
    elem_size,
    E=2200.0,
    nu=0.35,
    prescribed_disp=-0.4
):
    """
    单轴压缩有限元。

    底部：ux=uy=uz=0
    顶部：uz=prescribed_disp
    """

    Lx, Ly, Lz = size
    dx, dy, dz = elem_size

    nx, ny, nz = solid.shape

    K, used_nodes, used_dofs = assemble_global_stiffness(
        solid,
        dx, dy, dz,
        E=E,
        nu=nu
    )

    # 找 active node 的 k 坐标范围
    ijk = np.array([node_ijk_from_id(n, ny, nz) for n in used_nodes])
    k_values = ijk[:, 2]

    k_min = np.min(k_values)
    k_max = np.max(k_values)

    bottom_nodes = used_nodes[k_values == k_min]
    top_nodes = used_nodes[k_values == k_max]

    if len(bottom_nodes) == 0 or len(top_nodes) == 0:
        raise RuntimeError("没有找到顶部或底部节点，模型可能没有贯通压缩方向。")

    print("边界节点统计")
    print(f"bottom nodes: {len(bottom_nodes)}")
    print(f"top nodes: {len(top_nodes)}")
    print(f"k_min: {k_min}, k_max: {k_max}")
    print("--------------------------------")

    prescribed = {}

    # 底面全固定
    for n in bottom_nodes:
        prescribed[3 * n + 0] = 0.0
        prescribed[3 * n + 1] = 0.0
        prescribed[3 * n + 2] = 0.0

    # 顶面施加 z 向位移
    for n in top_nodes:
        prescribed[3 * n + 2] = prescribed_disp

    prescribed_dofs = np.array(sorted(prescribed.keys()), dtype=int)
    prescribed_vals = np.array([prescribed[d] for d in prescribed_dofs], dtype=float)

    active_dofs = used_dofs
    free_dofs = np.setdiff1d(active_dofs, prescribed_dofs)

    print("自由度统计")
    print(f"active dofs: {len(active_dofs)}")
    print(f"prescribed dofs: {len(prescribed_dofs)}")
    print(f"free dofs: {len(free_dofs)}")
    print("--------------------------------")

    F = np.zeros(K.shape[0], dtype=float)

    K_ff = K[free_dofs[:, None], free_dofs]
    K_fc = K[free_dofs[:, None], prescribed_dofs]

    rhs = F[free_dofs] - K_fc @ prescribed_vals

    print("开始求解线性方程...")
    U_free = spsolve(K_ff, rhs)
    print("求解完成")

    U = np.zeros(K.shape[0], dtype=float)
    U[free_dofs] = U_free
    U[prescribed_dofs] = prescribed_vals

    # 反力
    R = K @ U - F

    top_z_dofs = np.array([3 * n + 2 for n in top_nodes], dtype=int)
    reaction_force = np.sum(R[top_z_dofs])

    force_abs = abs(reaction_force)

    # 使用实际 active 高度估算应变
    height_active = (k_max - k_min) * dz
    strain = abs(prescribed_disp) / height_active

    area = Lx * Ly
    stress = force_abs / area

    E_eff = stress / strain
    compliance = float(U @ (K @ U))

    print("--------------------------------")
    print("压缩有限元结果")
    print(f"reaction force: {reaction_force:.6f} N")
    print(f"|force|: {force_abs:.6f} N")
    print(f"area: {area:.6f} mm^2")
    print(f"stress: {stress:.6f} MPa")
    print(f"strain: {strain:.6f}")
    print(f"E_eff: {E_eff:.6f} MPa")
    print(f"compliance U^T K U: {compliance:.6f}")
    print("--------------------------------")

    return {
        "U": U,
        "reaction_force": reaction_force,
        "force_abs": force_abs,
        "stress": stress,
        "strain": strain,
        "E_eff": E_eff,
        "compliance": compliance,
        "bottom_nodes": bottom_nodes,
        "top_nodes": top_nodes,
    }


# ============================================================
# 6. 查找 t1, t2 参数的函数
# ============================================================

def find_t1_t2_by_lookup(target_rho, target_w):
    """
    根据目标rho*和w值查找对应的t1和t2参数
    
    这里使用预设的参数表，实际应用中可能需要更精确的参数优化
    """
    # 预设参数表，基于经验值
    # 格式：(target_rho, target_w, t1, t2)
    param_table = [
        # rho=0.20
        (0.20, 0.0, 10.0, -3.0),
        (0.20, 0.2, -3.0, -2.8),
        (0.20, 0.4, -2.9, -2.9),
        (0.20, 0.6, -2.8, -3.0),
        (0.20, 0.8, -2.7, -3.1),
        (0.20, 1.0, -3.0, 10.0),
        # rho=0.25
        (0.25, 0.0, 10.0, -2.8),
        (0.25, 0.2, -2.8, -2.6),
        (0.25, 0.4, -2.7, -2.7),
        (0.25, 0.6, -2.6, -2.8),
        (0.25, 0.8, -2.5, -2.9),
        (0.25, 1.0, -2.8, 10.0),
        # rho=0.30
        (0.30, 0.0, 10.0, -2.5),
        (0.30, 0.2, -2.6, -2.4),
        (0.30, 0.4, -2.5, -2.5),
        (0.30, 0.6, -2.4, -2.6),
        (0.30, 0.8, -2.3, -2.7),
        (0.30, 1.0, -2.5, 10.0),
        # rho=0.35
        (0.35, 0.0, 10.0, -2.3),
        (0.35, 0.2, -2.4, -2.2),
        (0.35, 0.4, -2.3, -2.3),
        (0.35, 0.6, -2.2, -2.4),
        (0.35, 0.8, -2.1, -2.5),
        (0.35, 1.0, -2.3, 10.0),
        # rho=0.40
        (0.40, 0.0, 10.0, -2.0),
        (0.40, 0.2, -2.2, -1.9),
        (0.40, 0.4, -2.1, -2.0),
        (0.40, 0.6, -2.0, -2.1),
        (0.40, 0.8, -1.9, -2.2),
        (0.40, 1.0, -2.0, 10.0),
    ]
    
    # 查找最接近的参数组合
    min_error = float('inf')
    best_t1 = -2.0
    best_t2 = -2.0
    
    for rho, w, t1, t2 in param_table:
        error = abs(rho - target_rho) + abs(w - target_w)
        if error < min_error:
            min_error = error
            best_t1 = t1
            best_t2 = t2
    
    return best_t1, best_t2

# ============================================================
# 7. 主程序 - 参数扫描分析
# ============================================================

if __name__ == "__main__":
    import time
    import csv
    
    # FDM 复现尺寸
    CELL_SIZE = 20.0
    CELLS = (2, 2, 2)

    # 有限元分辨率
    N_PER_CELL_FE = 16

    # 目标参数列表
    target_rho_list = [0.20, 0.25, 0.30, 0.35, 0.40]
    target_w_list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    # 几何参数
    ALPHA = 5.0

    # 材料参数
    # PLA 可先取 2200 MPa，nu=0.35
    # 如果用论文树脂参数，可改 E=1993.4, nu=0.3
    E_SOLID = 2200.0
    NU_SOLID = 0.35

    # 压缩位移
    # 40 mm 高度，0.4 mm 位移约等于 1% 应变
    DISP = -0.4
    
    # 记录结果
    results = []
    
    print("开始参数扫描分析...\n")
    
    # 遍历所有参数组合
    for target_rho in target_rho_list:
        for target_w in target_w_list:
            print(f"====================================")
            print(f"测试参数: rho*={target_rho:.2f}, w={target_w:.1f}")
            print(f"====================================")
            
            # 记录开始时间
            start_time = time.time()
            
            # 查找t1和t2参数
            t1, t2 = find_t1_t2_by_lookup(target_rho, target_w)
            print(f"查找得到的参数: t1={t1:.2f}, t2={t2:.2f}")
            
            # 生成体素模型
            solid, size, elem_size = generate_ipc_voxels_threshold(
                cell_size=CELL_SIZE,
                cells=CELLS,
                n_per_cell_fe=N_PER_CELL_FE,
                t1=t1,
                t2=t2,
                alpha=ALPHA,
                keep_largest=True
            )

            # 计算实际rho*和w
            solid_bcc = F_BCC(
                np.linspace(-20, 20, 32),
                np.linspace(-20, 20, 32),
                np.linspace(-20, 20, 32),
                2.0 * np.pi / CELL_SIZE,
                2.0 * np.pi / CELL_SIZE,
                2.0 * np.pi / CELL_SIZE
            ) <= t1
            solid_iwp = F_IWP_modified(
                np.linspace(-20, 20, 32),
                np.linspace(-20, 20, 32),
                np.linspace(-20, 20, 32),
                2.0 * np.pi / CELL_SIZE,
                2.0 * np.pi / CELL_SIZE,
                2.0 * np.pi / CELL_SIZE,
                t2=t2,
                alpha=ALPHA
            ) <= t2
            actual_rho = np.mean(solid)
            actual_w = np.mean(solid_bcc) / (np.mean(solid_bcc) + np.mean(solid_iwp)) if (np.mean(solid_bcc) + np.mean(solid_iwp)) > 0 else 0
            
            # 进行有限元压缩分析
            result = solve_compression(
                solid=solid,
                size=size,
                elem_size=elem_size,
                E=E_SOLID,
                nu=NU_SOLID,
                prescribed_disp=DISP
            )
            
            # 记录结束时间
            end_time = time.time()
            run_time = end_time - start_time
            
            # 记录结果
            results.append({
                'target_rho': target_rho,
                'target_w': target_w,
                'actual_rho': actual_rho,
                'actual_w': actual_w,
                't1': t1,
                't2': t2,
                'E_eff': result['E_eff'],
                'compliance': result['compliance'],
                'reaction_force': result['reaction_force'],
                'run_time': run_time
            })
            
            print(f"运行时间: {run_time:.2f} 秒")
            print(f"====================================\n")
    
    # 打印汇总结果
    print("\n====================================")
    print("参数扫描分析结果汇总")
    print("====================================")
    print(f"{'target ρ*':<10} {'target w':<10} {'actual ρ*':<10} {'actual w':<10} {'t1':<8} {'t2':<8} {'E_eff(MPa)':<12} {'柔度':<10}")
    print("-" * 80)
    
    for res in results:
        print(f"{res['target_rho']:<10.2f} {res['target_w']:<10.1f} {res['actual_rho']:<10.4f} {res['actual_w']:<10.4f} {res['t1']:<8.2f} {res['t2']:<8.2f} {res['E_eff']:<12.2f} {res['compliance']:<10.2f}")
    
    # 保存结果到CSV文件
    csv_file = "ipc_param_scan_results.csv"
    with open(csv_file, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow([
            'target_rho', 'target_w', 'actual_rho', 'actual_w', 
            't1', 't2', 'E_eff', 'compliance', 'reaction_force', 'run_time'
        ])
        for res in results:
            writer.writerow([
                res['target_rho'], res['target_w'], res['actual_rho'], res['actual_w'],
                res['t1'], res['t2'], res['E_eff'], res['compliance'],
                res['reaction_force'], res['run_time']
            ])
    
    print(f"\n结果已保存到: {csv_file}")
    print("====================================")