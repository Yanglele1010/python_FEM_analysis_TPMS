# -*- coding: utf-8 -*-
"""
12_macro_cantilever_density_optimization.py

功能：
1. 读取 D(rho,w) 插值模型系数
2. 建立二维悬臂梁平面应力 FEM 模型
3. 固定 w = 0.6，只优化 rho
4. 使用 OC 方法进行柔顺度最小化
5. 输出密度分布、位移云图、柔顺度迭代曲线、体积分数曲线

说明：
这是宏观多尺度优化的第一版验证程序。
暂时不做 rho + w 双变量 MMA，只先保证 FEM + D(rho,w) 耦合跑通。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 配置参数
# ============================================================================
class Config:
    nelx = 60  # 减少网格大小，提高计算速度
    nely = 20
    rho_target = 0.30  # 目标相对密度
    w_fixed = 0.6
    rmin = 3.0
    ft = 1
    move = 0.1  # 减小移动步长
    max_iter = 50  # 减少最大迭代次数
    tol = 1e-4

    rho_min = 0.01  # 最小相对密度
    rho_max = 1.0   # 最大相对密度

    force_magnitude = -1.0

    OUTPUT_DIR = r"E:\code\new\04_Final_Results"
    D_COEFF_FILE = r"E:\code\new\04_Final_Results\D_interpolation_coefficients.csv"

# ============================================================================
# 读取 D(rho,w) 插值系数
# ============================================================================
def load_D_coefficients(filepath):
    df = pd.read_csv(filepath)
    coeffs = {}
    for component in ['D1111', 'D1122', 'D1212']:
        coeffs[component] = df[component].values
    return coeffs

# ============================================================================
# D(rho,w) 插值函数
# ============================================================================
def D_interpolation(rho, w, coeffs):
    b = coeffs
    D1111 = (b['D1111'][0] * rho +
             b['D1111'][1] * rho**2 +
             b['D1111'][2] * rho**3 +
             b['D1111'][3] * rho**4 +
             b['D1111'][4] * rho**5 +
             b['D1111'][5] * rho * w +
             b['D1111'][6] * rho * w**2 +
             b['D1111'][7] * rho**2 * w +
             b['D1111'][8] * rho**2 * w**2)

    D1122 = (b['D1122'][0] * rho +
             b['D1122'][1] * rho**2 +
             b['D1122'][2] * rho**3 +
             b['D1122'][3] * rho**4 +
             b['D1122'][4] * rho**5 +
             b['D1122'][5] * rho * w +
             b['D1122'][6] * rho * w**2 +
             b['D1122'][7] * rho**2 * w +
             b['D1122'][8] * rho**2 * w**2)

    D1212 = (b['D1212'][0] * rho +
             b['D1212'][1] * rho**2 +
             b['D1212'][2] * rho**3 +
             b['D1212'][3] * rho**4 +
             b['D1212'][4] * rho**5 +
             b['D1212'][5] * rho * w +
             b['D1212'][6] * rho * w**2 +
             b['D1212'][7] * rho**2 * w +
             b['D1212'][8] * rho**2 * w**2)

    return D1111, D1122, D1212

# ============================================================================
# 分析 D(rho,w) 模型
# ============================================================================
def analyze_D_model(coeffs, w=0.6):
    rho_values = np.linspace(0, 1, 100)
    D1111_values = []
    D1122_values = []
    D1212_values = []

    for rho in rho_values:
        D1111, D1122, D1212 = D_interpolation(rho, w, coeffs)
        D1111_values.append(D1111)
        D1122_values.append(D1122)
        D1212_values.append(D1212)

    plt.figure(figsize=(10, 6))
    plt.plot(rho_values, D1111_values, label='D1111')
    plt.plot(rho_values, D1212_values, label='D1212')
    plt.xlabel('rho')
    plt.ylabel('D values')
    plt.title(f'D(rho, w={w}) Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = os.path.join(Config.OUTPUT_DIR, 'D_model_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved D model analysis to: {output_path}")

    # 检查趋势
    D1111_min = min(D1111_values)
    D1111_argmin = rho_values[np.argmin(D1111_values)]
    print(f"D1111 minimum: {D1111_min:.6f} at rho={D1111_argmin:.3f}")

    D1212_min = min(D1212_values)
    D1212_argmin = rho_values[np.argmin(D1212_values)]
    print(f"D1212 minimum: {D1212_min:.6f} at rho={D1212_argmin:.3f}")

# ============================================================================
# 构造平面应力 D 矩阵
# ============================================================================
def construct_D_matrix(rho, w, coeffs):
    D1111, D1122, D1212 = D_interpolation(rho, w, coeffs)

    D_e = np.array([[D1111, D1122, 0],
                    [D1122, D1111, 0],
                    [0,     0,     D1212]])

    if D1111 <= 0 or D1212 <= 0:
        raise ValueError(f"Negative stiffness detected: D1111={D1111}, D1212={D1212}")

    return D_e

# ============================================================================
# 单元刚度矩阵 (平面应力，四节点矩形单元)
# ============================================================================
def quad4_element_stiffness(D_e, hx=1.0, hy=1.0):
    k = np.zeros((8, 8))

    gp = 1.0 / np.sqrt(3.0)
    wg = 1.0

    C = D_e

    for xi in [-gp, gp]:
        for eta in [-gp, gp]:
            dN = np.array([
                [-0.25*(1-eta),  0.25*(1-eta),  0.25*(1+eta), -0.25*(1+eta)],
                [-0.25*(1-xi),  -0.25*(1+xi),  0.25*(1+xi),  0.25*(1-xi)]
            ])

            B = np.zeros((3, 8))
            for i in range(4):
                B[0, 2*i] = dN[0, i]
                B[1, 2*i+1] = dN[1, i]
                B[2, 2*i] = dN[1, i]
                B[2, 2*i+1] = dN[0, i]

            det_jac = hx * hy / 4
            k += B.T @ C @ B * det_jac * wg * wg

    return k

# ============================================================================
# 密度过滤
# ============================================================================
def density_filter(x, rmin, nelx, nely):
    x_filtered = np.zeros((nely, nelx))

    for i in range(nelx):
        for j in range(nely):
            sum_h = 0.0
            sum_h_x = 0.0
            for k in range(max(0, int(i - rmin - 1)), min(nelx, int(i + rmin))):
                for l in range(max(0, int(j - rmin - 1)), min(nely, int(j + rmin))):
                    distance = np.sqrt((i - k)**2 + (j - l)**2)
                    if distance < rmin:
                        weight = rmin - distance
                        sum_h += weight
                        sum_h_x += weight * x[l, k]
            if sum_h > 0:
                x_filtered[j, i] = sum_h_x / sum_h
            else:
                x_filtered[j, i] = x[j, i]

    return x_filtered

# ============================================================================
# 设计变量到实际密度的转换
# ============================================================================
def x_to_rho(x, rho_min, rho_max):
    return rho_min + x * (rho_max - rho_min)

# ============================================================================
# 实际密度到设计变量的转换
# ============================================================================
def rho_to_x(rho, rho_min, rho_max):
    return (rho - rho_min) / (rho_max - rho_min)

# ============================================================================
# 组装全局刚度矩阵
# ============================================================================
def assemble_stiffness_matrix(x, w, coeffs, nelx, nely, rho_min, rho_max):
    hx = 1.0
    hy = 1.0
    n_dof = 2 * (nelx + 1) * (nely + 1)

    K = csr_matrix((n_dof, n_dof))

    for ex in range(nelx):
        for ey in range(nely):
            x_e = x[ey, ex]
            rho_e = x_to_rho(x_e, rho_min, rho_max)

            try:
                D_e = construct_D_matrix(rho_e, w, coeffs)
            except ValueError as e:
                print(f"Warning: {e} at element ({ex}, {ey}), using rho_min")
                D_e = construct_D_matrix(rho_min, w, coeffs)

            k_e = quad4_element_stiffness(D_e, hx, hy)

            n1 = ey * (nelx + 1) + ex
            n2 = ey * (nelx + 1) + ex + 1
            n3 = (ey + 1) * (nelx + 1) + ex + 1
            n4 = (ey + 1) * (nelx + 1) + ex

            dofs = [2*n1, 2*n1+1, 2*n2, 2*n2+1, 2*n3, 2*n3+1, 2*n4, 2*n4+1]

            for i in range(8):
                for j in range(8):
                    K[dofs[i], dofs[j]] += k_e[i, j]

    return K

# ============================================================================
# 施加载荷和边界条件
# ============================================================================
def apply_boundary_conditions(K, F, nelx, nely):
    n_dof = 2 * (nelx + 1) * (nely + 1)

    fixed_dofs = []

    for ey in range(nely + 1):
        node = ey * (nelx + 1)
        fixed_dofs.append(2 * node)
        fixed_dofs.append(2 * node + 1)

    for dof in fixed_dofs:
        K[dof, :] = 0
        K[:, dof] = 0
        K[dof, dof] = 1.0

    mid_y = nely // 2
    load_node = mid_y * (nelx + 1) + nelx
    load_dof = 2 * load_node + 1

    F[load_dof] = Config.force_magnitude

    return K, F, fixed_dofs

# ============================================================================
# 求解器
# ============================================================================
def solve_fem(K, F, fixed_dofs):
    n_dof = len(F)

    u = np.zeros(n_dof)

    free_dofs = np.setdiff1d(np.arange(n_dof), fixed_dofs)

    if len(free_dofs) == 0:
        raise ValueError("No free degrees of freedom")

    K_ff = K[free_dofs][:, free_dofs]
    F_f = F[free_dofs]

    if K_ff.shape[0] == 0:
        raise ValueError("Global stiffness matrix is empty after boundary conditions")

    try:
        u_free = spsolve(K_ff, F_f)
    except Exception as e:
        raise ValueError(f"Linear solver failed: {e}")

    u[free_dofs] = u_free

    return u

# ============================================================================
# 计算柔顺度
# ============================================================================
def compute_compliance(u, F):
    C = np.dot(F, u)

    if C < 0:
        print(f"Warning: Negative compliance detected: C = {C}")

    return C

# ============================================================================
# 计算灵敏度（使用有限差分）
# ============================================================================
def compute_sensitivity(x, u, w, coeffs, nelx, nely, rho_min, rho_max):
    dc = np.zeros((nely, nelx))
    hx = 1.0
    hy = 1.0
    eps = 1e-6

    for ex in range(nelx):
        for ey in range(nely):
            x_e = x[ey, ex]
            rho_e = x_to_rho(x_e, rho_min, rho_max)

            n1 = ey * (nelx + 1) + ex
            n2 = ey * (nelx + 1) + ex + 1
            n3 = (ey + 1) * (nelx + 1) + ex + 1
            n4 = (ey + 1) * (nelx + 1) + ex
            dofs = [2*n1, 2*n1+1, 2*n2, 2*n2+1, 2*n3, 2*n3+1, 2*n4, 2*n4+1]
            u_e = u[dofs]

            try:
                # 直接计算单元刚度矩阵对 x 的导数（更高效）
                rho_e_plus = min(rho_max, rho_e + eps * (rho_max - rho_min))
                rho_e_minus = max(rho_min, rho_e - eps * (rho_max - rho_min))

                D_e_plus = construct_D_matrix(rho_e_plus, w, coeffs)
                D_e_minus = construct_D_matrix(rho_e_minus, w, coeffs)

                k_e_plus = quad4_element_stiffness(D_e_plus, hx, hy)
                k_e_minus = quad4_element_stiffness(D_e_minus, hx, hy)

                dk_e_dx = (k_e_plus - k_e_minus) / (2 * eps)

                dc[ey, ex] = -np.dot(u_e, np.dot(dk_e_dx, u_e))

            except Exception as e:
                # 简化处理，避免打印过多警告
                dc[ey, ex] = 0.0

    return dc

# ============================================================================
# OC 更新方法（修正版）
# ============================================================================
def oc_update(x, dc, dv, rho_target, rho_min, rho_max, move, nelx, nely):
    x_new = np.copy(x)
    l1 = 0
    l2 = 1e9

    # 计算目标 x 值
    x_target = rho_to_x(rho_target, rho_min, rho_max)

    for _ in range(20):  # 减少迭代次数，提高效率
        lmid = 0.5 * (l1 + l2)

        for i in range(nelx):
            for j in range(nely):
                if dc[j, i] > 0:
                    tmp = -dc[j, i] / (dv[j, i] * lmid)
                    if tmp > 0:
                        x_new[j, i] = max(0, min(1, x[j, i] * (tmp ** 0.5)))
                    else:
                        x_new[j, i] = max(0, min(1, x[j, i] - move))
                else:
                    x_new[j, i] = max(0, min(1, x[j, i] + move))

        x_new = np.clip(x_new, 0, 1)
        current_x = np.mean(x_new)

        if current_x > x_target:
            l1 = lmid
        else:
            l2 = lmid

        if (l2 - l1) < 1e-3:
            break

    # 强制体积约束：直接缩放以满足目标密度
    current_x = np.mean(x_new)
    if current_x != 0:
        scale_factor = x_target / current_x
        # 确保缩放后有足够的变化
        if abs(scale_factor - 1.0) > 1e-6:
            x_new = np.clip(x_new * scale_factor, 0, 1)

    return x_new

# ============================================================================
# 主优化循环
# ============================================================================
def optimize(x, w, coeffs, nelx, nely, rho_target, rmin, move, max_iter, tol, ft, rho_min, rho_max):
    history = {
        'iteration': [],
        'compliance': [],
        'mean_x': [],
        'mean_rho': [],
        'change': []
    }

    for iteration in range(max_iter):
        K = assemble_stiffness_matrix(x, w, coeffs, nelx, nely, rho_min, rho_max)

        F = np.zeros(2 * (nelx + 1) * (nely + 1))
        K, F, fixed_dofs = apply_boundary_conditions(K, F, nelx, nely)

        try:
            u = solve_fem(K, F, fixed_dofs)
        except ValueError as e:
            print(f"Solver error at iteration {iteration}: {e}")
            break

        try:
            C = compute_compliance(u, F)
        except ValueError as e:
            print(f"Compliance error at iteration {iteration}: {e}")
            break

        dc = compute_sensitivity(x, u, w, coeffs, nelx, nely, rho_min, rho_max)
        dv = np.ones((nely, nelx))

        x_old = x.copy()

        if ft == 1:
            x_filtered = density_filter(x, rmin, nelx, nely)
            dc_filtered = density_filter(dc * x, rmin, nelx, nely)
            with np.errstate(divide='ignore', invalid='ignore'):
                dc_filtered = np.where(x_filtered > 0, dc_filtered / x_filtered, 0)
            x = oc_update(x_filtered, dc_filtered, dv, rho_target, rho_min, rho_max, move, nelx, nely)
        else:
            x = oc_update(x, dc, dv, rho_target, rho_min, rho_max, move, nelx, nely)

        change = np.max(np.abs(x - x_old))

        mean_x = np.mean(x)
        mean_rho = np.mean(x_to_rho(x, rho_min, rho_max))

        history['iteration'].append(iteration)
        history['compliance'].append(C)
        history['mean_x'].append(mean_x)
        history['mean_rho'].append(mean_rho)
        history['change'].append(change)

        print(f"Iter {iteration}: C = {C:.6f}, mean_x = {mean_x:.4f}, mean_rho = {mean_rho:.4f}, Change = {change:.6f}")

        # 改进收敛条件
        if iteration > 0:
            if change < tol:
                print(f"Converged at iteration {iteration}")
                break
            # 添加最大迭代次数限制
            if iteration >= max_iter - 1:
                print(f"Reached maximum iterations: {max_iter}")
                break

    return x, u, history

# ============================================================================
# 主函数
# ============================================================================
def main():
    print("=" * 60)
    print("Macro Cantilever Density Optimization")
    print("=" * 60)

    print("\nLoading D(rho,w) interpolation coefficients...")
    coeffs = load_D_coefficients(Config.D_COEFF_FILE)
    print(f"Coefficients loaded: D1111, D1122, D1212")

    # 分析 D(rho,w) 模型
    print("\nAnalyzing D(rho,w) model...")
    analyze_D_model(coeffs, Config.w_fixed)

    nelx = Config.nelx
    nely = Config.nely
    rho_target = Config.rho_target
    w = Config.w_fixed
    rho_min = Config.rho_min
    rho_max = Config.rho_max

    x_initial = rho_to_x(rho_target, rho_min, rho_max)

    print(f"\nConfiguration:")
    print(f"  Grid size: {nelx} x {nely}")
    print(f"  Target relative density: {rho_target}")
    print(f"  Initial x: {x_initial:.4f}")
    print(f"  Fixed w: {w}")
    print(f"  Density filter radius: {Config.rmin}")
    print(f"  rho_min: {rho_min}, rho_max: {rho_max}")

    x = x_initial * np.ones((nely, nelx))

    # 添加初始扰动，确保优化能够开始
    np.random.seed(42)  # 固定随机种子，保证结果可重复
    x = x + np.random.normal(0, 0.05, x.shape)
    x = np.clip(x, 0, 1)

    print("\nStarting optimization...")
    x_opt, u_opt, history = optimize(
        x, w, coeffs, nelx, nely, rho_target,
        Config.rmin, Config.move, Config.max_iter, Config.tol, Config.ft,
        rho_min, rho_max
    )

    print("\nOptimization completed!")
    if len(history['compliance']) > 0:
        final_mean_x = history['mean_x'][-1]
        final_mean_rho = history['mean_rho'][-1]
        print(f"Final compliance: {history['compliance'][-1]:.6f}")
        print(f"Final mean(x): {final_mean_x:.4f}")
        print(f"Final mean(rho_actual): {final_mean_rho:.4f}")
        print(f"Target rho: {rho_target:.4f}")
        print(f"Error: {abs(final_mean_rho - rho_target):.4f}")

    print("\nGenerating output files...")

    if len(history['compliance']) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        ax = axes[0, 0]
        im = ax.imshow(x_opt, cmap='gray', aspect='auto')
        ax.set_title('Optimal Density Distribution (x)')
        ax.set_xlabel('Element X')
        ax.set_ylabel('Element Y')
        plt.colorbar(im, ax=ax, label='x')

        ax = axes[0, 1]
        rho_opt = x_to_rho(x_opt, rho_min, rho_max)
        im = ax.imshow(rho_opt, cmap='gray', aspect='auto')
        ax.set_title('Optimal Density Distribution (rho)')
        ax.set_xlabel('Element X')
        ax.set_ylabel('Element Y')
        plt.colorbar(im, ax=ax, label='rho')

        ax = axes[1, 0]
        ax.plot(history['iteration'], history['compliance'], 'b-', linewidth=2)
        ax.set_title('Compliance History')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Compliance C')
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        ax.plot(history['iteration'], history['mean_rho'], 'r-', linewidth=2)
        ax.axhline(y=rho_target, color='k', linestyle='--', label='Target rho')
        ax.set_title('Mean rho History')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Mean rho')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        output_path = os.path.join(Config.OUTPUT_DIR, 'macro_optimization_rho_corrected_result.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")

        plt.figure(figsize=(10, 6))
        plt.imshow(x_opt, cmap='gray', aspect='auto')
        plt.title('Optimal Density Distribution (x, w=0.6)')
        plt.xlabel('Element X')
        plt.ylabel('Element Y')
        plt.colorbar(label='x')
        output_path = os.path.join(Config.OUTPUT_DIR, 'macro_optimization_rho_corrected_distribution.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")

        plt.figure(figsize=(10, 6))
        u_magnitude = np.sqrt(u_opt[0::2].reshape(nely+1, nelx+1)**2 +
                              u_opt[1::2].reshape(nely+1, nelx+1)**2)
        plt.imshow(u_magnitude, cmap='viridis', aspect='auto')
        plt.title('Displacement Magnitude (mm)')
        plt.xlabel('Node X')
        plt.ylabel('Node Y')
        plt.colorbar(label='Displacement')
        output_path = os.path.join(Config.OUTPUT_DIR, 'macro_optimization_rho_corrected_displacement.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")

        plt.figure(figsize=(10, 6))
        plt.plot(history['iteration'], history['compliance'], 'b-', linewidth=2)
        plt.title('Compliance History')
        plt.xlabel('Iteration')
        plt.ylabel('Compliance C')
        plt.grid(True, alpha=0.3)
        output_path = os.path.join(Config.OUTPUT_DIR, 'macro_optimization_rho_corrected_compliance_history.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")

        plt.figure(figsize=(10, 6))
        plt.plot(history['iteration'], history['mean_rho'], 'r-', linewidth=2)
        plt.axhline(y=rho_target, color='k', linestyle='--', label='Target rho')
        plt.title('Mean rho History')
        plt.xlabel('Iteration')
        plt.ylabel('Mean rho')
        plt.legend()
        plt.grid(True, alpha=0.3)
        output_path = os.path.join(Config.OUTPUT_DIR, 'macro_optimization_rho_corrected_volume_history.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")

        results_df = pd.DataFrame({
            'iteration': history['iteration'],
            'compliance': history['compliance'],
            'mean_x': history['mean_x'],
            'mean_rho': history['mean_rho'],
            'change': history['change']
        })
        output_path = os.path.join(Config.OUTPUT_DIR, 'macro_optimization_rho_corrected_result.csv')
        results_df.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")

        converged = len(history['iteration']) < Config.max_iter

        report = f"""# Macro Cantilever Density Optimization Report (Corrected)

## Configuration
- Grid size: {nelx} x {nely}
- Target relative density: {rho_target}
- Fixed w: {w}
- Density filter radius: {Config.rmin}
- Max iterations: {Config.max_iter}
- Convergence tolerance: {Config.tol}
- rho_min: {rho_min}
- rho_max: {rho_max}

## Results
- Final compliance: {history['compliance'][-1]:.6f}
- Final mean(x): {final_mean_x:.4f}
- Final mean(rho_actual): {final_mean_rho:.4f}
- Target rho: {rho_target:.4f}
- Error: {abs(final_mean_rho - rho_target):.4f}
- Number of iterations: {len(history['iteration'])}

## Convergence Analysis
The optimization {'converged' if converged else 'reached maximum iterations'}.

## Files Generated
- macro_optimization_rho_corrected_distribution.png
- macro_optimization_rho_corrected_displacement.png
- macro_optimization_rho_corrected_compliance_history.png
- macro_optimization_rho_corrected_volume_history.png
- macro_optimization_rho_corrected_result.csv

## Notes
This is a verification version that:
1. Fixes w = 0.6 and only optimizes rho
2. Uses OC (Optimality Criteria) method with corrected density transformation
3. Includes density filtering to avoid checkerboard patterns
4. Validates the coupling between D(rho,w) interpolation and macro FEM
5. Uses finite difference method for sensitivity analysis

This version is specifically designed to verify the correct coupling between the D(rho,w) interpolation model and the macro FEM optimization framework, not to produce final rho + w bi-variable MMA optimization results.
"""

        output_path = os.path.join(Config.OUTPUT_DIR, 'macro_optimization_rho_corrected_report.md')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Saved: {output_path}")

    print("\n" + "=" * 60)
    print("All outputs saved to:", Config.OUTPUT_DIR)
    print("=" * 60)

if __name__ == "__main__":
    main()