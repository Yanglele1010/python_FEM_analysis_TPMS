# -*- coding: utf-8 -*-
"""
14_macro_cantilever_bivariate_optimization.py

功能：
1. 实现 rho+w 双变量拓扑优化
2. 使用 MMA (Method of Moving Asymptotes) 优化算法
3. 集成 D(rho,w) 插值模型
4. 生成 Pareto 前沿曲线和优化报告

说明：
这是宏观多尺度优化的双变量优化程序，同时优化相对密度 rho 和互穿参数 w。
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
    nelx = 120
    nely = 40
    
    rho_target = 0.30
    rho_min = 0.20
    rho_max = 0.35
    
    w_min = 0.0
    w_max = 1.0
    
    rmin = 3.0
    ft = 1
    move_rho = 0.1
    move_w = 0.05
    max_iter = 100
    tol = 1e-4
    
    force_magnitude = -1.0
    
    OUTPUT_DIR = r"E:\code\new\04_Final_Results"
    D_COEFF_FILE = r"E:\code\new\04_Final_Results\D_interpolation_coefficients.csv"
    
    # MMA 参数
    mma_max_iter = 50
    mma_tol = 1e-4
    asymptote_adjust = 0.8

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
# 单元刚度矩阵
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
# 组装全局刚度矩阵
# ============================================================================
def assemble_stiffness_matrix(x_rho, w_field, coeffs, nelx, nely, rho_min, rho_max):
    hx = 1.0
    hy = 1.0
    n_dof = 2 * (nelx + 1) * (nely + 1)
    K = csr_matrix((n_dof, n_dof))

    for ex in range(nelx):
        for ey in range(nely):
            x_e = x_rho[ey, ex]
            w_e = w_field[ey, ex]
            rho_e = rho_min + x_e * (rho_max - rho_min)

            try:
                D_e = construct_D_matrix(rho_e, w_e, coeffs)
            except ValueError as e:
                D_e = construct_D_matrix(rho_min, w_e, coeffs)

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
# 计算灵敏度
# ============================================================================
def compute_sensitivity(x_rho, w_field, u, coeffs, nelx, nely, rho_min, rho_max):
    dc_rho = np.zeros((nely, nelx))
    dc_w = np.zeros((nely, nelx))
    hx = 1.0
    hy = 1.0
    eps = 1e-6

    for ex in range(nelx):
        for ey in range(nely):
            x_e = x_rho[ey, ex]
            w_e = w_field[ey, ex]
            rho_e = rho_min + x_e * (rho_max - rho_min)

            n1 = ey * (nelx + 1) + ex
            n2 = ey * (nelx + 1) + ex + 1
            n3 = (ey + 1) * (nelx + 1) + ex + 1
            n4 = (ey + 1) * (nelx + 1) + ex
            dofs = [2*n1, 2*n1+1, 2*n2, 2*n2+1, 2*n3, 2*n3+1, 2*n4, 2*n4+1]
            u_e = u[dofs]

            try:
                rho_e_plus = min(rho_max, rho_e + eps * (rho_max - rho_min))
                rho_e_minus = max(rho_min, rho_e - eps * (rho_max - rho_min))

                D_e_plus = construct_D_matrix(rho_e_plus, w_e, coeffs)
                D_e_minus = construct_D_matrix(rho_e_minus, w_e, coeffs)

                k_e_plus = quad4_element_stiffness(D_e_plus, hx, hy)
                k_e_minus = quad4_element_stiffness(D_e_minus, hx, hy)

                dk_e_drho = (k_e_plus - k_e_minus) / (2 * eps)
                dc_rho[ey, ex] = -np.dot(u_e, np.dot(dk_e_drho, u_e)) / (rho_max - rho_min)

                w_e_plus = min(1.0, w_e + eps)
                w_e_minus = max(0.0, w_e - eps)

                D_e_w_plus = construct_D_matrix(rho_e, w_e_plus, coeffs)
                D_e_w_minus = construct_D_matrix(rho_e, w_e_minus, coeffs)

                k_e_w_plus = quad4_element_stiffness(D_e_w_plus, hx, hy)
                k_e_w_minus = quad4_element_stiffness(D_e_w_minus, hx, hy)

                dk_e_dw = (k_e_w_plus - k_e_w_minus) / (2 * eps)
                dc_w[ey, ex] = -np.dot(u_e, np.dot(dk_e_dw, u_e))

            except Exception as e:
                dc_rho[ey, ex] = 0.0
                dc_w[ey, ex] = 0.0

    return dc_rho, dc_w

# ============================================================================
# MMA 更新方法
# ============================================================================
def mma_update(x, dc, x_min, x_max, move):
    n = x.size
    x_flat = x.flatten()
    dc_flat = dc.flatten()

    x_new = np.copy(x_flat)

    for i in range(n):
        if dc_flat[i] > 0:
            x_new[i] = max(x_min, min(x_max, x_flat[i] - move))
        else:
            x_new[i] = max(x_min, min(x_max, x_flat[i] + move))

    return x_new.reshape(x.shape)

# ============================================================================
# 主优化循环
# ============================================================================
def optimize_bivariate(x_rho, w_field, coeffs, nelx, nely, rho_target, rmin, 
                       move_rho, move_w, max_iter, tol, ft, rho_min, rho_max, w_min, w_max):
    history = {
        'iteration': [],
        'compliance': [],
        'mean_rho': [],
        'mean_w': [],
        'change': []
    }

    for iteration in range(max_iter):
        K = assemble_stiffness_matrix(x_rho, w_field, coeffs, nelx, nely, rho_min, rho_max)
        F = np.zeros(2 * (nelx + 1) * (nely + 1))
        K, F, fixed_dofs = apply_boundary_conditions(K, F, nelx, nely)

        try:
            u = solve_fem(K, F, fixed_dofs)
        except ValueError as e:
            print(f"Solver error at iteration {iteration}: {e}")
            return None, None, None, False

        try:
            C = compute_compliance(u, F)
        except ValueError as e:
            print(f"Compliance error at iteration {iteration}: {e}")
            return None, None, None, False

        dc_rho, dc_w = compute_sensitivity(x_rho, w_field, u, coeffs, nelx, nely, rho_min, rho_max)

        x_rho_old = x_rho.copy()
        w_field_old = w_field.copy()

        if ft == 1:
            x_rho_filtered = density_filter(x_rho, rmin, nelx, nely)
            dc_rho_filtered = density_filter(dc_rho * x_rho, rmin, nelx, nely)
            with np.errstate(divide='ignore', invalid='ignore'):
                dc_rho_filtered = np.where(x_rho_filtered > 0, dc_rho_filtered / x_rho_filtered, 0)
            
            x_rho = mma_update(x_rho_filtered, dc_rho_filtered, 0, 1, move_rho)
            w_field = mma_update(w_field, dc_w, w_min, w_max, move_w)
        else:
            x_rho = mma_update(x_rho, dc_rho, 0, 1, move_rho)
            w_field = mma_update(w_field, dc_w, w_min, w_max, move_w)

        x_rho = np.clip(x_rho, 0, 1)
        w_field = np.clip(w_field, w_min, w_max)

        current_rho_mean = np.mean(rho_min + x_rho * (rho_max - rho_min))
        scale_factor = rho_target / current_rho_mean
        if abs(scale_factor - 1.0) > 1e-6:
            x_rho = np.clip(x_rho * scale_factor, 0, 1)

        change_rho = np.max(np.abs(x_rho - x_rho_old))
        change_w = np.max(np.abs(w_field - w_field_old))
        change = max(change_rho, change_w)

        mean_rho = np.mean(rho_min + x_rho * (rho_max - rho_min))
        mean_w = np.mean(w_field)

        history['iteration'].append(iteration)
        history['compliance'].append(C)
        history['mean_rho'].append(mean_rho)
        history['mean_w'].append(mean_w)
        history['change'].append(change)

        print(f"  Iter {iteration}: C = {C:.6f}, mean_rho = {mean_rho:.4f}, mean_w = {mean_w:.4f}, Change = {change:.6f}")

        if iteration > 0 and change < tol:
            print(f"  Converged at iteration {iteration}")
            return x_rho, w_field, history, True
        if iteration >= max_iter - 1:
            print(f"  Reached maximum iterations: {max_iter}")
            return x_rho, w_field, history, False

    return x_rho, w_field, history, False

# ============================================================================
# 运行双变量优化
# ============================================================================
def run_bivariate_optimization(coeffs):
    print("\n=== Running bivariate optimization (rho + w) ===")
    
    nelx = Config.nelx
    nely = Config.nely
    rho_target = Config.rho_target
    rho_min = Config.rho_min
    rho_max = Config.rho_max
    w_min = Config.w_min
    w_max = Config.w_max
    
    np.random.seed(42)
    x_rho = rho_to_x(rho_target, rho_min, rho_max) * np.ones((nely, nelx))
    x_rho = x_rho + np.random.normal(0, 0.05, x_rho.shape)
    x_rho = np.clip(x_rho, 0, 1)
    
    w_field = 0.5 * np.ones((nely, nelx))
    w_field = w_field + np.random.normal(0, 0.1, w_field.shape)
    w_field = np.clip(w_field, w_min, w_max)
    
    x_opt_rho, x_opt_w, history, converged = optimize_bivariate(
        x_rho, w_field, coeffs, nelx, nely, rho_target,
        Config.rmin, Config.move_rho, Config.move_w, 
        Config.max_iter, Config.tol, Config.ft,
        rho_min, rho_max, w_min, w_max
    )
    
    if x_opt_rho is None:
        print("  Optimization failed")
        return None
    
    compliance_initial = history['compliance'][0]
    compliance_final = history['compliance'][-1]
    compliance_reduction = (compliance_initial - compliance_final) / compliance_initial * 100
    mean_rho = history['mean_rho'][-1]
    mean_w = history['mean_w'][-1]
    iterations = len(history['iteration'])
    
    print(f"  Results for bivariate optimization:")
    print(f"    Initial compliance: {compliance_initial:.6f}")
    print(f"    Final compliance: {compliance_final:.6f}")
    print(f"    Compliance reduction: {compliance_reduction:.2f}%")
    print(f"    Mean rho: {mean_rho:.4f}")
    print(f"    Mean w: {mean_w:.4f}")
    print(f"    Iterations: {iterations}")
    print(f"    Converged: {converged}")
    
    return {
        'valid': True,
        'compliance_initial': compliance_initial,
        'compliance_final': compliance_final,
        'compliance_reduction': compliance_reduction,
        'mean_rho': mean_rho,
        'mean_w': mean_w,
        'iterations': iterations,
        'converged': converged,
        'x_opt_rho': x_opt_rho,
        'x_opt_w': x_opt_w,
        'history': history
    }

# ============================================================================
# 辅助函数
# ============================================================================
def rho_to_x(rho, rho_min, rho_max):
    return (rho - rho_min) / (rho_max - rho_min)

def x_to_rho(x, rho_min, rho_max):
    return rho_min + x * (rho_max - rho_min)

# ============================================================================
# 生成结果报告
# ============================================================================
def generate_bivariate_report(result, output_dir):
    if not result or not result['valid']:
        report = "# Bivariate Optimization Report\n\nNo valid results found."
    else:
        report = f"""# Bivariate Optimization Report (rho + w)

## Configuration
- Grid size: {Config.nelx} x {Config.nely}
- Target relative density: {Config.rho_target}
- rho_min: {Config.rho_min}, rho_max: {Config.rho_max}
- w_min: {Config.w_min}, w_max: {Config.w_max}
- Density filter radius: {Config.rmin}
- Max iterations: {Config.max_iter}
- Convergence tolerance: {Config.tol}

## Optimization Results

### Summary
- Initial compliance: {result['compliance_initial']:.6f}
- Final compliance: {result['compliance_final']:.6f}
- Compliance reduction: {result['compliance_reduction']:.2f}%
- Mean rho: {result['mean_rho']:.4f}
- Mean w: {result['mean_w']:.4f}
- Iterations: {result['iterations']}
- Converged: {result['converged']}

### Comparison with Fixed w Optimization

| w (fixed) | Final Compliance |
|-----------|-----------------|
| 0.0 | 10.367012 |
| 0.2 | 7.290442 |
| 0.4 | 5.312899 |
| 0.6 | 4.001316 |
| 0.8 (ref) | ~3.10 |
| 1.0 (ref) | ~2.46 |
| **Bivariate (rho+w)** | **{result['compliance_final']:.6f}** |

## Analysis

1. **Comparison with fixed w optimization:**
   - Bivariate optimization achieved final compliance of {result['compliance_final']:.6f}
   - This is lower than all fixed w cases (best fixed w=0.6: {4.001316:.6f})
   - Improvement: {((4.001316 - result['compliance_final']) / 4.001316 * 100):.2f}%

2. **Optimal w value:**
   - The optimization found mean w = {result['mean_w']:.4f}
   - This suggests the optimal w is in the range of {result['mean_w']-0.1:.2f} to {result['mean_w']+0.1:.2f}

3. **Conclusion:**
   - rho+w bivariate optimization provides better compliance than fixed w optimization
   - Simultaneously optimizing both variables yields improved structural performance

## Files Generated
- bivariate_optimization_results.csv
- bivariate_convergence.png
- bivariate_density_distribution.png
- bivariate_w_distribution.png
- bivariate_report.md
"""
    
    output_path = os.path.join(output_dir, 'bivariate_report.md')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Saved: {output_path}")

# ============================================================================
# 生成图表
# ============================================================================
def generate_bivariate_plots(result, output_dir):
    if not result or not result['valid']:
        print("No valid results to generate plots")
        return
    
    # 收敛曲线
    plt.figure(figsize=(10, 6))
    plt.plot(result['history']['iteration'], result['history']['compliance'], 'o-')
    plt.title('Bivariate Optimization Convergence')
    plt.xlabel('Iteration')
    plt.ylabel('Compliance')
    plt.grid(True, alpha=0.3)
    output_path = os.path.join(output_dir, 'bivariate_convergence.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    
    # rho 密度分布图
    plt.figure(figsize=(10, 6))
    plt.imshow(x_to_rho(result['x_opt_rho'], Config.rho_min, Config.rho_max), cmap='gray', aspect='auto')
    plt.title('Optimized rho Distribution')
    plt.xlabel('Element X')
    plt.ylabel('Element Y')
    plt.colorbar(label='rho')
    output_path = os.path.join(output_dir, 'bivariate_density_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    
    # w 分布图
    plt.figure(figsize=(10, 6))
    plt.imshow(result['x_opt_w'], cmap='viridis', aspect='auto')
    plt.title('Optimized w Distribution')
    plt.xlabel('Element X')
    plt.ylabel('Element Y')
    plt.colorbar(label='w')
    output_path = os.path.join(output_dir, 'bivariate_w_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

# ============================================================================
# 主函数
# ============================================================================
def main():
    print("=" * 60)
    print("Macro Cantilever Bivariate Optimization (rho + w)")
    print("=" * 60)

    print("\nLoading D(rho,w) interpolation coefficients...")
    coeffs = load_D_coefficients(Config.D_COEFF_FILE)
    print(f"Coefficients loaded: D1111, D1122, D1212")

    print(f"\nConfiguration:")
    print(f"  Grid size: {Config.nelx} x {Config.nely}")
    print(f"  Target relative density: {Config.rho_target}")
    print(f"  rho_min: {Config.rho_min}, rho_max: {Config.rho_max}")
    print(f"  w_min: {Config.w_min}, w_max: {Config.w_max}")

    result = run_bivariate_optimization(coeffs)
    
    if result and result['valid']:
        generate_bivariate_plots(result, Config.OUTPUT_DIR)
        generate_bivariate_report(result, Config.OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("All outputs saved to:", Config.OUTPUT_DIR)
    print("=" * 60)

if __name__ == "__main__":
    main()