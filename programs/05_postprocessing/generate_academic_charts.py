# -*- coding: utf-8 -*-
"""
生成学术风格图表（Nature/Science 风格）
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from mpl_toolkits.mplot3d import Axes3D
import os

# ============================================================
# 字体和样式配置
# ============================================================
# 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Nature/Science 风格配色
COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b']
MARKERS = ['o', 's', '^', 'D', 'v', 'p']
LINE_STYLES = ['-', '--', '-.', ':', '-', '--']

# 输出目录
OUTPUT_DIR = r"E:\code\new\04_Final_Results"

# ============================================================
# 图表 1: RVE 等效弹性常数 vs w（按 rho 分组）
# ============================================================
def plot_rve_properties():
    """绘制 RVE 等效弹性常数随 w 变化的曲线"""
    df = pd.read_csv(os.path.join(OUTPUT_DIR, 'rve_pbc_effective_valid.csv'), encoding='utf-8-sig')

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle('RVE-PBC 等效弹性常数随互穿参数 w 的变化', fontsize=14, fontweight='bold')

    properties = [
        ('D1111', 'D1111 (MPa)', axes[0, 0]),
        ('D1122', 'D1122 (MPa)', axes[0, 1]),
        ('D1212', 'D1212 (MPa)', axes[1, 0]),
        ('E_eff', 'E_eff (MPa)', axes[1, 1]),
    ]

    rho_values = sorted(df['target_rho'].unique())

    for prop_name, prop_label, ax in properties:
        for i, rho in enumerate(rho_values):
            mask = df['target_rho'] == rho
            subset = df[mask].sort_values('target_w')
            if len(subset) > 0:
                ax.plot(subset['target_w'], subset[prop_name],
                       color=COLORS[i % len(COLORS)],
                       marker=MARKERS[i % len(MARKERS)],
                       linestyle=LINE_STYLES[i % len(LINE_STYLES)],
                       linewidth=1.5, markersize=6,
                       label=f'ρ = {rho:.2f}')

        ax.set_xlabel('互穿参数 w', fontsize=11)
        ax.set_ylabel(prop_label, fontsize=11)
        ax.set_title(prop_label, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 1.05)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'fig_rve_properties_vs_w.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


# ============================================================
# 图表 2: 固定 w 优化柔顺度对比
# ============================================================
def plot_compliance_comparison():
    """绘制固定 w 优化的初始/最终柔顺度对比柱状图"""
    df = pd.read_csv(os.path.join(OUTPUT_DIR, 'fixed_w_sweep_summary.csv'), encoding='utf-8-sig')

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('不同互穿参数 w 下的悬臂梁柔顺度优化结果', fontsize=14, fontweight='bold')

    w_values = df['w'].values
    x = np.arange(len(w_values))
    width = 0.35

    # 初始柔顺度
    bars1 = ax.bar(x - width/2, df['compliance_initial'], width,
                   label='初始柔顺度', color=COLORS[0], alpha=0.8)
    # 最终柔顺度
    bars2 = ax.bar(x + width/2, df['compliance_final'], width,
                   label='最终柔顺度', color=COLORS[1], alpha=0.8)

    # 双变量优化参考线
    bivariate_compliance = 2.2936
    ax.axhline(y=bivariate_compliance, color=COLORS[2], linestyle='--',
               linewidth=2, label=f'双变量优化 (C = {bivariate_compliance:.3f})')

    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('互穿参数 w', fontsize=12)
    ax.set_ylabel('柔顺度 (N·mm)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{w:.1f}' for w in w_values])
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 40)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'fig_compliance_comparison.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


# ============================================================
# 图表 3: D(rho,w) 插值模型拟合曲面
# ============================================================
def plot_d_interpolation_surface():
    """绘制 D(rho,w) 插值模型的 3D 曲面图"""
    # 读取系数
    coeffs_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'D_interpolation_coefficients.csv'), encoding='utf-8-sig')
    coeffs = {}
    for comp in ['D1111', 'D1122', 'D1212']:
        coeffs[comp] = coeffs_df[comp].values.astype(np.float64)

    # 读取实际数据点
    data_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'rve_pbc_effective_valid.csv'), encoding='utf-8-sig')

    # 创建网格
    rho = np.linspace(0.20, 0.42, 50)
    w = np.linspace(0.0, 1.0, 50)
    RHO, W = np.meshgrid(rho, w)

    # 计算 D 值
    def compute_D(rho_vec, w_vec, coeffs):
        r = rho_vec
        r2 = r * r
        r3 = r2 * r
        r4 = r3 * r
        r5 = r4 * r
        rw = r * w_vec
        rw2 = r * w_vec * w_vec
        r2w = r2 * w_vec
        r2w2 = r2 * w_vec * w_vec

        D = (coeffs[0]*r + coeffs[1]*r2 + coeffs[2]*r3 +
             coeffs[3]*r4 + coeffs[4]*r5 + coeffs[5]*rw +
             coeffs[6]*rw2 + coeffs[7]*r2w + coeffs[8]*r2w2)
        return D

    fig = plt.figure(figsize=(15, 5))

    for idx, comp in enumerate(['D1111', 'D1122', 'D1212']):
        ax = fig.add_subplot(1, 3, idx+1, projection='3d')

        # 计算曲面
        D_surface = np.zeros_like(RHO)
        for i in range(len(w)):
            for j in range(len(rho)):
                D_surface[i, j] = compute_D(
                    np.array([rho[j]]), np.array([w[i]]), coeffs[comp]
                )[0]

        # 绘制曲面
        surf = ax.plot_surface(RHO, W, D_surface, cmap='viridis', alpha=0.7)

        # 叠加实际数据点
        ax.scatter(data_df['actual_rho_rve'], data_df['actual_w_rve'],
                  data_df[comp], color='red', s=50, zorder=5, label='实际数据')

        ax.set_xlabel('ρ', fontsize=10)
        ax.set_ylabel('w', fontsize=10)
        ax.set_zlabel(f'{comp} (MPa)', fontsize=10)
        ax.set_title(f'{comp} 插值曲面', fontsize=12, fontweight='bold')

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'fig_d_interpolation_surface.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


# ============================================================
# 图表 4: 优化策略对比
# ============================================================
def plot_optimization_comparison():
    """绘制不同优化策略的对比图"""
    # 固定 w 优化数据
    fixed_w_data = {
        'w': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        'compliance_final': [10.367, 7.290, 5.313, 3.868, 3.084, 2.411],
        'reduction': [68.69, 43.97, 30.54, 25.18, 18.13, 16.32],
        'iterations': [23, 23, 27, 82, 56, 57],
    }

    # 双变量优化数据
    bivariate = {
        'compliance_final': 2.294,
        'reduction': 63.44,
        'iterations': 24,
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle('不同优化策略性能对比', fontsize=14, fontweight='bold')

    # 子图1: 最终柔顺度
    ax1 = axes[0]
    x = np.arange(len(fixed_w_data['w']))
    bars = ax1.bar(x, fixed_w_data['compliance_final'], color=COLORS[0], alpha=0.8, label='固定 w 优化')
    ax1.axhline(y=bivariate['compliance_final'], color=COLORS[1], linestyle='--',
                linewidth=2, label=f'双变量优化 ({bivariate["compliance_final"]:.3f})')
    ax1.set_xlabel('互穿参数 w', fontsize=11)
    ax1.set_ylabel('最终柔顺度 (N·mm)', fontsize=11)
    ax1.set_title('最终柔顺度', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{w:.1f}' for w in fixed_w_data['w']])
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')

    # 子图2: 柔顺度减少率
    ax2 = axes[1]
    bars2 = ax2.bar(x, fixed_w_data['reduction'], color=COLORS[2], alpha=0.8, label='固定 w 优化')
    ax2.axhline(y=bivariate['reduction'], color=COLORS[1], linestyle='--',
                linewidth=2, label=f'双变量优化 ({bivariate["reduction"]:.1f}%)')
    ax2.set_xlabel('互穿参数 w', fontsize=11)
    ax2.set_ylabel('柔顺度减少率 (%)', fontsize=11)
    ax2.set_title('柔顺度减少率', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{w:.1f}' for w in fixed_w_data['w']])
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')

    # 子图3: 迭代次数
    ax3 = axes[2]
    bars3 = ax3.bar(x, fixed_w_data['iterations'], color=COLORS[3], alpha=0.8, label='固定 w 优化')
    ax3.axhline(y=bivariate['iterations'], color=COLORS[1], linestyle='--',
                linewidth=2, label=f'双变量优化 ({bivariate["iterations"]})')
    ax3.set_xlabel('互穿参数 w', fontsize=11)
    ax3.set_ylabel('迭代次数', fontsize=11)
    ax3.set_title('收敛迭代次数', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'{w:.1f}' for w in fixed_w_data['w']])
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'fig_optimization_comparison.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("生成学术风格图表")
    print("=" * 50)

    plot_rve_properties()
    plot_compliance_comparison()
    plot_d_interpolation_surface()
    plot_optimization_comparison()

    print("\n所有图表生成完成！")
    print(f"输出目录: {OUTPUT_DIR}")
