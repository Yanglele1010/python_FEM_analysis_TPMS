#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成拟合曲面图
生成 D1111_fit_surface.png、D1122_fit_surface.png、D1212_fit_surface.png
图中同时显示原始有效数据点、拟合曲面、坐标轴 rho, w, D
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义文件路径
INPUT_DATA = r"E:\code\new\04_Final_Results\rve_pbc_effective_valid.csv"
INPUT_COEFF = r"E:\code\new\04_Final_Results\D_interpolation_coefficients.csv"
OUTPUT_DIR = r"E:\code\new\04_Final_Results"

# 读取数据
df_data = pd.read_csv(INPUT_DATA)
df_coeff = pd.read_csv(INPUT_COEFF)

# 确定自变量
if 'actual_rho_rve' in df_data.columns and 'actual_w_rve' in df_data.columns:
    print("使用 actual_rho_rve 和 actual_w_rve 作为自变量")
    rho = df_data['actual_rho_rve'].values
    w = df_data['actual_w_rve'].values
elif 'actual_rho' in df_data.columns and 'actual_w' in df_data.columns:
    print("使用 actual_rho 和 actual_w 作为自变量")
    rho = df_data['actual_rho'].values
    w = df_data['actual_w'].values
else:
    print("使用 target_rho 和 target_w 作为自变量")
    rho = df_data['target_rho'].values
    w = df_data['target_w'].values

# 定义拟合函数
def fit_func(rho, w, coeffs):
    b1, b2, b3, b4, b5, b6, b7, b8, b9 = coeffs
    return (b1 * rho + b2 * rho**2 + b3 * rho**3 + b4 * rho**4 + b5 * rho**5 +
            b6 * rho * w + b7 * rho * w**2 + b8 * rho**2 * w + b9 * rho**2 * w**2)

# 获取系数
coeffs_d1111 = df_coeff['D1111'].values
coeffs_d1122 = df_coeff['D1122'].values
coeffs_d1212 = df_coeff['D1212'].values

# 生成网格点
rho_min, rho_max = rho.min(), rho.max()
w_min, w_max = w.min(), w.max()
rho_grid, w_grid = np.meshgrid(np.linspace(rho_min, rho_max, 50),
                              np.linspace(w_min, w_max, 50))

# 计算拟合值
d1111_grid = fit_func(rho_grid, w_grid, coeffs_d1111)
d1122_grid = fit_func(rho_grid, w_grid, coeffs_d1122)
d1212_grid = fit_func(rho_grid, w_grid, coeffs_d1212)

# 绘制 D1111 拟合曲面
print("绘制 D1111 拟合曲面...")
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制原始数据点
ax.scatter(rho, w, df_data['D1111'].values, c='r', s=50, label='Original Data')

# 绘制拟合曲面
surf = ax.plot_surface(rho_grid, w_grid, d1111_grid, alpha=0.7, cmap='viridis', label='Fitted Surface')

# 设置标签和标题
ax.set_xlabel('rho')
ax.set_ylabel('w')
ax.set_zlabel('D1111')
ax.set_title('D1111 Fit Surface')
ax.legend()

# 添加颜色条
fig.colorbar(surf, ax=ax)

# 保存图像
output_file = os.path.join(OUTPUT_DIR, 'D1111_fit_surface.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"D1111 拟合曲面已保存到: {output_file}")

# 绘制 D1122 拟合曲面
print("绘制 D1122 拟合曲面...")
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制原始数据点
ax.scatter(rho, w, df_data['D1122'].values, c='r', s=50, label='Original Data')

# 绘制拟合曲面
surf = ax.plot_surface(rho_grid, w_grid, d1122_grid, alpha=0.7, cmap='viridis', label='Fitted Surface')

# 设置标签和标题
ax.set_xlabel('rho')
ax.set_ylabel('w')
ax.set_zlabel('D1122')
ax.set_title('D1122 Fit Surface')
ax.legend()

# 添加颜色条
fig.colorbar(surf, ax=ax)

# 保存图像
output_file = os.path.join(OUTPUT_DIR, 'D1122_fit_surface.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"D1122 拟合曲面已保存到: {output_file}")

# 绘制 D1212 拟合曲面
print("绘制 D1212 拟合曲面...")
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制原始数据点
ax.scatter(rho, w, df_data['D1212'].values, c='r', s=50, label='Original Data')

# 绘制拟合曲面
surf = ax.plot_surface(rho_grid, w_grid, d1212_grid, alpha=0.7, cmap='viridis', label='Fitted Surface')

# 设置标签和标题
ax.set_xlabel('rho')
ax.set_ylabel('w')
ax.set_zlabel('D1212')
ax.set_title('D1212 Fit Surface')
ax.legend()

# 添加颜色条
fig.colorbar(surf, ax=ax)

# 保存图像
output_file = os.path.join(OUTPUT_DIR, 'D1212_fit_surface.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"D1212 拟合曲面已保存到: {output_file}")

print("所有拟合曲面图已生成完成!")