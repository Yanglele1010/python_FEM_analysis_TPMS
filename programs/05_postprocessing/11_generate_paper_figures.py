#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成论文结果图
生成 E_vs_w_all_rho_valid.png、G_vs_w_all_rho_valid.png、nu_vs_w_all_rho_valid.png、A_zener_vs_w_all_rho_valid.png
生成 E_heatmap_valid.png、G_heatmap_valid.png、A_zener_heatmap_valid.png
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# 定义文件路径
INPUT_FILE = r"E:\code\new\04_Final_Results\rve_pbc_effective_valid.csv"
OUTPUT_DIR = r"E:\code\new\04_Final_Results"

# 读取数据
df = pd.read_csv(INPUT_FILE)
print(f"读取数据点数量: {len(df)}")

# 确定自变量
if 'actual_rho_rve' in df.columns and 'actual_w_rve' in df.columns:
    print("使用 actual_rho_rve 和 actual_w_rve 作为自变量")
    rho_col = 'actual_rho_rve'
    w_col = 'actual_w_rve'
elif 'actual_rho' in df.columns and 'actual_w' in df.columns:
    print("使用 actual_rho 和 actual_w 作为自变量")
    rho_col = 'actual_rho'
    w_col = 'actual_w'
else:
    print("使用 target_rho 和 target_w 作为自变量")
    rho_col = 'target_rho'
    w_col = 'target_w'

# 计算 A_zener
df['A_zener'] = df['E_eff'] / (2 * df['G_eff'] * (1 + df['nu_eff']))

# 生成 E_vs_w_all_rho_valid.png
print("生成 E_vs_w_all_rho_valid.png...")
plt.figure(figsize=(10, 6))

# 按 rho 分组
rho_values = sorted(df[rho_col].unique())
colors = plt.cm.viridis(np.linspace(0, 1, len(rho_values)))

for i, rho in enumerate(rho_values):
    df_rho = df[df[rho_col] == rho]
    plt.scatter(df_rho[w_col], df_rho['E_eff'], label=f'rho={rho:.2f}', color=colors[i], s=50)

plt.xlabel('w')
plt.ylabel('E_eff (MPa)')
plt.title('Young\'s Modulus vs w for Different rho')
plt.legend()
plt.grid(True)

output_file = os.path.join(OUTPUT_DIR, 'E_vs_w_all_rho_valid.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"E_vs_w_all_rho_valid.png 已保存到: {output_file}")

# 生成 G_vs_w_all_rho_valid.png
print("生成 G_vs_w_all_rho_valid.png...")
plt.figure(figsize=(10, 6))

for i, rho in enumerate(rho_values):
    df_rho = df[df[rho_col] == rho]
    plt.scatter(df_rho[w_col], df_rho['G_eff'], label=f'rho={rho:.2f}', color=colors[i], s=50)

plt.xlabel('w')
plt.ylabel('G_eff (MPa)')
plt.title('Shear Modulus vs w for Different rho')
plt.legend()
plt.grid(True)

output_file = os.path.join(OUTPUT_DIR, 'G_vs_w_all_rho_valid.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"G_vs_w_all_rho_valid.png 已保存到: {output_file}")

# 生成 nu_vs_w_all_rho_valid.png
print("生成 nu_vs_w_all_rho_valid.png...")
plt.figure(figsize=(10, 6))

for i, rho in enumerate(rho_values):
    df_rho = df[df[rho_col] == rho]
    plt.scatter(df_rho[w_col], df_rho['nu_eff'], label=f'rho={rho:.2f}', color=colors[i], s=50)

plt.xlabel('w')
plt.ylabel('nu_eff')
plt.title('Poisson\'s Ratio vs w for Different rho')
plt.legend()
plt.grid(True)

output_file = os.path.join(OUTPUT_DIR, 'nu_vs_w_all_rho_valid.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"nu_vs_w_all_rho_valid.png 已保存到: {output_file}")

# 生成 A_zener_vs_w_all_rho_valid.png
print("生成 A_zener_vs_w_all_rho_valid.png...")
plt.figure(figsize=(10, 6))

for i, rho in enumerate(rho_values):
    df_rho = df[df[rho_col] == rho]
    plt.scatter(df_rho[w_col], df_rho['A_zener'], label=f'rho={rho:.2f}', color=colors[i], s=50)

plt.xlabel('w')
plt.ylabel('A_zener')
plt.title('Zener Ratio vs w for Different rho')
plt.legend()
plt.grid(True)

output_file = os.path.join(OUTPUT_DIR, 'A_zener_vs_w_all_rho_valid.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"A_zener_vs_w_all_rho_valid.png 已保存到: {output_file}")

# 生成热图
from scipy.interpolate import griddata

# 准备数据
x = df[w_col].values
y = df[rho_col].values

# 生成 E_heatmap_valid.png
print("生成 E_heatmap_valid.png...")

# 创建网格
xi = np.linspace(x.min(), x.max(), 100)
yi = np.linspace(y.min(), y.max(), 100)
xi, yi = np.meshgrid(xi, yi)

# 插值
zi = griddata((x, y), df['E_eff'].values, (xi, yi), method='cubic')

plt.figure(figsize=(10, 8))
plt.contourf(xi, yi, zi, 50, cmap='viridis')
plt.colorbar(label='E_eff (MPa)')
plt.scatter(x, y, c='white', s=30, edgecolors='black')
plt.xlabel('w')
plt.ylabel('rho')
plt.title('Young\'s Modulus Heatmap')

output_file = os.path.join(OUTPUT_DIR, 'E_heatmap_valid.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"E_heatmap_valid.png 已保存到: {output_file}")

# 生成 G_heatmap_valid.png
print("生成 G_heatmap_valid.png...")

# 插值
zi = griddata((x, y), df['G_eff'].values, (xi, yi), method='cubic')

plt.figure(figsize=(10, 8))
plt.contourf(xi, yi, zi, 50, cmap='viridis')
plt.colorbar(label='G_eff (MPa)')
plt.scatter(x, y, c='white', s=30, edgecolors='black')
plt.xlabel('w')
plt.ylabel('rho')
plt.title('Shear Modulus Heatmap')

output_file = os.path.join(OUTPUT_DIR, 'G_heatmap_valid.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"G_heatmap_valid.png 已保存到: {output_file}")

# 生成 A_zener_heatmap_valid.png
print("生成 A_zener_heatmap_valid.png...")

# 插值
zi = griddata((x, y), df['A_zener'].values, (xi, yi), method='cubic')

plt.figure(figsize=(10, 8))
plt.contourf(xi, yi, zi, 50, cmap='viridis')
plt.colorbar(label='A_zener')
plt.scatter(x, y, c='white', s=30, edgecolors='black')
plt.xlabel('w')
plt.ylabel('rho')
plt.title('Zener Ratio Heatmap')

output_file = os.path.join(OUTPUT_DIR, 'A_zener_heatmap_valid.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"A_zener_heatmap_valid.png 已保存到: {output_file}")

print("所有论文结果图已生成完成!")