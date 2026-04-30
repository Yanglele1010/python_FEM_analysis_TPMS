#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
建立 D(rho,w) 插值模型
拟合 D1111 = f(rho, w)、D1122 = f(rho, w)、D1212 = f(rho, w)
使用指定的多项式形式进行拟合
输出 D_interpolation_coefficients.csv 和 D_fit_quality_report.md
"""

import os
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error

# 定义文件路径
INPUT_FILE = r"E:\code\new\04_Final_Results\rve_pbc_effective_valid.csv"
OUTPUT_COEFF = r"E:\code\new\04_Final_Results\D_interpolation_coefficients.csv"
OUTPUT_REPORT = r"E:\code\new\04_Final_Results\D_fit_quality_report.md"

# 读取数据
df = pd.read_csv(INPUT_FILE)
print(f"读取数据点数量: {len(df)}")

# 确定自变量
if 'actual_rho_rve' in df.columns and 'actual_w_rve' in df.columns:
    print("使用 actual_rho_rve 和 actual_w_rve 作为自变量")
    rho = df['actual_rho_rve'].values
    w = df['actual_w_rve'].values
elif 'actual_rho' in df.columns and 'actual_w' in df.columns:
    print("使用 actual_rho 和 actual_w 作为自变量")
    rho = df['actual_rho'].values
    w = df['actual_w'].values
else:
    print("使用 target_rho 和 target_w 作为自变量")
    rho = df['target_rho'].values
    w = df['target_w'].values

# 定义拟合函数
def fit_func(x, b1, b2, b3, b4, b5, b6, b7, b8, b9):
    rho, w = x
    return (b1 * rho + b2 * rho**2 + b3 * rho**3 + b4 * rho**4 + b5 * rho**5 +
            b6 * rho * w + b7 * rho * w**2 + b8 * rho**2 * w + b9 * rho**2 * w**2)

# 准备数据
x_data = (rho, w)

# 拟合 D1111
print("拟合 D1111...")
y_data = df['D1111'].values
popt_d1111, pcov_d1111 = curve_fit(fit_func, x_data, y_data)

# 计算拟合质量指标
y_pred_d1111 = fit_func(x_data, *popt_d1111)
r2_d1111 = r2_score(y_data, y_pred_d1111)
rmse_d1111 = np.sqrt(mean_squared_error(y_data, y_pred_d1111))
max_rel_error_d1111 = np.max(np.abs((y_pred_d1111 - y_data) / y_data)) * 100

# 拟合 D1122
print("拟合 D1122...")
y_data = df['D1122'].values
popt_d1122, pcov_d1122 = curve_fit(fit_func, x_data, y_data)

# 计算拟合质量指标
y_pred_d1122 = fit_func(x_data, *popt_d1122)
r2_d1122 = r2_score(y_data, y_pred_d1122)
rmse_d1122 = np.sqrt(mean_squared_error(y_data, y_pred_d1122))
max_rel_error_d1122 = np.max(np.abs((y_pred_d1122 - y_data) / y_data)) * 100

# 拟合 D1212
print("拟合 D1212...")
y_data = df['D1212'].values
popt_d1212, pcov_d1212 = curve_fit(fit_func, x_data, y_data)

# 计算拟合质量指标
y_pred_d1212 = fit_func(x_data, *popt_d1212)
r2_d1212 = r2_score(y_data, y_pred_d1212)
rmse_d1212 = np.sqrt(mean_squared_error(y_data, y_pred_d1212))
max_rel_error_d1212 = np.max(np.abs((y_pred_d1212 - y_data) / y_data)) * 100

# 保存系数
coefficients = {
    'coefficient': ['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9'],
    'D1111': popt_d1111,
    'D1122': popt_d1122,
    'D1212': popt_d1212
}
df_coeff = pd.DataFrame(coefficients)
df_coeff.to_csv(OUTPUT_COEFF, index=False)
print(f"系数已保存到: {OUTPUT_COEFF}")

# 生成质量报告
with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
    f.write("# D(rho,w) 插值模型拟合质量报告\n\n")
    f.write(f"## 数据概况\n")
    f.write(f"- 使用的有效数据点数量: {len(df)}\n")
    f.write(f"- 自变量: {'actual_rho_rve 和 actual_w_rve' if 'actual_rho_rve' in df.columns and 'actual_w_rve' in df.columns else 'actual_rho 和 actual_w' if 'actual_rho' in df.columns and 'actual_w' in df.columns else 'target_rho 和 target_w'}\n\n")
    
    f.write("## 拟合系数\n")
    f.write("### D1111\n")
    for i, coeff in enumerate(popt_d1111):
        f.write(f"b{i+1}: {coeff:.6e}\n")
    f.write("\n")
    
    f.write("### D1122\n")
    for i, coeff in enumerate(popt_d1122):
        f.write(f"b{i+1}: {coeff:.6e}\n")
    f.write("\n")
    
    f.write("### D1212\n")
    for i, coeff in enumerate(popt_d1212):
        f.write(f"b{i+1}: {coeff:.6e}\n")
    f.write("\n")
    
    f.write("## 拟合质量指标\n")
    f.write("| 指标 | D1111 | D1122 | D1212 |\n")
    f.write("|------|-------|-------|-------|\n")
    f.write(f"| R² | {r2_d1111:.6f} | {r2_d1122:.6f} | {r2_d1212:.6f} |\n")
    f.write(f"| RMSE | {rmse_d1111:.6e} | {rmse_d1122:.6e} | {rmse_d1212:.6e} |\n")
    f.write(f"| 最大相对误差 (%) | {max_rel_error_d1111:.2f} | {max_rel_error_d1122:.2f} | {max_rel_error_d1212:.2f} |\n")
    f.write("\n")
    
    f.write("## 排除的点\n")
    f.write("- 无，所有点均为有效点\n")

print(f"质量报告已保存到: {OUTPUT_REPORT}")

# 打印拟合结果
print("\n拟合结果:")
print(f"D1111 R²: {r2_d1111:.6f}")
print(f"D1122 R²: {r2_d1122:.6f}")
print(f"D1212 R²: {r2_d1212:.6f}")