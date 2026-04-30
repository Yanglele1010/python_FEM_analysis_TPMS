# -*- coding: utf-8 -*-
"""
Voxel FEM 查找表结果绘图脚本。

该脚本保留旧入口，但输出不再写入项目根目录。
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(r"E:\code\new")
FINAL_DIR = PROJECT_ROOT / "04_Final_Results"
FDM_OUT_DIR = FINAL_DIR / "02_FDM_压缩结果"
LOOKUP_OUT_DIR = FINAL_DIR / "04_查找表与旧FEM结果"

FDM_OUT_DIR.mkdir(parents=True, exist_ok=True)
LOOKUP_OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("找不到输入文件，已尝试：" + "；".join(str(p) for p in paths))


def save_current(output_file):
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"生成图表：{output_file}")


csv_file = first_existing(
    [
        LOOKUP_OUT_DIR / "fem_results_by_rho_w_full.csv",
        PROJECT_ROOT / "fem_results_by_rho_w_full.csv",
        PROJECT_ROOT / "01_Programs" / "02_voxel-based FEM" / "fem_results_by_rho_w_full.csv",
    ]
)

df = pd.read_csv(csv_file)
df = df.drop_duplicates(subset=["target_rho", "target_w"], keep="last")

print(f"读取文件：{csv_file}")
print(
    df[
        [
            "target_rho",
            "target_w",
            "actual_rho_lookup",
            "actual_w_lookup",
            "E_eff_MPa",
            "reaction_force_N",
            "compliance",
        ]
    ]
)


plt.figure(figsize=(7, 5))
for rho, group in df.groupby("target_rho"):
    group = group.sort_values("actual_w_lookup")
    plt.plot(
        group["actual_w_lookup"],
        group["E_eff_MPa"],
        marker="o",
        label=f"目标相对密度 ρ*={rho:.2f}",
    )

plt.xlabel("实际互穿权重 w")
plt.ylabel("等效弹性模量 E_eff (MPa)")
plt.title("互穿权重 w 对等效弹性模量的影响")
plt.grid(True)
plt.legend()
save_current(FDM_OUT_DIR / "E_eff_vs_w.png")


plt.figure(figsize=(7, 5))
for w, group in df.groupby("target_w"):
    group = group.sort_values("actual_rho_lookup")
    plt.plot(
        group["actual_rho_lookup"],
        group["E_eff_MPa"],
        marker="s",
        label=f"目标权重 w={w:.2f}",
    )

plt.xlabel("实际相对密度 ρ*")
plt.ylabel("等效弹性模量 E_eff (MPa)")
plt.title("相对密度对等效弹性模量的影响")
plt.grid(True)
plt.legend()
save_current(FDM_OUT_DIR / "E_eff_vs_rho.png")


plt.figure(figsize=(7, 5))
for rho, group in df.groupby("target_rho"):
    group = group.sort_values("actual_w_lookup")
    plt.plot(
        group["actual_w_lookup"],
        abs(group["reaction_force_N"]),
        marker="^",
        label=f"目标相对密度 ρ*={rho:.2f}",
    )

plt.xlabel("实际互穿权重 w")
plt.ylabel("压缩反力 |F| (N)")
plt.title("互穿权重 w 对压缩反力的影响")
plt.grid(True)
plt.legend()
save_current(FDM_OUT_DIR / "reaction_force_vs_w.png")


df["rho_error"] = df["actual_rho_lookup"] - df["target_rho"]
df["w_error"] = df["actual_w_lookup"] - df["target_w"]

plt.figure(figsize=(7, 5))
plt.scatter(df["rho_error"], df["w_error"])
plt.axhline(0, linestyle="--")
plt.axvline(0, linestyle="--")
plt.xlabel("相对密度误差 Δρ*")
plt.ylabel("权重误差 Δw")
plt.title("查找表目标值与实际值匹配误差")
plt.grid(True)
save_current(LOOKUP_OUT_DIR / "lookup_error.png")
