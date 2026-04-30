[README.md](https://github.com/user-attachments/files/27227569/README.md)
# python_FEM_analysis_TPMS
python_FEM_analysis_TPMS
# BCC-IWP 互穿 TPMS 点阵结构多尺度优化研究

[English](#english) | 中文

## 项目简介

本项目研究基于 **BCC（体心立方）** 和 **IWP（I-WP）** 两种 TPMS（三周期极小曲面）的互穿点阵结构，通过多尺度优化方法实现轻量化设计。

### 核心创新

1. **双变量参数化**：引入密度 ρ 和互穿参数 w 联合控制结构性能
2. **RVE-PBC 均匀化**：基于周期性边界条件的代表性体积单元分析
3. **D 模型插值**：构建弹性张量的高精度插值模型
4. **多尺度优化**：微观（ρ, w）→ 宏观（柔度最小化）

## 目录结构

```
├── programs/                    # 源代码
│   ├── 01_core/                # 核心 TPMS 函数
│   │   ├── python_voxel_fem_ipc_compression.py  # F_BCC, F_IWP_modified
│   │   └── tpms_strict_core.py                  # TPMS 单元类
│   ├── 02_model_generation/    # 模型生成
│   │   ├── generate_inspection_model.py          # 单模型生成
│   │   └── generate_batch_inspection_models.py   # 批量生成
│   ├── 03_fem_analysis/        # 有限元分析
│   │   ├── 02_lookup_and_run_fem_by_rho_w.py     # FEM 查找表
│   │   ├── 06_rve_pbc_homogenization.py          # RVE-PBC 均匀化
│   │   ├── 04_mesh_convergence.py                # 网格收敛分析
│   │   └── 07_fdm_cells3_compression.py          # FDM 压缩模拟
│   ├── 04_optimization/        # 优化算法
│   │   ├── 09_fit_D_interpolation_model.py       # D 模型拟合
│   │   ├── 11_fit_eff_model.py                   # 有效性能拟合
│   │   ├── 12_macro_cantilever_density_optimization.py  # 密度优化
│   │   └── 14_macro_cantilever_bivariate_optimization.py # 双变量优化
│   └── 05_postprocessing/      # 后处理与绘图
│       ├── 03_plot_fem_results.py
│       ├── 11_generate_paper_figures.py
│       └── generate_academic_charts.py
├── data/                        # 数据文件
│   ├── fem_results_by_rho_w_full.csv      # FEM 结果数据库
│   ├── rve_pbc_homogenization_results.csv # RVE-PBC 结果
│   ├── D_interpolation_coefficients.csv   # D 模型系数
│   ├── model_parameters.csv               # 模型参数记录
│   └── lookup_table_cell20_fine.csv       # 查找表
├── figures/                     # 图表
│   ├── 01_structure/           # 结构示意图
│   ├── 02_fem_analysis/        # FEM 分析图
│   ├── 03_optimization/        # 优化结果图
│   └── 04_paper_figures/       # 论文图表
├── models/                      # STL 模型
│   └── STL_2x2x2/             # 2×2×2 单胞模型（40mm）
└── results/                     # 结果报告
```

## 快速开始

### 环境要求

```bash
Python >= 3.8
numpy
scipy
matplotlib
trimesh
scikit-image
abapy  # 用于 ABAQUS 交互
```

### 安装依赖

```bash
pip install numpy scipy matplotlib trimesh scikit-image
```

### 运行示例

#### 1. 生成单个 TPMS 模型

```python
from programs.01_core.tpms_strict_core import TPMSUnitCell

# 创建 BCC-IWP 互穿结构
cell = TPMSUnitCell(cell_size_mm=20.0, alpha=5.0)
t1, t2 = cell.solve_ipc_thresholds(w=0.6, target_rho=0.30)
```

#### 2. 批量生成 STL 模型

```bash
cd programs/02_model_generation
python generate_batch_inspection_models.py
```

#### 3. 运行 FEM 分析

```bash
cd programs/03_fem_analysis
python 06_rve_pbc_homogenization.py
```

#### 4. 执行优化

```bash
cd programs/04_optimization
python 14_macro_cantilever_bivariate_optimization.py
```

## 核心算法

### 1. TPMS 隐式函数

**BCC 结构**：
```
F_BCC = cos(2kx·x) + cos(2ky·y) + cos(2kz·z)
      - 2[cos(kx·x)cos(ky·y) + cos(ky·y)cos(kz·z) + cos(kz·z)cos(kx·x)]
```

**IWP 结构**：
```
F_IWP = -cos(2kx·x) + cos(2ky·y) + cos(2kz·z)
      + (t2 - α)[cos(kx·x)cos(ky·y) + cos(ky·y)cos(kz·z) + cos(kz·z)cos(kx·x)]
```

其中 `kx = ky = kz = 2π / cell_size`

### 2. 实体判定

- **阈值法（FEM 仿真用）**：`F(x,y,z) ≤ t`
- **隐式曲面带法（STL 打印用）**：`|φ/|∇φ|| ≤ thickness/2`

### 3. 密度与互穿参数

- **密度 ρ**：实体体积占比
- **互穿参数 w**：BCC 相占总体积比例，`w = ρ_BCC / ρ_total`

### 4. D 模型插值

构建弹性张量 D(ρ, w) 的插值模型：

```
D_ij(ρ, w) = Σ_k Σ_l C_kl · ρ^k · w^l
```

## 关键结果

### FEM 分析结果

| 参数 | 范围 | 说明 |
|------|------|------|
| ρ | 0.10 - 0.50 | 相对密度 |
| w | 0.0 - 1.0 | 互穿参数 |
| E_eff | 50 - 800 MPa | 有效弹性模量 |
| G_eff | 20 - 300 MPa | 有效剪切模量 |

### 优化结果

- **单变量优化**（固定 w=0.6）：柔度降低 15-20%
- **双变量优化**（ρ, w 联合）：柔度降低 25-30%

## 应用场景

1. **轻量化设计**：航空航天、汽车结构
2. **能量吸收**：冲击防护、缓冲结构
3. **热管理**：散热器、热交换器
4. **生物医学**：骨植入物、支架结构

## 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@mastersthesis{Yang_mortal-yu2026,
  title={基于 BCC 和 I-WP 的互穿点阵结构设计与力学性能研究},
  author={Yang_mortal-yu},
  year={2026},
  school={North China University of Water Resources and Electric Power（NCWU）}
}
```

## 许可证

本项目采用 [MIT License](LICENSE) 许可证。

## 联系方式
- 作者：[Yang_mortal-yu]
- 邮箱：[y9915630@gmail.com]
- 项目主页：[(https://github.com/Yanglele1010)]

---

<a name="english"></a>
## English

### Project Overview

This project investigates **interpenetrating lattice structures** based on two TPMS (Triply Periodic Minimal Surface) types: **BCC (Body-Centered Cubic)** and **IWP (I-WP)**, achieving lightweight design through multi-scale optimization.

### Key Innovations

1. **Bivariate Parameterization**: Joint control of density ρ and interpenetration parameter w
2. **RVE-PBC Homogenization**: Representative Volume Element analysis with Periodic Boundary Conditions
3. **D-model Interpolation**: High-precision interpolation for elastic tensor
4. **Multi-scale Optimization**: Micro (ρ, w) → Macro (compliance minimization)

### Quick Start

```bash
# Install dependencies
pip install numpy scipy matplotlib trimesh scikit-image

# Generate single TPMS model
python programs/02_model_generation/generate_inspection_model.py

# Batch generate STL models
python programs/02_model_generation/generate_batch_inspection_models.py

# Run FEM analysis
python programs/03_fem_analysis/06_rve_pbc_homogenization.py

# Execute optimization
python programs/04_optimization/14_macro_cantilever_bivariate_optimization.py
```

### Core Algorithms

#### TPMS Implicit Functions

**BCC Structure**:
```
F_BCC = cos(2kx·x) + cos(2ky·y) + cos(2kz·z)
      - 2[cos(kx·x)cos(ky·y) + cos(ky·y)cos(kz·z) + cos(kz·z)cos(kx·x)]
```

**IWP Structure**:
```
F_IWP = -cos(2kx·x) + cos(2ky·y) + cos(2kz·z)
      + (t2 - α)[cos(kx·x)cos(ky·y) + cos(ky·y)cos(kz·z) + cos(kz·z)cos(kx·x)]
```

Where `kx = ky = kz = 2π / cell_size`

#### Solid/Void Determination

- **Threshold Method (FEM)**: `F(x,y,z) ≤ t`
- **Implicit Surface Band Method (STL)**: `|φ/|∇φ|| ≤ thickness/2`

#### Density and Interpenetration Parameter

- **Density ρ**: Solid volume fraction
- **Interpenetration w**: BCC phase volume ratio, `w = ρ_BCC / ρ_total`

### Key Results

| Parameter | Range | Description |
|-----------|-------|-------------|
| ρ | 0.10 - 0.50 | Relative density |
| w | 0.0 - 1.0 | Interpenetration parameter |
| E_eff | 50 - 800 MPa | Effective elastic modulus |
| G_eff | 20 - 300 MPa | Effective shear modulus |

### Applications

1. **Lightweight Design**: Aerospace, automotive structures
2. **Energy Absorption**: Impact protection, cushioning
3. **Thermal Management**: Heat sinks, heat exchangers
4. **Biomedical**: Bone implants, scaffolds

### Citation

If this project helps your research, please cite:

```bibtex
@mastersthesis{your_name2026,
  title={Design and Mechanical Properties of Interpenetrating Lattice Structures Based on BCC and I-WP},
  author={Your Name},
  year={2026},
  school={Your University}
}
```

### License

This project is licensed under the [MIT License](LICENSE).

### Contact

- Author: [yanglele]
- Email: [y9915630@gmail.com]
