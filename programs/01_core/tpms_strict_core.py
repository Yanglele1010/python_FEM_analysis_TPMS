"""
TPMS严格核心模块

功能说明：
- BCC和I-WP点阵结构的精确数学定义
- 互穿点阵结构(IPC)的参数化生成
- 相对密度和互穿比例的精确求解
- 网格生成与mesh后处理

作者：自动生成
日期：2026-04-24
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import trimesh
from skimage import measure


# ==================== 项目路径和默认参数 ====================

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 默认单胞大小（毫米）
DEFAULT_CELL_SIZE_MM = 10.0
# 默认目标相对密度
DEFAULT_TARGET_RHO = 0.30
# 默认重复次数
DEFAULT_REPEATS = 4
# 默认互穿参数w的取值序列
DEFAULT_MAIN_W_VALUES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
# 默认I-WP修正参数alpha
DEFAULT_ALPHA = 5.0


@dataclass(frozen=True)
class PaperCompressionParameters:
    """
    论文压缩实验参数数据类
    
    存储论文中压缩实验的所有相关参数，包括：
    - 样本尺寸和单胞大小
    - 打印材料和设备参数
    - 压缩测试设备和参数
    """
    source_document: str
    paper_sample_size_mm: float = 24.0  # 样本尺寸（毫米）
    paper_cell_size_mm: float = 6.0     # 单胞尺寸（毫米）
    repeats_per_axis: int = 4           # 每个轴的重复次数
    rho_star: float = DEFAULT_TARGET_RHO  # 目标相对密度
    w_values: tuple[float, ...] = DEFAULT_MAIN_W_VALUES  # 互穿参数序列
    default_alpha: float = DEFAULT_ALPHA  # 默认alpha值
    printer: str = "Asiga Max X27"       # 3D打印机型号
    material: str = "Nova3d standard resin"  # 打印材料
    layer_thickness_um: float = 100.0   # 层厚（微米）
    exposure_time_s: float = 1.5         # 曝光时间（秒）
    light_intensity_mw_cm2: float = 5.0  # 光照强度（mW/cm²）
    post_cure_device: str = "Asiga Flash"  # 后固化设备
    post_cure_time_h: float = 1.0        # 后固化时间（小时）
    compression_tester: str = "Shimadzu AG25-TB"  # 万能试验机
    compression_speed_mm_min: float = 2.0  # 压缩速度（mm/min）
    repetitions_per_configuration: int = 2  # 每个配置的重复次数

    @property
    def adapted_sample_size_mm(self) -> float:
        """计算实际样本尺寸"""
        return self.paper_cell_size_mm * self.repeats_per_axis


class TPMSUnitCell:
    """
    TPMS单胞类
    
    实现BCC和I-WP点阵结构的水平集函数计算，
    支持相对密度和互穿参数的精确求解。
    
    属性：
        cell_size_mm: 单胞尺寸（毫米）
        samples_per_axis: 每个轴的采样点数
        pitch_mm: 采样间距（毫米）
        pair: 预计算的cos乘积项
        bcc_field: BCC水平集场
        iwp_part1: I-WP水平集场的第一部分
    """
    
    def __init__(self, cell_size_mm: float = DEFAULT_CELL_SIZE_MM, samples_per_axis: int = 40) -> None:
        """
        初始化TPMS单胞
        
        参数：
            cell_size_mm: 单胞尺寸（毫米），默认10.0mm
            samples_per_axis: 每个轴的采样点数，默认40
        """
        self.cell_size_mm = float(cell_size_mm)
        self.samples_per_axis = int(samples_per_axis)
        self.pitch_mm = self.cell_size_mm / self.samples_per_axis
        
        # 生成采样坐标网格
        coords = (np.arange(self.samples_per_axis, dtype=np.float32) + 0.5) * self.pitch_mm
        x, y, z = np.meshgrid(coords, coords, coords, indexing="ij")
        k = 2.0 * np.pi / self.cell_size_mm

        # 预计算三角函数值以提高性能
        cos_x = np.cos(k * x, dtype=np.float32)
        cos_y = np.cos(k * y, dtype=np.float32)
        cos_z = np.cos(k * z, dtype=np.float32)
        cos2_x = np.cos(2.0 * k * x, dtype=np.float32)
        cos2_y = np.cos(2.0 * k * y, dtype=np.float32)
        cos2_z = np.cos(2.0 * k * z, dtype=np.float32)

        # 预计算cos乘积项
        pair = cos_x * cos_y + cos_y * cos_z + cos_z * cos_x
        self.pair = pair.astype(np.float32, copy=False)
        
        # BCC水平集场：cos(2kx) + cos(2ky) + cos(2kz) - 2*(cos(kx)*cos(ky) + cos(ky)*cos(kz) + cos(kz)*cos(kx))
        self.bcc_field = (cos2_x + cos2_y + cos2_z - 2.0 * self.pair).astype(np.float32, copy=False)
        
        # I-WP水平集场的第一部分：-cos(2kx) + cos(2ky) + cos(2kz)
        self.iwp_part1 = (-cos2_x + cos2_y + cos2_z).astype(np.float32, copy=False)

    def bcc_mask(self, t1: float) -> np.ndarray:
        """
        生成BCC结构掩膜
        
        参数：
            t1: BCC阈值参数
            
        返回：
            布尔数组，True表示实体区域
        """
        return self.bcc_field <= t1

    def iwp_residual(self, alpha: float, t2: float) -> np.ndarray:
        """
        计算I-WP结构残差
        
        参数：
            alpha: I-WP修正参数
            t2: I-WP阈值参数
            
        返回：
            I-WP水平集残差值
        """
        return self.iwp_part1 + (t2 - alpha) * self.pair - t2

    def iwp_mask(self, alpha: float, t2: float) -> np.ndarray:
        """
        生成I-WP结构掩膜
        
        参数：
            alpha: I-WP修正参数
            t2: I-WP阈值参数
            
        返回：
            布尔数组，True表示实体区域
        """
        return self.iwp_residual(alpha, t2) <= 0.0

    def solve_bcc_threshold(self, target_fraction: float, iterations: int = 40) -> tuple[float, float]:
        """
        二分法求解BCC目标密度对应的阈值
        
        参数：
            target_fraction: 目标体积分数
            iterations: 迭代次数
            
        返回：
            (阈值, 实际体积分数)
        """
        low = float(self.bcc_field.min())
        high = float(self.bcc_field.max())
        mid = low
        fraction = 0.0
        for _ in range(iterations):
            mid = 0.5 * (low + high)
            fraction = float((self.bcc_field <= mid).mean())
            if fraction < target_fraction:
                low = mid
            else:
                high = mid
        return mid, fraction

    def solve_scalar_parameter(
        self,
        evaluator: Callable[[float], float],
        target_fraction: float,
        lower: float = -8.0,
        upper: float = 6.0,
        coarse_steps: int = 97,
        iterations: int = 32,
    ) -> tuple[float, float]:
        """
        通用标量参数求解器
        
        使用二分法求解目标体积分数对应的参数值。
        
        参数：
            evaluator: 评估函数，输入参数值，返回体积分数
            target_fraction: 目标体积分数
            lower: 参数下界
            upper: 参数上界
            coarse_steps: 粗搜索步数
            iterations: 精搜索迭代次数
            
        返回：
            (最优参数值, 实际体积分数)
        """
        values = np.linspace(lower, upper, coarse_steps, dtype=np.float64)
        fractions = np.array([evaluator(float(v)) for v in values], dtype=np.float64)

        # 寻找目标分数所在的区间
        bracket_index: int | None = None
        best_score = None
        for idx in range(len(values) - 1):
            left = fractions[idx] - target_fraction
            right = fractions[idx + 1] - target_fraction
            if left == 0.0:
                return float(values[idx]), float(fractions[idx])
            if right == 0.0:
                return float(values[idx + 1]), float(fractions[idx + 1])
            if left * right <= 0.0:
                score = abs(left) + abs(right)
                if best_score is None or score < best_score:
                    best_score = score
                    bracket_index = idx

        # 如果没有找到区间，使用最近的点
        if bracket_index is None:
            nearest = int(np.argmin(np.abs(fractions - target_fraction)))
            return float(values[nearest]), float(fractions[nearest])

        # 二分法精搜索
        low = float(values[bracket_index])
        high = float(values[bracket_index + 1])
        frac_low = float(fractions[bracket_index])
        frac_high = float(fractions[bracket_index + 1])
        increasing = frac_high >= frac_low

        mid = low
        frac_mid = frac_low
        for _ in range(iterations):
            mid = 0.5 * (low + high)
            frac_mid = float(evaluator(mid))
            if increasing:
                if frac_mid < target_fraction:
                    low = mid
                else:
                    high = mid
            else:
                if frac_mid > target_fraction:
                    low = mid
                else:
                    high = mid
        return mid, frac_mid

    def solve_iwp_threshold(self, alpha: float, target_fraction: float) -> tuple[float, float]:
        """
        求解I-WP目标密度对应的阈值
        
        参数：
            alpha: I-WP修正参数
            target_fraction: 目标体积分数
            
        返回：
            (阈值, 实际体积分数)
        """
        return self.solve_scalar_parameter(
            evaluator=lambda t2: float(self.iwp_mask(alpha, t2).mean()),
            target_fraction=target_fraction,
        )

    def solve_ipc_thresholds(self, alpha: float, target_rho: float, w: float) -> dict[str, float | np.ndarray]:
        """
        求解IPC（互穿点阵结构）的阈值参数
        
        参数：
            alpha: I-WP修正参数
            target_rho: 目标相对密度
            w: 互穿参数（0=IWP, 1=BCC）
            
        返回：
            包含求解结果的字典：
            - kind: 结构类型（'BCC', 'IWP', 'IPC'）
            - t1, t2: 阈值参数
            - rho_actual: 实际相对密度
            - w_actual: 实际互穿比例
            - bcc_fraction: BCC相体积分数
            - overlap_fraction: 重叠区域分数
            - unit_mask: 单胞掩膜
        """
        # 纯I-WP情况
        if w <= 0.0:
            t2, rho_actual = self.solve_iwp_threshold(alpha=alpha, target_fraction=target_rho)
            mask = self.iwp_mask(alpha, t2)
            return {
                "kind": "IWP",
                "t1": None,
                "t2": t2,
                "rho_actual": float(mask.mean()),
                "w_actual": 0.0,
                "bcc_fraction": 0.0,
                "overlap_fraction": 0.0,
                "unit_mask": mask,
            }

        # 纯BCC情况
        if w >= 1.0:
            t1, rho_actual = self.solve_bcc_threshold(target_fraction=target_rho)
            mask = self.bcc_mask(t1)
            return {
                "kind": "BCC",
                "t1": t1,
                "t2": None,
                "rho_actual": float(mask.mean()),
                "w_actual": 1.0,
                "bcc_fraction": float(mask.mean()),
                "overlap_fraction": 0.0,
                "unit_mask": mask,
            }

        # 互穿结构情况
        # 首先求解BCC部分
        bcc_target_fraction = target_rho * w
        t1, bcc_fraction = self.solve_bcc_threshold(target_fraction=bcc_target_fraction)
        mask_bcc = self.bcc_mask(t1)

        # 然后求解联合掩膜
        def union_fraction_for_t2(t2: float) -> float:
            return float((mask_bcc | self.iwp_mask(alpha, t2)).mean())

        t2, rho_actual = self.solve_scalar_parameter(
            evaluator=union_fraction_for_t2,
            target_fraction=target_rho,
        )
        mask_iwp = self.iwp_mask(alpha, t2)
        unit_mask = mask_bcc | mask_iwp
        overlap_fraction = float((mask_bcc & mask_iwp).mean())
        rho_actual = float(unit_mask.mean())
        w_actual = float(mask_bcc.mean() / rho_actual) if rho_actual > 0 else 0.0
        return {
            "kind": "IPC",
            "t1": t1,
            "t2": t2,
            "rho_actual": rho_actual,
            "w_actual": w_actual,
            "bcc_fraction": float(mask_bcc.mean()),
            "overlap_fraction": overlap_fraction,
            "unit_mask": unit_mask,
        }


def build_macro_mask(unit_mask: np.ndarray, repeats: int = DEFAULT_REPEATS) -> np.ndarray:
    """
    将单胞掩膜扩展为宏观结构
    
    参数：
        unit_mask: 单胞掩膜数组
        repeats: 每个轴的重复次数
        
    返回：
        扩展后的宏观掩膜数组
    """
    return np.tile(unit_mask, (repeats, repeats, repeats))


def mask_to_mesh(mask: np.ndarray, pitch_mm: float, size_mm: float) -> trimesh.Trimesh:
    """
    将3D掩膜转换为三角网格
    
    使用marching cubes算法从掩膜提取等值面，
    并进行缩放以匹配目标尺寸。
    
    参数：
        mask: 3D布尔掩膜数组
        pitch_mm: 采样间距（毫米）
        size_mm: 目标模型尺寸（毫米）
        
    返回：
        trimesh网格对象
    """
    # 填充边界以确保闭合网格
    padded = np.pad(mask.astype(np.float32), pad_width=1, mode="constant", constant_values=0.0)
    
    # Marching cubes提取等值面
    verts, faces, _, _ = measure.marching_cubes(padded, level=0.5, spacing=(pitch_mm, pitch_mm, pitch_mm))
    
    # 创建网格对象
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    if hasattr(mesh, "remove_duplicate_faces"):
        mesh.remove_duplicate_faces()
    if hasattr(mesh, "remove_unreferenced_vertices"):
        mesh.remove_unreferenced_vertices()
    mesh.fix_normals()

    # 缩放到目标尺寸
    bounds = mesh.bounds
    extents = bounds[1] - bounds[0]
    mesh.apply_translation(-bounds[0])
    scale = np.divide(
        np.full(3, size_mm, dtype=np.float64),
        extents,
        out=np.ones(3, dtype=np.float64),
        where=extents > 0,
    )
    transform = np.eye(4)
    transform[0, 0] = scale[0]
    transform[1, 1] = scale[1]
    transform[2, 2] = scale[2]
    mesh.apply_transform(transform)
    mesh.fix_normals()
    return mesh


def overhang_area(mesh: trimesh.Trimesh, downward_normal_limit: float = -0.5) -> float:
    """
    计算网格的悬挑面积
    
    统计朝下方向（z轴负方向）的面片面积，
    用于评估FDM打印的支撑需求。
    
    参数：
        mesh: trimesh网格对象
        downward_normal_limit: 朝下法向阈值
        
    返回：
        悬挑面积（平方毫米）
    """
    normals = mesh.face_normals
    area_faces = mesh.area_faces
    return float(area_faces[normals[:, 2] < downward_normal_limit].sum())


def mesh_metrics(mesh: trimesh.Trimesh) -> dict[str, object]:
    """
    计算网格的几何度量指标
    
    参数：
        mesh: trimesh网格对象
        
    返回：
        包含各项指标的字典：
        - n_vertices: 顶点数
        - n_faces: 面片数
        - surface_area_mm2: 表面积（mm²）
        - volume_mm3: 体积（mm³）
        - is_watertight: 是否水密
        - bounds_mm: 边界框
        - dimensions_mm: 尺寸
        - overhang_area_mm2: 悬挑面积
    """
    bounds = mesh.bounds
    extents = bounds[1] - bounds[0]
    return {
        "n_vertices": int(len(mesh.vertices)),
        "n_faces": int(len(mesh.faces)),
        "surface_area_mm2": float(mesh.area),
        "volume_mm3": float(abs(mesh.volume)),
        "is_watertight": bool(mesh.is_watertight),
        "bounds_mm": bounds.tolist(),
        "dimensions_mm": extents.tolist(),
        "overhang_area_mm2": overhang_area(mesh),
    }


def w_code(w: float) -> str:
    """
    将互穿参数w转换为文件命名编码
    
    参数：
        w: 互穿参数（0-1）
        
    返回：
        三位数字字符串，如0.5→"050"
    """
    return f"{int(round(w * 100.0)):03d}"


def alpha_code(alpha: float) -> str:
    """
    将alpha参数转换为文件命名编码
    
    参数：
        alpha: I-WP修正参数
        
    返回：
        两位数字字符串，如5.0→"05"
    """
    return f"{int(round(alpha)):02d}"

