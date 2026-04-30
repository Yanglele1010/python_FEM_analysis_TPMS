#!/usr/bin/env python3
"""RVE + periodic boundary condition homogenization for IPC voxel cells."""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from python_voxel_fem_ipc_compression import (
    F_BCC,
    F_IWP_modified,
    elasticity_matrix,
    hex8_element_stiffness,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOOKUP_CSV = PROJECT_ROOT / "fem_results_by_rho_w_full.csv"
OUTPUT_CSV = PROJECT_ROOT / "rve_pbc_homogenization_results.csv"

CELL_SIZE = 20.0
ALPHA = 5.0
E_SOLID = 2200.0
NU_SOLID = 0.35
DEFAULT_N_RVE = 24

FULL_CASES = [
    (0.20, 0.0), (0.20, 0.2), (0.20, 0.4), (0.20, 0.6), (0.20, 0.8), (0.20, 1.0),
    (0.25, 0.0), (0.25, 0.2), (0.25, 0.4), (0.25, 0.6), (0.25, 0.8), (0.25, 1.0),
    (0.30, 0.0), (0.30, 0.2), (0.30, 0.4), (0.30, 0.6), (0.30, 0.8), (0.30, 1.0),
    (0.35, 0.0), (0.35, 0.2), (0.35, 0.4), (0.35, 0.6), (0.35, 0.8), (0.35, 1.0),
    (0.40, 0.0), (0.40, 0.2), (0.40, 0.4), (0.40, 0.6), (0.40, 0.8), (0.40, 1.0),
]
# Only run rho=0.20 cases (others already completed)
RHO020_CASES = [
    (0.20, 0.0), (0.20, 0.2), (0.20, 0.4), (0.20, 0.6), (0.20, 0.8), (0.20, 1.0),
]
SMOKE_CASES = [(0.30, 0.6)]

FIELDNAMES = [
    "case_name",
    "target_rho",
    "target_w",
    "actual_rho_lookup",
    "actual_w_lookup",
    "actual_rho_rve",
    "actual_w_rve",
    "rho_bcc_rve",
    "rho_iwp_rve",
    "t1",
    "t2",
    "n_rve",
    "dx",
    "D1111",
    "D1122",
    "D1212",
    "E_eff",
    "G_eff",
    "nu_eff",
    "A_zener",
    "runtime_s",
    "status",
    "error_message",
]


def values_close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) <= tol


def case_name(rho: float, w: float) -> str:
    return f"rho{rho:.2f}_w{w:.1f}"


def read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def read_cases_from_fem_csv(target_cases: list[tuple[float, float]]) -> list[dict[str, Any]]:
    rows = read_csv_rows(LOOKUP_CSV)
    cases: list[dict[str, Any]] = []

    for rho, w in target_cases:
        matches = [
            row for row in rows
            if values_close(float(row["target_rho"]), rho)
            and values_close(float(row["target_w"]), w)
        ]
        if not matches:
            print(f"[skip] target_rho={rho:.2f}, target_w={w:.1f} not found in {LOOKUP_CSV.name}")
            continue

        row = matches[-1]
        cases.append(
            {
                "case_name": case_name(rho, w),
                "target_rho": float(row["target_rho"]),
                "target_w": float(row["target_w"]),
                "actual_rho_lookup": float(row["actual_rho_lookup"]),
                "actual_w_lookup": float(row["actual_w_lookup"]),
                "t1": float(row["t1"]),
                "t2": float(row["t2"]),
            }
        )

    return cases


def generate_rve_voxels(
    n: int,
    cell_size: float,
    t1: float,
    t2: float,
    alpha: float,
) -> tuple[np.ndarray, float, dict[str, float]]:
    L = cell_size
    dx = L / n

    axis = np.linspace(dx / 2.0, L - dx / 2.0, n)
    X, Y, Z = np.meshgrid(axis, axis, axis, indexing="ij")
    Xc = X - L / 2.0
    Yc = Y - L / 2.0
    Zc = Z - L / 2.0

    kx = ky = kz = 2.0 * np.pi / cell_size
    solid_bcc = F_BCC(Xc, Yc, Zc, kx, ky, kz) <= t1
    solid_iwp = F_IWP_modified(Xc, Yc, Zc, kx, ky, kz, t2=t2, alpha=alpha) <= t2
    solid = solid_bcc | solid_iwp

    rho_total = float(np.mean(solid))
    rho_bcc = float(np.mean(solid_bcc))
    rho_iwp = float(np.mean(solid_iwp))

    return solid, dx, {
        "rho_total": rho_total,
        "rho_bcc": rho_bcc,
        "rho_iwp": rho_iwp,
        "w_actual": rho_bcc / rho_total if rho_total > 1e-12 else 0.0,
    }


def periodic_node_id(i: int, j: int, k: int, n: int) -> int:
    i %= n
    j %= n
    k %= n
    return i * n * n + j * n + k


def periodic_element_nodes(i: int, j: int, k: int, n: int) -> np.ndarray:
    return np.array(
        [
            periodic_node_id(i, j, k, n),
            periodic_node_id(i + 1, j, k, n),
            periodic_node_id(i + 1, j + 1, k, n),
            periodic_node_id(i, j + 1, k, n),
            periodic_node_id(i, j, k + 1, n),
            periodic_node_id(i + 1, j, k + 1, n),
            periodic_node_id(i + 1, j + 1, k + 1, n),
            periodic_node_id(i, j + 1, k + 1, n),
        ],
        dtype=int,
    )


def local_physical_node_coords(i: int, j: int, k: int, dx: float) -> np.ndarray:
    return np.array(
        [
            [i * dx, j * dx, k * dx],
            [(i + 1) * dx, j * dx, k * dx],
            [(i + 1) * dx, (j + 1) * dx, k * dx],
            [i * dx, (j + 1) * dx, k * dx],
            [i * dx, j * dx, (k + 1) * dx],
            [(i + 1) * dx, j * dx, (k + 1) * dx],
            [(i + 1) * dx, (j + 1) * dx, (k + 1) * dx],
            [i * dx, (j + 1) * dx, (k + 1) * dx],
        ],
        dtype=float,
    )


def macro_displacement(coords: np.ndarray, strain: np.ndarray) -> np.ndarray:
    exx, eyy, ezz, gxy, gyz, gxz = strain
    u = np.zeros((coords.shape[0], 3), dtype=float)
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]
    u[:, 0] = exx * x + gxy * y + gxz * z
    u[:, 1] = eyy * y + gyz * z
    u[:, 2] = ezz * z
    return u.reshape(-1)


def element_dofs(enodes: np.ndarray) -> np.ndarray:
    edofs = np.zeros(24, dtype=int)
    for a, node in enumerate(enodes):
        edofs[3 * a + 0] = 3 * node + 0
        edofs[3 * a + 1] = 3 * node + 1
        edofs[3 * a + 2] = 3 * node + 2
    return edofs


def compute_B_matrix(dx: float, dy: float, dz: float, xi: float, eta: float, zeta: float) -> np.ndarray:
    node_nat = np.array(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=float,
    )
    invJ = np.linalg.inv(np.diag([dx / 2.0, dy / 2.0, dz / 2.0]))
    dN_dnat = np.zeros((8, 3), dtype=float)

    for a in range(8):
        xi_a, eta_a, zeta_a = node_nat[a]
        dN_dnat[a, 0] = 0.125 * xi_a * (1 + eta_a * eta) * (1 + zeta_a * zeta)
        dN_dnat[a, 1] = 0.125 * eta_a * (1 + xi_a * xi) * (1 + zeta_a * zeta)
        dN_dnat[a, 2] = 0.125 * zeta_a * (1 + xi_a * xi) * (1 + eta_a * eta)

    dN = dN_dnat @ invJ.T
    B = np.zeros((6, 24), dtype=float)

    for a in range(8):
        dNx, dNy, dNz = dN[a]
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

    return B


def compute_average_stress(
    solid: np.ndarray,
    dx: float,
    E: float,
    nu: float,
    q: np.ndarray,
    macro_strain: np.ndarray,
) -> np.ndarray:
    n = solid.shape[0]
    D = elasticity_matrix(E, nu)
    gauss = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]
    detJ = (dx / 2.0) ** 3
    total_volume = (n * dx) ** 3
    total_stress_integral = np.zeros(6, dtype=float)

    for i, j, k in np.argwhere(solid):
        enodes = periodic_element_nodes(int(i), int(j), int(k), n)
        edofs = element_dofs(enodes)
        u_e = macro_displacement(local_physical_node_coords(int(i), int(j), int(k), dx), macro_strain) + q[edofs]

        for xi in gauss:
            for eta in gauss:
                for zeta in gauss:
                    strain_gp = compute_B_matrix(dx, dx, dx, xi, eta, zeta) @ u_e
                    total_stress_integral += (D @ strain_gp) * detJ

    return total_stress_integral / total_volume


def solve_rve_pbc_case(solid: np.ndarray, dx: float, E: float, nu: float, macro_strain: np.ndarray) -> np.ndarray:
    n = solid.shape[0]
    if solid.shape != (n, n, n):
        raise ValueError("RVE solid grid must be cubic")

    num_dofs = n * n * n * 3
    Ke = hex8_element_stiffness(dx, dx, dx, E, nu)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []
    f_macro = np.zeros(num_dofs, dtype=float)
    used_nodes: set[int] = set()

    active = np.argwhere(solid)
    if active.size == 0:
        raise ValueError("RVE has no active solid voxels")

    for i, j, k in active:
        ii, jj, kk = int(i), int(j), int(k)
        enodes = periodic_element_nodes(ii, jj, kk, n)
        used_nodes.update(int(node) for node in enodes)
        edofs = element_dofs(enodes)
        u_macro_e = macro_displacement(local_physical_node_coords(ii, jj, kk, dx), macro_strain)

        rows.append(np.repeat(edofs, 24))
        cols.append(np.tile(edofs, 24))
        data.append(Ke.ravel())
        f_macro[edofs] += Ke @ u_macro_e

    K = coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(num_dofs, num_dofs),
    ).tocsr()

    used_nodes_array = np.array(sorted(used_nodes), dtype=int)
    used_dofs = np.concatenate([3 * used_nodes_array + 0, 3 * used_nodes_array + 1, 3 * used_nodes_array + 2])
    fixed_node = int(used_nodes_array[0])
    fixed = np.array([3 * fixed_node + 0, 3 * fixed_node + 1, 3 * fixed_node + 2], dtype=int)
    free = np.setdiff1d(used_dofs, fixed)

    q = np.zeros(num_dofs, dtype=float)
    q[free] = spsolve(K[free, :][:, free], -f_macro[free])

    return compute_average_stress(solid=solid, dx=dx, E=E, nu=nu, q=q, macro_strain=macro_strain)


def empty_row(case: dict[str, Any], n_rve: int, status: str, error_message: str = "") -> dict[str, Any]:
    row = {field: "" for field in FIELDNAMES}
    row.update(
        {
            "case_name": case["case_name"],
            "target_rho": case["target_rho"],
            "target_w": case["target_w"],
            "actual_rho_lookup": case["actual_rho_lookup"],
            "actual_w_lookup": case["actual_w_lookup"],
            "t1": case["t1"],
            "t2": case["t2"],
            "n_rve": n_rve,
            "status": status,
            "error_message": error_message,
        }
    )
    return row


def homogenize_one_case(case: dict[str, Any], n_rve: int) -> dict[str, Any]:
    print(f"\nRVE-PBC case {case['case_name']} | N_RVE={n_rve}")
    start = time.time()
    row = empty_row(case, n_rve, status="ok")

    try:
        solid, dx, geom = generate_rve_voxels(n=n_rve, cell_size=CELL_SIZE, t1=case["t1"], t2=case["t2"], alpha=ALPHA)
        eps0 = 1e-3
        gamma0 = 1e-3
        stress_uniaxial = solve_rve_pbc_case(solid, dx, E_SOLID, NU_SOLID, np.array([eps0, 0, 0, 0, 0, 0], dtype=float))
        stress_shear = solve_rve_pbc_case(solid, dx, E_SOLID, NU_SOLID, np.array([0, 0, 0, gamma0, 0, 0], dtype=float))

        D1111 = float(stress_uniaxial[0] / eps0)
        D1122 = float(stress_uniaxial[1] / eps0)
        D1212 = float(stress_shear[3] / gamma0)
        denom = max(D1111 + D1122, 1e-12)
        shear_denom = max(D1111 - D1122, 1e-12)
        E_eff = float((D1111**2 + D1111 * D1122 - 2.0 * D1122**2) / denom)
        G_eff = D1212
        nu_eff = float(D1122 / denom)
        A_zener = float(2.0 * D1212 / shear_denom)

        row.update(
            {
                "actual_rho_rve": geom["rho_total"],
                "actual_w_rve": geom["w_actual"],
                "rho_bcc_rve": geom["rho_bcc"],
                "rho_iwp_rve": geom["rho_iwp"],
                "dx": dx,
                "D1111": D1111,
                "D1122": D1122,
                "D1212": D1212,
                "E_eff": E_eff,
                "G_eff": G_eff,
                "nu_eff": nu_eff,
                "A_zener": A_zener,
            }
        )
        print(f"D1111={D1111:.6f}, D1122={D1122:.6f}, D1212={D1212:.6f}, E={E_eff:.6f}")
    except Exception as exc:
        row["status"] = "error"
        row["error_message"] = repr(exc)
        print(f"[error] {case['case_name']}: {exc}")
    finally:
        row["runtime_s"] = time.time() - start

    return row


def write_rows(rows: list[dict[str, Any]], append: bool) -> None:
    mode = "a" if append and OUTPUT_CSV.exists() else "w"
    with OUTPUT_CSV.open(mode, newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if mode == "w":
            writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RVE-PBC homogenization for IPC voxel cells.")
    parser.add_argument("--list-cases", action="store_true", help="List selected cases and exit.")
    parser.add_argument("--rho", type=float, help="Target rho for a focused run.")
    parser.add_argument("--w", type=float, help="Target w for a focused run.")
    parser.add_argument("--n-rve", type=int, default=DEFAULT_N_RVE, help="RVE voxels per side.")
    parser.add_argument("--full", action="store_true", help="Run all 30 rho,w cases.")
    parser.add_argument("--append", action="store_true", help="Append to output CSV.")
    return parser


def selected_targets(args: argparse.Namespace) -> list[tuple[float, float]]:
    if args.full:
        return FULL_CASES
    if args.rho is not None or args.w is not None:
        if args.rho is None or args.w is None:
            raise ValueError("Both --rho and --w are required for a focused run")
        return [(args.rho, args.w)]
    return RHO020_CASES  # Default: only run rho=0.20 cases


def main() -> None:
    args = build_parser().parse_args()
    cases = read_cases_from_fem_csv(selected_targets(args))

    print(f"Input CSV: {LOOKUP_CSV}")
    for case in cases:
        print(f"{case['case_name']}: t1={case['t1']:.6g}, t2={case['t2']:.6g}")

    if args.list_cases:
        return

    rows = [homogenize_one_case(case, args.n_rve) for case in cases]
    write_rows(rows, append=args.append)
    print(f"\nOutput written: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
