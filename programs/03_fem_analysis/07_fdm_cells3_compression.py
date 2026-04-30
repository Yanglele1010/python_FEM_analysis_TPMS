#!/usr/bin/env python3
"""FDM-oriented cells=(3,3,3) compression simulations for selected IPC cases."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import binary_closing, label

from python_voxel_fem_ipc_compression import (
    generate_ipc_voxels_threshold,
    solve_compression,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOOKUP_CSV = PROJECT_ROOT / "fem_results_by_rho_w_full.csv"
OUTPUT_CSV = PROJECT_ROOT / "fdm_cells3_compression_results.csv"

CELL_SIZE = 20.0
CELLS = (3, 3, 3)
ALPHA = 5.0
E_SOLID = 2200.0
NU_SOLID = 0.35
DISP = -0.6
CLOSING_ITERATIONS = 1
KEEP_LARGEST = False

DEFAULT_CASES = [(0.30, 0.2), (0.30, 0.6), (0.30, 0.8), (0.25, 0.6), (0.35, 0.6)]
DEFAULT_N_LIST = [12]
FULL_N_LIST = [12, 16, 20]

FIELDNAMES = [
    "case_name",
    "target_rho",
    "target_w",
    "source_target_rho",
    "source_target_w",
    "actual_rho_lookup",
    "actual_w_lookup",
    "actual_rho_cells3",
    "t1",
    "t2",
    "cell_size",
    "cells_x",
    "cells_y",
    "cells_z",
    "overall_size",
    "n_per_cell_fe",
    "closing_iterations",
    "keep_largest",
    "active_elements",
    "connected_components",
    "largest_component_ratio",
    "reaction_force_N",
    "force_abs_N",
    "stress_MPa",
    "strain",
    "E_eff_MPa",
    "energy_index_UKU",
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


def read_selected_cases(target_cases: list[tuple[float, float]]) -> list[dict[str, Any]]:
    rows = read_csv_rows(LOOKUP_CSV)
    cases: list[dict[str, Any]] = []

    for rho, w in target_cases:
        exact = [
            row for row in rows
            if values_close(float(row["target_rho"]), rho)
            and values_close(float(row["target_w"]), w)
        ]
        if exact:
            row = exact[-1]
        else:
            same_rho = [row for row in rows if values_close(float(row["target_rho"]), rho)]
            if not same_rho:
                print(f"[skip] no source row for target_rho={rho:.2f}, target_w={w:.1f}")
                continue
            row = min(same_rho, key=lambda item: abs(float(item["target_w"]) - w))
            print(
                f"[warn] no exact source for rho={rho:.2f}, w={w:.1f}; "
                f"using source w={float(row['target_w']):.1f}"
            )

        cases.append(
            {
                "case_name": case_name(rho, w),
                "target_rho": rho,
                "target_w": w,
                "source_target_rho": float(row["target_rho"]),
                "source_target_w": float(row["target_w"]),
                "actual_rho_lookup": float(row["actual_rho_lookup"]),
                "actual_w_lookup": float(row["actual_w_lookup"]),
                "t1": float(row["t1"]),
                "t2": float(row["t2"]),
            }
        )

    return cases


def connectivity_stats(solid: np.ndarray) -> dict[str, Any]:
    labeled, num = label(solid)
    active = int(np.count_nonzero(solid))
    if num == 0 or active == 0:
        return {"connected_components": int(num), "largest_component_ratio": 0.0, "active_elements": active}

    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    largest = int(np.max(counts))
    return {
        "connected_components": int(num),
        "largest_component_ratio": float(largest / active),
        "active_elements": active,
    }


def existing_success_keys() -> set[tuple[str, int, int]]:
    if not OUTPUT_CSV.exists():
        return set()
    keys: set[tuple[str, int, int]] = set()
    for row in read_csv_rows(OUTPUT_CSV):
        if row.get("status") == "ok":
            keys.add((row["case_name"], int(row["n_per_cell_fe"]), int(row["closing_iterations"])))
    return keys


def empty_row(case: dict[str, Any], n_per_cell_fe: int, status: str, error_message: str = "") -> dict[str, Any]:
    row = {field: "" for field in FIELDNAMES}
    row.update(
        {
            "case_name": case["case_name"],
            "target_rho": case["target_rho"],
            "target_w": case["target_w"],
            "source_target_rho": case["source_target_rho"],
            "source_target_w": case["source_target_w"],
            "actual_rho_lookup": case["actual_rho_lookup"],
            "actual_w_lookup": case["actual_w_lookup"],
            "t1": case["t1"],
            "t2": case["t2"],
            "cell_size": CELL_SIZE,
            "cells_x": CELLS[0],
            "cells_y": CELLS[1],
            "cells_z": CELLS[2],
            "overall_size": CELL_SIZE * CELLS[0],
            "n_per_cell_fe": n_per_cell_fe,
            "closing_iterations": CLOSING_ITERATIONS,
            "keep_largest": KEEP_LARGEST,
            "status": status,
            "error_message": error_message,
        }
    )
    return row


def run_one_cells3_case(case: dict[str, Any], n_per_cell_fe: int) -> dict[str, Any]:
    print(f"\nFDM cells3 case {case['case_name']} | N={n_per_cell_fe}")
    start = time.time()
    row = empty_row(case, n_per_cell_fe, status="ok")

    try:
        solid, size, elem_size = generate_ipc_voxels_threshold(
            cell_size=CELL_SIZE,
            cells=CELLS,
            n_per_cell_fe=n_per_cell_fe,
            t1=case["t1"],
            t2=case["t2"],
            alpha=ALPHA,
            keep_largest=KEEP_LARGEST,
        )

        if CLOSING_ITERATIONS > 0:
            solid = binary_closing(solid, iterations=CLOSING_ITERATIONS)

        actual_rho = float(np.mean(solid))
        conn = connectivity_stats(solid)
        result = solve_compression(
            solid=solid,
            size=size,
            elem_size=elem_size,
            E=E_SOLID,
            nu=NU_SOLID,
            prescribed_disp=DISP,
        )

        row.update(
            {
                "actual_rho_cells3": actual_rho,
                "active_elements": conn["active_elements"],
                "connected_components": conn["connected_components"],
                "largest_component_ratio": conn["largest_component_ratio"],
                "reaction_force_N": result["reaction_force"],
                "force_abs_N": result["force_abs"],
                "stress_MPa": result["stress"],
                "strain": result["strain"],
                "E_eff_MPa": result["E_eff"],
                "energy_index_UKU": result["compliance"],
            }
        )
        print(
            f"rho={actual_rho:.6f}, components={conn['connected_components']}, "
            f"largest={conn['largest_component_ratio']:.6f}, E={result['E_eff']:.6f}"
        )
    except Exception as exc:
        row["status"] = "error"
        row["error_message"] = repr(exc)
        print(f"[error] {case['case_name']}, N={n_per_cell_fe}: {exc}")
    finally:
        row["runtime_s"] = time.time() - start

    return row


def append_row(row: dict[str, Any]) -> None:
    mode = "a" if OUTPUT_CSV.exists() else "w"
    with OUTPUT_CSV.open(mode, newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if mode == "w":
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FDM cells=(3,3,3) compression simulations.")
    parser.add_argument("--list-cases", action="store_true", help="List selected cases and exit.")
    parser.add_argument("--rho", type=float, help="Target rho for a focused run.")
    parser.add_argument("--w", type=float, help="Target w for a focused run.")
    parser.add_argument("--n-list", type=int, nargs="+", default=DEFAULT_N_LIST, help="N_PER_CELL_FE values.")
    parser.add_argument("--full", action="store_true", help="Run default cases with N=12,16,20.")
    parser.add_argument("--append", action="store_true", help="Accepted for CLI symmetry; output is always appended.")
    parser.add_argument("--force", action="store_true", help="Re-run even if a matching successful row exists.")
    return parser


def selected_targets(args: argparse.Namespace) -> list[tuple[float, float]]:
    if args.rho is not None or args.w is not None:
        if args.rho is None or args.w is None:
            raise ValueError("Both --rho and --w are required for a focused run")
        return [(args.rho, args.w)]
    return DEFAULT_CASES


def main() -> None:
    args = build_parser().parse_args()
    cases = read_selected_cases(selected_targets(args))
    n_list = FULL_N_LIST if args.full else args.n_list

    print(f"Input CSV: {LOOKUP_CSV}")
    for case in cases:
        print(
            f"{case['case_name']}: t1={case['t1']:.6g}, t2={case['t2']:.6g}, "
            f"source_w={case['source_target_w']:.1f}"
        )

    if args.list_cases:
        return

    done = existing_success_keys()
    for case in cases:
        for n in n_list:
            key = (case["case_name"], int(n), CLOSING_ITERATIONS)
            if key in done and not args.force:
                print(f"[skip existing] {case['case_name']} N={n}")
                continue
            append_row(run_one_cells3_case(case, n))

    print(f"\nOutput written: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
