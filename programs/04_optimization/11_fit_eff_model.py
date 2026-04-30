#!/usr/bin/env python3
"""Fit E_eff(ρ, w) prediction model."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RVE_CSV = PROJECT_ROOT / "rve_pbc_homogenization_results.csv"


def read_rve_results() -> list[dict[str, Any]]:
    """Read RVE results from CSV."""
    rows: list[dict[str, Any]] = []
    if not RVE_CSV.exists():
        raise FileNotFoundError(f"RVE results CSV not found: {RVE_CSV}")
    
    with RVE_CSV.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") != "ok":
                continue
            rows.append({
                "target_rho": float(row["target_rho"]),
                "target_w": float(row["target_w"]),
                "actual_rho_rve": float(row["actual_rho_rve"]),
                "E_eff": float(row["E_eff"]),
                "G_eff": float(row["G_eff"]),
                "nu_eff": float(row["nu_eff"]),
                "A_zener": float(row["A_zener"]),
            })
    return rows


def power_law_model(x: tuple[float, float], a: float, b: float, c: float, d: float) -> float:
    """Power law model for E_eff(ρ, w)."""
    rho, w = x
    return a * (rho ** b) * (1 + c * w ** d)


def exponential_model(x: tuple[float, float], a: float, b: float, c: float, d: float, e: float) -> float:
    """Exponential model for E_eff(ρ, w)."""
    rho, w = x
    return a * np.exp(b * rho) * (1 + c * (1 - np.exp(-d * w)) ** e)


def polynomial_model(x: tuple[float, float], a: float, b: float, c: float, d: float, e: float, f: float) -> float:
    """Polynomial model for E_eff(ρ, w)."""
    rho, w = x
    return a + b*rho + c*w + d*rho*w + e*rho**2 + f*w**2


def fit_model(model_func, x_data: np.ndarray, y_data: np.ndarray, initial_guess: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Fit model to data."""
    popt, pcov = curve_fit(model_func, x_data.T, y_data, p0=initial_guess)
    return popt, pcov


def calculate_r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate R-squared value."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def plot_3d_surface(x: np.ndarray, y: np.ndarray, z: np.ndarray, z_pred: np.ndarray, title: str, filename: str) -> None:
    """Plot 3D surface of the model."""
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create meshgrid for surface plot
    rho_grid, w_grid = np.meshgrid(np.linspace(min(x[:, 0]), max(x[:, 0]), 50), 
                                  np.linspace(min(x[:, 1]), max(x[:, 1]), 50))
    
    # Predict values for meshgrid
    grid_points = np.array([rho_grid.ravel(), w_grid.ravel()]).T
    z_grid = np.zeros_like(rho_grid)
    for i in range(rho_grid.shape[0]):
        for j in range(rho_grid.shape[1]):
            z_grid[i, j] = power_law_model((rho_grid[i, j], w_grid[i, j]), 
                                          *popt_power)
    
    # Plot actual data points
    sc = ax.scatter(x[:, 0], x[:, 1], z, c='blue', marker='o', label='Actual')
    
    # Plot fitted surface
    surf = ax.plot_surface(rho_grid, w_grid, z_grid, cmap='viridis', alpha=0.5, label='Fitted')
    
    ax.set_xlabel('ρ*')
    ax.set_ylabel('w')
    ax.set_zlabel('E_eff (MPa)')
    ax.set_title(title)
    ax.legend()
    
    # Add color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    output_path = PROJECT_ROOT / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()


def plot_2d_cuts(x: np.ndarray, y_data: np.ndarray, model_func: callable, popt: np.ndarray, title: str, filename: str) -> None:
    """Plot 2D cuts of the model at different w values."""
    plt.figure(figsize=(10, 6))
    
    # Get unique w values
    w_values = sorted(set(x[:, 1]))
    
    for w in w_values:
        # Filter data for this w
        mask = x[:, 1] == w
        rho_data = x[mask, 0]
        E_data = y_data[mask]
        
        if len(rho_data) > 0:
            # Sort data by rho
            sorted_indices = np.argsort(rho_data)
            rho_sorted = rho_data[sorted_indices]
            E_sorted = E_data[sorted_indices]
            
            # Predict values
            E_pred = [model_func((r, w), *popt) for r in rho_sorted]
            
            # Plot
            plt.plot(rho_sorted, E_sorted, 'o-', label=f'w={w:.1f} (actual)')
            plt.plot(rho_sorted, E_pred, '--', label=f'w={w:.1f} (predicted)')
    
    plt.xlabel('ρ*')
    plt.ylabel('E_eff (MPa)')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    output_path = PROJECT_ROOT / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()


def main() -> None:
    """Main function."""
    print(f"Reading RVE results from: {RVE_CSV}")
    rows = read_rve_results()
    
    if len(rows) < 5:
        print("Insufficient data for fitting. Need at least 5 data points.")
        return
    
    # Prepare data
    x_data = np.array([[row['target_rho'], row['target_w']] for row in rows])
    y_data = np.array([row['E_eff'] for row in rows])
    
    print(f"Found {len(rows)} data points for fitting")
    
    # Fit power law model
    print("\nFitting power law model...")
    initial_guess_power = [1000, 2, 1, 1]
    popt_power, pcov_power = fit_model(power_law_model, x_data, y_data, initial_guess_power)
    y_pred_power = np.array([power_law_model((r, w), *popt_power) for r, w in x_data])
    r2_power = calculate_r_squared(y_data, y_pred_power)
    
    print(f"Power law model parameters: {popt_power}")
    print(f"R-squared: {r2_power:.4f}")
    
    # Fit exponential model
    print("\nFitting exponential model...")
    initial_guess_exp = [100, 5, 1, 5, 1]
    try:
        popt_exp, pcov_exp = fit_model(exponential_model, x_data, y_data, initial_guess_exp)
        y_pred_exp = np.array([exponential_model((r, w), *popt_exp) for r, w in x_data])
        r2_exp = calculate_r_squared(y_data, y_pred_exp)
        print(f"Exponential model parameters: {popt_exp}")
        print(f"R-squared: {r2_exp:.4f}")
    except Exception as e:
        print(f"Exponential model fitting failed: {e}")
        r2_exp = 0
    
    # Fit polynomial model
    print("\nFitting polynomial model...")
    initial_guess_poly = [0, 1000, 100, 100, 1000, 100]
    popt_poly, pcov_poly = fit_model(polynomial_model, x_data, y_data, initial_guess_poly)
    y_pred_poly = np.array([polynomial_model((r, w), *popt_poly) for r, w in x_data])
    r2_poly = calculate_r_squared(y_data, y_pred_poly)
    
    print(f"Polynomial model parameters: {popt_poly}")
    print(f"R-squared: {r2_poly:.4f}")
    
    # Compare models
    print("\nModel comparison:")
    print(f"Power law: R² = {r2_power:.4f}")
    print(f"Exponential: R² = {r2_exp:.4f}")
    print(f"Polynomial: R² = {r2_poly:.4f}")
    
    # Select best model
    best_model = max([('power', r2_power), ('exponential', r2_exp), ('polynomial', r2_poly)], key=lambda x: x[1])
    print(f"\nBest model: {best_model[0]} with R² = {best_model[1]:.4f}")
    
    # Plot results using the best model
    if best_model[0] == 'power':
        plot_3d_surface(x_data, y_data, y_pred_power, 
                       "Power Law Model: E_eff(ρ, w)", 
                       "E_eff_model_power_law_3d.png")
        plot_2d_cuts(x_data, y_data, power_law_model, popt_power, 
                     "Power Law Model: E_eff vs ρ* at different w", 
                     "E_eff_model_power_law_2d.png")
    elif best_model[0] == 'exponential':
        plot_3d_surface(x_data, y_data, y_pred_exp, 
                       "Exponential Model: E_eff(ρ, w)", 
                       "E_eff_model_exponential_3d.png")
        plot_2d_cuts(x_data, y_data, exponential_model, popt_exp, 
                     "Exponential Model: E_eff vs ρ* at different w", 
                     "E_eff_model_exponential_2d.png")
    else:
        plot_3d_surface(x_data, y_data, y_pred_poly, 
                       "Polynomial Model: E_eff(ρ, w)", 
                       "E_eff_model_polynomial_3d.png")
        plot_2d_cuts(x_data, y_data, polynomial_model, popt_poly, 
                     "Polynomial Model: E_eff vs ρ* at different w", 
                     "E_eff_model_polynomial_2d.png")
    
    # Save model parameters
    with open(PROJECT_ROOT / "E_eff_model_parameters.txt", "w") as f:
        f.write(f"Best model: {best_model[0]}\n")
        f.write(f"R-squared: {best_model[1]:.4f}\n\n")
        
        if best_model[0] == 'power':
            f.write("Power law model: E = a * ρ^b * (1 + c * w^d)\n")
            f.write(f"a = {popt_power[0]:.4f}\n")
            f.write(f"b = {popt_power[1]:.4f}\n")
            f.write(f"c = {popt_power[2]:.4f}\n")
            f.write(f"d = {popt_power[3]:.4f}\n")
        elif best_model[0] == 'exponential':
            f.write("Exponential model: E = a * exp(b * ρ) * (1 + c * (1 - exp(-d * w))^e)\n")
            f.write(f"a = {popt_exp[0]:.4f}\n")
            f.write(f"b = {popt_exp[1]:.4f}\n")
            f.write(f"c = {popt_exp[2]:.4f}\n")
            f.write(f"d = {popt_exp[3]:.4f}\n")
            f.write(f"e = {popt_exp[4]:.4f}\n")
        else:
            f.write("Polynomial model: E = a + b*ρ + c*w + d*ρ*w + e*ρ² + f*w²\n")
            f.write(f"a = {popt_poly[0]:.4f}\n")
            f.write(f"b = {popt_poly[1]:.4f}\n")
            f.write(f"c = {popt_poly[2]:.4f}\n")
            f.write(f"d = {popt_poly[3]:.4f}\n")
            f.write(f"e = {popt_poly[4]:.4f}\n")
            f.write(f"f = {popt_poly[5]:.4f}\n")
    
    print("\nModel parameters saved to E_eff_model_parameters.txt")
    print("\nFitting complete!")


if __name__ == "__main__":
    main()
