#!/usr/bin/env python
"""Evaluate trained PINN models and generate diagnostic plots."""

import argparse
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from pinn_qushion.inference import POTENTIAL_CONFIGS, ModelManager


def evaluate_model(manager: ModelManager, potential_name: str, output_dir: Path):
    """Evaluate a single model and generate diagnostic plots."""
    print(f"\nEvaluating: {potential_name}")
    print("-" * 40)

    model = manager.get_model(potential_name)
    if model is None:
        print(f"  Model weights not found, skipping")
        return

    config = POTENTIAL_CONFIGS[potential_name]
    potential = config["class"](**config["params"])

    # Spatial grid
    x = jnp.linspace(-10, 10, 256)
    dx = float(x[1] - x[0])

    # Test parameters
    x0, k0 = 0.0, 2.0
    n = len(x)
    x0_arr = jnp.full(n, x0)
    k0_arr = jnp.full(n, k0)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"{potential_name.replace('_', ' ').title()}", fontsize=14)

    # 1. Probability density at different times
    ax1 = axes[0, 0]
    times = [0.0, 2.0, 5.0, 10.0]
    for t in times:
        t_arr = jnp.full(n, t)
        psi_r, psi_i = model.psi(x, t_arr, x0_arr, k0_arr)
        prob = psi_r**2 + psi_i**2
        ax1.plot(x, prob, label=f"t={t:.1f}")
    ax1.set_xlabel("x")
    ax1.set_ylabel("|Ψ|²")
    ax1.set_title("Probability Density Evolution")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Potential overlay
    ax2 = axes[0, 1]
    V = potential(x)
    V_clipped = np.clip(np.array(V), -20, 20)
    ax2.plot(x, V_clipped, 'k-', linewidth=2)
    ax2.set_xlabel("x")
    ax2.set_ylabel("V(x)")
    ax2.set_title("Potential Shape")
    ax2.grid(True, alpha=0.3)

    # 3. Normalization over time
    ax3 = axes[1, 0]
    t_range = jnp.linspace(0, 20, 100)
    norms = []
    for t in t_range:
        t_arr = jnp.full(n, float(t))
        psi_r, psi_i = model.psi(x, t_arr, x0_arr, k0_arr)
        prob = psi_r**2 + psi_i**2
        norm = float(jnp.sum(prob) * dx)
        norms.append(norm)
    ax3.plot(t_range, norms, 'b-')
    ax3.axhline(y=1.0, color='r', linestyle='--', label='Expected (normalized)')
    ax3.set_xlabel("Time")
    ax3.set_ylabel("∫|Ψ|²dx")
    ax3.set_title("Probability Conservation")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Initial condition comparison
    ax4 = axes[1, 1]
    t0_arr = jnp.zeros(n)
    psi_r, psi_i = model.psi(x, t0_arr, x0_arr, k0_arr)
    prob_pred = psi_r**2 + psi_i**2

    # Analytical initial wavepacket
    sigma = 1.0
    envelope = jnp.exp(-((x - x0) ** 2) / (4 * sigma**2))
    psi_analytic = envelope  # Real part for k0=0 is just envelope
    prob_analytic = psi_analytic**2
    prob_analytic = prob_analytic / (jnp.sum(prob_analytic) * dx)  # Normalize

    ax4.plot(x, prob_pred, 'b-', label='PINN prediction')
    ax4.plot(x, prob_analytic, 'r--', label='Analytical IC')
    ax4.set_xlabel("x")
    ax4.set_ylabel("|Ψ(x,0)|²")
    ax4.set_title("Initial Condition Comparison")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    fig_path = output_dir / f"{potential_name}_diagnostics.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  Saved: {fig_path}")

    # Print summary statistics
    print(f"  Initial norm: {norms[0]:.4f}")
    print(f"  Final norm:   {norms[-1]:.4f}")
    print(f"  Norm range:   [{min(norms):.4f}, {max(norms):.4f}]")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained PINN models")
    parser.add_argument(
        "--weights-dir",
        type=str,
        default="weights",
        help="Directory containing model weights",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="diagnostics",
        help="Directory to save diagnostic plots",
    )
    parser.add_argument(
        "--potentials",
        type=str,
        nargs="+",
        default=list(POTENTIAL_CONFIGS.keys()),
        help="Which potentials to evaluate (default: all)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    manager = ModelManager(weights_dir=args.weights_dir)

    print("=" * 60)
    print("PINN Model Evaluation")
    print("=" * 60)

    for potential_name in args.potentials:
        if potential_name not in POTENTIAL_CONFIGS:
            print(f"Unknown potential: {potential_name}, skipping")
            continue
        evaluate_model(manager, potential_name, output_dir)

    print("\n" + "=" * 60)
    print(f"Diagnostics saved to: {output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
