#!/usr/bin/env python
"""Train all PINN models for each potential type with loss curve logging."""

import argparse
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from tqdm import tqdm

from pinn_qushion.inference import POTENTIAL_CONFIGS
from pinn_qushion.models import PINN
from pinn_qushion.training import CollocationSampler, PINNLoss, Trainer


def plot_loss_curves(loss_history: dict, output_path: Path, potential_name: str, log_every: int = 100):
    """Generate and save loss curve plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Training Curves: {potential_name.replace('_', ' ').title()}", fontsize=14)

    n_logged = len(loss_history["total"])
    iterations = np.arange(1, n_logged + 1) * log_every

    # 1. Total loss (log scale)
    ax1 = axes[0, 0]
    ax1.semilogy(iterations, loss_history["total"], 'b-', alpha=0.7, linewidth=0.5)
    # Add smoothed line
    window = min(1000, len(iterations) // 10)
    if window > 1:
        smoothed = np.convolve(loss_history["total"], np.ones(window)/window, mode='valid')
        ax1.semilogy(np.arange(window//2, len(iterations) - window//2 + 1), smoothed, 'b-', linewidth=2, label='Smoothed')
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Total Loss")
    ax1.set_title("Total Loss (log scale)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. Component losses (log scale) — shade phase boundary if curriculum used
    ax2 = axes[0, 1]
    if "phase" in loss_history and loss_history["phase"]:
        phases = np.array(loss_history["phase"])
        phase2_start = int(np.argmax(phases == 2)) if 2 in phases else 0
        if phase2_start > 0:
            ax2.axvline(phase2_start, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Phase 2 start')
    phys = np.array(loss_history["physics"])
    # Avoid semilogy with zeros (phase 1 has phys=0)
    phys_safe = np.where(phys > 0, phys, np.nan)
    norm_arr = np.array(loss_history.get("norm", []))
    norm_safe = np.where(norm_arr > 0, norm_arr, np.nan) if len(norm_arr) else None
    ax2.semilogy(iterations, phys_safe, 'r-', alpha=0.5, linewidth=0.5, label='Physics')
    ax2.semilogy(iterations, loss_history["ic"], 'g-', alpha=0.5, linewidth=0.5, label='IC')
    ax2.semilogy(iterations, loss_history["bc"], 'b-', alpha=0.5, linewidth=0.5, label='BC')
    if norm_safe is not None:
        ax2.semilogy(iterations, norm_safe, 'm-', alpha=0.5, linewidth=0.5, label='Norm')
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Loss Component")
    ax2.set_title("Loss Components (log scale)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Loss ratios over time
    ax3 = axes[1, 0]
    total = np.array(loss_history["total"])
    physics_ratio = np.array(loss_history["physics"]) / (total + 1e-10)
    ic_ratio = np.array(loss_history["ic"]) / (total + 1e-10)
    bc_ratio = np.array(loss_history["bc"]) / (total + 1e-10)
    ax3.plot(iterations, physics_ratio, 'r-', alpha=0.5, linewidth=0.5, label='Physics/Total')
    ax3.plot(iterations, ic_ratio, 'g-', alpha=0.5, linewidth=0.5, label='IC/Total')
    ax3.plot(iterations, bc_ratio, 'b-', alpha=0.5, linewidth=0.5, label='BC/Total')
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("Fraction of Total Loss")
    ax3.set_title("Loss Component Ratios")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)

    # 4. Final statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    stats_text = f"""
Training Statistics for {potential_name}
{'='*50}

Total Iterations: {len(iterations):,}

Final Losses (last 1000 avg):
  Total:   {np.mean(loss_history['total'][-1000:]):.6f}
  Physics: {np.mean(loss_history['physics'][-1000:]):.6f}
  IC:      {np.mean(loss_history['ic'][-1000:]):.6f}
  BC:      {np.mean(loss_history['bc'][-1000:]):.6f}

Initial Losses (first 100 avg):
  Total:   {np.mean(loss_history['total'][:100]):.6f}
  Physics: {np.mean(loss_history['physics'][:100]):.6f}
  IC:      {np.mean(loss_history['ic'][:100]):.6f}
  BC:      {np.mean(loss_history['bc'][:100]):.6f}

Improvement Ratio (initial/final):
  Total:   {np.mean(loss_history['total'][:100]) / (np.mean(loss_history['total'][-1000:]) + 1e-10):.1f}x
  Physics: {np.mean(loss_history['physics'][:100]) / (np.mean(loss_history['physics'][-1000:]) + 1e-10):.1f}x
  IC:      {np.mean(loss_history['ic'][:100]) / (np.mean(loss_history['ic'][-1000:]) + 1e-10):.1f}x
  BC:      {np.mean(loss_history['bc'][:100]) / (np.mean(loss_history['bc'][-1000:]) + 1e-10):.1f}x
"""
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def train_model(
    potential_name: str,
    output_dir: Path,
    log_dir: Path,
    n_iterations: int = 100000,
    batch_size_interior: int = 10000,
    batch_size_ic: int = 2000,
    batch_size_bc: int = 2000,
    batch_size_norm: int = 256,
    learning_rate: float = 1e-3,
    lambda_phys: float = 1.0,
    lambda_ic: float = 10.0,
    lambda_bc: float = 0.0,
    lambda_norm: float = 10.0,
    checkpoint_every: int = 10000,
    log_every: int = 100,
    seed: int = 42,
    curriculum: bool = True,
    curriculum_ic_steps: int = 20000,
    warm_start: str = None,
    norm_late_times: list = None,
):
    """Train a single PINN model for a given potential with component loss logging."""
    print(f"\n{'='*60}")
    print(f"Training: {potential_name}")
    print(f"{'='*60}")
    print(f"  Iterations: {n_iterations:,}")
    print(f"  Lambda weights: phys={lambda_phys}, ic={lambda_ic}, bc={lambda_bc}, norm={lambda_norm}")
    print(f"  Learning rate: {learning_rate}")

    config = POTENTIAL_CONFIGS[potential_name]
    potential = config["class"](**config["params"])

    # Initialize model
    key = jax.random.PRNGKey(seed)
    model = PINN(
        potential=potential,
        hidden_dim=128,
        num_layers=5,
        key=key,
    )

    if warm_start:
        model = eqx.tree_deserialise_leaves(warm_start, model)
        print(f"  Warm start from: {warm_start}")
        # Skip IC curriculum — weights are already trained
        curriculum = False

    # Optimizer with cosine decay and gradient clipping
    schedule = optax.cosine_decay_schedule(
        init_value=learning_rate,
        decay_steps=n_iterations,
        alpha=0.01,  # Minimum learning rate ratio
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),  # Gradient clipping for stability
        optax.adam(schedule),
    )

    # Sampler with potential-aware domain
    # For infinite square well, sample only inside the well and put BC at walls
    if potential_name == "infinite_square_well":
        well_width = config["params"].get("width", 8.0)
        half_width = well_width / 2
        x_range = (-half_width, half_width)  # BC at walls, interior strictly inside
        x0_range = (-half_width / 2, half_width / 2)  # Initial position well inside
    else:
        x_range = (-10, 10)
        x0_range = (-5, 5)

    sampler = CollocationSampler(
        x_range=x_range,
        t_range=(0, 20),
        x0_range=x0_range,
        k0_range=(-3, 3),
    )

    # Curriculum Phase 1: IC + norm only — nail the initial condition before introducing physics
    # Phase 2: full loss with physics ramped in
    ic_steps = curriculum_ic_steps if curriculum else 0
    phys_steps = n_iterations - ic_steps

    if curriculum:
        print(f"  Curriculum: {ic_steps:,} IC-only steps, then {phys_steps:,} full-loss steps")

    # Phase 1 trainer: IC + norm only (physics disabled)
    if curriculum and ic_steps > 0:
        trainer_ic = Trainer(
            model=model,
            optimizer=optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adam(optax.cosine_decay_schedule(learning_rate, ic_steps, alpha=0.1)),
            ),
            sigma=1.0,
            lambda_phys=0.0,
            lambda_ic=lambda_ic * 10,   # strong IC enforcement
            lambda_bc=lambda_bc,
            lambda_norm=lambda_norm * 10,  # strong norm enforcement
            x_range=x_range,
        )
    else:
        trainer_ic = None

    # Phase 2 trainer: full loss
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        sigma=1.0,
        lambda_phys=lambda_phys,
        lambda_ic=lambda_ic,
        lambda_bc=lambda_bc,
        lambda_norm=lambda_norm,
        x_range=x_range,
    )

    dx = (x_range[1] - x_range[0]) / 256
    # Shared loss function for logging component values
    loss_fn = PINNLoss(
        sigma=1.0,
        lambda_phys=lambda_phys,
        lambda_ic=lambda_ic,
        lambda_bc=lambda_bc,
        lambda_norm=lambda_norm,
        dx=dx,
    )

    key = jax.random.PRNGKey(seed + 1)
    loss_history = {
        "total": [],
        "physics": [],
        "ic": [],
        "bc": [],
        "norm": [],
        "phase": [],   # 1 = IC curriculum, 2 = full loss
    }

    # --- Phase 1: IC curriculum ---
    if trainer_ic is not None:
        print(f"\n  Phase 1: IC curriculum ({ic_steps:,} steps)")
        for i in tqdm(range(ic_steps), desc=f"{potential_name} [IC]"):
            key, *subkeys = jax.random.split(key, 7)
            x_ic, t_ic, x0_ic, k0_ic = sampler.sample_initial(subkeys[1], batch_size_ic * 5)
            # dummy interior/bc (unused — lambda_phys=0)
            x_int, t_int, x0_int, k0_int = sampler.sample_interior(subkeys[0], 100)
            x_bc, t_bc, x0_bc, k0_bc = sampler.sample_boundary(subkeys[2], 100)
            # Independent keys for t_norm, x0_norm, k0_norm — previously all used subkeys[3]
            # which made them perfectly correlated linear functions of the same variate.
            t_norm = jax.random.uniform(subkeys[3], minval=0.0, maxval=20.0)
            x0_norm = jax.random.uniform(subkeys[4], minval=x0_range[0], maxval=x0_range[1])
            k0_norm = jax.random.uniform(subkeys[5], minval=-3.0, maxval=3.0)

            loss = trainer_ic.step(
                x_int, t_int, x0_int, k0_int,
                x_ic, t_ic, x0_ic, k0_ic,
                x_bc, t_bc, x0_bc, k0_bc,
                t_norm=float(t_norm),
                x0_norm=float(x0_norm),
                k0_norm=float(k0_norm),
                extra_norm_times=list(norm_late_times) if norm_late_times else [0.0, 5.0, 10.0, 20.0],
            )

            if (i + 1) % log_every == 0:
                current_model = trainer_ic.get_model()
                l_ic = float(loss_fn.initial_condition_loss(current_model, x_ic, t_ic, x0_ic, k0_ic))
                x_ng = trainer_ic.x_norm_grid
                n_ng = len(x_ng)
                l_norm = float(loss_fn.normalization_loss(
                    current_model,
                    x_ng, jnp.zeros(n_ng),
                    jnp.full(n_ng, float(x0_norm)), jnp.full(n_ng, float(k0_norm)),
                ))
                loss_history["total"].append(float(loss))
                loss_history["physics"].append(0.0)
                loss_history["ic"].append(l_ic)
                loss_history["bc"].append(0.0)
                loss_history["norm"].append(l_norm)
                loss_history["phase"].append(1)

        # Hand model off to phase 2 trainer
        trainer.model = trainer_ic.get_model()
        trainer.opt_state = trainer.optimizer.init(eqx.filter(trainer.model, eqx.is_array))
        print(f"  Phase 1 complete. Handing model to phase 2.")

    # --- Phase 2: full loss ---
    print(f"\n  Phase 2: Full loss ({phys_steps:,} steps)")
    for i in tqdm(range(phys_steps), desc=f"{potential_name} [full]"):
        # Sample batches
        key, *subkeys = jax.random.split(key, 5)
        x_int, t_int, x0_int, k0_int = sampler.sample_interior(subkeys[0], batch_size_interior)
        x_ic, t_ic, x0_ic, k0_ic = sampler.sample_initial(subkeys[1], batch_size_ic)
        x_bc, t_bc, x0_bc, k0_bc = sampler.sample_boundary(subkeys[2], batch_size_bc)

        # Use the IC batch's (x0, k0) for norm enforcement — same parameters per step
        # so the model sees "IC amplitude → norm at t>0" as a unified constraint.
        # Sample a random t_norm from exponential (bias toward early t where decay begins).
        t_norm_raw = jax.random.exponential(subkeys[3]) * (20.0 / 5.0)
        t_norm = jnp.clip(t_norm_raw, 0.0, 20.0)
        x0_norm = float(x0_ic[0])
        k0_norm = float(k0_ic[0])

        extra_times = list(norm_late_times) if norm_late_times else [0.0, 10.0]
        loss = trainer.step(
            x_int, t_int, x0_int, k0_int,
            x_ic, t_ic, x0_ic, k0_ic,
            x_bc, t_bc, x0_bc, k0_bc,
            t_norm=float(t_norm),
            x0_norm=x0_norm,
            k0_norm=k0_norm,
            extra_norm_times=extra_times,
        )

        # Log all losses periodically (keep lists the same length)
        if (i + 1) % log_every == 0:
            current_model = trainer.get_model()
            l_phys = float(loss_fn.physics_loss(current_model, x_int, t_int, x0_int, k0_int))
            l_ic = float(loss_fn.initial_condition_loss(current_model, x_ic, t_ic, x0_ic, k0_ic))
            l_bc = float(loss_fn.boundary_condition_loss(current_model, x_bc, t_bc, x0_bc, k0_bc))
            x_ng = trainer.x_norm_grid
            n_ng = len(x_ng)
            # Log norm at t=0 and t=5 for the current (x0,k0) to track late-time health
            log_times = [0.0, 5.0]
            x_log = jnp.concatenate([x_ng] * len(log_times))
            t_log = jnp.concatenate([jnp.full(n_ng, t) for t in log_times])
            l_norm = float(loss_fn.normalization_loss(
                current_model,
                x_log, t_log,
                jnp.full(len(x_log), x0_norm), jnp.full(len(x_log), k0_norm),
            ))
            loss_history["total"].append(float(loss))
            loss_history["physics"].append(l_phys)
            loss_history["ic"].append(l_ic)
            loss_history["bc"].append(l_bc)
            loss_history["norm"].append(l_norm)
            loss_history["phase"].append(2)

        # Checkpoint (offset by IC steps so filenames reflect total training progress)
        global_step = ic_steps + i + 1
        if global_step % checkpoint_every == 0:
            checkpoint_path = output_dir / f"{potential_name}_checkpoint_{global_step}.eqx"
            eqx.tree_serialise_leaves(checkpoint_path, trainer.get_model())
            window = checkpoint_every // log_every
            avg_loss = np.mean(loss_history["total"][-window:])
            avg_phys = np.mean(loss_history["physics"][-window:])
            avg_ic = np.mean(loss_history["ic"][-window:])
            avg_bc = np.mean(loss_history["bc"][-window:])
            avg_norm = np.mean(loss_history["norm"][-window:])
            print(f"\n  Step {global_step}: Total={avg_loss:.6f} | Phys={avg_phys:.6f} | IC={avg_ic:.6f} | BC={avg_bc:.6f} | Norm={avg_norm:.6f}")

    # Save final model
    final_path = output_dir / config["weight_file"]
    eqx.tree_serialise_leaves(final_path, trainer.get_model())
    print(f"\n  Final model saved to: {final_path}")

    # Save loss history as JSON
    loss_json_path = log_dir / f"{potential_name}_losses.json"
    with open(loss_json_path, "w") as f:
        json.dump({
            "potential": potential_name,
            "iterations": n_iterations,
            "lambda_phys": lambda_phys,
            "lambda_ic": lambda_ic,
            "lambda_bc": lambda_bc,
            "lambda_norm": lambda_norm,
            "learning_rate": learning_rate,
            "curriculum": curriculum,
            "curriculum_ic_steps": curriculum_ic_steps if curriculum else 0,
            "log_every": log_every,
            "losses": loss_history,
        }, f)
    print(f"  Loss history saved to: {loss_json_path}")

    # Generate and save loss curve plots
    plot_path = log_dir / f"{potential_name}_loss_curves.png"
    plot_loss_curves(loss_history, plot_path, potential_name, log_every=log_every)
    print(f"  Loss curves saved to: {plot_path}")

    return loss_history


def main():
    parser = argparse.ArgumentParser(description="Train PINN models with loss logging")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="weights",
        help="Directory to save trained weights",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="training_logs",
        help="Directory to save loss curves and logs",
    )
    parser.add_argument(
        "--potentials",
        type=str,
        nargs="+",
        default=list(POTENTIAL_CONFIGS.keys()),
        help="Which potentials to train (default: all)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100000,
        help="Number of training iterations per model",
    )
    parser.add_argument(
        "--lambda-phys",
        type=float,
        default=1.0,
        help="Weight for physics loss",
    )
    parser.add_argument(
        "--lambda-ic",
        type=float,
        default=10.0,
        help="Weight for initial condition loss",
    )
    parser.add_argument(
        "--lambda-bc",
        type=float,
        default=0.0,
        help="Weight for boundary condition loss",
    )
    parser.add_argument(
        "--lambda-norm",
        type=float,
        default=10.0,
        help="Weight for normalization loss",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--no-curriculum",
        action="store_true",
        default=False,
        help="Disable curriculum training (train with full loss from step 0)",
    )
    parser.add_argument(
        "--curriculum-ic-steps",
        type=int,
        default=20000,
        help="Number of IC-only steps in curriculum phase 1",
    )
    parser.add_argument(
        "--warm-start",
        type=str,
        default=None,
        help="Path to .eqx weights file to warm-start from (skips phase 1)",
    )
    parser.add_argument(
        "--norm-late-times",
        type=float,
        nargs="+",
        default=None,
        help="Fixed late times at which to evaluate norm loss every step (e.g. 0 5 10 20)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Pinn-Qushion Training")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Log directory: {log_dir}")
    print(f"Potentials: {args.potentials}")
    print(f"Iterations per model: {args.iterations:,}")
    print(f"Loss weights: λ_phys={args.lambda_phys}, λ_ic={args.lambda_ic}, λ_bc={args.lambda_bc}, λ_norm={args.lambda_norm}")
    print(f"Learning rate: {args.learning_rate}")

    for potential_name in args.potentials:
        if potential_name not in POTENTIAL_CONFIGS:
            print(f"Unknown potential: {potential_name}, skipping")
            continue

        train_model(
            potential_name=potential_name,
            output_dir=output_dir,
            log_dir=log_dir,
            n_iterations=args.iterations,
            learning_rate=args.learning_rate,
            lambda_phys=args.lambda_phys,
            lambda_ic=args.lambda_ic,
            lambda_bc=args.lambda_bc,
            lambda_norm=args.lambda_norm,
            seed=args.seed,
            curriculum=not args.no_curriculum,
            curriculum_ic_steps=args.curriculum_ic_steps,
            warm_start=args.warm_start,
            norm_late_times=args.norm_late_times,
        )

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Weights saved to: {output_dir}/")
    print(f"Loss curves saved to: {log_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
