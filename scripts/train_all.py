#!/usr/bin/env python
"""Train all PINN models for each potential type."""

import argparse
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm

from pinn_qushion.inference import POTENTIAL_CONFIGS
from pinn_qushion.models import PINN
from pinn_qushion.training import CollocationSampler, Trainer


def train_model(
    potential_name: str,
    output_dir: Path,
    n_iterations: int = 50000,
    batch_size_interior: int = 10000,
    batch_size_ic: int = 2000,
    batch_size_bc: int = 2000,
    learning_rate: float = 1e-3,
    checkpoint_every: int = 5000,
    seed: int = 42,
):
    """Train a single PINN model for a given potential."""
    print(f"\n{'='*60}")
    print(f"Training: {potential_name}")
    print(f"{'='*60}")

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

    # Optimizer with cosine decay
    schedule = optax.cosine_decay_schedule(
        init_value=learning_rate,
        decay_steps=n_iterations,
    )
    optimizer = optax.adam(schedule)

    # Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        sigma=1.0,
        lambda_phys=1.0,
        lambda_ic=10.0,
        lambda_bc=10.0,
    )

    # Sampler
    sampler = CollocationSampler(
        x_range=(-10, 10),
        t_range=(0, 20),
        x0_range=(-5, 5),
        k0_range=(-3, 3),
    )

    # Training loop
    key = jax.random.PRNGKey(seed + 1)
    losses = []

    for i in tqdm(range(n_iterations), desc=potential_name):
        # Sample batches
        key, *subkeys = jax.random.split(key, 4)
        x_int, t_int, x0_int, k0_int = sampler.sample_interior(subkeys[0], batch_size_interior)
        x_ic, t_ic, x0_ic, k0_ic = sampler.sample_initial(subkeys[1], batch_size_ic)
        x_bc, t_bc, x0_bc, k0_bc = sampler.sample_boundary(subkeys[2], batch_size_bc)

        # Training step
        loss = trainer.step(
            x_int, t_int, x0_int, k0_int,
            x_ic, t_ic, x0_ic, k0_ic,
            x_bc, t_bc, x0_bc, k0_bc,
        )
        losses.append(float(loss))

        # Checkpoint
        if (i + 1) % checkpoint_every == 0:
            checkpoint_path = output_dir / f"{potential_name}_checkpoint_{i+1}.eqx"
            eqx.tree_serialise_leaves(checkpoint_path, trainer.get_model())
            avg_loss = sum(losses[-checkpoint_every:]) / checkpoint_every
            print(f"\n  Iteration {i+1}: Avg Loss = {avg_loss:.6f}")

    # Save final model
    final_path = output_dir / config["weight_file"]
    eqx.tree_serialise_leaves(final_path, trainer.get_model())
    print(f"\n  Final model saved to: {final_path}")

    return losses


def main():
    parser = argparse.ArgumentParser(description="Train PINN models")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="weights",
        help="Directory to save trained weights",
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
        default=50000,
        help="Number of training iterations per model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("Pinn-Qushion Training")
    print(f"Output directory: {output_dir}")
    print(f"Potentials: {args.potentials}")
    print(f"Iterations per model: {args.iterations}")

    for potential_name in args.potentials:
        if potential_name not in POTENTIAL_CONFIGS:
            print(f"Unknown potential: {potential_name}, skipping")
            continue

        train_model(
            potential_name=potential_name,
            output_dir=output_dir,
            n_iterations=args.iterations,
            seed=args.seed,
        )

    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)


if __name__ == "__main__":
    main()
