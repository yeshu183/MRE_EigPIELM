#!/usr/bin/env python
"""
Optimized training script for Lightning AI with GPU support
This script trains MRE-PINN models on the BIOQIC simulation dataset.

Usage:
    python train_lightning.py --example_id 60 --frequency 90 --n_iters 100000
"""

import sys, os
import numpy as np
import torch

# Set DeepXDE backend before importing
os.environ['DDEBACKEND'] = 'pytorch'
import deepxde

import mre_pinn
from mre_pinn.utils import main
from mre_pinn.training.losses import msae_loss


def print_system_info():
    """Print system and GPU information"""
    print("\n" + "="*60)
    print("SYSTEM INFORMATION")
    print("="*60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    else:
        print("WARNING: CUDA not available. Training will be slow on CPU.")

    print(f"DeepXDE version: {deepxde.__version__}")
    print("="*60 + "\n")


@main
def train(
    # data settings
    xarray_dir='data/BIOQIC/fem_box',
    example_id='60',
    frequency='auto',
    noise_ratio=0.0,
    anatomical=False,

    # pde settings
    pde_name='hetero',

    # baseline settings
    savgol_filter=False,

    # model settings
    omega=30,
    n_layers=5,
    n_hidden=128,
    activ_fn='ss',
    polar_input=False,

    # training settings
    optimizer='adam',
    learning_rate=1e-4,
    u_loss_weight=1.0,
    mu_loss_weight=0.0,
    a_loss_weight=0.0,
    pde_loss_weight=1e-16,
    pde_warmup_iters=10000,
    pde_init_weight=1e-18,
    pde_step_iters=5000,
    pde_step_factor=10,
    n_points=1024,
    n_iters=100000,

    # testing settings
    test_every=1000,
    save_every=10000,
    save_prefix=None,

    # device settings
    device='auto'  # 'auto', 'cuda', or 'cpu'
):
    """
    Train MRE-PINN model on BIOQIC simulation dataset.

    Args:
        xarray_dir: Directory containing preprocessed xarray data
        example_id: Example ID to train on (frequency in Hz)
        frequency: Override frequency (default: auto-detect from data)
        device: Device to use ('auto', 'cuda', or 'cpu')
        ... (see function signature for all parameters)
    """

    # Print system information
    print_system_info()

    # Determine device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Reset GPU memory stats if using CUDA
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)

    # Load the training data
    example = mre_pinn.data.MREExample.load_xarrays(
        xarray_dir=xarray_dir,
        example_id=example_id,
        anat=anatomical
    )

    if frequency == 'auto':  # infer from data
        frequency = float(example.wave.frequency.item())
    else:
        frequency = float(frequency)

    print(f"Training example: {example_id}")
    print(f"Frequency: {frequency} Hz")

    if noise_ratio > 0:
        print(f"Adding Gaussian noise (ratio={noise_ratio})")
        example.add_gaussian_noise(noise_ratio)

    print("\n" + "="*60)
    print("COMPUTING BASELINES")
    print("="*60)

    # Compute baselines
    print("Computing AHI baseline...")
    mre_pinn.baseline.eval_ahi_baseline(
        example, frequency=frequency, savgol_filter=savgol_filter
    )

    print("Computing FEM baseline...")
    mre_pinn.baseline.eval_fem_baseline(
        example,
        frequency=frequency,
        hetero=(pde_name == 'hetero'),
        savgol_filter=savgol_filter
    )

    print("\n" + "="*60)
    print("BUILDING MODEL")
    print("="*60)

    # Define PDE that we want to solve
    pde = mre_pinn.pde.WaveEquation.from_name(
        pde_name, omega=frequency, detach=True
    )
    print(f"PDE: {pde_name}")

    # Define the model architecture
    pinn = mre_pinn.model.MREPINN(
        example,
        omega=omega,
        n_layers=n_layers,
        n_hidden=n_hidden,
        activ_fn=activ_fn,
        polar_input=polar_input
    )
    print(pinn)

    # Compile model and configure training settings
    model = mre_pinn.training.MREPINNModel(
        example, pinn, pde,
        loss_weights=[u_loss_weight, mu_loss_weight, a_loss_weight, pde_loss_weight],
        pde_warmup_iters=pde_warmup_iters,
        pde_step_iters=pde_step_iters,
        pde_step_factor=pde_step_factor,
        pde_init_weight=pde_init_weight,
        n_points=n_points,
        device=device
    )
    model.compile(optimizer=optimizer, lr=learning_rate, loss=msae_loss)

    print("\n" + "="*60)
    print("BENCHMARKING")
    print("="*60)
    model.benchmark(100)

    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    print(f"Training for {n_iters:,} iterations")
    print(f"Test every: {test_every}")
    print(f"Save every: {save_every}")

    # Create test evaluator
    test_eval = mre_pinn.testing.TestEvaluator(
        test_every=test_every,
        save_every=save_every,
        save_prefix=save_prefix,
        interact=False  # Disable interactive plots on Lightning AI
    )
    test_eval.model = model

    # Train the model
    model.train(n_iters, display_every=100, callbacks=[test_eval])

    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    test_eval.test()
    print(test_eval.metrics)

    # Print GPU memory stats if using CUDA
    if device == 'cuda':
        peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"\nPeak GPU memory usage: {peak_memory_gb:.2f} GB")

        # Print per-device memory stats
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            reserved = torch.cuda.memory_reserved(i) / (1024**3)
            print(f"GPU {i} - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)


if __name__ == '__main__':
    # This will be called automatically by the @main decorator
    pass
