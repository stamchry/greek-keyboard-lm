#!/usr/bin/env python3
"""
plot_training.py - Real-time Training Curves Visualizer for Greek Keyboard LM

Reads training.log dynamically and plots:
1. Training Loss (raw + smoothed moving average)
2. Learning Rate Schedule
3. Validation Loss and Perplexity

Outputs:
- Saves high-res chart to training_curves.png
- Optionally prints Unicode/ASCII chart directly in the terminal (via plotext)
"""

import re
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any

import matplotlib.pyplot as plt
import numpy as np


def parse_training_log(log_path: Path) -> Dict[str, Any]:
    """Parse training.log to extract training and validation metrics."""
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found at {log_path}")

    steps: List[int] = []
    losses: List[float] = []
    lrs: List[float] = []

    val_steps: List[int] = []
    val_losses: List[float] = []
    val_ppls: List[float] = []

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # Regex for progress bar postfix: 95/8048 ... loss=8.5965, lr=2.82e-05
    matches = re.findall(r'(\d+)/\d+.*?loss=([0-9\.]+),\s*lr=([0-9\.e\-+]+)', content)
    for step_str, loss_str, lr_str in matches:
        step_idx = int(step_str)
        loss = float(loss_str)
        lr = float(lr_str)
        steps.append(step_idx)
        losses.append(loss)
        lrs.append(lr)

    # Regex for periodic validation: [Step 250] Validation Loss: 4.1234 | Perplexity: 61.77
    val_matches = re.findall(r'\[Step\s+(\d+)\]\s+Validation Loss:\s+([0-9\.]+)\s+\|\s+Perplexity:\s+([0-9\.]+)', content)
    for step_str, loss_str, ppl_str in val_matches:
        val_steps.append(int(step_str))
        val_losses.append(float(loss_str))
        val_ppls.append(float(ppl_str))

    return {
        "steps": steps,
        "losses": losses,
        "lrs": lrs,
        "val_steps": val_steps,
        "val_losses": val_losses,
        "val_ppls": val_ppls
    }


def moving_average(values: List[float], window_size: int = 10) -> List[float]:
    """Compute simple moving average."""
    if len(values) < window_size:
        return values
    cumsum = np.cumsum(np.insert(values, 0, 0))
    ma = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
    pad = [values[0]] * (window_size - 1)
    return pad + list(ma)


def plot_matplotlib(data: Dict[str, Any], output_png: Path):
    """Generate high-resolution PNG chart with Matplotlib."""
    steps = data["steps"]
    losses = data["losses"]
    lrs = data["lrs"]
    val_steps = data["val_steps"]
    val_losses = data["val_losses"]
    val_ppls = data["val_ppls"]

    if not steps:
        print("[!] No training steps found in log yet.")
        return

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Subplot 1: Training & Validation Loss
    x_indices = list(range(1, len(losses) + 1))
    ax1.plot(x_indices, losses, alpha=0.35, color="#1f77b4", label="Train Loss (Raw)")
    if len(losses) >= 10:
        smoothed_loss = moving_average(losses, window_size=min(15, len(losses)))
        ax1.plot(x_indices, smoothed_loss, color="#1f77b4", linewidth=2.0, label="Train Loss (Smoothed)")

    if val_steps:
        ax1.plot(val_steps, val_losses, "ro--", label="Val Loss", linewidth=2.0, markersize=6)

    ax1.set_ylabel("Cross Entropy Loss", fontsize=12, fontweight="bold")
    ax1.set_title("Greek Keyboard LM (~36M Mini-LLaMA) Training Progress", fontsize=14, fontweight="bold", pad=12)
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Subplot 2: Learning Rate & Validation Perplexity
    ax2.plot(x_indices, lrs, color="#2ca02c", linewidth=1.8, label="Learning Rate (AdamW Cosine)")
    ax2.set_xlabel("Logged Training Steps", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Learning Rate", fontsize=12, fontweight="bold", color="#2ca02c")
    ax2.tick_params(axis="y", labelcolor="#2ca02c")
    ax2.grid(True, linestyle="--", alpha=0.6)

    if val_steps and any(p > 0 for p in val_ppls):
        ax2_ppl = ax2.twinx()
        ax2_ppl.plot(val_steps, val_ppls, "mo-.", linewidth=2.0, markersize=6, label="Val Perplexity (PPL)")
        ax2_ppl.set_ylabel("Validation Perplexity", fontsize=12, fontweight="bold", color="#9467bd")
        ax2_ppl.tick_params(axis="y", labelcolor="#9467bd")

    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200)
    plt.close(fig)
    print(f"[✓] Saved chart: {output_png} (Total Points: {len(steps)})")


def plot_terminal(data: Dict[str, Any]):
    """Render ASCII / Unicode plot directly in terminal."""
    losses = data["losses"]
    if not losses:
        print("[!] No training steps found in log yet.")
        return

    try:
        import plotext as plt_term
        x = list(range(1, len(losses) + 1))
        fig = plt_term.figure()
        if len(losses) >= 10:
            smoothed = moving_average(losses, window_size=min(10, len(losses)))
            fig.plot(x, smoothed, label="Smoothed Loss")
        else:
            fig.plot(x, losses, label="Train Loss")
        fig.title("Live Training Loss Curve")
        fig.xlabel("Step")
        fig.ylabel("Loss")
        fig.show()
    except Exception as e:
        # Fallback simple text summary
        min_loss = min(losses)
        max_loss = max(losses)
        curr_loss = losses[-1]
        print(f"Loss Summary -> Initial: {losses[0]:.4f} | Min: {min_loss:.4f} | Current: {curr_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Visualize Greek LM Training Curves")
    parser.add_argument("--log_file", type=str, default="training.log", help="Path to training.log")
    parser.add_argument("--output", type=str, default="training_curves.png", help="Path to output PNG image")
    parser.add_argument("--terminal", action="store_true", help="Also display ASCII plot in terminal")
    args = parser.parse_args()

    data = parse_training_log(Path(args.log_file))
    print(f"[-] Parsed {len(data['steps'])} step data points from {args.log_file}")
    if data["steps"]:
        print(f"[-] Initial Loss: {data['losses'][0]:.4f} -> Current Loss: {data['losses'][-1]:.4f} | Current LR: {data['lrs'][-1]:.2e}")

    plot_matplotlib(data, Path(args.output))

    if args.terminal:
        print("\n" + "=" * 50)
        plot_terminal(data)
        print("=" * 50)


if __name__ == "__main__":
    main()
