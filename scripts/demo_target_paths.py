"""Demo: compare target-blend path types (linear, cosine, quadratic).

Visualizes 1D trajectories and alpha/velocity for three target-blend
approaches. In all cases the model starts from noise, initially heads
toward v1, then redirects toward v2 — without ever reaching v1.

Usage:
    uv run python scripts/demo_target_paths.py          # tau=0.5
    uv run python scripts/demo_target_paths.py --tau 0.3
"""

import argparse
import math

import matplotlib.pyplot as plt
import numpy as np


def target_linear(sigma: np.ndarray, tau: float, noise: float, v1: float, v2: float):
    """Hard switch at tau (C0 velocity, continuous x_t)."""
    high = sigma >= tau

    # High segment
    x_t_h = sigma * noise + (1 - sigma) * v1
    vel_h = np.full_like(sigma, noise - v1)

    # Low segment: from x_theta toward v2
    x_theta = tau * noise + (1 - tau) * v1
    s_l = sigma / tau
    x_t_l = s_l * x_theta + (1 - s_l) * v2
    vel_l = np.full_like(sigma, (x_theta - v2) / tau)

    x_t = np.where(high, x_t_h, x_t_l)
    vel = np.where(high, vel_h, vel_l)

    # alpha: effective blend weight toward v1
    # x_eff = alpha * v1 + (1-alpha) * v2, and x_t = sigma*noise + (1-sigma)*x_eff
    # => x_eff = (x_t - sigma*noise) / (1 - sigma)  when sigma < 1
    # => alpha = (x_eff - v2) / (v1 - v2)            when v1 != v2
    with np.errstate(divide="ignore", invalid="ignore"):
        x_eff = np.where(sigma < 1 - 1e-8, (x_t - sigma * noise) / (1 - sigma), v1)
        alpha = np.where(abs(v1 - v2) > 1e-8, (x_eff - v2) / (v1 - v2), 0.5)

    return x_t, vel, alpha


def target_cosine(sigma: np.ndarray, tau: float, noise: float, v1: float, v2: float):
    """Smooth cosine blend (C1)."""
    pi = math.pi
    high = sigma >= tau

    # Alpha: piecewise cosine
    t_l = sigma / tau
    alpha_l = 0.25 * (1 - np.cos(pi * t_l))
    dalpha_l = 0.25 * pi / tau * np.sin(pi * t_l)

    t_h = (sigma - tau) / (1 - tau)
    alpha_h = 0.5 + 0.25 * (1 - np.cos(pi * t_h))
    dalpha_h = 0.25 * pi / (1 - tau) * np.sin(pi * t_h)

    alpha = np.where(high, alpha_h, alpha_l)
    dalpha = np.where(high, dalpha_h, dalpha_l)

    x_eff = alpha * v1 + (1 - alpha) * v2
    x_t = sigma * noise + (1 - sigma) * x_eff
    vel = noise - x_eff + (1 - sigma) * dalpha * (v1 - v2)

    return x_t, vel, alpha


def target_quadratic(sigma: np.ndarray, tau: float, noise: float, v1: float, v2: float):
    """Quadratic alpha: a*sigma^2 + (1-a)*sigma, with alpha(tau)=0.5."""
    # alpha(0)=0, alpha(1)=1, alpha(tau)=0.5
    # a*tau^2 + (1-a)*tau = 0.5  =>  a = (0.5 - tau) / (tau^2 - tau)
    denom = tau * tau - tau
    a = 0.0 if abs(denom) < 1e-10 else (0.5 - tau) / denom

    alpha = a * sigma**2 + (1 - a) * sigma
    dalpha = 2 * a * sigma + (1 - a)

    x_eff = alpha * v1 + (1 - alpha) * v2
    x_t = sigma * noise + (1 - sigma) * x_eff
    vel = noise - x_eff + (1 - sigma) * dalpha * (v1 - v2)

    return x_t, vel, alpha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--output", type=str, default="scripts/demo_target_paths.png")
    args = parser.parse_args()

    tau = args.tau
    noise, v1, v2 = 1.0, 0.3, -0.2  # 1D scalar values

    sigma = np.linspace(0, 1, 500)

    methods = [
        ("target_linear", target_linear),
        ("target_cosine", target_cosine),
        ("target_quadratic", target_quadratic),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    colors = ["#e74c3c", "#2ecc71", "#3498db"]

    # --- Plot 1: trajectory x_t(sigma) ---
    ax = axes[0]
    for (name, fn), c in zip(methods, colors, strict=False):
        x_t, _, _ = fn(sigma, tau, noise, v1, v2)
        ax.plot(sigma, x_t, label=name, color=c, linewidth=2)
    ax.axvline(tau, color="gray", linestyle="--", alpha=0.5, label=f"τ={tau}")
    ax.axhline(v1, color="orange", linestyle=":", alpha=0.5, label=f"v1={v1}")
    ax.axhline(v2, color="purple", linestyle=":", alpha=0.5, label=f"v2={v2}")
    ax.set_xlabel("σ (1=noise, 0=clean)")
    ax.set_ylabel("x_t")
    ax.set_title("Trajectory x_t(σ)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Plot 2: alpha(sigma) ---
    ax = axes[1]
    for (name, fn), c in zip(methods, colors, strict=False):
        _, _, alpha = fn(sigma, tau, noise, v1, v2)
        ax.plot(sigma, alpha, label=name, color=c, linewidth=2)
    ax.axvline(tau, color="gray", linestyle="--", alpha=0.5, label=f"τ={tau}")
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.3)
    ax.set_xlabel("σ")
    ax.set_ylabel("α (blend weight toward v1)")
    ax.set_title("α(σ): 0=target v2, 1=target v1")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Plot 3: velocity ---
    ax = axes[2]
    for (name, fn), c in zip(methods, colors, strict=False):
        _, vel, _ = fn(sigma, tau, noise, v1, v2)
        ax.plot(sigma, vel, label=name, color=c, linewidth=2)
    ax.axvline(tau, color="gray", linestyle="--", alpha=0.5, label=f"τ={tau}")
    ax.set_xlabel("σ")
    ax.set_ylabel("dx/dσ")
    ax.set_title("Velocity target dx/dσ")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Target-blend paths comparison (τ={tau}, noise={noise}, v1={v1}, v2={v2})", fontsize=13)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
