"""COS (Chain-of-Step) interpolation paths for piecewise flow matching.

Controls how the flow-matching path traverses noise -> x_tau -> x_final
in sigma-space. The default ``linear`` path produces a piecewise-linear
trajectory with a velocity discontinuity at sigma = tau.  Alternative
paths smooth this transition while still passing through all three points.

There are two families of paths:

**Passthrough** – the path literally passes through x_tau at sigma = tau.

**Target-blend** – the path does NOT pass through x_tau.  Instead the
velocity field initially points toward x_tau and gradually (or abruptly)
shifts to point toward x_final.

Path types
----------
Passthrough paths (noise -> x_tau -> x_final):
  linear           Piecewise linear (C0 at boundary). Original behaviour.
  cosine           Cosine reparameterisation per segment (C1, velocity -> 0 at tau).
  cubic_hermite    Catmull-Rom cubic Hermite spline (C1, smooth velocity at tau).
  smooth_blend     Linear with smoothstep blending in a window around tau (C1 locally).
  quadratic_bezier Quadratic polynomial through the three points (C1 globally).

Target-blend paths (no passthrough):
  target_linear    Hard target switch at tau (C0). Standard FM per segment.
  target_cosine    Smooth cosine target blend around tau (C1).
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor

PathType = Literal[
    "linear", "cosine", "cubic_hermite", "smooth_blend", "quadratic_bezier",
    "target_linear", "target_cosine",
]


def compute_cos_path(
    path_type: PathType,
    sigma: Tensor,
    tau: float,
    noise: Tensor,
    x_tau: Tensor,
    x_final: Tensor,
    *,
    boundary_noise_std: float = 0.0,
    smooth_blend_delta: float = 0.05,
) -> tuple[Tensor, Tensor]:
    """Compute interpolated sample *x_t* and velocity target *dx/dsigma*.

    Args:
        path_type: Interpolation strategy.
        sigma: Per-sample sigma, broadcastable ``(B, 1, 1, 1, 1)``.
        tau: Boundary sigma separating the two segments.
        noise: Pure Gaussian noise, same shape as *x_final*.
        x_tau: Intermediate (search) state.
        x_final: Final target state.
        boundary_noise_std: Gaussian std added to *x_tau* for low-stage
            samples (regularisation).  Applied only to samples with
            ``sigma < tau`` so that the high stage sees clean *x_tau*.
        smooth_blend_delta: Half-width of the blending window for
            ``smooth_blend`` (ignored by other paths).

    Returns:
        ``(x_t, target)`` with the same shape as *x_final*.
    """
    # Boundary noise: perturb x_tau only for low-stage samples.
    if boundary_noise_std > 0:
        high = sigma >= tau
        x_tau_noisy = x_tau + torch.randn_like(x_tau) * boundary_noise_std
        x_tau = torch.where(high, x_tau, x_tau_noisy)

    if path_type == "linear":
        return _linear(sigma, tau, noise, x_tau, x_final)
    if path_type == "cosine":
        return _cosine(sigma, tau, noise, x_tau, x_final)
    if path_type == "cubic_hermite":
        return _cubic_hermite(sigma, tau, noise, x_tau, x_final)
    if path_type == "smooth_blend":
        return _smooth_blend(sigma, tau, noise, x_tau, x_final, smooth_blend_delta)
    if path_type == "quadratic_bezier":
        return _quadratic_bezier(sigma, tau, noise, x_tau, x_final)
    if path_type == "target_linear":
        return _target_linear(sigma, tau, noise, x_tau, x_final)
    if path_type == "target_cosine":
        return _target_cosine(sigma, tau, noise, x_tau, x_final)
    raise ValueError(f"Unknown COS path type: {path_type!r}")


# ======================================================================
# Path implementations
# ======================================================================


def _linear(
    sigma: Tensor, tau: float, noise: Tensor, x_tau: Tensor, x_final: Tensor,
) -> tuple[Tensor, Tensor]:
    """Piecewise linear (original).  C0 at boundary."""
    high = sigma >= tau

    s_h = (sigma - tau) / (1.0 - tau)
    x_t_h = s_h * noise + (1.0 - s_h) * x_tau
    tgt_h = (noise - x_tau) / (1.0 - tau)

    s_l = sigma / tau
    x_t_l = s_l * x_tau + (1.0 - s_l) * x_final
    tgt_l = (x_tau - x_final) / tau

    return torch.where(high, x_t_h, x_t_l), torch.where(high, tgt_h, tgt_l)


def _cosine(
    sigma: Tensor, tau: float, noise: Tensor, x_tau: Tensor, x_final: Tensor,
) -> tuple[Tensor, Tensor]:
    """Cosine reparameterisation per segment.  C1 at boundary (velocity -> 0)."""
    high = sigma >= tau
    pi = math.pi

    # High segment: x_tau -> noise as sigma goes tau -> 1
    t_h = (sigma - tau) / (1.0 - tau)
    s_h = 0.5 * (1.0 - torch.cos(pi * t_h))
    ds_dsigma_h = 0.5 * pi * torch.sin(pi * t_h) / (1.0 - tau)
    x_t_h = s_h * noise + (1.0 - s_h) * x_tau
    tgt_h = ds_dsigma_h * (noise - x_tau)

    # Low segment: x_final -> x_tau as sigma goes 0 -> tau
    t_l = sigma / tau
    s_l = 0.5 * (1.0 - torch.cos(pi * t_l))
    ds_dsigma_l = 0.5 * pi * torch.sin(pi * t_l) / tau
    x_t_l = s_l * x_tau + (1.0 - s_l) * x_final
    tgt_l = ds_dsigma_l * (x_tau - x_final)

    return torch.where(high, x_t_h, x_t_l), torch.where(high, tgt_h, tgt_l)


def _cubic_hermite(
    sigma: Tensor, tau: float, noise: Tensor, x_tau: Tensor, x_final: Tensor,
) -> tuple[Tensor, Tensor]:
    """Catmull-Rom cubic Hermite spline.  C1 at boundary.

    Tangent at sigma=tau is estimated via Catmull-Rom from the two
    neighbours (noise at sigma=1, x_final at sigma=0):
        v_tau = noise - x_final

    End tangents use the one-sided chord:
        v_1 = (noise - x_tau) / (1 - tau)
        v_0 = (x_tau - x_final) / tau
    """
    high = sigma >= tau
    v_tau = noise - x_final  # Catmull-Rom tangent (dx/dsigma) at tau

    # --- High segment: x_tau -> noise, t in [0, 1] ---
    t = (sigma - tau) / (1.0 - tau)
    t2 = t * t
    t3 = t2 * t
    # Hermite basis
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    # Tangents in t-space (scaled by segment length 1-tau)
    m0_h = (1.0 - tau) * v_tau   # Catmull-Rom at tau
    m1_h = noise - x_tau         # chord at sigma=1
    x_t_h = h00 * x_tau + h10 * m0_h + h01 * noise + h11 * m1_h
    # Derivative basis (d/dt)
    dh00 = 6.0 * t2 - 6.0 * t
    dh10 = 3.0 * t2 - 4.0 * t + 1.0
    dh01 = -6.0 * t2 + 6.0 * t
    dh11 = 3.0 * t2 - 2.0 * t
    dx_dt_h = dh00 * x_tau + dh10 * m0_h + dh01 * noise + dh11 * m1_h
    tgt_h = dx_dt_h / (1.0 - tau)

    # --- Low segment: x_final -> x_tau, t in [0, 1] ---
    t = sigma / tau
    t2 = t * t
    t3 = t2 * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    m0_l = x_tau - x_final   # chord at sigma=0
    m1_l = tau * v_tau        # Catmull-Rom at tau
    x_t_l = h00 * x_final + h10 * m0_l + h01 * x_tau + h11 * m1_l
    dh00 = 6.0 * t2 - 6.0 * t
    dh10 = 3.0 * t2 - 4.0 * t + 1.0
    dh01 = -6.0 * t2 + 6.0 * t
    dh11 = 3.0 * t2 - 2.0 * t
    dx_dt_l = dh00 * x_final + dh10 * m0_l + dh01 * x_tau + dh11 * m1_l
    tgt_l = dx_dt_l / tau

    return torch.where(high, x_t_h, x_t_l), torch.where(high, tgt_h, tgt_l)


def _smooth_blend(
    sigma: Tensor, tau: float, noise: Tensor, x_tau: Tensor, x_final: Tensor,
    delta: float,
) -> tuple[Tensor, Tensor]:
    """Linear with smoothstep blending in ``[tau - delta, tau + delta]``.

    Outside the blending window the path is identical to ``linear``.
    Inside, the two linear branches are mixed with a C1 smoothstep and
    the velocity target includes the blending derivative term.
    """
    # Compute both linear branches for all samples
    s_h = (sigma - tau) / (1.0 - tau)
    x_t_h = s_h * noise + (1.0 - s_h) * x_tau
    tgt_h = (noise - x_tau) / (1.0 - tau)

    s_l = sigma / tau
    x_t_l = s_l * x_tau + (1.0 - s_l) * x_final
    tgt_l = (x_tau - x_final) / tau

    # Smoothstep blend: alpha = 0 (low) at tau-delta, 1 (high) at tau+delta
    lo = tau - delta
    hi = tau + delta
    u = ((sigma - lo) / (hi - lo)).clamp(0.0, 1.0)
    alpha = 3.0 * u * u - 2.0 * u * u * u          # smoothstep(u)
    dalpha_dsigma = 6.0 * u * (1.0 - u) / (hi - lo)

    x_t = (1.0 - alpha) * x_t_l + alpha * x_t_h
    target = (
        (1.0 - alpha) * tgt_l
        + alpha * tgt_h
        + dalpha_dsigma * (x_t_h - x_t_l)
    )
    return x_t, target


def _quadratic_bezier(
    sigma: Tensor, tau: float, noise: Tensor, x_tau: Tensor, x_final: Tensor,
) -> tuple[Tensor, Tensor]:
    """Quadratic polynomial through ``(0, x_final)``, ``(tau, x_tau)``, ``(1, noise)``.

    Fits ``x(sigma) = a * sigma^2 + b * sigma + x_final`` and returns
    the analytic derivative ``dx/dsigma = 2a * sigma + b``.  The path is
    C-infinity (a single polynomial) and passes exactly through all three
    anchor points.
    """
    diff_tau = x_tau - x_final
    diff_1 = noise - x_final
    denom = tau * (tau - 1.0)  # negative for tau in (0, 1)
    a = (diff_tau - tau * diff_1) / denom
    b = diff_1 - a

    sigma2 = sigma * sigma
    x_t = a * sigma2 + b * sigma + x_final
    target = 2.0 * a * sigma + b
    return x_t, target


# ======================================================================
# Target-blend paths (no passthrough)
# ======================================================================


def _target_linear(
    sigma: Tensor, tau: float, noise: Tensor, x_tau: Tensor, x_final: Tensor,
) -> tuple[Tensor, Tensor]:
    """No-passthrough, hard target switch at tau.  C0 in velocity, continuous in x_t.

    High segment (sigma >= tau):
        Standard FM with clean target = x_tau.
        x_t = sigma * noise + (1 - sigma) * x_tau
        velocity = noise - x_tau

    Low segment (sigma < tau):
        Continues from x_theta = tau * noise + (1 - tau) * x_tau (the position
        at the switch point) and linearly interpolates toward x_final.
        x_t = (sigma / tau) * x_theta + (1 - sigma / tau) * x_final
        velocity = (x_theta - x_final) / tau
    """
    high = sigma >= tau

    # High: standard FM targeting x_tau
    x_t_h = sigma * noise + (1.0 - sigma) * x_tau
    tgt_h = noise - x_tau

    # Low: from x_theta toward x_final
    x_theta = tau * noise + (1.0 - tau) * x_tau
    s_l = sigma / tau
    x_t_l = s_l * x_theta + (1.0 - s_l) * x_final
    tgt_l = (x_theta - x_final) / tau

    return torch.where(high, x_t_h, x_t_l), torch.where(high, tgt_h, tgt_l)


def _target_cosine(
    sigma: Tensor, tau: float, noise: Tensor, x_tau: Tensor, x_final: Tensor,
) -> tuple[Tensor, Tensor]:
    """No-passthrough, smooth cosine target blend around tau.  C1.

    A blending weight ``alpha(sigma)`` goes smoothly from 0 at sigma=0
    to 1 at sigma=1, passing through 0.5 at sigma=tau, with C1
    continuity (derivative = 0 at tau from both sides).

    Effective clean-sample target:
        x_eff = alpha * x_tau + (1 - alpha) * x_final

    The noisy sample and velocity target are:
        x_t    = sigma * noise + (1 - sigma) * x_eff
        target = noise - x_eff + (1 - sigma) * dalpha/dsigma * (x_tau - x_final)
    """
    high = sigma >= tau
    pi = math.pi

    # Low segment [0, tau]: alpha from 0 to 0.5
    t_l = sigma / tau
    alpha_l = 0.25 * (1.0 - torch.cos(pi * t_l))
    dalpha_l = 0.25 * pi / tau * torch.sin(pi * t_l)

    # High segment [tau, 1]: alpha from 0.5 to 1
    t_h = (sigma - tau) / (1.0 - tau)
    alpha_h = 0.5 + 0.25 * (1.0 - torch.cos(pi * t_h))
    dalpha_h = 0.25 * pi / (1.0 - tau) * torch.sin(pi * t_h)

    alpha = torch.where(high, alpha_h, alpha_l)
    dalpha = torch.where(high, dalpha_h, dalpha_l)

    x_eff = alpha * x_tau + (1.0 - alpha) * x_final
    x_t = sigma * noise + (1.0 - sigma) * x_eff
    target = noise - x_eff + (1.0 - sigma) * dalpha * (x_tau - x_final)

    return x_t, target
