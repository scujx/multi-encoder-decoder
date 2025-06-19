#!/usr/bin/env python
"""synthetic_mixed_frequency_data_generator.py

Generate complex synthetic mixed-frequency data for experimentation with
multi-encoder–decoder (MED) models or other time-series forecasting
approaches.

High-frequency (monthly) series **x** (6 dimensions) and low-frequency
(quarterly) series **y** (4 dimensions) are created on a shared timeline
with configurable regime shifts, seasonality, trends, GARCH noise and
missing values.

The script can be used as a library (import the functions) or executed as
a stand-alone program to preview the generated series and export them to
an Excel workbook with separate sheets for *x* and *y*.

Author: Jiaxi Liu (liujiaxi@stu.scu.edu.cn)
License: MIT
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple

# =============================================================
# I/O UTILITIES
# =============================================================

def save_to_excel(
    x: np.ndarray,
    y_q: np.ndarray,
    filename: str = "synthetic_data.xlsx",
) -> None:
    """Save monthly *x* and quarterly *y* data to an Excel workbook.

    Parameters
    ----------
    x : ndarray, shape (T_months, n_x)
        High-frequency data (monthly).
    y_q : ndarray, shape (T_quarters, n_y)
        Low-frequency data (quarterly).
    filename : str, default "synthetic_data.xlsx"
        Destination file name (``.xlsx``).
    """
    x_time = pd.date_range("1970-01-01", periods=x.shape[0], freq="MS")
    y_time = pd.date_range("1970-01-01", periods=y_q.shape[0], freq="QS")

    x_df = pd.DataFrame(x, columns=[f"x_{i}" for i in range(x.shape[1])])
    y_df = pd.DataFrame(y_q, columns=[f"y_{i}" for i in range(y_q.shape[1])])

    x_df.insert(0, "Date", x_time)
    y_df.insert(0, "Date", y_time)

    with pd.ExcelWriter(filename) as writer:
        x_df.to_excel(writer, sheet_name="x", index=False)
        y_df.to_excel(writer, sheet_name="y", index=False)

    print(f"Data exported to {filename}")


# =============================================================
# NOISE GENERATION
# =============================================================

def garch_noise(
    length: int,
    omega: float = 0.1,
    alpha: float = 0.2,
    beta: float = 0.6,
    base_sigma: float = 0.5,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a GARCH(1,1) noise series.

    sigma_t^2 = omega + alpha * eps_{t-1}^2 + beta * sigma_{t-1}^2
    eps_t     = sigma_t * z_t, with z_t ~ N(0, 1)
    """
    if seed is not None:
        np.random.seed(seed)

    eps = np.zeros(length)
    sigma = np.full(length, base_sigma)
    z = np.random.randn(length)

    for t in range(length):
        if t:  # t > 0
            sigma[t] = np.sqrt(
                omega + alpha * eps[t - 1] ** 2 + beta * sigma[t - 1] ** 2
            )
        eps[t] = sigma[t] * z[t]
    return eps


# =============================================================
# SYNTHETIC DATA GENERATOR
# =============================================================

def generate_complex_mixed_data(
    T_months: int = 600,
    regime_shift_point: int = 60,
    seed: int | None = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate high-/low-frequency synthetic data with rich dynamics.

    Parameters
    ----------
    T_months : int, default 600
        Total number of monthly observations.
    regime_shift_point : int, default 60
        Month index at which all series experience a structural break.
    seed : int | None, default 42
        Random seed for reproducibility.

    Returns
    -------
    x_missing : ndarray, shape (T_months, 6)
        High-frequency series with *NaN* injected (5 % missing).
    y_quarterly : ndarray, shape (T_months // 3, 4)
        Low-frequency (quarterly) series.
    x_full : ndarray, shape (T_months, 6)
        High-frequency series **without** missing values.
    mask_missing : ndarray[bool], shape (T_months, 6)
        Boolean mask indicating missing positions in *x_missing*.
    """
    np.random.seed(seed)
    T_quarters = T_months // 3

    # -------------------- Low-frequency Y -------------------- #
    y = np.zeros((T_quarters, 4))

    freq = [12.0, 8.0, 10.0, 6.0]
    phase = [0.0, 0.5, -1.0, 2.0]
    trend = [0.01, -0.02, 0.03, 0.0]
    amp = [1.0, 2.0, 1.5, 0.8]
    shift_y = np.array([0.5, -1.0, 1.0, 0.0])

    y_noise = np.random.randn(T_quarters, 4) * 0.3

    for q in range(T_quarters):
        regime_offset = shift_y if (3 * q) >= regime_shift_point else 0.0
        for i in range(4):
            sinusoid = amp[i] * np.sin(2 * np.pi * q / freq[i] + phase[i])
            y[q, i] = (
                sinusoid
                + trend[i] * q
                + (regime_offset[i] if isinstance(regime_offset, np.ndarray) else 0.0)
                + y_noise[q, i]
            )

    # -------------------- High-frequency X (baseline) -------- #
    x = np.zeros((T_months, 6))

    # x_1 … x_3 : trend + seasonality + shift
    a = [0.01, -0.005, 0.02]
    b = [1.0, 0.8, 1.2]
    freq_x = [12.0, 10.0, 8.0]
    shift_x = np.array([0.5, -0.5, 1.0])

    for t in range(T_months):
        in_regime_b = t >= regime_shift_point
        for k in range(3):
            val = a[k] * t + b[k] * np.sin(2 * np.pi * t / freq_x[k])
            if in_regime_b:
                val += shift_x[k]
            val += np.random.randn() * 0.3  # white noise
            x[t, k + 1] = val

    # x_4 : non-linear exponential trend
    x[:, 4] = np.exp(0.001 * np.arange(T_months) + 0.1 * np.sin(2 * np.pi * np.arange(T_months) / 6)) + np.random.randn(T_months) * 0.2

    # x_5 : random walk with regime drift
    x_5 = np.cumsum(np.random.randn(T_months) * 0.1)
    x_5[regime_shift_point:] += 0.01
    x[:, 5] = x_5

    # x_0 : depends on lagged y and x + GARCH noise
    eps_garch = garch_noise(T_months, omega=0.05, alpha=0.2, beta=0.5, base_sigma=0.3, seed=seed)

    coeffs = {
        "y0_lag6": 0.2,
        "y0_lag12": 0.1,
        "y1_lag6": 0.15,
        "y1_lag12": 0.08,
        "x1_lag1": 0.5,
        "x1_lag12": 0.3,
        "x4_lag3": 0.4,
        "x5_lag5": 0.2,
    }

    for t in range(T_months):
        val = 0.0

        # low-frequency lags (map month → quarter)
        if t >= 6:
            val += coeffs["y0_lag6"] * y[(t - 6) // 3, 0] + coeffs["y1_lag6"] * y[(t - 6) // 3, 1]
        if t >= 12:
            val += coeffs["y0_lag12"] * y[(t - 12) // 3, 0] + coeffs["y1_lag12"] * y[(t - 12) // 3, 1]

        # high-frequency lags
        if t >= 1:
            val += coeffs["x1_lag1"] * x[t - 1, 1]
        if t >= 12:
            val += coeffs["x1_lag12"] * x[t - 12, 1]
        if t >= 3:
            val += coeffs["x4_lag3"] * x[t - 3, 4]
        if t >= 5:
            val += coeffs["x5_lag5"] * x[t - 5, 5]

        # non-linear interaction term
        if t >= 12:
            val += 10 * y[(t - 6) // 3, 0] * x[t - 12, 1]

        # regime shift intercept
        if t >= regime_shift_point:
            val += 1.0

        # add GARCH noise
        x[t, 0] = val + eps_garch[t]

    # holiday shocks (illustrative)
    for start, end in [(120, 125), (360, 365)]:
        x[start : end + 1, 0] += start * 0.3

    # -------------------- Inject missing values -------------- #
    miss_prob = 0.05
    mask_missing = np.random.rand(T_months, 6) < miss_prob
    x_missing = x.copy()
    x_missing[mask_missing] = np.nan

    return x_missing, y, x, mask_missing


# =============================================================
# UTILITY: QUARTERLY → MONTHLY INTERPOLATION
# =============================================================

def interpolate_y_to_monthly(y: np.ndarray, T_months: int) -> np.ndarray:
    """Linearly upsample quarterly *y* to monthly frequency."""
    T_quarters, n_y = y.shape
    monthly_index = np.linspace(0, T_quarters - 1, T_months)
    y_monthly = np.vstack(
        [np.interp(monthly_index, np.arange(T_quarters), y[:, i]) for i in range(n_y)]
    ).T
    return y_monthly


# =============================================================
# DEMO USAGE
# =============================================================

if __name__ == "__main__":
    XM, YQ, XF, MISSING = generate_complex_mixed_data(T_months=600, regime_shift_point=60)

    print("x_missing shape:", XM.shape)
    print("y_quarterly shape:", YQ.shape)
    print("Missing values in x:", np.isnan(XM).sum())

    # -- quick visualization -------------------------------------------------
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # (1) Target series x_0
    axs[0].plot(XF[:, 0], label="x_0 (clean)")
    axs[0].plot(np.where(MISSING[:, 0], np.nan, XM[:, 0]), "x", label="x_0 missing", ms=4)
    axs[0].set_title("High-frequency target x_0 with missing periods")
    axs[0].legend()

    # (2) Other high-frequency predictors
    for i, color in zip(range(1, 6), ["g", "orange", "purple", "blue", "red"]):
        axs[1].plot(XF[:, i], label=f"x_{i}", color=color)
    axs[1].set_title("High-frequency predictors x_1 … x_5")
    axs[1].legend(ncol=3)

    # (3) Low-frequency series (upsampled for display)
    y_monthly = np.repeat(YQ, 3, axis=0)[: XF.shape[0]]
    for i, color in zip(range(4), ["brown", "gray", "cyan", "magenta"]):
        axs[2].plot(y_monthly[:, i], label=f"y_{i}", color=color)
    axs[2].set_title("Low-frequency predictors y_0 … y_3 (quarterly, repeated)")
    axs[2].legend(ncol=4)

    plt.tight_layout()
    plt.show()

    # -- simple missing-value imputation -------------------------------------
    XM_filled = pd.DataFrame(XM).ffill().bfill().values

    save_to_excel(XM_filled, YQ, filename="synthetic_mixed_data.xlsx")