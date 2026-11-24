"""Pure forecast-formatting helpers.

These functions enforce the numeric invariants Metaculus expects:
- binary probabilities stay strictly inside (0, 1) -- we clamp to [0.01, 0.99];
- multiple-choice probabilities are floored and renormalized to sum to 1;
- numeric percentile values are non-decreasing in the percentile;
- a full CDF built from declared percentiles is non-decreasing, stays within
  [0, 1], and spans the question's [lower_bound, upper_bound] range.
"""

from __future__ import annotations

import math
from bisect import bisect_right

MIN_BINARY_PROBABILITY = 0.01
MAX_BINARY_PROBABILITY = 0.99
DEFAULT_OPTION_FLOOR = 0.005
CDF_NUM_POINTS = 201  # Metaculus represents numeric CDFs with 201 points.


def clamp_binary_probability(probability: float) -> float:
    """Clamp a binary probability into [0.01, 0.99].

    Raises ValueError for NaN / infinite inputs so bad forecasts fail loudly
    instead of being posted.
    """
    value = float(probability)
    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"Invalid probability: {probability!r}")
    return max(MIN_BINARY_PROBABILITY, min(MAX_BINARY_PROBABILITY, value))


def ensure_monotonic(values: list[float]) -> list[float]:
    """Return a non-decreasing copy of ``values`` (running maximum)."""
    result: list[float] = []
    running = -math.inf
    for value in values:
        running = max(running, float(value))
        result.append(running)
    return result


def normalize_option_probabilities(
    probabilities: dict[str, float], floor: float = DEFAULT_OPTION_FLOOR
) -> dict[str, float]:
    """Floor each option's probability and renormalize so they sum to 1.

    Every option keeps at least ``floor`` probability (forecasters should
    leave some mass on unexpected outcomes); the remaining mass is
    distributed in proportion to the original values.
    """
    if not probabilities:
        raise ValueError("No options to normalize.")
    n = len(probabilities)
    if floor * n >= 1.0:
        raise ValueError(f"Floor {floor} too large for {n} options.")
    cleaned = {}
    for name, value in probabilities.items():
        value = float(value)
        if math.isnan(value) or math.isinf(value) or value < 0:
            raise ValueError(f"Invalid probability for option {name!r}: {value!r}")
        cleaned[name] = value
    total = sum(cleaned.values())
    if total <= 0:
        # Degenerate input: fall back to uniform.
        return {name: 1.0 / n for name in cleaned}
    remaining = 1.0 - floor * n
    return {
        name: floor + remaining * (value / total) for name, value in cleaned.items()
    }


def _interpolate(x: float, xs: list[float], ys: list[float]) -> float:
    """Linear interpolation of ys(xs) at x, clamped to the end values."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    idx = bisect_right(xs, x) - 1
    x0, x1 = xs[idx], xs[idx + 1]
    y0, y1 = ys[idx], ys[idx + 1]
    if x1 == x0:
        return y1
    fraction = (x - x0) / (x1 - x0)
    return y0 + fraction * (y1 - y0)


def build_cdf(
    percentiles: list[tuple[float, float]],
    lower_bound: float,
    upper_bound: float,
    num_points: int = CDF_NUM_POINTS,
) -> list[float]:
    """Build a full CDF from declared percentiles.

    ``percentiles`` is a list of (percentile, value) pairs with percentile in
    [0, 1]. The returned list has ``num_points`` probabilities corresponding
    to x positions evenly spaced over [lower_bound, upper_bound]. The CDF is
    anchored to 0 at/below the lower bound and 1 at/above the upper bound,
    passes through the declared percentiles, and is non-decreasing.
    """
    if upper_bound <= lower_bound:
        raise ValueError(
            f"upper_bound ({upper_bound}) must exceed lower_bound ({lower_bound})."
        )
    if not percentiles:
        raise ValueError("At least one percentile is required to build a CDF.")
    ordered = sorted(percentiles, key=lambda pair: pair[1])
    values = ensure_monotonic([v for _, v in ordered])
    levels = [p for p, _ in ordered]
    # Anchor points: CDF(lower_bound) = 0 and CDF(upper_bound) = 1.
    anchor_values = [lower_bound] + values + [upper_bound]
    anchor_levels = [0.0] + levels + [1.0]
    anchor_values = ensure_monotonic(anchor_values)

    cdf: list[float] = []
    for i in range(num_points):
        x = lower_bound + (upper_bound - lower_bound) * i / (num_points - 1)
        cdf.append(_interpolate(x, anchor_values, anchor_levels))
    cdf = ensure_monotonic(cdf)
    return [max(0.0, min(1.0, p)) for p in cdf]
