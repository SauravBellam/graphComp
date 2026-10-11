"""Small, dependency-light statistics used by every Plan A summary.

All intervals are 95% unless ``alpha`` says otherwise.
  * wilson(k, n)              : proportion of tasks (task-level yes/no outcomes)
  * ratio_ci(nums, dens)      : pooled ratio sum(nums)/sum(dens) with a task-cluster bootstrap
                                (edges within a task are not independent, so resample tasks)
  * mean_ci(xs)               : mean of per-task values, bootstrap over tasks
  * mcnemar(a, b)             : exact two-sided McNemar test for paired yes/no outcomes on the same tasks
"""

from __future__ import annotations

import math
import random
from typing import Sequence

Z95 = 1.959963984540054


def wilson(k: int, n: int, z: float = Z95) -> dict:
    if n <= 0:
        return {"k": k, "n": n, "p": None, "lo": None, "hi": None}
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return {"k": k, "n": n, "p": round(p, 4), "lo": round(max(0.0, c - h), 4), "hi": round(min(1.0, c + h), 4)}


def _quantiles(xs: list[float], alpha: float) -> tuple[float, float]:
    xs = sorted(xs)
    lo = xs[int(math.floor(alpha / 2 * (len(xs) - 1)))]
    hi = xs[int(math.ceil((1 - alpha / 2) * (len(xs) - 1)))]
    return lo, hi


def ratio_ci(nums: Sequence[float], dens: Sequence[float], b: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    pairs = [(float(x), float(y)) for x, y in zip(nums, dens) if y is not None and x is not None]
    tot_n = sum(x for x, _ in pairs)
    tot_d = sum(y for _, y in pairs)
    if tot_d <= 0:
        return {"num": tot_n, "den": tot_d, "ratio": None, "lo": None, "hi": None, "n_tasks": len(pairs)}
    rng = random.Random(seed)
    boots = []
    m = len(pairs)
    for _ in range(b):
        sn = sd = 0.0
        for _ in range(m):
            x, y = pairs[rng.randrange(m)]
            sn += x
            sd += y
        if sd > 0:
            boots.append(sn / sd)
    lo, hi = _quantiles(boots, alpha) if boots else (None, None)
    return {"num": tot_n, "den": tot_d, "ratio": round(tot_n / tot_d, 4),
            "lo": None if lo is None else round(lo, 4), "hi": None if hi is None else round(hi, 4), "n_tasks": m}


def mean_ci(xs: Sequence[float | None], b: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    v = [float(x) for x in xs if x is not None]
    if not v:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    rng = random.Random(seed)
    boots = [sum(v[rng.randrange(len(v))] for _ in v) / len(v) for _ in range(b)]
    lo, hi = _quantiles(boots, alpha)
    return {"mean": round(sum(v) / len(v), 4), "lo": round(lo, 4), "hi": round(hi, 4), "n": len(v)}


def mcnemar(a: Sequence[bool], b: Sequence[bool]) -> dict:
    """Exact McNemar on paired outcomes. b01 = a wrong & b right, b10 = a right & b wrong."""
    b10 = sum(1 for x, y in zip(a, b) if x and not y)
    b01 = sum(1 for x, y in zip(a, b) if y and not x)
    n = b10 + b01
    if n == 0:
        p = 1.0
    else:
        k = min(b10, b01)
        p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"a_only": b10, "b_only": b01, "both": sum(1 for x, y in zip(a, b) if x and y),
            "neither": sum(1 for x, y in zip(a, b) if not x and not y), "p_exact": round(p, 6)}
