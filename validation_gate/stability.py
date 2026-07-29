"""
Seed stability — is the result a property of the system, or of a random draw?

A real case, from the research this gate was built to audit. One line:

    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())      # no randomSeed

The 3-D conformer was redrawn on every run. Running the same script five times gave
rho = 0.855 / 0.127 / -0.100 / 0.818 / 0.297 — mean 0.40 +/- 0.38. The number that got
written down was the best draw. Nobody lied; nobody re-ran it either.

The same shape of bug lives in any pipeline with an unseeded stochastic step:
initialization, augmentation, negative sampling, train/test shuffling, dropout at
inference. This module makes it hard to get away with:

1. `descriptor_ensemble()` — never compute a descriptor from a single draw. Always an
   average over a deterministic ensemble, with the spread reported alongside.
2. `assert_seed_stable()` — a descriptor whose rho swings with the seed is REJECTED
   before it becomes a number on a slide.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
from scipy import stats

# Fixed seeds — the same ones everywhere, always.
DEFAULT_SEEDS: tuple[int, ...] = (1337, 42, 7, 2024, 999)


class UnstableDescriptor(RuntimeError):
    """The descriptor depends on which draw you got. That is not a result."""


@dataclass(frozen=True)
class EnsembleValue:
    """A descriptor as it actually is: a mean with an uncertainty."""
    mean: float
    std: float
    n_draws: int
    values: tuple[float, ...]

    @property
    def cv(self) -> float:
        """Coefficient of variation. Above 0.30, the draw dominates the number."""
        return abs(self.std / self.mean) if self.mean else float("inf")

    def __float__(self) -> float:
        return self.mean


def descriptor_ensemble(
    compute: Callable[[str, int], Optional[float]],
    item: str,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    min_success: int = 3,
) -> Optional[EnsembleValue]:
    """Compute a descriptor over a deterministic ensemble of draws.

    `compute(item, seed) -> float | None` is your engine. Returns None if fewer than
    `min_success` draws converge — because an "average" over one draw is precisely the
    error being eliminated here.
    """
    vals = []
    for s in seeds:
        try:
            v = compute(item, s)
        except Exception:
            v = None
        if v is not None and np.isfinite(v):
            vals.append(float(v))
    if len(vals) < min_success:
        return None
    a = np.asarray(vals)
    return EnsembleValue(float(a.mean()), float(a.std()), len(vals), tuple(vals))


def assert_seed_stable(
    per_seed_rho: dict[int, float],
    max_spread: float = 0.15,
    max_relative_spread: float = 0.35,
    min_mean: float = 0.0,
) -> None:
    """Reject a descriptor whose correlation swings with the random seed.

    `per_seed_rho`: {seed: rho at that seed}, each computed over the whole dataset with
    that draw.

    Two criteria, both must hold:

    - **Absolute** (`max_spread`): the standard deviation of rho across seeds.
    - **Relative** (`max_relative_spread`): spread / |mean|. This is the one that
      catches the real pathology — the case above had spread 0.216 over mean 0.280,
      i.e. **77% of the "signal" was the draw**. A generous absolute threshold alone
      would have let it through.

    A real method gives roughly the same rho under any reasonable draw. If it does not,
    what is being measured is the draw.
    """
    rhos = np.array(list(per_seed_rho.values()), float)
    if len(rhos) < 3:
        raise UnstableDescriptor(
            f"only {len(rhos)} seeds tested. Minimum 3 — preferably 5."
        )
    spread = float(rhos.std())
    mean = float(rhos.mean())
    rel = abs(spread / mean) if mean else float("inf")
    detail = "  ".join(f"seed={s}: rho={r:+.3f}" for s, r in sorted(per_seed_rho.items()))

    if spread > max_spread or rel > max_relative_spread:
        raise UnstableDescriptor(
            f"the descriptor is NOT stable across seeds.\n"
            f"  mean rho = {mean:+.3f}   spread across seeds = {spread:.3f}"
            f"   ({rel:.0%} of the signal)\n"
            f"  limits: spread <= {max_spread}  and  relative spread <= "
            f"{max_relative_spread:.0%}\n"
            f"  {detail}\n"
            f"What is being measured is the random draw, not the system."
        )
    if mean < min_mean:
        raise UnstableDescriptor(
            f"mean rho across seeds = {mean:+.3f} < {min_mean}. No signal."
        )


def seed_stability_report(
    descriptor_by_seed: dict[int, Sequence[float]],
    observable: Sequence[float],
    metric: str = "spearman",
) -> dict:
    """Run the descriptor under each seed and report the spread of rho.

    Call this BEFORE reporting any correlation. It is cheap, and it would have caught
    every false claim that motivated this package.
    """
    f = {"spearman": lambda x, y: stats.spearmanr(x, y).statistic,
         "pearson":  lambda x, y: stats.pearsonr(x, y).statistic}[metric]
    per_seed = {s: float(f(v, observable)) for s, v in descriptor_by_seed.items()}
    rhos = np.array(list(per_seed.values()))
    try:
        assert_seed_stable(per_seed)
        stable, warning = True, None
    except UnstableDescriptor as exc:
        stable, warning = False, str(exc)
    return {
        "per_seed_rho": per_seed,
        "mean": float(rhos.mean()),
        "std": float(rhos.std()),
        "min": float(rhos.min()),
        "max": float(rhos.max()),
        "stable": stable,
        "warning": warning,
    }
