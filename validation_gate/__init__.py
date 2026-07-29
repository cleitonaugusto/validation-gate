"""
validation-gate — does your model beat a trivial baseline on a blind holdout?

A validation gate for ML-in-science. It refuses results obtained through the shortcuts
that produce claims which do not reproduce: unsealed hypotheses, strawman baselines,
post-hoc exclusions, best-of-N reporting, and holdouts that get "one more look".

I wrote it to audit my own computational-chemistry research. Applied to five months of
my own results, it passed **zero of five** benchmarks. Those are the claims I was about
to publish:

  flagship metal sensing   a 1963 lookup table beat the quantum pipeline (0.905 vs 0.685)
  selectivity study        pre-registered, but comparable data was never collected
  protein target (n=148)   the signal was molecular weight; the method added noise
  6-compound series        too small for any interpretable result
  multi-target sweep       free docking software won across the board

A validation tool that has never told its author "no" is not validation — it is
rationalization.

Minimal use:

    from validation_gate import PreRegistration, run_blind_holdout

    pre = PreRegistration(...)        # before seeing any result
    pre.seal("PREREG.json")           # commit this
    ...                               # compute your descriptor
    result = run_blind_holdout(PreRegistration.load("PREREG.json"), records)
    print(result["verdict"])
"""
from .baselines import BASELINES, Baseline, applicable_baselines, register_baseline
from .protocol import (
    MIN_HOLDOUT_N,
    DegenerateSeries,
    PreRegistration,
    ProtocolViolation,
    bootstrap_ci,
    null_reference,
    paired_gain_ci,
    run_blind_holdout,
)
from .stability import (
    DEFAULT_SEEDS,
    EnsembleValue,
    UnstableDescriptor,
    assert_seed_stable,
    descriptor_ensemble,
    seed_stability_report,
)

__version__ = "0.1.0"

__all__ = [
    "BASELINES", "Baseline", "applicable_baselines", "register_baseline",
    "PreRegistration", "ProtocolViolation", "DegenerateSeries", "run_blind_holdout",
    "bootstrap_ci", "paired_gain_ci", "null_reference", "MIN_HOLDOUT_N",
    "DEFAULT_SEEDS", "EnsembleValue", "UnstableDescriptor",
    "assert_seed_stable", "descriptor_ensemble", "seed_stability_report",
    "__version__",
]
