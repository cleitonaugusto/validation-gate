"""
Generate two demonstration scenarios for the validation gate:

  PASS  — the descriptor carries signal independent of the trivial baseline.
  FAIL  — the descriptor only recodes the baseline (donor softness), which is the
          failure mode that quietly invalidates most "promising" correlations.

    python examples/make_demo.py
    validation-gate --prereg examples/prereg_pass.json --data examples/records_pass.csv
    validation-gate --prereg examples/prereg_fail.json --data examples/records_fail.csv
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from validation_gate import PreRegistration

HERE = Path(__file__).resolve().parent
DONORS = ["S", "N", "O"]
SOFT = {"S": 3.0, "N": 2.0, "O": 1.0}


def _write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dq", "y", "donor_element"])
        w.writeheader()
        w.writerows(rows)


def _seal(path: Path, n: int) -> None:
    """Seal a pre-registration and clear its ledger.

    Each scenario gets its OWN pre-registration: one pre-registration means one holdout,
    touched exactly once. Regenerating the demo starts a new study, so the old ledger
    goes with it — this is the one place where deleting it is legitimate.
    """
    PreRegistration(
        hypothesis="descriptor dq predicts observable y beyond donor softness (HSAB)",
        descriptor="dq", observable="y", metric="spearman",
        baselines=["hsab_donor"],
        decision_rule="gain over the strongest baseline with 95% CI excluding zero",
        n_expected=n,
    ).seal(path)
    ledger = Path(str(path) + ".runs.json")
    if ledger.exists():
        ledger.unlink()


def main() -> None:
    rng = np.random.default_rng(7)
    # 120 points because the verdict comes from the HOLDOUT (30%), and a holdout with
    # fewer than 30 points is inconclusive by construction — demo included.
    n = 120

    _seal(HERE / "prereg_pass.json", n)
    _seal(HERE / "prereg_fail.json", n)

    donors = rng.choice(DONORS, size=n)
    soft = np.array([SOFT[d] for d in donors])

    # PASS: y follows a latent signal; dq tracks that signal; the donor is noise.
    latent = rng.normal(size=n)
    y_pass = latent + 0.3 * rng.normal(size=n)
    dq_pass = latent + 0.3 * rng.normal(size=n)
    _write_csv(HERE / "records_pass.csv", [
        {"dq": f"{dq_pass[i]:.4f}", "y": f"{y_pass[i]:.4f}", "donor_element": donors[i]}
        for i in range(n)
    ])

    # FAIL: y follows donor softness; dq merely recodes softness (+ noise). The
    # expensive descriptor adds nothing over a lookup table from 1963.
    y_fail = soft + 0.5 * rng.normal(size=n)
    dq_fail = soft + 0.5 * rng.normal(size=n)
    _write_csv(HERE / "records_fail.csv", [
        {"dq": f"{dq_fail[i]:.4f}", "y": f"{y_fail[i]:.4f}", "donor_element": donors[i]}
        for i in range(n)
    ])

    print("written:")
    print("  prereg_pass.json / prereg_fail.json  (sealed, one holdout each)")
    print("  records_pass.csv  (descriptor with real signal -> should PASS)")
    print("  records_fail.csv  (descriptor = recoded baseline -> should FAIL)")


if __name__ == "__main__":
    main()
