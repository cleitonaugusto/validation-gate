"""
The protocol — pre-registration, blind holdout, honest statistics.

This module is not a statistics utility. It is a gate. It refuses results obtained
through the shortcuts that produce false claims:

  Mandatory baseline   no trivial baseline declared, no run
  No post-hoc dropping  n_final != n_initial aborts the study
  No best-of-N          the metric AND the expected sign are declared up front
  Blind holdout         every run is recorded; the second one is refused

Correct use is two phases, separated in time:

    # PHASE 1 — before looking at any result
    pre = PreRegistration(
        hypothesis="descriptor X predicts observable Y beyond donor softness",
        descriptor="dq",
        observable="log_beta",
        metric="spearman",
        baselines=["hsab_donor", "steric_bulk"],
        decision_rule="gain over the strongest baseline with 95% CI excluding zero",
        direction="positive",
        n_expected=100,
    )
    pre.seal("PREREG.json")        # commit this BEFORE running anything

    # PHASE 2 — once, at the end
    result = run_blind_holdout(PreRegistration.load("PREREG.json"), records)

The verdict has four states — PASS, FAIL, INCONCLUSIVE (holdout too small to decide
anything) and PROTOCOL_FAIL (holdout already touched). Missing data is not a failure
and certainly not a pass: it is missing data, and the gate says so instead of
inventing a conclusion the sample size cannot support.

Why this exists: with n=10, the standard deviation of Spearman's rho under the null
hypothesis is 0.33. Run ~56 tests with no real signal anywhere and the best rho you
expect to see is 0.71. That is the range most "promising correlations" live in.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from scipy import stats

from .baselines import Baseline

_METRICS = {
    "spearman": lambda x, y: stats.spearmanr(x, y).statistic,
    "pearson":  lambda x, y: stats.pearsonr(x, y).statistic,
    "kendall":  lambda x, y: stats.kendalltau(x, y).statistic,
}

_DIRECTIONS = ("positive", "negative")

# Below this, no verdict is interpretable. It applies to the HOLDOUT — where the
# decision actually happens — not to the full dataset. With n=9 the standard
# deviation of Spearman's rho under the null is 0.35: anything at all can come out.
MIN_HOLDOUT_N = 30


class ProtocolViolation(RuntimeError):
    """The protocol was violated. Do not catch this — fix the study."""


class DegenerateSeries(ValueError):
    """A series does not vary: the correlation is undefined, not merely weak."""


# ─────────────────────────────────────────────────────────────────────────────
# Pre-registration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PreRegistration:
    """What you declare BEFORE looking at any result.

    Once sealed and committed, changing any field invalidates the study.
    """
    hypothesis: str
    descriptor: str            # key of your model's prediction in each record
    observable: str            # key of the ground-truth value
    metric: str                # "spearman" | "pearson" | "kendall" — exactly one
    baselines: list[str]       # registered baseline names
    decision_rule: str
    n_expected: int
    direction: str = "positive"   # expected sign of the correlation — declared up front
    holdout_frac: float = 0.30
    split_seed: int = 20260714
    n_bootstrap: int = 10_000
    notes: str = ""
    sealed_sha256: Optional[str] = field(default=None)

    # A plain attribute, NOT a field: stays out of asdict() and therefore out of the
    # seal. Records where the pre-registration came from, so the run ledger can find
    # its counterpart on disk.
    source_path = None

    def __post_init__(self) -> None:
        if self.metric not in _METRICS:
            raise ProtocolViolation(f"unknown metric: {self.metric}")
        if self.direction not in _DIRECTIONS:
            raise ProtocolViolation(
                f"direction={self.direction!r}: declare one of {_DIRECTIONS}. A hypothesis "
                "that accepts either sign is best-of-2 in disguise — if you do not know "
                "which way it should go, you do not have a hypothesis yet."
            )
        if not self.baselines:
            raise ProtocolViolation(
                "no baseline declared. A study without a trivial baseline proves nothing: "
                "you cannot show your method adds anything if you never checked what the "
                "free rule already gives you."
            )
        # Imported late on purpose: register_baseline() rebinds the name in the module,
        # and a top-level import would freeze the tuple as of import time.
        from .baselines import BASELINES as _known
        known = {b.name for b in _known}
        for b in self.baselines:
            if b not in known:
                raise ProtocolViolation(
                    f"unknown baseline: {b} (known: {sorted(known)}). "
                    "Use register_baseline() to add one from your own domain."
                )
        if self.n_expected < 30:
            raise ProtocolViolation(
                f"n_expected={self.n_expected} < 30. Below n=30 the null-hypothesis spread "
                "of Spearman's rho exceeds 0.19 and no result is interpretable."
            )
        if not 0 < self.holdout_frac < 1:
            raise ProtocolViolation(f"holdout_frac={self.holdout_frac} outside (0, 1).")

    # NB: nothing here may MUTATE a field — load() re-runs __post_init__ and the seal
    # is recomputed from the fields; a mutation would raise a false tamper alarm.

    @property
    def n_for_conclusive(self) -> int:
        """Smallest n_expected whose holdout reaches MIN_HOLDOUT_N points."""
        return int(np.ceil(MIN_HOLDOUT_N / self.holdout_frac))

    # -- sealing ------------------------------------------------------------
    @staticmethod
    def _digest(d: dict) -> str:
        """Canonical hash of the pre-registration, over the dict AS WRITTEN.

        Hashing the file's dict (rather than the dataclass fields) keeps old seals
        valid when a new field with a default is added to the protocol: the seal
        covers what was declared, not the version of the code that reads it.
        """
        d = {k: v for k, v in d.items() if k != "sealed_sha256"}
        return hashlib.sha256(
            json.dumps(d, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()

    def seal(self, path: str | Path) -> str:
        """Write the pre-registration and return its hash. COMMIT IT BEFORE RUNNING."""
        d = asdict(self)
        d.pop("sealed_sha256", None)
        self.sealed_sha256 = self._digest(d)
        d["sealed_sha256"] = self.sealed_sha256
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
        self.source_path = p
        return self.sealed_sha256

    @classmethod
    def load(cls, path: str | Path) -> "PreRegistration":
        p = Path(path)
        d = json.loads(p.read_text(encoding="utf-8"))
        declared = d.get("sealed_sha256")
        if not declared:
            raise ProtocolViolation(
                f"{p} is NOT SEALED (no sealed_sha256).\n"
                "An unsealed pre-registration proves nothing — anyone wanting to rewrite "
                "the hypothesis after seeing the result would just delete the hash field. "
                "Call PreRegistration(...).seal(path) and commit it BEFORE running."
            )
        recomputed = cls._digest(d)
        if declared != recomputed:
            raise ProtocolViolation(
                "the pre-registration was MODIFIED after being sealed. "
                f"stored hash={declared[:12]}… recomputed={recomputed[:12]}…\n"
                "The study is void. Do not rewrite the hypothesis after seeing the result."
            )
        pre = cls(**{k: v for k, v in d.items() if k != "sealed_sha256"})
        pre.sealed_sha256 = declared
        pre.source_path = p
        return pre


# ─────────────────────────────────────────────────────────────────────────────
# Honest statistics
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_ci(
    x: Sequence[float], y: Sequence[float], metric: str,
    n_boot: int = 10_000, seed: int = 0, alpha: float = 0.05,
) -> tuple[float, float, float]:
    """(estimate, CI_low, CI_high) by percentile bootstrap."""
    f = _METRICS[metric]
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    est = float(f(x, y))
    rng = np.random.default_rng(seed)
    n = len(x)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.std(x[idx]) == 0 or np.std(y[idx]) == 0:
            boots[i] = np.nan
            continue
        boots[i] = f(x[idx], y[idx])
    boots = boots[~np.isnan(boots)]
    if boots.size < 0.5 * n_boot:
        raise DegenerateSeries(
            f"only {boots.size} of {n_boot} resamples produced a defined value — one of "
            "the series is constant (or nearly so). A correlation against a constant does "
            "not exist; that is not a weak result, it is an absence of variation."
        )
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return est, float(lo), float(hi)


def paired_gain_ci(
    y: Sequence[float], model: Sequence[float], baseline: Sequence[float],
    metric: str, n_boot: int = 10_000, seed: int = 0, alpha: float = 0.05,
    direction: str = "positive",
) -> tuple[float, float, float]:
    """Gain of the model OVER the baseline, with a paired CI (same resampling).

    This is the number that decides everything. If the interval includes zero, the
    model demonstrated nothing — however pretty its standalone rho looks.

    Deliberately asymmetric, conservative in both directions:

    - the MODEL counts **with its sign**, in the direction declared in the
      pre-registration. A descriptor that predicts the opposite of the hypothesis did
      not "get it right backwards" — it got it wrong. Taking |rho| here would accept
      best-of-two-signs, the very thing pre-registration exists to prevent.
    - the BASELINE counts in **absolute value**, i.e. at its best. It never declared
      anything; the burden of proof is on the expensive method, not the free rule.
    """
    f = _METRICS[metric]
    sign = 1.0 if direction == "positive" else -1.0
    y = np.asarray(y, float)
    m = np.asarray(model, float)
    b = np.asarray(baseline, float)
    gain = float(sign * f(m, y) - abs(f(b, y)))
    rng = np.random.default_rng(seed)
    n = len(y)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.std(y[idx]) == 0 or np.std(m[idx]) == 0 or np.std(b[idx]) == 0:
            boots[i] = np.nan
            continue
        boots[i] = sign * f(m[idx], y[idx]) - abs(f(b[idx], y[idx]))
    boots = boots[~np.isnan(boots)]
    if boots.size < 0.5 * n_boot:
        raise DegenerateSeries(
            f"only {boots.size} of {n_boot} gain resamples stayed defined — the model or "
            "the baseline is constant across the holdout."
        )
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return gain, float(lo), float(hi)


def null_reference(n: int, metric: str = "spearman", n_sim: int = 20_000,
                   seed: int = 0) -> dict:
    """What pure chance produces at this n. Always print it next to the result.

    At n=10: spread 0.33 and P(rho >= 0.70) = 1.3%. Run 56 tests with no signal at all
    and the best rho you expect is 0.709 — which is exactly where most exciting
    preliminary correlations live.
    """
    f = _METRICS[metric]
    rng = np.random.default_rng(seed)
    draws = np.array([f(rng.normal(size=n), rng.normal(size=n)) for _ in range(n_sim)])
    return {
        "n": n,
        "sd_under_null": float(draws.std()),
        "p_ge_0.70": float((draws >= 0.70).mean()),
        "p_ge_0.85": float((draws >= 0.85).mean()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# The run ledger — this is what makes "touched once" true
# ─────────────────────────────────────────────────────────────────────────────

def _records_digest(records: Sequence[dict]) -> str:
    return hashlib.sha256(
        json.dumps(records, sort_keys=True, ensure_ascii=False, default=str).encode()
    ).hexdigest()


def _ledger_path(pre: PreRegistration, explicit: str | Path | None) -> Optional[Path]:
    if explicit is not None:
        return Path(explicit)
    return Path(str(pre.source_path) + ".runs.json") if pre.source_path else None


def _previous_runs(path: Optional[Path], prereg_hash: Optional[str]) -> list[dict]:
    if path is None or not path.exists():
        return []
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [e for e in entries if e.get("prereg_sha256") == prereg_hash]


def _append_run(path: Optional[Path], entry: dict) -> None:
    if path is None:
        return
    entries = []
    if path.exists():
        try:
            entries = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            entries = []
    entries.append(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# The gate
# ─────────────────────────────────────────────────────────────────────────────

def run_blind_holdout(
    pre: PreRegistration,
    records: Sequence[dict],
    ledger: str | Path | None = None,
    allow_rerun: bool = False,
) -> dict:
    """Evaluate the model against its baselines on a blind holdout. ONCE.

    Aborts on any protocol violation. Returns an explicit verdict.

    "Touched once" is only real if something counts the touches: every run is recorded
    in a ledger next to the pre-registration (`PREREG.json.runs.json`), and a second run
    of the SAME pre-registration is refused. `allow_rerun=True` does not hand back the
    approval — it lets the run happen, but the verdict comes out PROTOCOL_FAIL, because
    a holdout you have already seen is no longer blind. A pre-registration built in
    memory (never sealed to disk) has no ledger: there the rule is on the operator, and
    the result says so via `r38_enforced`.
    """
    from .baselines import BASELINES as _B
    by_name = {b.name: b for b in _B}

    n0 = len(records)

    def _present(v) -> bool:
        """Really present: NaN/inf are absence wearing a number's clothes."""
        if v is None or v == "":
            return False
        try:
            return bool(np.isfinite(float(v)))
        except (TypeError, ValueError):
            return False

    # No silent exclusions
    usable = [
        r for r in records
        if _present(r.get(pre.descriptor)) and _present(r.get(pre.observable))
    ]
    if len(usable) != n0:
        raise ProtocolViolation(
            f"{n0 - len(usable)} of {n0} records are missing the descriptor or the "
            "observable (absent, empty, NaN or infinite).\n"
            "Dropping points after computing them is how a rho of 0.685 over n=10 becomes "
            "a reported 0.952 over n=8. Complete the data, or declare the exclusion rule "
            "in the pre-registration BEFORE running."
        )

    # The holdout is blind exactly once; a second look is a different thing
    led = _ledger_path(pre, ledger)
    prior = _previous_runs(led, pre.sealed_sha256)
    if prior and not allow_rerun:
        raise ProtocolViolation(
            f"this pre-registration has ALREADY been evaluated {len(prior)}x "
            f"(first: {prior[0].get('timestamp')}, verdict: {prior[0].get('verdict')}).\n"
            f"Ledger: {led}\n"
            "The holdout stopped being blind the moment you saw the first result. Running "
            "it again after adjusting the method is best-of-N in slow motion.\n"
            "If the data genuinely changed, write a NEW pre-registration, with a new "
            "hypothesis and new data. To run anyway (the verdict will be PROTOCOL_FAIL): "
            "allow_rerun=True."
        )
    if n0 < pre.n_expected:
        raise ProtocolViolation(
            f"n={n0} < n_expected={pre.n_expected} declared in the pre-registration."
        )

    # Blind, deterministic split
    rng = np.random.default_rng(pre.split_seed)
    idx = rng.permutation(n0)
    n_test = int(round(pre.holdout_frac * n0))
    test_idx = sorted(idx[:n_test].tolist())
    test = [records[i] for i in test_idx]

    y = [float(r[pre.observable]) for r in test]
    model = [float(r[pre.descriptor]) for r in test]

    metric = pre.metric
    try:
        m_est, m_lo, m_hi = bootstrap_ci(model, y, metric, pre.n_bootstrap, pre.split_seed)
    except DegenerateSeries as exc:
        raise ProtocolViolation(
            f"descriptor '{pre.descriptor}' or observable '{pre.observable}' does not vary "
            f"across the holdout: {exc}"
        ) from exc

    # Every declared baseline, no more and no fewer
    base_rows = []
    for name in pre.baselines:
        b: Baseline = by_name[name]
        vals = [b.fn(r) for r in test]
        if any(v is None for v in vals):
            base_rows.append({"baseline": name, "status": "insufficient data"})
            continue
        try:
            b_est, b_lo, b_hi = bootstrap_ci(vals, y, metric, pre.n_bootstrap, pre.split_seed)
            g, g_lo, g_hi = paired_gain_ci(y, model, vals, metric, pre.n_bootstrap,
                                           pre.split_seed, direction=pre.direction)
        except DegenerateSeries as exc:
            # A baseline that does not vary across the holdout is not a beaten baseline
            # — it is an untested one. Report it; never count it as a win.
            base_rows.append({"baseline": name, "status": f"constant across holdout ({exc})"})
            continue
        base_rows.append({
            "baseline": name,
            "reference": b.reference,
            "estimate": b_est, "ci": [b_lo, b_hi],
            "gain_of_model_over_baseline": g,
            "gain_ci95": [g_lo, g_hi],
            "model_beats_baseline": bool(g_lo > 0),
        })

    evaluated = [r for r in base_rows if "gain_ci95" in r]
    if not evaluated:
        raise ProtocolViolation("no baseline could be evaluated — incomplete data.")

    # The model must beat the STRONGEST baseline, not the weakest
    best = max(evaluated, key=lambda r: abs(r["estimate"]))
    beat_baseline = bool(best["gain_ci95"][0] > 0)

    # Four states, not two. The holdout decides, and a holdout that is too small decides
    # nothing — neither for nor against. Calling that PASS or FAIL would be inventing a
    # conclusion the sample size does not support.
    underpowered = n_test < MIN_HOLDOUT_N
    gain_txt = (f"gain={best['gain_of_model_over_baseline']:+.3f}, "
                f"CI95=[{best['gain_ci95'][0]:+.3f}, {best['gain_ci95'][1]:+.3f}]")

    if prior:  # only reachable with allow_rerun=True
        outcome = "PROTOCOL_FAIL"
        verdict = (
            f"PROTOCOL FAILURE — this holdout had already been evaluated {len(prior)}x. "
            "What follows is diagnostic, not a verdict: a revisited holdout is no longer "
            "blind, and no number drawn from it supports a claim."
        )
    elif underpowered:
        outcome = "INCONCLUSIVE"
        verdict = (
            f"INCONCLUSIVE — holdout of {n_test} points (< {MIN_HOLDOUT_N}). {gain_txt}. "
            f"At n={n_test} the null-hypothesis spread of {metric} is "
            f"{null_reference(n_test, metric)['sd_under_null']:.2f}: this result is "
            "consistent with chance, whether it looks good or bad. It is neither a pass "
            f"nor a failure — it is missing data. For a conclusive verdict: "
            f"n >= {pre.n_for_conclusive}."
        )
    elif beat_baseline:
        outcome = "PASS"
        verdict = (
            f"PASSED — the model beats the strongest baseline ({best['baseline']}) on a "
            f"blind holdout of {n_test} points, in the declared direction "
            f"({pre.direction}), with a 95% CI excluding zero."
        )
    else:
        outcome = "FAIL"
        verdict = (
            f"FAILED — the model does NOT beat baseline '{best['baseline']}' ({gain_txt}). "
            "No advantage demonstrated."
        )

    result = {
        "prereg_sha256": pre.sealed_sha256,
        "hypothesis": pre.hypothesis,
        "metric": metric,
        "direction": pre.direction,
        "n_total": n0,
        "n_holdout": n_test,
        "model": {"descriptor": pre.descriptor, "estimate": m_est, "ci95": [m_lo, m_hi]},
        "baselines": base_rows,
        "hardest_baseline": best["baseline"],
        "gain_over_hardest": best["gain_of_model_over_baseline"],
        "gain_ci95": best["gain_ci95"],
        "null_reference": null_reference(n_test, metric),
        "beats_hardest_baseline": beat_baseline,
        "outcome": outcome,               # PASS | FAIL | INCONCLUSIVE | PROTOCOL_FAIL
        "PASSED": outcome == "PASS",
        "holdout_previously_touched": len(prior),
        "r38_enforced": led is not None,
        "verdict": verdict,
    }

    _append_run(led, {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "prereg_sha256": pre.sealed_sha256,
        "records_sha256": _records_digest(list(records)),
        "n_total": n0,
        "n_holdout": n_test,
        "outcome": outcome,
        "verdict": verdict.split("—")[0].strip(),
    })
    return result
