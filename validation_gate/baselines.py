"""
Trivial baselines — the free rule your model has to beat.

If a baseline wins, there is no method: the expensive computation is only recoding a
rule a practitioner applies from memory. That is not a hypothetical failure mode. It
is the single most common one, and it is invisible unless you check for it.

The built-in baselines are from coordination chemistry, because that is where this gate
was written. The question they encode is not chemical: *does the free rule already do
this?* Register your own before sealing a pre-registration:

    from validation_gate import register_baseline

    register_baseline("days_since_last_purchase", "churn",
                      lambda r: r.get("recency_days"), "business rule, 2019")
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

# ── HSAB — Pearson (1963) ────────────────────────────────────────────────────
# Donor softness: the softer the donor, the more strongly it binds soft metals
# (Hg2+, Au+, Ag+). A lookup table published in 1963.
_DONOR_SOFTNESS = {"S": 3.0, "SE": 3.5, "P": 3.0, "I": 3.0,
                   "N": 2.0, "C": 2.5, "CL": 1.5, "BR": 2.0,
                   "O": 1.0, "F": 0.5}

# ── Irving-Williams series (1953) ────────────────────────────────────────────
# Stability of divalent transition-metal complexes, high spin:
#   Mn2+ < Fe2+ < Co2+ < Ni2+ < Cu2+ > Zn2+
_IRVING_WILLIAMS = {
    "MN": 1.0, "FE": 2.0, "CO": 3.0, "NI": 4.0, "CU": 5.0, "ZN": 3.5,
}


@dataclass(frozen=True)
class Baseline:
    """A trivial baseline: a name, what it predicts, and how to read it off a record.

    `fn` takes one record (dict) and returns a float, or None where it does not apply.
    """
    name: str
    predicts: str
    fn: Callable[[dict], Optional[float]]
    reference: str


def _donor_softness(rec: dict) -> Optional[float]:
    d = rec.get("donor_element")
    return _DONOR_SOFTNESS.get(str(d).upper()) if d else None


def _irving_williams(rec: dict) -> Optional[float]:
    m = rec.get("metal")
    return _IRVING_WILLIAMS.get(str(m).upper()) if m else None


def _irving_williams_pair(rec: dict) -> Optional[float]:
    """For SELECTIVITY observables between two metals (metal_a vs metal_b)."""
    a = _IRVING_WILLIAMS.get(str(rec.get("metal_a", "")).upper())
    b = _IRVING_WILLIAMS.get(str(rec.get("metal_b", "")).upper())
    return None if a is None or b is None else a - b


def _ligand_pka(rec: dict) -> Optional[float]:
    v = rec.get("pka")
    return float(v) if v not in (None, "") else None


def _denticity(rec: dict) -> Optional[float]:
    v = rec.get("denticity")
    return float(v) if v not in (None, "") else None


def _steric_bulk(rec: dict) -> Optional[float]:
    """Ligand steric bulk (Tolman cone angle, or a proxy)."""
    v = rec.get("cone_angle", rec.get("steric_bulk"))
    return float(v) if v not in (None, "") else None


def _molecular_weight(rec: dict) -> Optional[float]:
    """The dumbest baseline there is, and it wins more often than anyone admits."""
    v = rec.get("mw")
    return float(v) if v not in (None, "") else None


def _docking(rec: dict) -> Optional[float]:
    """A docking score. For protein targets this is the one to beat — free software,
    twenty years old, and still the strongest baseline in most published comparisons."""
    v = rec.get("vina_score")
    return float(v) if v not in (None, "") else None


BASELINES: tuple[Baseline, ...] = (
    Baseline("hsab_donor", "coordination strength", _donor_softness,
             "Pearson, J. Am. Chem. Soc. 85, 3533 (1963)"),
    Baseline("irving_williams", "stability across divalent metals", _irving_williams,
             "Irving & Williams, J. Chem. Soc. 3192 (1953)"),
    Baseline("irving_williams_pair", "selectivity between two metals", _irving_williams_pair,
             "Irving & Williams (1953)"),
    Baseline("ligand_pka", "donor basicity", _ligand_pka, "—"),
    Baseline("denticity", "chelate effect", _denticity, "—"),
    Baseline("steric_bulk", "steric hindrance", _steric_bulk,
             "Tolman, Chem. Rev. 77, 313 (1977)"),
    Baseline("molecular_weight", "trivial size trend", _molecular_weight, "—"),
    Baseline("vina_docking", "protein-ligand affinity", _docking,
             "Trott & Olson, J. Comput. Chem. 31, 455 (2010)"),
)


def register_baseline(name: str, predicts: str, fn: Callable[[dict], Optional[float]],
                      reference: str = "—") -> Baseline:
    """Add a trivial baseline from YOUR domain and make it declarable in a pre-registration.

    The honest baseline is what your field would use *without* the model. Picking a
    deliberately weak one is the most common way to manufacture a result — and the
    easiest to spot afterwards, which is why it belongs in a sealed pre-registration.
    """
    global BASELINES
    if any(b.name == name for b in BASELINES):
        raise ValueError(f"baseline '{name}' already registered — pick another name")
    b = Baseline(name, predicts, fn, reference)
    BASELINES = BASELINES + (b,)
    return b


def applicable_baselines(records: Sequence[dict]) -> list[Baseline]:
    """Baselines with enough data to be evaluated (>= 80% populated)."""
    out = []
    n = len(records)
    if n == 0:
        return out
    for b in BASELINES:
        filled = sum(1 for r in records if b.fn(r) is not None)
        if filled >= 0.8 * n:
            out.append(b)
    return out
