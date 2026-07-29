"""
Regression traps for the gate.

Each test reproduces a way a false claim gets manufactured and requires the gate to
refuse it. If one of these starts passing, someone loosened the gate — and the next
irreproducible result is already on its way.

    pytest tests/test_gate.py -v
"""
import json

import pytest

from validation_gate import (
    MIN_HOLDOUT_N, PreRegistration, ProtocolViolation, run_blind_holdout,
    assert_seed_stable, UnstableDescriptor,
)


def _valid_prereg(**over):
    kw = dict(
        hypothesis="test", descriptor="dq", observable="y", metric="spearman",
        baselines=["hsab_donor"], decision_rule="gain CI95 > 0", n_expected=30,
        n_bootstrap=400,
    )
    kw.update(over)
    return PreRegistration(**kw)


def _sealed_prereg(tmp_path, name="prereg.json", **over):
    """A pre-registration sealed to disk — that is what activates the run ledger."""
    pre = _valid_prereg(**over)
    pre.seal(tmp_path / name)
    return PreRegistration.load(tmp_path / name)


def _records(n, *, signal=True):
    """n records. `signal=False` makes the descriptor merely recode the baseline."""
    donors = ["S", "N", "O"]
    soft = {"S": 3.0, "N": 2.0, "O": 1.0}
    out = []
    for i in range(n):
        d = donors[i % 3]
        out.append({"dq": float(i) if signal else soft[d] + 0.01 * i,
                    "y": float(i) if signal else soft[d] * 10 + 0.01 * i,
                    "donor_element": d})
    return out


# ── a baseline is mandatory ──────────────────────────────────────────────────
def test_no_baseline_is_refused():
    with pytest.raises(ProtocolViolation, match="no baseline"):
        _valid_prereg(baselines=[])


def test_unknown_baseline_is_refused():
    with pytest.raises(ProtocolViolation, match="unknown baseline"):
        _valid_prereg(baselines=["something_i_made_up"])


def test_custom_domain_baseline_can_be_registered(tmp_path):
    """The built-ins are chemistry, but the question is not. Any domain must fit."""
    from validation_gate import register_baseline
    register_baseline("recency_days", "churn", lambda r: r.get("recency"), "business rule")
    records = [{"dq": float(i % 7), "y": float(i), "recency": float(i)} for i in range(120)]
    r = run_blind_holdout(
        _sealed_prereg(tmp_path, n_expected=120, baselines=["recency_days"]), records)
    assert r["hardest_baseline"] == "recency_days"
    assert r["outcome"] == "FAIL"          # dq is noise against a perfect baseline


# ── sample size ──────────────────────────────────────────────────────────────
def test_tiny_n_is_refused():
    with pytest.raises(ProtocolViolation, match="< 30"):
        _valid_prereg(n_expected=10)


def test_small_holdout_is_inconclusive_not_a_pass(tmp_path):
    """n=30 with a 30% holdout decides on 9 points. That is not a pass and not a
    failure — it is missing data, and saying otherwise is the original sin."""
    r = run_blind_holdout(_sealed_prereg(tmp_path, n_expected=30), _records(30))
    assert r["n_holdout"] < MIN_HOLDOUT_N
    assert r["outcome"] == "INCONCLUSIVE"
    assert r["PASSED"] is False


# ── the sealed pre-registration ──────────────────────────────────────────────
def test_tampered_prereg_is_detected(tmp_path):
    fp = tmp_path / "prereg.json"
    _valid_prereg().seal(fp)
    d = json.loads(fp.read_text())
    d["hypothesis"] = "rewritten after seeing the result"
    fp.write_text(json.dumps(d))
    with pytest.raises(ProtocolViolation, match="MODIFIED"):
        PreRegistration.load(fp)


def test_unsealed_prereg_is_refused(tmp_path):
    """Deleting the hash field must not be a way around the check."""
    fp = tmp_path / "prereg.json"
    _valid_prereg().seal(fp)
    d = json.loads(fp.read_text())
    d["hypothesis"] = "rewritten after seeing the result"
    d.pop("sealed_sha256")
    fp.write_text(json.dumps(d))
    with pytest.raises(ProtocolViolation, match="NOT SEALED"):
        PreRegistration.load(fp)


def test_old_seal_survives_a_new_field(tmp_path):
    """The seal covers what was DECLARED. Adding a field with a default to the protocol
    must not turn honest pre-registrations into 'tampered' ones."""
    import hashlib
    d = dict(hypothesis="h", descriptor="dq", observable="y", metric="spearman",
             baselines=["hsab_donor"], decision_rule="r", n_expected=30,
             holdout_frac=0.3, split_seed=20260714, n_bootstrap=400, notes="")
    d["sealed_sha256"] = hashlib.sha256(
        json.dumps(d, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    fp = tmp_path / "old.json"
    fp.write_text(json.dumps(d))
    assert PreRegistration.load(fp).direction == "positive"


# ── no post-hoc exclusion ────────────────────────────────────────────────────
def test_posthoc_exclusion_is_refused():
    recs = _records(30)
    recs[5]["dq"] = None                       # the inconvenient point, removed
    with pytest.raises(ProtocolViolation, match="missing the descriptor"):
        run_blind_holdout(_valid_prereg(), recs)


def test_nan_is_exclusion_too():
    """NaN is absence dressed as a number: `is not None` let it through, and the
    correlation came out NaN with nobody complaining."""
    recs = _records(30)
    recs[5]["dq"] = float("nan")
    with pytest.raises(ProtocolViolation, match="missing the descriptor"):
        run_blind_holdout(_valid_prereg(), recs)


# ── the sign is declared too ─────────────────────────────────────────────────
def test_anticorrelated_model_does_not_pass(tmp_path):
    """Using |rho| treated perfect anti-correlation as a win: best-of-two-signs."""
    recs = [{"dq": -float(i), "y": float(i), "donor_element": ["S", "N", "O"][i % 3]}
            for i in range(120)]
    r = run_blind_holdout(_sealed_prereg(tmp_path, n_expected=120), recs)
    assert r["model"]["estimate"] < 0
    assert r["outcome"] == "FAIL"

    # the SAME series, with the direction declared up front, is a legitimate result
    r2 = run_blind_holdout(
        _sealed_prereg(tmp_path, "neg.json", n_expected=120, direction="negative"), recs)
    assert r2["outcome"] == "PASS"


def test_invalid_direction_is_refused():
    with pytest.raises(ProtocolViolation, match="direction"):
        _valid_prereg(direction="two_sided")


# ── the holdout is blind exactly once ────────────────────────────────────────
def test_second_run_is_refused(tmp_path):
    pre = _sealed_prereg(tmp_path, n_expected=120)
    recs = _records(120)
    run_blind_holdout(pre, recs)
    with pytest.raises(ProtocolViolation, match="ALREADY been evaluated"):
        run_blind_holdout(pre, recs)


def test_forced_rerun_never_passes(tmp_path):
    """A revisited holdout may be inspected, but it can no longer approve anything."""
    pre = _sealed_prereg(tmp_path, n_expected=120)
    recs = _records(120)
    assert run_blind_holdout(pre, recs)["outcome"] == "PASS"
    forced = run_blind_holdout(pre, recs, allow_rerun=True)
    assert forced["outcome"] == "PROTOCOL_FAIL"
    assert forced["PASSED"] is False


def test_in_memory_prereg_reports_unenforced(tmp_path):
    """No file, no ledger — and the result must admit it rather than imply a guarantee."""
    r = run_blind_holdout(_valid_prereg(n_expected=120), _records(120))
    assert r["r38_enforced"] is False


# ── degenerate series ────────────────────────────────────────────────────────
def test_constant_baseline_is_not_a_win(tmp_path):
    """A constant baseline used to crash the gate with IndexError mid-bootstrap."""
    recs = [{"dq": float(i), "y": float(i), "donor_element": "S", "mw": float(i % 5)}
            for i in range(120)]
    r = run_blind_holdout(
        _sealed_prereg(tmp_path, n_expected=120,
                       baselines=["hsab_donor", "molecular_weight"]), recs)
    hsab = next(b for b in r["baselines"] if b["baseline"] == "hsab_donor")
    assert "constant" in hsab["status"]
    assert r["hardest_baseline"] == "molecular_weight"


# ── seed stability ───────────────────────────────────────────────────────────
def test_seed_unstable_descriptor_is_refused():
    # real numbers from the audit: mean 0.28, spread 0.22
    measured = {1337: 0.503, 42: 0.576, 7: 0.164, 2024: 0.103, 999: 0.055}
    with pytest.raises(UnstableDescriptor):
        assert_seed_stable(measured)


def test_seed_stable_descriptor_passes():
    good = {1337: 0.82, 42: 0.80, 7: 0.85, 2024: 0.81, 999: 0.83}
    assert_seed_stable(good)                    # must not raise
