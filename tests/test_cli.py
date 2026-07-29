"""
Tests for the command-line gate.

Generates the demo data and requires the CLI to PASS the descriptor with real signal
(exit 0) and FAIL the one that only recodes the trivial baseline (exit 1).

    pytest tests/test_cli.py -v
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "examples"


@pytest.fixture(scope="module")
def demo_data():
    subprocess.run([sys.executable, str(EX / "make_demo.py")], cwd=ROOT, check=True)
    return EX


def _run(prereg, data, *extra, ledger=None):
    cmd = [sys.executable, "-m", "validation_gate.cli",
           "--prereg", str(prereg), "--data", str(data), *extra]
    if ledger is not None:
        cmd += ["--ledger", str(ledger)]
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)


def test_passes_real_signal(demo_data, tmp_path):
    r = _run(demo_data / "prereg_pass.json", demo_data / "records_pass.csv",
             ledger=tmp_path / "runs.json")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "PASSED" in r.stdout


def test_fails_recoded_baseline(demo_data, tmp_path):
    r = _run(demo_data / "prereg_fail.json", demo_data / "records_fail.csv",
             ledger=tmp_path / "runs.json")
    assert r.returncode == 1, r.stdout + r.stderr
    assert "FAILED" in r.stdout
    assert "No advantage demonstrated" in r.stdout


def test_json_output(demo_data, tmp_path):
    r = _run(demo_data / "prereg_pass.json", demo_data / "records_pass.csv",
             "--format", "json", ledger=tmp_path / "runs.json")
    payload = json.loads(r.stdout)
    assert payload["PASSED"] is True
    assert payload["outcome"] == "PASS"
    assert payload["hardest_baseline"] == "hsab_donor"


def test_refuses_second_run_on_same_holdout(demo_data, tmp_path):
    """The blind holdout, from the command line: a second --data on the same
    pre-registration is refused."""
    led = tmp_path / "runs.json"
    first = _run(demo_data / "prereg_pass.json", demo_data / "records_pass.csv", ledger=led)
    assert first.returncode == 0, first.stdout + first.stderr

    second = _run(demo_data / "prereg_pass.json", demo_data / "records_fail.csv", ledger=led)
    assert second.returncode == 3, second.stdout + second.stderr
    assert "ALREADY been evaluated" in second.stderr

    forced = _run(demo_data / "prereg_pass.json", demo_data / "records_fail.csv",
                  "--allow-rerun", ledger=led)
    assert forced.returncode == 3, forced.stdout + forced.stderr
    assert "PROTOCOL FAILURE" in forced.stdout
