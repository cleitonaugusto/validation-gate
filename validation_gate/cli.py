"""
Command line for the validation gate — usable without writing any Python.

    validation-gate --prereg PREREG.json --data records.csv
    validation-gate --prereg PREREG.json --data records.json --format json

`--data` holds the records (one per row in CSV, or a list of objects in JSON). Each
record needs the descriptor column, the observable column, and whatever columns the
baselines declared in the pre-registration consume.

The output is a verdict: the model beats the STRONGEST baseline with a 95% CI
excluding zero, or it does not. No best-of-N.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from .protocol import PreRegistration, ProtocolViolation, run_blind_holdout


def _coerce(v: str) -> Any:
    """CSV gives everything as text; parse numbers, keep the rest, empty -> None."""
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def load_records(path: str | Path) -> list[dict]:
    """Read records from a .csv (DictReader) or .json (list of objects)."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"{path}: JSON must be a LIST of records")
        return data
    rows = list(csv.DictReader(text.splitlines()))
    return [{k: _coerce(v) for k, v in row.items()} for row in rows]


def format_report(result: dict) -> str:
    """Human-readable verdict. Deterministic — paste it into an issue, PR or memo."""
    L: list[str] = []
    bar = "=" * 66
    L.append(bar)
    L.append("  VALIDATION GATE — verdict")
    L.append(bar)
    L.append(f"  hypothesis : {result['hypothesis']}")
    L.append(f"  metric     : {result['metric']} ({result['direction']})")
    L.append(f"  n total    : {result['n_total']}   |   blind holdout: {result['n_holdout']}")
    if result.get("prereg_sha256"):
        L.append(f"  pre-reg    : {result['prereg_sha256'][:16]}... (sealed)")
    L.append("")

    m = result["model"]
    L.append(f"  MODEL  [{m['descriptor']}]")
    L.append(f"    {result['metric']} = {m['estimate']:+.3f}   "
             f"CI95 [{m['ci95'][0]:+.3f}, {m['ci95'][1]:+.3f}]")
    L.append("")

    L.append("  TRIVIAL BASELINES (the model must beat the strongest):")
    header = f"    {'baseline':<22}{'rho':>8}{'gain':>9}{'gain CI95':>20}  beats?"
    L.append(header)
    L.append("    " + "-" * (len(header) - 4))
    for b in result["baselines"]:
        if "gain_ci95" not in b:
            L.append(f"    {b['baseline']:<22}{'-':>8}   ({b.get('status', 'no data')})")
            continue
        beat = "YES" if b["model_beats_baseline"] else "no"
        L.append(
            f"    {b['baseline']:<22}{b['estimate']:>+8.3f}"
            f"{b['gain_of_model_over_baseline']:>+9.3f}"
            f"   [{b['gain_ci95'][0]:+.3f}, {b['gain_ci95'][1]:+.3f}]   {beat}"
        )
    L.append("")

    nr = result["null_reference"]
    L.append(f"  CHANCE (n={nr['n']}): sd under null = {nr['sd_under_null']:.3f}   "
             f"P(rho>=0.70) = {nr['p_ge_0.70']*100:.1f}%   "
             f"P(rho>=0.85) = {nr['p_ge_0.85']*100:.1f}%")
    L.append("")

    L.append(bar)
    tag = {
        "PASS":          "PASSED",
        "FAIL":          "FAILED",
        "INCONCLUSIVE":  "INCONCLUSIVE",
        "PROTOCOL_FAIL": "PROTOCOL FAILURE",
    }[result["outcome"]]
    L.append(f"  {tag}   (strongest baseline: {result['hardest_baseline']})")
    L.append(f"  {result['verdict']}")
    if not result.get("r38_enforced"):
        L.append("  note: the pre-registration did not come from a file, so 'holdout")
        L.append("        touched once' could not be verified for this run.")
    L.append(bar)
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="validation-gate",
        description="Does your model beat the strongest trivial baseline, on a blind "
                    "holdout, with a 95% CI excluding zero — or does it prove nothing?",
    )
    ap.add_argument("--prereg", required=True,
                    help="sealed PREREG.json (declared BEFORE running)")
    ap.add_argument("--data", required=True, help="records: .csv (DictReader) or .json (list)")
    ap.add_argument("--format", choices=["text", "json"], default="text", help="output format")
    ap.add_argument("--out", help="write the result here instead of stdout")
    ap.add_argument("--ledger", help="run ledger (default: <prereg>.runs.json)")
    ap.add_argument("--allow-rerun", action="store_true",
                    help="run even though the holdout was already touched; "
                         "the verdict will be PROTOCOL_FAIL")
    args = ap.parse_args(argv)

    try:
        pre = PreRegistration.load(args.prereg)
        records = load_records(args.data)
        result = run_blind_holdout(pre, records, ledger=args.ledger,
                                   allow_rerun=args.allow_rerun)
    except ProtocolViolation as e:
        print(f"PROTOCOL VIOLATION — study aborted:\n{e}", file=sys.stderr)
        return 3
    except (OSError, ValueError, KeyError) as e:
        print(f"could not load or run: {e}", file=sys.stderr)
        return 2

    out_text = json.dumps(result, indent=2, ensure_ascii=False) if args.format == "json" \
        else format_report(result)
    if args.out:
        Path(args.out).write_text(out_text + "\n", encoding="utf-8")
        print(f"result written to {args.out}")
    else:
        print(out_text)

    # exit code — drops straight into CI or a Makefile.
    #   0 pass · 1 fail · 2 I/O error · 3 protocol violation · 4 inconclusive
    return {"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 4, "PROTOCOL_FAIL": 3}[result["outcome"]]


if __name__ == "__main__":
    raise SystemExit(main())
