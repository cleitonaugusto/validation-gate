# validation-gate

**Does your ML-for-science model beat a trivial baseline on a blind holdout?**

Most models that look impressive on a slide have never been asked the only two questions
that matter: does it beat the *dumb rule*, and does it survive data it has never touched?
When you finally ask, the number often evaporates.

I built this to audit my own computational-chemistry research. Applied to five months of
my own results, it passed **zero of five** benchmarks. Those were the claims I was about
to publish. [The full story is here.](https://cleitonaugusto.github.io/validation/)

```
pip install validation-gate
```

## The verdict

Real output from the bundled demo — a descriptor correlating at **rho = +0.78**, which
would look excellent on any slide, and it still fails:

```
$ validation-gate --prereg examples/prereg_fail.json --data examples/records_fail.csv

==================================================================
  VALIDATION GATE — verdict
==================================================================
  hypothesis : descriptor dq predicts observable y beyond donor softness (HSAB)
  metric     : spearman (positive)
  n total    : 120   |   blind holdout: 36
  pre-reg    : 9c17ebaec7536e8e... (sealed)

  MODEL  [dq]
    spearman = +0.778   CI95 [+0.553, +0.897]

  TRIVIAL BASELINES (the model must beat the strongest):
    baseline                   rho     gain           gain CI95  beats?
    -------------------------------------------------------------------
    hsab_donor              +0.866   -0.088   [-0.254, +0.020]   no

  CHANCE (n=36): sd under null = 0.169   P(rho>=0.70) = 0.0%   P(rho>=0.85) = 0.0%

==================================================================
  FAILED   (strongest baseline: hsab_donor)
  FAILED — the model does NOT beat baseline 'hsab_donor'
  (gain=-0.088, CI95=[-0.254, +0.020]). No advantage demonstrated.
==================================================================
```

A lookup table published in 1963 scores 0.866 on the same data. The rho of 0.78 was never
the model's own signal — it was the free rule, recoded. Nothing but an explicit baseline
comparison shows you that, which is why this one is not optional.

Exit codes drop straight into CI: `0` pass · `1` fail · `2` I/O error · `3` protocol
violation · `4` inconclusive.

## What it enforces

**Pre-registration, hash-sealed.** You declare the hypothesis, the metric, *and the
expected sign* before seeing results. The file is sealed with a SHA-256; edit it
afterwards and the tool voids the study. Delete the hash to dodge the check and it
refuses to load at all.

**A trivial baseline is mandatory.** The model must beat the *strongest* declared
baseline, not the weakest — and beat it in the direction you declared. A descriptor that
predicts the opposite of your hypothesis did not get it right backwards; it got it wrong.

**The blind holdout is counted, not promised.** Every run is written to a ledger beside
the pre-registration. The second run of the same pre-registration is *refused*: running
again after adjusting the method is best-of-N in slow motion. `--allow-rerun` lets you
look, but the verdict comes back `PROTOCOL_FAIL`, because a holdout you have already seen
is not blind.

**Honest statistics.** Paired bootstrap confidence intervals on the *gain over the
baseline* — the only number that decides anything — plus what pure chance produces at
your sample size, printed next to every result.

## Four verdicts, not two

| outcome | exit | meaning |
|---|---|---|
| `PASS` | 0 | beats the strongest baseline, declared direction, 95% CI excluding zero |
| `FAIL` | 1 | does not beat it. No advantage demonstrated |
| `INCONCLUSIVE` | 4 | holdout under 30 points: missing data, not a verdict |
| `PROTOCOL_FAIL` | 3 | holdout already touched. Diagnostic only, never an approval |

`INCONCLUSIVE` exists because the alternative is worse. At a 9-point holdout the null
spread of Spearman's rho is 0.35 — calling anything there a "pass" is exactly the mistake
this package was written to prevent. With n=10 and ~56 tests run against no real signal
at all, the best rho you *expect* is 0.71. That is the range most exciting preliminary
correlations live in.

## Use it

```python
from validation_gate import PreRegistration, run_blind_holdout

# PHASE 1 — before looking at any result. Commit this file.
pre = PreRegistration(
    hypothesis="descriptor dq predicts binding beyond donor identity",
    descriptor="dq",
    observable="log_beta",
    metric="spearman",            # one metric, chosen now, forever
    direction="positive",         # the expected sign, chosen now
    baselines=["hsab_donor", "steric_bulk"],
    decision_rule="gain over the strongest baseline with 95% CI excluding zero",
    n_expected=100,               # the holdout (30%) has to reach 30
)
pre.seal("PREREG.json")

# PHASE 2 — after computing the descriptor. Once.
result = run_blind_holdout(PreRegistration.load("PREREG.json"), records)
print(result["verdict"])
```

Records are plain dicts (or CSV/JSON rows for the CLI): one key for your model's
prediction, one for the ground truth, plus whatever the baselines read.

### Outside chemistry

The built-in baselines come from coordination chemistry because that is where this was
written. The question they encode — *does the free rule already do this?* — is not
chemical. Register your own before sealing:

```python
from validation_gate import register_baseline

register_baseline("days_since_last_purchase", "churn",
                  lambda r: r.get("recency_days"), "business rule, 2019")
```

The honest baseline is what your field would use *without* the model. Picking a
deliberately weak one is the most common way to manufacture a result — which is why the
baseline belongs in the sealed pre-registration, where it cannot be swapped later.

### Unseeded randomness

A separate trap, from a real case: a 3-D conformer generated without a fixed seed made
the same script return rho = 0.855 / 0.127 / -0.100 / 0.818 / 0.297 across five runs.
Mean 0.40 ± 0.38. The number that got written down was the best draw — nobody lied, and
nobody re-ran it either. The same shape of bug lives in any unseeded initialization,
augmentation, or shuffle.

```python
from validation_gate import assert_seed_stable

assert_seed_stable({1337: 0.503, 42: 0.576, 7: 0.164, 2024: 0.103, 999: 0.055})
# UnstableDescriptor: 77% of the signal is the random draw, not the system
```

## Demo

```bash
python examples/make_demo.py
validation-gate --prereg examples/prereg_pass.json --data examples/records_pass.csv  # PASS
validation-gate --prereg examples/prereg_fail.json --data examples/records_fail.csv  # FAIL
```

The `fail` case is the interesting one: a descriptor that correlates well with the
observable, and adds nothing over a lookup table from 1963.

## Independent validation

I also do this as a service — for investors doing technical due diligence on an
AI-for-science company, for founders who want an independent verdict to hand an
investor, and for R&D teams deciding whether to commit lab budget to a model's
predictions. [Details and contact.](https://cleitonaugusto.github.io/validation/)

## Citation and credit

MIT licensed — use it commercially, fork it, ship it. The licence already requires the
copyright notice to travel with any copy or substantial portion of the code.

Beyond that, a request rather than a legal condition: **if this tool informs a paper, a
due-diligence memo, a model card, or any published or client-facing analysis, cite it and
credit the author.** GitHub's "Cite this repository" button reads
[`CITATION.cff`](CITATION.cff) and will give you BibTeX or APA.

```
Bezerra, C. A. C. (2026). validation-gate: a gate that refuses the shortcuts behind
irreproducible claims (v0.1.0) [Computer software].
https://github.com/cleitonaugusto/validation-gate
ORCID: 0009-0003-5543-8026
```

If a verdict from this gate changed a decision — a claim dropped, an investment
reconsidered, a wet-lab budget redirected — I would genuinely like to hear about it.
Open an issue or [get in touch](https://cleitonaugusto.github.io/validation/).
