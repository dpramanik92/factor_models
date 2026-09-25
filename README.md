# factor_models

A from-scratch Barra-style cross-sectional equity factor risk model for a subset of NSE (India)
equities: factor exposures → daily cross-sectional WLS regression → factor return time series →
EWMA factor covariance matrix.

See [CLAUDE.md](CLAUDE.md) for the full model specification (universe, factors, regression spec,
known limitations) and project conventions.

## Setup

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/pip install -e .
```

Requires the (git-ignored, locally-provided) `data/` and `fundamentals/` folders to be present at
the repo root — this repo ships code only, not data.

## Running

```bash
PYTHONPATH=src ./.venv/bin/python -m factor_model.cli run-all      # full pipeline
PYTHONPATH=src ./.venv/bin/python -m factor_model.cli run-regression
PYTHONPATH=src ./.venv/bin/python -m factor_model.cli validate
./.venv/bin/pytest tests/                                            # unit tests, no data/ needed
```

(`PYTHONPATH=src` is a workaround for an editable-install quirk observed in this environment;
if `pip install -e .` resolves cleanly for you, `factor-model run-all` works directly.)

Outputs land in `output/` (mirrored into `data/`) — daily factor returns, regression diagnostics,
the EWMA factor covariance matrix, specific risk, and `output/validation_report.md`.
