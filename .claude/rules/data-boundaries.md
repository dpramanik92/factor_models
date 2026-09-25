---
scope: src/**
---

# Data boundary rules

- `data/`, `fundamentals/`, `model_data/`, and `output/` are all git-ignored and must never be
  committed. `data/` and `fundamentals/` are raw, read-only inputs - nothing in the pipeline ever
  writes to them. `model_data/` holds pipeline-built intermediates (universe mapping, parsed
  fundamentals, the factor exposure panel); `output/` holds final regression/risk/validation
  deliverables.
- Every path to these folders is routed through `src/factor_model/config.py` - never hardcode a
  path like `"data/nse198_pe_full.parquet"` inside a factor module; import the path constant or
  a loader function from `config.py`/`io/loaders.py` instead.
- The 99-ticker universe (`model_data/universe_99.csv`) is the single source of truth for which
  symbols are in scope. Factor modules subset to it rather than each re-deriving "which tickers
  do we have fundamentals for".
