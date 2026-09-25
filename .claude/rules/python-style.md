---
scope: src/**/*.py
---

# Python style rules

- Type hints on all function signatures (params and return type).
- One-line docstring on public functions that references the exact formula/convention being
  implemented (e.g. "ROA = Net profit / Total assets"), not a restatement of the function name.
- Use `logging`, never `print`, for anything other than the final CLI-facing regression/validation
  summary that's meant to go straight to the user.
- Vectorize with pandas/numpy; a per-date or per-file loop is acceptable only where the operation
  is inherently per-date (the cross-sectional regression) or per-file (parsing 99 separate xlsx
  files) - not as a substitute for a vectorizable groupby/apply.
- No bare `except:` - catch specific exceptions, and when a parsing/data issue is recoverable
  (e.g. one company's fiscal year total-check fails), log and continue rather than raising and
  killing the whole pipeline run.
