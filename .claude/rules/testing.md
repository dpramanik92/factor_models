---
scope: tests/**
---

# Testing rules

- Unit tests use synthetic fixtures only - never a dependency on the real `data/` or
  `fundamentals/` folders, so the suite runs without the (git-ignored, 51MB+) real data present.
- The regression engine's correctness is tested via a synthetic recovery test: simulate returns
  from known exposures and known factor returns plus noise, fit, and assert recovered
  coefficients are close to the truth (see `.claude/skills/statistics/SKILL.md`).
- The point-in-time reporting-lag logic (`factors/pit.py`) gets a dedicated test asserting a
  fixture FY-end value does not appear in the daily-expanded series before
  `FY_end + REPORTING_LAG_DAYS`.
- Name test files `test_<module>.py` mirroring the `src/factor_model/` module they cover.
- Statistical validation of the *real, fitted* model (VIF, residual diagnostics, look-ahead check
  on real data, etc.) is not a pytest suite - it's `src/factor_model/validation/`, run via the
  CLI's `validate` command and reviewed by the `model_reviewer` agent. Don't conflate the two:
  pytest here answers "does the code work", `validate` answers "is the fitted model sound".
