# RoadSight

Traffic event detection (Part A) and causal accident risk (Part B) for one fixed CCTV camera.
The full specification is in `SPEC.md` (source: the RoadSight technical specification PDF in the repo root).

## Hard rules

- Never edit `run_submission.py` or `evaluate.py` once the organizer's versions are in place. The current
  ones are stand-ins and must be replaced unchanged.
- No network at inference: weights are local files, nothing may download (tests block sockets).
- Deterministic: two runs must produce identical `predictions.json`.
- `detect_events` output: floats rounded to 2 decimals, `0 <= start < end <= duration`, labels from
  `CLASSES`, no two segments of the same class overlapping. Never raise; return `[]` on failure.
- `RiskEstimator.step` is causal: only past frames, no file access, no Part A outputs. Returns [0, 1].
- Runtime: Part A + Part B within 3x video duration on a T4-class GPU (target 1x).

## Working here

- Logic lives in `roadsight/`; `solution.py` stays a thin adapter.
- Every threshold goes in `configs/default.yaml`, not in code.
- One rule per file in `roadsight/events/rules/`; add a synthetic scenario to `tests/test_rules.py`.
- After changes: `make test`, then `make run && make eval`.
- Use `ROADSIGHT_CACHE_DIR=.cache` to cache perception while iterating on rules (dev only).
