# Write-up (draft): Scripted Franka pick-and-place — engineering against VLA baselines

> Draft skeleton for the Physical AI Competition open-source prize (judged on
> reproducibility, clarity, usefulness). Sections marked [DRAFT] get filled as
> data arrives. Status: policy v2 (grammar + drawer + budget 400 + mode-gates),
> polygon-gated locally.

## 1. What this is
A deterministic, CPU-only policy for the Positronic Franka simulators: a
per-session state machine (servo → descend → grasp → lift → carry → place)
driven by HSV blob detection on the wrist camera, with closed-loop axis-sign
calibration (probe Δee per chunk, no flip oscillation) and an affordance-guarded
task-grammar parser covering the catalog families (pick / place / open / close /
move-dir / give_up).

## 2. Why scripted (the honest answer)
VLA baselines on the board score 0.29–0.34. Our first scored submission scored
0.047. We publish anyway, because the method answers a question the leaderboard
cannot: how much of the task family a rules-based policy covers with zero
training data, and exactly why each failure happens (every zero has a named
mechanism — see the failure atlas below).

## 3. Architecture
- Per-tick DLS-IK re-planning toward the blob-centroid (wrist cam, HSV / colorless
  contrast candidates), rotation fixed from the first observation.
- Closed-loop gain-sign calibration: probe Δee per axis per chunk — deterministic
  camera-mount compensation, no flip oscillation (0 flips in the last 320-episode run).
- Rate-limited servo targets (0.02 rad/tick) inside the measured servo class.
- Task grammar parser: catalog ids → {skill, object, target}; flat-surface
  affordance ban; hinged vs slide drawer geometry.

## 4. Receipts culture
Every change ships with: unit gates, polygon episode rates, byte-verified
builds, and a failure atlas (symptom → root cause → fix → regression test).
Nothing is claimed without a receipt — the repo history is the log.

## 5. Results
[DRAFT] Filled per scored submission: per-family success rates, videos, and the
failure atlas. Current: pick 3/3, place 32-chunk, open/close drawer gated,
move-left 2/2 — all on the local FR3 polygon; platform scores land with each
daily submission.

## 6. Reproduce
[DRAFT] Local polygon: `python run_episode.py --task pick-red --episodes 3`;
platform: `positronic eval run --policy-image=…@sha256:<digest>`.

## 7. Honest boundaries
- No training data, no learning: the ceiling of a rules-based policy is real.
- Simulator-specific tuning is labeled as such; cross-render claims wait for frames.
