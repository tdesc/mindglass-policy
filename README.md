# mindglass · scripted Franka pick-and-place (Physical AI Competition)

Deterministic, CPU-only policy for the Positronic platform (Nebius × Positronic
Physical AI Competition, competition.phail.ai). No neural networks: an HSV
blob tracker on the wrist camera drives a pick-and-place state machine
(servo → descend → grasp → lift → carry → release), with adaptive sign-flip
calibration for unknown camera mounts, spiral search on detection loss, and
grip-feedback grasp verification.

- `scripted_policy.py` — the policy and the per-session state machine
- `policy_server.py` — Positronic policy API server (port 8000)
- `Dockerfile` — linux/amd64, CPU-only; builds in GitHub Actions to
  `ghcr.io/tdesc/franka-scripted-policy:v0`

License: MIT.
