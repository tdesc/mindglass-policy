# Scripted Franka pick-and-place policy for the Positronic platform (MolmoSpaces practice evals).
#
# Contract the platform enforces (docs/nebius-competition.md:298-326, docs/submit-a-policy-image.md):
#   - linux/amd64 image, server starts with NO arguments, binds 0.0.0.0:8000;
#   - the only runtime env var is AUTH_TOKEN (bearer token of the run);
#   - no network at runtime: every dependency is baked in below (CPU-only, ~300 MB).
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY policy_server.py scripted_policy.py task_parser.py ./

# Drop privileges: the platform runs with --cap-drop ALL --security-opt no-new-privileges;
# port 8000 is unprivileged, and the server logs to stdout only.
USER 10001:10001

EXPOSE 8000
ENTRYPOINT ["python", "/app/policy_server.py"]
