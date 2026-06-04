#!/usr/bin/env bash
# Startup script for the ephemeral GPU runner VM (issue #48).
#
# Passed to `gcloud compute instances create` as metadata, so it runs at boot
# with no SSH in (which is why the provisioner SA needs only create/delete, not
# setMetadata). It registers this VM as a one-shot (`--ephemeral`) self-hosted
# GitHub Actions runner; after the single job it unregisters itself, and the
# `stop-runner` job (plus the VM's max-run-duration backstop) deletes the VM.
#
# Required instance metadata: gh_repo, runner_token, runner_name, runner_labels.
# The base image (Deep Learning VM, common-cu129-…-nvidia-580) already has the
# NVIDIA driver; world_engine deps are installed by the workflow's job steps.
set -euo pipefail
# Echo every command and announce the failing line. This output lands on the VM
# serial console, which the workflow's "Wait for runner to register" step dumps
# into the Actions log if registration doesn't complete in time.
set -x
trap 'echo "[runner-startup] FAILED at line ${LINENO} (exit $?)" >&2' ERR

RUNNER_VERSION="2.334.0"
meta() { curl -s -H "Metadata-Flavor: Google" \
  "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"; }

GH_REPO="$(meta gh_repo)"
RUNNER_TOKEN="$(meta runner_token)"
RUNNER_NAME="$(meta runner_name)"
RUNNER_LABELS="$(meta runner_labels)"

# The Actions runner refuses to run as root; this is a throwaway single-job VM,
# so allow it rather than provisioning a dedicated user.
export RUNNER_ALLOW_RUNASROOT=1

mkdir -p /actions-runner && cd /actions-runner
curl -sL -o runner.tar.gz \
  "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz"
tar xzf runner.tar.gz

# Install the runner's OS dependencies (libicu et al.). Missing libs are a
# common silent cause of config.sh failing on minimal/base images.
./bin/installdependencies.sh

./config.sh \
  --unattended \
  --ephemeral \
  --url "https://github.com/${GH_REPO}" \
  --token "${RUNNER_TOKEN}" \
  --name "${RUNNER_NAME}" \
  --labels "${RUNNER_LABELS}" \
  --replace

# Blocks until the single job completes, then the ephemeral runner exits and
# unregisters itself from GitHub.
./run.sh
