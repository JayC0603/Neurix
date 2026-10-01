#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

if ! docker image inspect durian-v0.1.0 >/dev/null 2>&1; then
  echo "Docker image durian-v0.1.0 does not exist. Run ./scripts/docker-build.sh first." >&2
  exit 1
fi

if docker compose version >/dev/null 2>&1; then
  exec docker compose up --detach --no-build
fi

exec docker-compose up --detach --no-build
