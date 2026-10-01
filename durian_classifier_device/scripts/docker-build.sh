#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTEXT_DIR="$(dirname "$PROJECT_DIR")"

exec docker build \
  --file "$PROJECT_DIR/Dockerfile" \
  --tag durian-v0.1.0 \
  "$CONTEXT_DIR"
