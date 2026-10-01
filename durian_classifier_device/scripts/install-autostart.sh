#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_NAME="durian-classifier.service"
SERVICE_TEMPLATE="$PROJECT_DIR/systemd/$SERVICE_NAME"
SERVICE_TARGET="/etc/systemd/system/$SERVICE_NAME"
RUN_USER="${SUDO_USER:-$(id -un)}"

if [[ ! -f "$SERVICE_TEMPLATE" ]]; then
    echo "ERROR: service template not found: $SERVICE_TEMPLATE" >&2
    exit 1
fi

if [[ ! -x "$PROJECT_DIR/.venv/bin/python" ]]; then
    echo "ERROR: Python environment not found: $PROJECT_DIR/.venv/bin/python" >&2
    echo "Run ./scripts/setup.sh before installing autostart." >&2
    exit 2
fi

if [[ ! -f "$PROJECT_DIR/.env" ]]; then
    echo "ERROR: configuration not found: $PROJECT_DIR/.env" >&2
    echo "Copy .env.example to .env and configure the hardware first." >&2
    exit 3
fi

if [[ $EUID -ne 0 ]]; then
    echo "ERROR: run this installer with sudo:" >&2
    echo "  sudo $PROJECT_DIR/scripts/install-autostart.sh" >&2
    exit 4
fi

sed \
    -e "s|@PROJECT_DIR@|$PROJECT_DIR|g" \
    -e "s|@RUN_USER@|$RUN_USER|g" \
    "$SERVICE_TEMPLATE" > "$SERVICE_TARGET"

chmod 0644 "$SERVICE_TARGET"
systemctl daemon-reload
systemctl enable --now "$SERVICE_NAME"

echo "Installed and started $SERVICE_NAME"
echo "Status: sudo systemctl status $SERVICE_NAME"
echo "Logs:   journalctl -u $SERVICE_NAME -f"
