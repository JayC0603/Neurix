#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BOOT_CONFIG=/boot/firmware/config.txt
SERVICE_NAME=durian-classifier.service
SERVICE_TEMPLATE="$PROJECT_DIR/systemd/$SERVICE_NAME"
SERVICE_TARGET="/etc/systemd/system/$SERVICE_NAME"
RUN_USER="${SUDO_USER:-raspberrypi}"
OVERLAY_LINE="dtoverlay=pwm,pin=18,func=2"

if [[ $EUID -ne 0 ]]; then
    echo "ERROR: run with sudo: sudo $0" >&2
    exit 1
fi

if [[ ! -f "$BOOT_CONFIG" ]]; then
    echo "ERROR: boot configuration not found: $BOOT_CONFIG" >&2
    exit 2
fi

backup="$BOOT_CONFIG.before-servo-hardware-pwm"
if [[ ! -f "$backup" ]]; then
    cp --preserve=all "$BOOT_CONFIG" "$backup"
fi

# GPIO18/PWM0 shares hardware PWM with the analogue audio output.
sed -i 's/^dtparam=audio=on$/dtparam=audio=off/' "$BOOT_CONFIG"
if ! grep -Fqx "$OVERLAY_LINE" "$BOOT_CONFIG"; then
    printf '\n# Stable hardware PWM for the MG996R Servo 1 on BCM18\n%s\n' \
        "$OVERLAY_LINE" >> "$BOOT_CONFIG"
fi

sed \
    -e "s|@PROJECT_DIR@|$PROJECT_DIR|g" \
    -e "s|@RUN_USER@|$RUN_USER|g" \
    "$SERVICE_TEMPLATE" > "$SERVICE_TARGET"
chmod 0644 "$SERVICE_TARGET"
systemctl daemon-reload
systemctl stop "$SERVICE_NAME" || true

echo "Hardware PWM configuration installed."
echo "Analogue audio is disabled; HDMI and USB audio are unaffected."
echo "Reboot now with: sudo reboot"
