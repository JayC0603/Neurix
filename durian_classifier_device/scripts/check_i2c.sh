#!/usr/bin/env bash
set -euo pipefail

if [[ ! -e /dev/i2c-1 ]]; then
    echo "ERROR: /dev/i2c-1 does not exist. Enable I2C using sudo raspi-config." >&2
    exit 1
fi
if ! command -v i2cdetect >/dev/null 2>&1; then
    echo "ERROR: i2cdetect not found. Install i2c-tools." >&2
    exit 2
fi

OUTPUT="$(i2cdetect -y 1)"
echo "$OUTPUT"
if grep -Eq '(^|[[:space:]])27([[:space:]]|$)' <<<"$OUTPUT"; then
    echo "Preferred LCD address detected: 0x27"
elif grep -Eq '(^|[[:space:]])3f([[:space:]]|$)' <<<"$OUTPUT"; then
    echo "Preferred LCD address detected: 0x3F"
else
    echo "No preferred LCD address (0x27 or 0x3F) detected."
fi
