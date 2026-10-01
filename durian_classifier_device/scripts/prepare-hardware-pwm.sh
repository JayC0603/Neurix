#!/usr/bin/env bash
set -euo pipefail

RUN_USER="${1:?Usage: prepare-hardware-pwm.sh RUN_USER}"
PWM_ROOT=/sys/class/pwm
PWM_PERIOD_NS=20000000

PWM_CHIP=""
for candidate in "$PWM_ROOT"/pwmchip*; do
    [[ -f "$candidate/npwm" ]] || continue
    if (( $(<"$candidate/npwm") >= 1 )); then
        PWM_CHIP="$candidate"
        break
    fi
done

if [[ -z "$PWM_CHIP" ]]; then
    echo "ERROR: PWM0 hardware PWM is unavailable." >&2
    echo "Run configure-hardware-pwm.sh and reboot the Raspberry Pi." >&2
    exit 1
fi

for channel in 0; do
    channel_dir="$PWM_CHIP/pwm$channel"
    if [[ ! -d "$channel_dir" ]]; then
        echo "$channel" > "$PWM_CHIP/export"
        for _attempt in {1..50}; do
            [[ -d "$channel_dir" ]] && break
            sleep 0.02
        done
    fi

    if [[ ! -d "$channel_dir" ]]; then
        echo "ERROR: could not export PWM channel $channel" >&2
        exit 2
    fi

    [[ $(<"$channel_dir/enable") == 0 ]] || echo 0 > "$channel_dir/enable"
    echo 0 > "$channel_dir/duty_cycle"
    echo "$PWM_PERIOD_NS" > "$channel_dir/period"

    chown "$RUN_USER:gpio" \
        "$channel_dir/enable" \
        "$channel_dir/duty_cycle" \
        "$channel_dir/period"
    chmod 0660 \
        "$channel_dir/enable" \
        "$channel_dir/duty_cycle" \
        "$channel_dir/period"
done

echo "Hardware PWM ready: Servo 1 PWM0=BCM18, period=${PWM_PERIOD_NS}ns"
