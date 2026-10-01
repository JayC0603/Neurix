#!/usr/bin/env python3
"""Servo 2 đã bị loại khỏi hệ thống và không được phép nhận PWM."""

from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "Servo 2 is permanently disabled. Only Servo 1 on BCM18 is supported."
    )


if __name__ == "__main__":
    main()
