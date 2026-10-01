#!/usr/bin/env python3
"""Show the team name on the 16x2 I2C LCD."""

from __future__ import annotations

import time

from hardware.lcd_display import LCDDisplay, LCD_IDLE_LINE_1, LCD_IDLE_LINE_2


def main() -> None:
    lcd = LCDDisplay(
        bus=1,
        address=0x27,
        columns=16,
        rows=2,
        enabled=True,
        mock=False,
    )
    lcd.initialize()
    if not lcd.is_connected:
        raise RuntimeError("Không tìm thấy LCD tại địa chỉ I2C 0x27")

    lcd.display_message(LCD_IDLE_LINE_1, LCD_IDLE_LINE_2)
    print("LCD: ĐH FPT Can Tho | Team: Neurix")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        lcd.close(clear=False)


if __name__ == "__main__":
    main()
