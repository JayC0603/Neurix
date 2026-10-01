"""Thread-safe LCD1602 I2C display with graceful fallback."""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable


LOGGER = logging.getLogger(__name__)
LCD_IDLE_LINE_1 = "ĐH FPT Can Tho"
LCD_IDLE_LINE_2 = "Team: Neurix"
LCD_CLASS_LABELS = {
    "unripe": "UNRIPE",
    "ripe": "RIPE",
    "overripe": "OVERRIPE",
    "unknown": "UNKNOWN",
}

# HD44780 A00 không có chữ Đ. Dùng ô CGRAM 0 để hiển thị đúng
# dòng nhận diện trường trên màn hình khởi động.
UPPERCASE_D_STROKE = (
    0b01110,
    0b01001,
    0b01001,
    0b11101,
    0b01001,
    0b01001,
    0b01110,
    0b00000,
)


class LCDDisplay:
    """Manage a PCF8574-backed character LCD without blocking the pipeline."""

    def __init__(
        self,
        bus: int = 1,
        address: int = 0x27,
        columns: int = 16,
        rows: int = 2,
        enabled: bool = True,
        mock: bool = False,
    ) -> None:
        self.bus = bus
        self.address = address
        self.columns = columns
        self.rows = rows
        self.enabled = enabled
        self.mock = mock
        self.is_connected = False
        self._lcd = None
        self._lock = threading.RLock()
        self._last_message: tuple[str, str] | None = None

    def initialize(self) -> None:
        """Detect and initialize the LCD; failure is intentionally non-fatal."""
        if not self.enabled:
            LOGGER.info("LCD disabled by configuration")
            return
        if self.mock:
            self.is_connected = True
            LOGGER.info("LCD initialized in mock mode")
            return

        try:
            from RPLCD.i2c import CharLCD

            candidates = self._candidate_addresses(self._scan_addresses())
            if not candidates:
                LOGGER.warning("No I2C devices detected on bus %s", self.bus)
                return

            last_error: Exception | None = None
            for address in candidates:
                try:
                    lcd = CharLCD(
                        i2c_expander="PCF8574",
                        address=address,
                        port=self.bus,
                        cols=self.columns,
                        rows=self.rows,
                        charmap="A00",
                        auto_linebreaks=False,
                    )
                    self._lcd = lcd
                    self.address = address
                    self.is_connected = True
                    lcd.create_char(0, UPPERCASE_D_STROKE)
                    self._set_backlight(True)
                    LOGGER.info("LCD initialized on I2C bus %s address 0x%02X", self.bus, address)
                    return
                except Exception as exc:  # Device drivers expose varying exceptions.
                    last_error = exc
                    LOGGER.debug("LCD probe failed at 0x%02X: %s", address, exc)
            LOGGER.warning("LCD was detected but could not initialize: %s", last_error)
        except Exception:
            self.is_connected = False
            LOGGER.exception("LCD initialization failed; continuing without LCD")

    def _scan_addresses(self) -> list[int]:
        from smbus2 import SMBus

        found: list[int] = []
        with SMBus(self.bus) as bus:
            for address in range(0x03, 0x78):
                try:
                    bus.write_quick(address)
                    found.append(address)
                except OSError:
                    continue
        LOGGER.info("I2C addresses detected on bus %s: %s", self.bus, [hex(x) for x in found])
        return found

    def _candidate_addresses(self, found: Iterable[int]) -> list[int]:
        found_set = set(found)
        preferred = [self.address, 0x27, 0x3F]
        ordered: list[int] = []
        for address in preferred:
            if address in found_set and address not in ordered:
                ordered.append(address)
        ordered.extend(sorted(found_set.difference(ordered)))
        return ordered

    def _format_line(self, text: str) -> str:
        return str(text)[: self.columns].ljust(self.columns)

    @staticmethod
    def _encode_custom_characters(text: str) -> str:
        """Thay chữ tiếng Việt bằng mã CGRAM tương ứng của LCD."""
        return text.replace("Đ", "\x00")

    def _set_backlight(self, enabled: bool) -> None:
        """Turn on the LCD backlight; contrast is adjusted on the I2C module."""
        if self.mock or self._lcd is None:
            return
        try:
            self._lcd.backlight_enabled = enabled
        except Exception:
            LOGGER.debug("LCD backlight control is not available", exc_info=True)

    def display_message(self, line1: str, line2: str) -> None:
        """Update two rows only when their formatted content has changed."""
        formatted = (self._format_line(line1), self._format_line(line2))
        LOGGER.info("LCD | %s | %s", formatted[0].rstrip(), formatted[1].rstrip())
        with self._lock:
            if formatted == self._last_message:
                return
            self._last_message = formatted
            if self.mock:
                return
            if not self.is_connected or self._lcd is None:
                return
            try:
                self._set_backlight(True)
                self._lcd.cursor_pos = (0, 0)
                self._lcd.write_string(self._encode_custom_characters(formatted[0]))
                self._lcd.cursor_pos = (1, 0)
                self._lcd.write_string(self._encode_custom_characters(formatted[1]))
            except Exception:
                self.is_connected = False
                LOGGER.exception("LCD write failed; disabling LCD output")

    def display_result(self, class_name: str, confidence: float) -> None:
        """Display a centered result within the 16x2 character limit."""
        label = LCD_CLASS_LABELS.get(class_name.lower(), class_name.upper())
        confidence_text = f"CONFIDENCE {confidence * 100:.1f}%"
        self.display_message(
            label[: self.columns].center(self.columns),
            confidence_text[: self.columns].center(self.columns),
        )

    def clear(self) -> None:
        """Clear the device explicitly; normal updates avoid clear to prevent flicker."""
        with self._lock:
            self._last_message = None
            if self.mock or not self.is_connected or self._lcd is None:
                return
            try:
                self._lcd.clear()
            except Exception:
                LOGGER.exception("LCD clear failed")

    def close(self, clear: bool = False) -> None:
        """Close the LCD without propagating hardware errors."""
        with self._lock:
            try:
                if not self.mock and self._lcd is not None:
                    if clear:
                        self._lcd.clear()
                    close = getattr(self._lcd, "close", None)
                    if callable(close):
                        close(clear=False)
            except Exception:
                LOGGER.exception("LCD cleanup failed")
            finally:
                self._lcd = None
                self.is_connected = False
                LOGGER.info("LCD closed")
