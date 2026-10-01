"""Classification output value object."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ClassificationResult:
    """A single durian classification and its diagnostics."""

    class_index: int
    class_name: str
    confidence: float
    inference_time_ms: float
    probabilities: list[float]
