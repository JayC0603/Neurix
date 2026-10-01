"""Domain-specific inference and hardware exceptions."""


class ModelError(RuntimeError):
    """Base error raised by the model pipeline."""


class ModelConfigurationError(ModelError):
    """Raised when model shapes or configured classes are incompatible."""


class ModelInferenceError(ModelError):
    """Raised when prediction cannot be completed."""


class CameraError(RuntimeError):
    """Raised when the camera cannot initialize or capture an image."""


class CameraBlurError(CameraError):
    """Raised when every frame in a capture burst is too blurry."""
