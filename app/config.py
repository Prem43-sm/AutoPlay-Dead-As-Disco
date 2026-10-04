from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class CaptureRegion:
    left: int
    top: int
    width: int
    height: int

    def validate(self) -> None:
        if self.left < 0 or self.top < 0:
            raise ValueError("Capture region left and top must be non-negative.")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Capture region width and height must be positive.")


@dataclass(frozen=True)
class CaptureConfig:
    monitor_index: int = 1
    max_fps: float = 60.0
    region: CaptureRegion | None = None

    def validate(self) -> None:
        if self.monitor_index < 1:
            raise ValueError("Monitor index must be 1 or greater.")
        if not isfinite(self.max_fps) or self.max_fps <= 0:
            raise ValueError("Maximum FPS must be greater than zero.")
        if self.region is not None:
            self.region.validate()
