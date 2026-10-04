from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from typing import Literal


@dataclass(frozen=True)
class BoundingBox:
    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Bounding box width and height must be positive.")

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)

    def clip(self, frame_width: int, frame_height: int) -> "BoundingBox | None":
        left = max(0, self.x)
        top = max(0, self.y)
        right = min(frame_width, self.x + self.width)
        bottom = min(frame_height, self.y + self.height)
        if right <= left or bottom <= top:
            return None
        return BoundingBox(left, top, right - left, bottom - top)


@dataclass(frozen=True)
class Detection:
    class_name: str
    bbox: BoundingBox
    confidence: float
    source: str
    frame_number: int | None = None
    timestamp: str | None = None
    limitation: str | None = None

    def __post_init__(self) -> None:
        if not self.class_name.strip():
            raise ValueError("Detection class name must not be empty.")
        if not self.source.strip():
            raise ValueError("Detection source must not be empty.")
        if not isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("Detection confidence must be between 0 and 1.")
        if self.frame_number is not None and self.frame_number < 1:
            raise ValueError("Frame number must be at least 1.")


@dataclass(frozen=True)
class DetectionResult:
    detections: tuple[Detection, ...]
    method: str
    limitation: str | None = None
    status: Literal["detected", "not_detected", "not_reliably_detected"] = (
        "not_detected"
    )

    @property
    def confidence(self) -> float | None:
        if not self.detections:
            return None
        return max(detection.confidence for detection in self.detections)


@dataclass(frozen=True)
class FrameContext:
    timestamp: datetime
    frame_number: int | None = None
    capture_fps: float | None = None

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError("Frame timestamp must include timezone information.")
        if self.frame_number is not None and self.frame_number < 1:
            raise ValueError("Frame number must be at least 1.")
        if self.capture_fps is not None and (
            not isfinite(self.capture_fps) or self.capture_fps < 0
        ):
            raise ValueError("Capture FPS must be finite and non-negative.")

    @classmethod
    def now(cls, frame_number: int | None = None) -> "FrameContext":
        return cls(datetime.now().astimezone(), frame_number=frame_number)
