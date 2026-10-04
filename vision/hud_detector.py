import cv2
import numpy as np
from numpy.typing import NDArray

from vision.models import BoundingBox, Detection, DetectionResult, FrameContext


class HudDetector:
    """Detects a coarse upper-left magenta HUD cue, not individual HUD values."""

    def detect(
        self, frame: NDArray[np.uint8], context: FrameContext | None = None
    ) -> DetectionResult:
        if frame.ndim != 3 or frame.shape[2] < 3:
            raise ValueError("HUD detection expects a color image.")
        height, width = frame.shape[:2]
        region_width = max(1, int(width * 0.30))
        region_height = max(1, int(height * 0.25))
        hsv = cv2.cvtColor(
            frame[:region_height, :region_width, :3], cv2.COLOR_BGR2HSV
        )
        magenta = cv2.inRange(
            hsv,
            np.array([140, 80, 80], dtype=np.uint8),
            np.array([179, 255, 255], dtype=np.uint8),
        )
        ratio = float(np.count_nonzero(magenta)) / magenta.size
        if ratio < 0.005:
            return DetectionResult(
                (),
                method="upper-left magenta HUD-cue heuristic",
                limitation="No upper-left magenta cue exceeded the detection threshold.",
                status="not_reliably_detected",
            )

        confidence = min(0.85, 0.5 + ratio * 5)
        limitation = (
            "Coarse upper-left HUD candidate only; health, Fever, Takedown, "
            "and score values are not decoded."
        )
        detection = Detection(
            class_name="hud_region_candidate",
            bbox=BoundingBox(0, 0, region_width, region_height),
            confidence=confidence,
            source="upper_left_magenta_cue",
            frame_number=context.frame_number if context else None,
            timestamp=context.timestamp.isoformat() if context else None,
            limitation=limitation,
        )
        return DetectionResult(
            (detection,),
            method="upper-left magenta HUD-cue heuristic",
            limitation=limitation,
            status="detected",
        )
