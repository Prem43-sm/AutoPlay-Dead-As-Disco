import cv2
import numpy as np
from numpy.typing import NDArray

from vision.models import BoundingBox, Detection, DetectionResult, FrameContext

Frame = NDArray[np.uint8]


class PlayerDetector:
    """Find a visible yellow-clothing candidate; this is not full-body tracking."""

    def __init__(self, hue_range: tuple[int, int] = (15, 40)) -> None:
        self._hue_range = hue_range

    def detect(
        self, frame: Frame, context: FrameContext | None = None
    ) -> DetectionResult:
        if frame.ndim != 3 or frame.shape[2] < 3:
            raise ValueError("Player detection expects a color image.")

        height, width = frame.shape[:2]
        hsv = cv2.cvtColor(frame[..., :3], cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array([self._hue_range[0], 70, 80], dtype=np.uint8),
            np.array([self._hue_range[1], 255, 255], dtype=np.uint8),
        )
        kernel_size = max(3, int(round(min(height, width) * 0.005)) | 1)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        min_area = max(20, int(height * width * 0.0002))
        candidates = [
            (int(stats[index, cv2.CC_STAT_AREA]), index)
            for index in range(1, count)
            if stats[index, cv2.CC_STAT_AREA] >= min_area
        ]
        if not candidates:
            return DetectionResult(
                (),
                method="HSV yellow-clothing connected-component heuristic",
                limitation="No component matched the current yellow appearance heuristic.",
                status="not_reliably_detected",
            )

        area, index = max(candidates)
        x = int(stats[index, cv2.CC_STAT_LEFT])
        y = int(stats[index, cv2.CC_STAT_TOP])
        box_width = int(stats[index, cv2.CC_STAT_WIDTH])
        box_height = int(stats[index, cv2.CC_STAT_HEIGHT])
        area_fraction = area / (height * width)
        confidence = min(0.65, 0.35 + area_fraction / 0.01)
        limitation = (
            "Yellow garment candidate only; bounding box does not represent "
            "the full player silhouette. Confidence is heuristic, not calibrated."
        )
        detection = Detection(
            class_name="player",
            bbox=BoundingBox(x, y, box_width, box_height),
            confidence=confidence,
            source="hsv_yellow_connected_component",
            frame_number=context.frame_number if context else None,
            timestamp=context.timestamp.isoformat() if context else None,
            limitation=limitation,
        )
        return DetectionResult(
            (detection,),
            method="HSV yellow-clothing connected-component heuristic",
            limitation=limitation,
            status="detected",
        )
