import numpy as np
from numpy.typing import NDArray

from vision.models import DetectionResult, FrameContext


class EnemyDetector:
    """Conservative placeholder until an enemy appearance model is validated."""

    def detect(
        self, frame: NDArray[np.uint8], context: FrameContext | None = None
    ) -> DetectionResult:
        if frame.ndim != 3 or frame.shape[2] < 3:
            raise ValueError("Enemy detection expects a color image.")
        frame_reference = (
            f" in frame {context.frame_number}"
            if context is not None and context.frame_number is not None
            else ""
        )
        return DetectionResult(
            detections=(),
            method="interface only; no validated enemy appearance model",
            limitation=(
                f"Enemy is visible{frame_reference}, but its appearance is not "
                "distinct enough for a reliable hand-written detector. "
                "NOT RELIABLY DETECTED YET."
            ),
            status="not_reliably_detected",
        )
