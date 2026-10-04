import re
from collections.abc import Callable
from importlib import import_module

import numpy as np
from numpy.typing import NDArray

import cv2

from vision.models import BoundingBox, Detection, DetectionResult, FrameContext

OCRReader = Callable[[NDArray[np.uint8]], list[tuple[str, BoundingBox, float]]]


def pytesseract_reader(
    frame: NDArray[np.uint8],
) -> list[tuple[str, BoundingBox, float]]:
    try:
        pytesseract = import_module("pytesseract")
    except ImportError as exc:
        raise RuntimeError(
            "Optional OCR requires pytesseract and the Tesseract OCR executable."
        ) from exc

    rgb_frame = cv2.cvtColor(frame[..., :3], cv2.COLOR_BGR2RGB)
    words = pytesseract.image_to_data(
        rgb_frame, output_type=pytesseract.Output.DICT
    )
    readings: list[tuple[str, BoundingBox, float]] = []
    for index, text in enumerate(words["text"]):
        text = text.strip()
        confidence = float(words["conf"][index])
        box_width = int(words["width"][index])
        box_height = int(words["height"][index])
        if not text or confidence < 0 or box_width <= 0 or box_height <= 0:
            continue
        readings.append(
            (
                text,
                BoundingBox(
                    int(words["left"][index]),
                    int(words["top"][index]),
                    box_width,
                    box_height,
                ),
                confidence / 100,
            )
        )
    return readings


class PromptDetector:
    """Optional OCR-backed prompt reader; no OCR package is required by default."""

    def __init__(self, ocr_reader: OCRReader | None = None) -> None:
        self._ocr_reader = ocr_reader

    def detect(
        self, frame: NDArray[np.uint8], context: FrameContext | None = None
    ) -> DetectionResult:
        if frame.ndim != 3 or frame.shape[2] < 3:
            raise ValueError("Prompt detection expects a color image.")
        if self._ocr_reader is None:
            return DetectionResult(
                (),
                method="OCR disabled",
                limitation=(
                    "No OCR backend is configured; action prompts are unknown. "
                    "The sample prompt is partially clipped."
                ),
                status="not_reliably_detected",
            )

        detections: list[Detection] = []
        for text, bbox, confidence in self._ocr_reader(frame):
            match = re.search(r"(?<![A-Z0-9])([EF])(?![A-Z0-9])", text.upper())
            if match is None or confidence < 0.5:
                continue
            detections.append(
                Detection(
                    class_name=f"prompt:{match.group(1)}",
                    bbox=bbox,
                    confidence=confidence,
                    source="optional_ocr",
                    frame_number=context.frame_number if context else None,
                    timestamp=context.timestamp.isoformat() if context else None,
                    limitation="OCR key label only; the F action meaning is contextual.",
                )
            )

        status = "detected" if detections else "not_detected"
        return DetectionResult(
            tuple(detections),
            method="optional OCR reader",
            limitation=(
                "OCR key label only; the F action meaning is contextual."
                if detections
                else "OCR reader found no standalone E or F prompt."
            ),
            status=status,
        )
