import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CaptureMetadata:
    timestamp: str
    frame_number: int
    capture_fps: float | None
    monitor_index: int
    monitor_bounds: dict[str, int]
    capture_region: dict[str, int]
    frame_resolution: dict[str, int]


def save_sample_frame(
    frame: NDArray[np.uint8],
    image_path: Path,
    *,
    frame_number: int,
    capture_fps: float | None,
    monitor_index: int,
    monitor_bounds: dict[str, int],
    capture_region: dict[str, int],
    timestamp: datetime | None = None,
) -> tuple[Path, Path]:
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Sample frame must be a three-channel image.")
    if frame_number < 1:
        raise ValueError("Frame number must be at least 1.")
    if capture_fps is not None and (
        not isfinite(capture_fps) or capture_fps < 0
    ):
        raise ValueError("Capture FPS must be finite and non-negative.")
    if monitor_index < 1:
        raise ValueError("Monitor index must be 1 or greater.")

    image_path = Path(image_path)
    timestamp = timestamp or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        raise ValueError("Capture timestamp must include timezone information.")

    metadata_path = image_path.with_suffix(".json")
    image_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(image_path), frame):
        raise OSError(f"Could not save captured frame to {image_path}.")

    metadata = CaptureMetadata(
        timestamp=timestamp.astimezone(timezone.utc).isoformat(),
        frame_number=frame_number,
        capture_fps=capture_fps,
        monitor_index=monitor_index,
        monitor_bounds=monitor_bounds,
        capture_region=capture_region,
        frame_resolution={"width": int(frame.shape[1]), "height": int(frame.shape[0])},
    )
    try:
        metadata_path.write_text(
            json.dumps(asdict(metadata), indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError:
        image_path.unlink(missing_ok=True)
        raise

    return image_path, metadata_path


def update_sample_capture_fps(metadata_path: Path, capture_fps: float) -> None:
    if not isfinite(capture_fps) or capture_fps < 0:
        raise ValueError("Capture FPS must be finite and non-negative.")

    metadata: dict[str, Any] = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["capture_fps"] = capture_fps
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
