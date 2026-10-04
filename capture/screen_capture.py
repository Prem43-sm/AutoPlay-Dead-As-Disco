import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from mss import mss
from numpy.typing import NDArray

from app.config import CaptureConfig
from app.logger import get_logger
from capture.frame_metadata import save_sample_frame, update_sample_capture_fps

Frame = NDArray[np.uint8]


class ScreenCapture:
    def __init__(self, config: CaptureConfig) -> None:
        config.validate()
        self._config = config
        self._capture = mss()
        monitors = self._capture.monitors
        if config.monitor_index >= len(monitors):
            self._capture.close()
            raise ValueError(
                f"Monitor {config.monitor_index} is unavailable; "
                f"choose an index from 1 to {len(monitors) - 1}."
            )

        self._monitor = {
            key: int(monitors[config.monitor_index][key])
            for key in ("left", "top", "width", "height")
        }
        try:
            self._bounds = self._make_capture_bounds()
        except ValueError:
            self._capture.close()
            raise

    @property
    def bounds(self) -> dict[str, int]:
        return self._bounds.copy()

    @property
    def monitor_bounds(self) -> dict[str, int]:
        return self._monitor.copy()

    def _make_capture_bounds(self) -> dict[str, int]:
        region = self._config.region
        if region is None:
            return self._monitor.copy()

        monitor_width = self._monitor["width"]
        monitor_height = self._monitor["height"]
        if (
            region.left + region.width > monitor_width
            or region.top + region.height > monitor_height
        ):
            raise ValueError(
                "Capture region must fit within the selected monitor "
                f"({monitor_width}x{monitor_height})."
            )

        return {
            "left": self._monitor["left"] + region.left,
            "top": self._monitor["top"] + region.top,
            "width": region.width,
            "height": region.height,
        }

    def capture(self) -> Frame:
        screenshot = self._capture.grab(self._bounds)
        return np.asarray(screenshot, dtype=np.uint8)[..., :3].copy()

    def close(self) -> None:
        self._capture.close()

    def __enter__(self) -> "ScreenCapture":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()


def run_live_capture(
    config: CaptureConfig,
    save_frame: Path | None = None,
    output_dir: Path = Path("recordings/samples"),
    save_first_frame: bool = False,
    max_frames: int | None = None,
    show_preview: bool = True,
) -> None:
    config.validate()
    if max_frames is not None and max_frames < 1:
        raise ValueError("Maximum frame count must be at least 1.")
    if save_frame is not None and save_first_frame:
        raise ValueError("Use either save_frame or save_first_frame, not both.")

    logger = get_logger(__name__)
    frame_interval = 1.0 / config.max_fps
    frame_number = 0
    first_frame_time: float | None = None
    measured_fps: float | None = None
    saved_metadata_path: Path | None = None
    window_name = "Dead As Disco AI - Screen Capture"

    try:
        with ScreenCapture(config) as capture:
            logger.info(
                "Capturing monitor %d, region %s (maximum %.1f FPS).",
                config.monitor_index,
                capture.bounds,
                config.max_fps,
            )
            while True:
                frame_start = time.perf_counter()
                frame = capture.capture()
                timestamp = datetime.now(timezone.utc)
                frame_number += 1
                frame_time = time.perf_counter()
                if first_frame_time is None:
                    first_frame_time = frame_time
                elif frame_number > 1:
                    measured_fps = (frame_number - 1) / (
                        frame_time - first_frame_time
                    )

                if (save_frame is not None or save_first_frame) and frame_number == 1:
                    image_path = (
                        save_frame
                        if save_frame is not None
                        else output_dir
                        / (
                            f"frame_{frame_number:06d}_"
                            f"{timestamp.strftime('%Y%m%dT%H%M%S_%fZ')}.png"
                        )
                    )
                    saved_image, saved_metadata = save_sample_frame(
                        frame,
                        image_path,
                        frame_number=frame_number,
                        capture_fps=measured_fps,
                        monitor_index=config.monitor_index,
                        monitor_bounds=capture.monitor_bounds,
                        capture_region=capture.bounds,
                        timestamp=timestamp,
                    )
                    saved_metadata_path = saved_metadata
                    logger.info(
                        "Saved sample frame %d: %s (metadata: %s; %dx%d; capture FPS: %s).",
                        frame_number,
                        saved_image,
                        saved_metadata,
                        frame.shape[1],
                        frame.shape[0],
                        f"{measured_fps:.2f}" if measured_fps is not None else "pending",
                    )

                if show_preview:
                    preview = frame.copy()
                    cv2.putText(
                        preview,
                        (
                            f"Capture: {measured_fps:.1f} FPS"
                            if measured_fps is not None
                            else "Capture: measuring FPS"
                        ),
                        (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow(window_name, preview)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break

                if max_frames is not None and frame_number >= max_frames:
                    break

                remaining = frame_interval - (time.perf_counter() - frame_start)
                if remaining > 0:
                    time.sleep(remaining)
    finally:
        try:
            if first_frame_time is not None and frame_number > 1:
                elapsed = time.perf_counter() - first_frame_time
                measured_fps = (frame_number - 1) / elapsed
                logger.info(
                    "Capture stopped after %d frames; average capture FPS: %.2f.",
                    frame_number,
                    measured_fps,
                )
                if saved_metadata_path is not None:
                    update_sample_capture_fps(saved_metadata_path, measured_fps)
            elif frame_number == 1:
                logger.info(
                    "Capture stopped after one frame; capture FPS is unavailable "
                    "until at least two frames are captured."
                )
        finally:
            if show_preview:
                cv2.destroyAllWindows()
