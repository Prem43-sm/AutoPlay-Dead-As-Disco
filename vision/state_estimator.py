from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from game.game_state import GameState
from vision.detector import VisionDetector
from vision.models import FrameContext


def load_frame_context(image_path: Path) -> FrameContext:
    metadata_path = image_path.with_suffix(".json")
    if not metadata_path.is_file():
        return FrameContext(timestamp=datetime.now(timezone.utc), frame_number=1)

    import json

    metadata: dict[str, Any] = json.loads(metadata_path.read_text(encoding="utf-8"))
    return FrameContext(
        timestamp=datetime.fromisoformat(metadata["timestamp"]),
        frame_number=metadata.get("frame_number"),
        capture_fps=metadata.get("capture_fps"),
    )


def estimate_game_state(
    frame: NDArray[np.uint8],
    context: FrameContext | None = None,
    detector: VisionDetector | None = None,
) -> GameState:
    return (detector or VisionDetector()).process(frame, context)


def draw_debug_frame(
    frame: NDArray[np.uint8], game_state: GameState
) -> NDArray[np.uint8]:
    debug_frame = frame[..., :3].copy()
    colors = {
        "player": (0, 255, 255),
        "hud_region_candidate": (255, 0, 255),
    }
    for detection in game_state.detections:
        box = detection.bbox.clip(debug_frame.shape[1], debug_frame.shape[0])
        if box is None:
            continue
        color = colors.get(detection.class_name, (0, 165, 255))
        cv2.rectangle(
            debug_frame,
            (box.x, box.y),
            (box.x + box.width, box.y + box.height),
            color,
            2,
        )
        label = f"{detection.class_name} {detection.confidence:.2f}"
        if detection.limitation:
            label += " (heuristic)"
        cv2.putText(
            debug_frame,
            label[:100],
            (box.x, max(18, box.y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    fps_text = (
        f"{game_state.capture_fps:.2f} FPS"
        if game_state.capture_fps is not None
        else "FPS unknown"
    )
    cv2.putText(
        debug_frame,
        f"Player: {'candidate' if game_state.player.detected else 'unknown'} | "
        f"Enemies: {game_state.enemies.count if game_state.enemies.count is not None else 'unknown'} | "
        f"HUD: {game_state.hud.visible} | Prompt: {game_state.prompt.key or 'unknown'} | {fps_text}",
        (10, max(22, debug_frame.shape[0] - 14)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return debug_frame


def save_vision_debug(
    frame: NDArray[np.uint8],
    game_state: GameState,
    output_dir: Path,
) -> tuple[Path, Path]:
    import json

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_number = game_state.frame_number or 1
    timestamp = datetime.fromisoformat(game_state.timestamp).strftime(
        "%Y%m%dT%H%M%S_%f"
    )
    stem = f"frame_{frame_number:06d}_{timestamp}"
    image_path = output_dir / f"{stem}.png"
    state_path = output_dir / f"{stem}.json"
    if not cv2.imwrite(str(image_path), draw_debug_frame(frame, game_state)):
        raise OSError(f"Could not save vision debug frame to {image_path}.")
    try:
        state_path.write_text(
            json.dumps(game_state.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError:
        image_path.unlink(missing_ok=True)
        raise
    return image_path, state_path
