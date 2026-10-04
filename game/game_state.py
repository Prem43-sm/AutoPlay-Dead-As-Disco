from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from vision.models import BoundingBox, Detection


class GameStatus(str, Enum):
    UNKNOWN = "UNKNOWN"
    MENU = "MENU"
    PLAYING = "PLAYING"
    DEAD = "DEAD"
    PAUSED = "PAUSED"


@dataclass(frozen=True)
class PlayerState:
    detected: bool
    bbox: BoundingBox | None = None
    center: tuple[float, float] | None = None
    confidence: float | None = None
    limitation: str | None = None


@dataclass(frozen=True)
class EnemiesState:
    count: int | None
    detections: tuple[Detection, ...] = ()
    centers: tuple[tuple[float, float], ...] = ()
    confidence: float | None = None
    limitation: str | None = None


@dataclass(frozen=True)
class HudState:
    visible: bool | None = None
    confidence: float | None = None
    region: BoundingBox | None = None
    health: int | None = None
    fever: float | None = None
    takedown_available: bool | None = None
    score: int | None = None
    limitation: str | None = None


@dataclass(frozen=True)
class PromptState:
    visible: bool | None = None
    key: str | None = None
    bbox: BoundingBox | None = None
    confidence: float | None = None
    limitation: str | None = None


@dataclass(frozen=True)
class GameState:
    timestamp: str
    frame_number: int | None
    capture_fps: float | None
    player: PlayerState
    enemies: EnemiesState
    hud: HudState
    prompt: PromptState
    game_status: GameStatus = GameStatus.UNKNOWN
    detections: tuple[Detection, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
