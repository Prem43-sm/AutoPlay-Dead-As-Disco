import base64
import json
import re
import tempfile
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from data.dataset_tools import (
    CLASS_NAMES,
    SPLITS,
    YOLO_CLASS_NAMES,
    YoloLabel,
    parse_yolo_labels,
    validate_yolo_label,
)

Frame = NDArray[np.uint8]
LABELING_STATE_FILENAME = "labeling_state.json"
CANVAS_MAX_WIDTH = 1400
CANVAS_MAX_HEIGHT = 820


@dataclass(frozen=True)
class DatasetImage:
    path: Path
    split: str
    session_id: str
    label_path: Path


@dataclass(frozen=True)
class LabelingProgress:
    reviewed: int
    labeled: int
    skipped: int
    remaining: int
    total: int


def pixel_box_to_yolo(
    class_id: int,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    image_width: int,
    image_height: int,
) -> YoloLabel:
    if class_id not in YOLO_CLASS_NAMES:
        raise ValueError(f"Unknown class ID: {class_id}.")
    if image_width < 1 or image_height < 1:
        raise ValueError("Image dimensions must be positive.")
    if not (0 <= x1 < x2 <= image_width and 0 <= y1 < y2 <= image_height):
        raise ValueError("Bounding box must be non-empty and inside the image.")
    return YoloLabel(
        class_id=class_id,
        x_center=((x1 + x2) / 2) / image_width,
        y_center=((y1 + y2) / 2) / image_height,
        width=(x2 - x1) / image_width,
        height=(y2 - y1) / image_height,
    )


def yolo_to_pixel_box(
    label: YoloLabel, image_width: int, image_height: int
) -> tuple[int, int, int, int]:
    error = validate_yolo_label(
        label, image_width, image_height, YOLO_CLASS_NAMES
    )
    if error is not None:
        raise ValueError(error)
    x1 = round((label.x_center - label.width / 2) * image_width)
    y1 = round((label.y_center - label.height / 2) * image_height)
    x2 = round((label.x_center + label.width / 2) * image_width)
    y2 = round((label.y_center + label.height / 2) * image_height)
    return x1, y1, x2, y2


def write_yolo_annotations(
    label_path: Path,
    labels: list[YoloLabel],
    image_width: int,
    image_height: int,
    *,
    allow_overwrite: bool = False,
) -> None:
    label_path = Path(label_path)
    if label_path.exists() and not allow_overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing annotation without explicit "
            f"permission: {label_path}"
        )
    lines: list[str] = []
    for label in labels:
        error = validate_yolo_label(
            label, image_width, image_height, YOLO_CLASS_NAMES
        )
        if error is not None:
            raise ValueError(f"Invalid annotation for {label_path}: {error}")
        lines.append(
            f"{label.class_id} {label.x_center:.8f} {label.y_center:.8f} "
            f"{label.width:.8f} {label.height:.8f}"
        )

    label_path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines) + ("\n" if lines else "")
    if not allow_overwrite:
        created = False
        try:
            with label_path.open("x", encoding="utf-8", newline="\n") as label_file:
                created = True
                label_file.write(content)
        except FileExistsError as exc:
            raise FileExistsError(
                f"Refusing to overwrite existing annotation without explicit "
                f"permission: {label_path}"
            ) from exc
        except Exception:
            if created:
                label_path.unlink(missing_ok=True)
            raise
        return

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=label_path.parent,
            prefix=f".{label_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(content)
        temporary_path.replace(label_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def read_yolo_annotations(
    label_path: Path, image_width: int, image_height: int
) -> list[YoloLabel]:
    labels = parse_yolo_labels(label_path)
    for label in labels:
        error = validate_yolo_label(
            label, image_width, image_height, YOLO_CLASS_NAMES
        )
        if error is not None:
            raise ValueError(f"Invalid annotation in {label_path}: {error}")
    return labels


def _image_session_id(image_path: Path) -> str:
    match = re.match(r"^(?P<session>.+)_frame_\d{6}_", image_path.name)
    return match.group("session") if match else image_path.stem


def list_dataset_images(
    dataset_dir: Path,
    *,
    split: str = "all",
    session_id: str | None = None,
) -> list[DatasetImage]:
    dataset_dir = Path(dataset_dir)
    if split != "all" and split not in SPLITS:
        raise ValueError(f"Split must be 'all' or one of: {', '.join(SPLITS)}.")
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    records: list[DatasetImage] = []
    splits = SPLITS if split == "all" else (split,)
    for dataset_split in splits:
        image_dir = dataset_dir / "images" / dataset_split
        label_dir = dataset_dir / "labels" / dataset_split
        if not image_dir.exists():
            continue
        for image_path in sorted(image_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in {
                ".png",
                ".jpg",
                ".jpeg",
                ".bmp",
            }:
                continue
            record = DatasetImage(
                path=image_path,
                split=dataset_split,
                session_id=_image_session_id(image_path),
                label_path=label_dir / f"{image_path.stem}.txt",
            )
            if session_id is None or record.session_id == session_id:
                records.append(record)
    return records


def load_labeling_state(dataset_dir: Path) -> dict[str, str]:
    state_path = Path(dataset_dir) / "metadata" / LABELING_STATE_FILENAME
    if not state_path.exists():
        return {}
    data = json.loads(state_path.read_text(encoding="utf-8"))
    statuses = data.get("images", {})
    if not isinstance(statuses, dict):
        raise ValueError(f"Invalid labeling progress file: {state_path}")
    return {
        str(image_path): status
        for image_path, status in statuses.items()
        if status in {"reviewed", "skipped"}
    }


def save_labeling_state(dataset_dir: Path, statuses: dict[str, str]) -> None:
    for image_path, status in statuses.items():
        if status not in {"reviewed", "skipped"}:
            raise ValueError(f"Invalid labeling status for {image_path}: {status}")
    state_path = Path(dataset_dir) / "metadata" / LABELING_STATE_FILENAME
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps({"images": statuses}, indent=2) + "\n",
        encoding="utf-8",
    )


def get_labeling_progress(
    dataset_dir: Path, records: list[DatasetImage] | None = None
) -> LabelingProgress:
    all_records = records if records is not None else list_dataset_images(dataset_dir)
    statuses = load_labeling_state(dataset_dir)
    reviewed_paths = {
        record.path.relative_to(dataset_dir).as_posix()
        for record in all_records
        if record.label_path.is_file()
        or statuses.get(record.path.relative_to(dataset_dir).as_posix()) == "reviewed"
    }
    skipped_paths = {
        record.path.relative_to(dataset_dir).as_posix()
        for record in all_records
        if statuses.get(record.path.relative_to(dataset_dir).as_posix()) == "skipped"
        and not record.label_path.is_file()
    }
    labeled_count = 0
    for record in all_records:
        if record.label_path.is_file():
            width, height = _read_image_size(record.path)
            if read_yolo_annotations(record.label_path, width, height):
                labeled_count += 1
    remaining = len(all_records) - len(reviewed_paths | skipped_paths)
    return LabelingProgress(
        reviewed=len(reviewed_paths),
        labeled=labeled_count,
        skipped=len(skipped_paths),
        remaining=max(0, remaining),
        total=len(all_records),
    )


def build_labeling_report(dataset_dir: Path) -> dict[str, Any]:
    records = list_dataset_images(dataset_dir)
    statuses = load_labeling_state(dataset_dir)
    class_counts = {name: 0 for name in CLASS_NAMES}
    reviewed_paths: set[str] = set()
    skipped_paths: set[str] = set()
    labeled_images = 0
    empty_reviewed_images = 0
    total_boxes = 0

    for record in records:
        relative_path = record.path.relative_to(dataset_dir).as_posix()
        if statuses.get(relative_path) == "skipped" and not record.label_path.is_file():
            skipped_paths.add(relative_path)
        if not record.label_path.is_file():
            continue
        width, height = _read_image_size(record.path)
        labels = read_yolo_annotations(record.label_path, width, height)
        reviewed_paths.add(relative_path)
        if labels:
            labeled_images += 1
        else:
            empty_reviewed_images += 1
        total_boxes += len(labels)
        for label in labels:
            class_counts[CLASS_NAMES[label.class_id]] += 1

    reviewed_paths.update(
        record.path.relative_to(dataset_dir).as_posix()
        for record in records
        if statuses.get(record.path.relative_to(dataset_dir).as_posix()) == "reviewed"
    )
    return {
        "images_total": len(records),
        "images_reviewed": len(reviewed_paths),
        "images_with_labels": labeled_images,
        "images_without_target_objects": empty_reviewed_images,
        "images_skipped": len(skipped_paths),
        "images_remaining": max(
            0, len(records) - len(reviewed_paths | skipped_paths)
        ),
        "total_bounding_boxes": total_boxes,
        "class_counts": class_counts,
    }


def _read_image_size(image_path: Path) -> tuple[int, int]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"Could not read image: {image_path}")
    return int(image.shape[1]), int(image.shape[0])


class DatasetLabeler:
    def __init__(
        self,
        dataset_dir: Path,
        *,
        split: str = "all",
        session_id: str | None = None,
        root: tk.Tk | None = None,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.records = list_dataset_images(
            self.dataset_dir, split=split, session_id=session_id
        )
        if not self.records:
            raise ValueError("No images match the requested dataset filters.")
        self.root = root or tk.Tk()
        self.root.title("Dead As Disco AI - Manual Dataset Labeler")
        self.root.geometry("1500x1000")
        self.root.minsize(900, 650)
        self.statuses = load_labeling_state(self.dataset_dir)
        self.index = 0
        self.selected_class = tk.IntVar(value=0)
        self.selected_box: int | None = None
        self.labels: list[YoloLabel] = []
        self.image: Frame | None = None
        self.image_width = 0
        self.image_height = 0
        self.dirty = False
        self.scale = 1.0
        self.display_width = 0
        self.display_height = 0
        self.offset_x = 0
        self.offset_y = 0
        self.photo: tk.PhotoImage | None = None
        self._drag_start: tuple[int, int] | None = None
        self._drag_rectangle: int | None = None
        self._build_ui()
        self._load_current_image()

    def _build_ui(self) -> None:
        toolbar = ttk.Frame(self.root, padding=6)
        toolbar.pack(fill=tk.X)
        ttk.Button(toolbar, text="Previous", command=self.previous_image).pack(
            side=tk.LEFT
        )
        ttk.Button(toolbar, text="Next", command=self.next_image).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Save", command=self.save_current).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        ttk.Button(toolbar, text="Mark reviewed", command=self.mark_reviewed).pack(
            side=tk.LEFT
        )
        ttk.Button(toolbar, text="Skip image", command=self.skip_image).pack(
            side=tk.LEFT
        )
        ttk.Button(toolbar, text="Delete selected box", command=self.delete_selected).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        ttk.Button(toolbar, text="Clear boxes", command=self.clear_boxes).pack(
            side=tk.LEFT
        )

        class_frame = ttk.LabelFrame(self.root, text="Class", padding=6)
        class_frame.pack(fill=tk.X, padx=8)
        for class_id, class_name in enumerate(CLASS_NAMES):
            ttk.Radiobutton(
                class_frame,
                text=f"{class_id}: {class_name}",
                variable=self.selected_class,
                value=class_id,
                command=self._update_status,
            ).pack(side=tk.LEFT, padx=5)

        body = ttk.Frame(self.root)
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.canvas = tk.Canvas(body, background="#202020", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sidebar = ttk.Frame(body, width=245, padding=8)
        sidebar.pack(side=tk.RIGHT, fill=tk.Y)
        sidebar.pack_propagate(False)
        self.image_info = ttk.Label(sidebar, text="", wraplength=225)
        self.image_info.pack(anchor=tk.W, fill=tk.X)
        self.progress_info = ttk.Label(sidebar, text="", justify=tk.LEFT)
        self.progress_info.pack(anchor=tk.W, fill=tk.X, pady=(10, 12))
        ttk.Label(sidebar, text="Boxes (select one to delete):").pack(anchor=tk.W)
        self.box_list = tk.Listbox(sidebar, height=18, exportselection=False)
        self.box_list.pack(fill=tk.X, pady=(3, 6))
        self.box_list.bind("<<ListboxSelect>>", self._select_box_from_list)
        ttk.Label(
            sidebar,
            text="Draw: left-drag on image\n"
            "Delete: select a box and press Delete\n"
            "Navigate: Previous / Next\n"
            "Saving an existing label asks before overwrite.",
            justify=tk.LEFT,
            wraplength=225,
        ).pack(anchor=tk.W, pady=(8, 0))

        self.canvas.bind("<ButtonPress-1>", self._mouse_down)
        self.canvas.bind("<B1-Motion>", self._mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._mouse_up)
        self.canvas.bind("<Configure>", self._canvas_resized)
        self.root.bind("<Delete>", lambda _event: self.delete_selected())
        self.root.bind("<Left>", lambda _event: self.previous_image())
        self.root.bind("<Right>", lambda _event: self.next_image())
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self._update_progress()

    def run(self) -> None:
        self.root.mainloop()

    def _record_key(self, record: DatasetImage | None = None) -> str:
        selected = record or self.records[self.index]
        return selected.path.relative_to(self.dataset_dir).as_posix()

    def _load_current_image(self) -> None:
        record = self.records[self.index]
        image = cv2.imread(str(record.path), cv2.IMREAD_COLOR)
        if image is None:
            messagebox.showerror("Image error", f"Could not read {record.path}")
            return
        self.image = image
        self.image_height, self.image_width = image.shape[:2]
        try:
            self.labels = (
                read_yolo_annotations(
                    record.label_path, self.image_width, self.image_height
                )
                if record.label_path.is_file()
                else []
            )
        except (OSError, ValueError) as exc:
            messagebox.showerror("Annotation error", str(exc))
            self.labels = []
        self.dirty = False
        self.selected_box = None
        self._render_image()
        self._update_status()
        self._update_progress()

    def _render_image(self) -> None:
        if self.image is None:
            return
        canvas_width = max(self.canvas.winfo_width(), 600)
        canvas_height = max(self.canvas.winfo_height(), 400)
        fit_scale = min(
            CANVAS_MAX_WIDTH / self.image_width,
            CANVAS_MAX_HEIGHT / self.image_height,
            canvas_width / self.image_width,
            canvas_height / self.image_height,
            1.0,
        )
        self.scale = max(fit_scale, 0.01)
        self.display_width = max(1, round(self.image_width * self.scale))
        self.display_height = max(1, round(self.image_height * self.scale))
        self.offset_x = max(0, (canvas_width - self.display_width) // 2)
        self.offset_y = max(0, (canvas_height - self.display_height) // 2)
        resized = cv2.resize(
            self.image,
            (self.display_width, self.display_height),
            interpolation=cv2.INTER_AREA if self.scale < 1 else cv2.INTER_LINEAR,
        )
        encoded, buffer = cv2.imencode(".png", resized)
        if not encoded:
            messagebox.showerror("Image error", "Could not render selected image.")
            return
        self.photo = tk.PhotoImage(data=base64.b64encode(buffer).decode("ascii"))
        self.canvas.delete("all")
        self.canvas.create_image(
            self.offset_x,
            self.offset_y,
            image=self.photo,
            anchor=tk.NW,
            tags=("image",),
        )
        for index, label in enumerate(self.labels):
            self._draw_label(index, label)
        self.canvas.configure(scrollregion=(0, 0, canvas_width, canvas_height))
        self._refresh_box_list()

    def _draw_label(self, index: int, label: YoloLabel) -> None:
        x1, y1, x2, y2 = yolo_to_pixel_box(
            label, self.image_width, self.image_height
        )
        x1 = self.offset_x + round(x1 * self.scale)
        y1 = self.offset_y + round(y1 * self.scale)
        x2 = self.offset_x + round(x2 * self.scale)
        y2 = self.offset_y + round(y2 * self.scale)
        selected = index == self.selected_box
        color = "#ff453a" if selected else "#00e676"
        self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=3)
        self.canvas.create_text(
            x1 + 4,
            y1 + 4,
            text=f"{label.class_id}: {CLASS_NAMES[label.class_id]}",
            fill="white",
            anchor=tk.NW,
            font=("Segoe UI", 10, "bold"),
        )

    def _canvas_resized(self, _event: tk.Event[Any]) -> None:
        self._render_image()

    def _canvas_to_image(self, x: int, y: int) -> tuple[int, int] | None:
        if (
            x < self.offset_x
            or y < self.offset_y
            or x >= self.offset_x + self.display_width
            or y >= self.offset_y + self.display_height
        ):
            return None
        image_x = min(self.image_width, max(0, round((x - self.offset_x) / self.scale)))
        image_y = min(self.image_height, max(0, round((y - self.offset_y) / self.scale)))
        return image_x, image_y

    def _mouse_down(self, event: tk.Event[Any]) -> None:
        point = self._canvas_to_image(event.x, event.y)
        if point is None:
            return
        self._drag_start = point
        x = self.offset_x + round(point[0] * self.scale)
        y = self.offset_y + round(point[1] * self.scale)
        self._drag_rectangle = self.canvas.create_rectangle(
            x, y, x, y, outline="#00e5ff", width=2
        )

    def _mouse_drag(self, event: tk.Event[Any]) -> None:
        if self._drag_start is None or self._drag_rectangle is None:
            return
        point = self._canvas_to_image(event.x, event.y)
        if point is None:
            point = (
                min(self.image_width, max(0, round((event.x - self.offset_x) / self.scale))),
                min(self.image_height, max(0, round((event.y - self.offset_y) / self.scale))),
            )
        x1, y1 = self._drag_start
        self.canvas.coords(
            self._drag_rectangle,
            self.offset_x + round(min(x1, point[0]) * self.scale),
            self.offset_y + round(min(y1, point[1]) * self.scale),
            self.offset_x + round(max(x1, point[0]) * self.scale),
            self.offset_y + round(max(y1, point[1]) * self.scale),
        )

    def _mouse_up(self, event: tk.Event[Any]) -> None:
        if self._drag_start is None:
            return
        start = self._drag_start
        end = self._canvas_to_image(event.x, event.y)
        if end is None:
            end = (
                min(self.image_width, max(0, round((event.x - self.offset_x) / self.scale))),
                min(self.image_height, max(0, round((event.y - self.offset_y) / self.scale))),
            )
        if self._drag_rectangle is not None:
            self.canvas.delete(self._drag_rectangle)
        self._drag_start = None
        self._drag_rectangle = None
        x1, x2 = sorted((start[0], end[0]))
        y1, y2 = sorted((start[1], end[1]))
        if x1 == x2 or y1 == y2:
            return
        self.labels.append(
            pixel_box_to_yolo(
                self.selected_class.get(),
                x1,
                y1,
                x2,
                y2,
                self.image_width,
                self.image_height,
            )
        )
        self.selected_box = len(self.labels) - 1
        self.dirty = True
        self._render_image()

    def _select_box_from_list(self, _event: tk.Event[Any]) -> None:
        selected = self.box_list.curselection()
        self.selected_box = selected[0] if selected else None
        self._render_image()

    def _refresh_box_list(self) -> None:
        self.box_list.delete(0, tk.END)
        for index, label in enumerate(self.labels):
            self.box_list.insert(
                tk.END,
                f"{index + 1}. {label.class_id}: {CLASS_NAMES[label.class_id]}",
            )
        if self.selected_box is not None and self.selected_box < len(self.labels):
            self.box_list.selection_set(self.selected_box)

    def _update_status(self) -> None:
        record = self.records[self.index]
        status = self.statuses.get(self._record_key())
        if record.label_path.is_file():
            status_text = "label file exists (reviewed)"
        elif status == "skipped":
            status_text = "skipped"
        elif status == "reviewed":
            status_text = "reviewed"
        else:
            status_text = "unreviewed"
        self.image_info.configure(
            text=(
                f"{self.index + 1} / {len(self.records)}\n"
                f"Split: {record.split}\nSession: {record.session_id}\n"
                f"Status: {status_text}\n\n{record.path.name}"
            )
        )

    def _update_progress(self) -> None:
        progress = get_labeling_progress(self.dataset_dir, self.records)
        self.progress_info.configure(
            text=(
                f"Progress ({progress.total} filtered images)\n"
                f"Reviewed: {progress.reviewed}\n"
                f"Labeled: {progress.labeled}\n"
                f"Skipped: {progress.skipped}\n"
                f"Remaining: {progress.remaining}"
            )
        )

    def _persist_status(self, status: str) -> None:
        self.statuses[self._record_key()] = status
        save_labeling_state(self.dataset_dir, self.statuses)

    def save_current(self) -> None:
        record = self.records[self.index]
        overwrite = False
        if record.label_path.exists():
            overwrite = messagebox.askyesno(
                "Confirm annotation overwrite",
                f"Replace the existing annotations for:\n{record.path.name}?",
                parent=self.root,
            )
            if not overwrite:
                return
        try:
            write_yolo_annotations(
                record.label_path,
                self.labels,
                self.image_width,
                self.image_height,
                allow_overwrite=overwrite,
            )
            self._persist_status("reviewed")
        except (OSError, ValueError) as exc:
            messagebox.showerror("Save failed", str(exc), parent=self.root)
            return
        self.dirty = False
        self._update_status()
        self._update_progress()

    def mark_reviewed(self) -> None:
        record = self.records[self.index]
        if record.label_path.exists():
            try:
                existing = read_yolo_annotations(
                    record.label_path, self.image_width, self.image_height
                )
            except (OSError, ValueError) as exc:
                messagebox.showerror("Annotation error", str(exc), parent=self.root)
                return
            if existing != self.labels:
                self.save_current()
                return
        else:
            try:
                write_yolo_annotations(
                    record.label_path,
                    self.labels,
                    self.image_width,
                    self.image_height,
                )
            except (OSError, ValueError) as exc:
                messagebox.showerror("Save failed", str(exc), parent=self.root)
                return
        self._persist_status("reviewed")
        self.dirty = False
        self._update_status()
        self._update_progress()

    def skip_image(self) -> None:
        if self.dirty and not messagebox.askyesno(
            "Discard unsaved boxes?",
            "Skip this image and discard unsaved annotation changes?",
            parent=self.root,
        ):
            return
        self._persist_status("skipped")
        self.next_image()

    def delete_selected(self) -> None:
        if self.selected_box is None or self.selected_box >= len(self.labels):
            return
        del self.labels[self.selected_box]
        self.selected_box = None
        self.dirty = True
        self._render_image()

    def clear_boxes(self) -> None:
        self.labels.clear()
        self.selected_box = None
        self.dirty = True
        self._render_image()

    def previous_image(self) -> None:
        if self.index > 0:
            if self.dirty and not messagebox.askyesno(
                "Discard unsaved boxes?",
                "Move to the previous image and discard unsaved annotation changes?",
                parent=self.root,
            ):
                return
            self.index -= 1
            self._load_current_image()

    def next_image(self) -> None:
        if self.index + 1 < len(self.records):
            if self.dirty and not messagebox.askyesno(
                "Discard unsaved boxes?",
                "Move to the next image and discard unsaved annotation changes?",
                parent=self.root,
            ):
                return
            self.index += 1
            self._load_current_image()

    def _close(self) -> None:
        self.root.destroy()


def run_dataset_labeler(
    dataset_dir: Path,
    *,
    split: str = "all",
    session_id: str | None = None,
) -> None:
    DatasetLabeler(dataset_dir, split=split, session_id=session_id).run()
