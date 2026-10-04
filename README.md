# Dead As Disco AI

A Windows/Python research project for **read-only** screen capture and early
computer-vision work for Dead as Disco. The project currently provides bounded
screen capture, preliminary vision/debugging tools, and a dataset collection,
validation, and manual-labeling workflow.

## Current status

- Phase 0: initial investigation
- Phase 1: project foundation
- Phase 2: screen capture verified against the game
- Phase 3: preliminary read-only vision (not reliable for full gameplay)
- Phase 4A: dataset tooling
- Phase 4B: real data collection and manual labeling not complete
- Dataset: **0 collected images, 0 manually labeled images**
- Training readiness: **NOT READY**

No keyboard, mouse, or controller input, autonomous gameplay, reinforcement
learning, or trained YOLO model is included. Do not treat heuristic vision
output as ground truth. See [GAME_SPEC.md](GAME_SPEC.md) for verified
observations and known limitations, [GAMEPLAY_ACTION_SPEC.md](GAMEPLAY_ACTION_SPEC.md)
for proposed future behavior (not implemented), and
[Dead_As_Disco_AI_Complete_Roadmap.txt](Dead_As_Disco_AI_Complete_Roadmap.txt)
for the project roadmap.

## Requirements

- Windows
- Python 3.10 or 3.11
- A running game or other visible content to capture

## Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Live screen capture

From the project directory:

```powershell
python -m app.main
```

This opens a preview of monitor 1 and reports the measured capture FPS. It
keeps only the current frame in memory and sends no keyboard, mouse, or
controller input. Press `q` or `Esc` in the preview window, or `Ctrl+C` in the
terminal, to stop.

Select monitor 2:

```powershell
python -m app.main --monitor 2
```

Select a monitor-relative region (`left,top,width,height`) and limit capture
to 30 FPS:

```powershell
python -m app.main --monitor 1 --region 100,100,1280,720 --fps 30
```

Capture 90 frames (about 3 seconds at 30 FPS), save the first frame and its
JSON metadata without opening a preview, then exit. Replace the example crop
with the measured game window rectangle:

```powershell
python -m app.main --monitor 1 --region 0,0,1536,864 --fps 30 --output-dir recordings/gameplay --save-first-frame --frames 90 --no-preview
```

The image and same-basename `.json` sidecar are written to the output
directory. Metadata includes UTC timestamp, frame number, measured average
capture FPS, monitor index and bounds, absolute capture region, and frame
resolution. Filenames include a timestamp so separate runs do not overwrite
one another.

To instead save the first frame at a specific path (also creates a JSON
sidecar):

```powershell
python -m app.main --save-frame recordings/gameplay/sample.png --frames 90 --no-preview
```

Stop ongoing preview capture with `q` or `Esc` in the preview window, or with
`Ctrl+C` in the terminal. Capture targets a monitor or monitor-relative
rectangle, not an isolated game window. Keep the game unobscured and ensure
the chosen region fits within the selected monitor. Capture FPS remains
unavailable for a one-frame run; a multi-frame run measures the average rate
and writes it to the sample metadata when capture stops.

## Tests

```powershell
python -m unittest discover -s tests -v
```

## Phase 3 vision foundation (read-only)

Run the vision pipeline against the saved gameplay image:

```powershell
python -m app.main --vision-image recordings/validation/live_capture.png
```

Display the annotated image in a local debug window (press `q` or `Esc` to
close), without sending input to the game:

```powershell
python -m app.main --vision-image recordings/validation/live_capture.png --vision-debug
```

Save exactly one annotated image and its detected `GameState` JSON:

```powershell
python -m app.main --vision-image recordings/validation/live_capture.png --save-vision-debug --vision-output-dir recordings/vision
```

Output filenames contain the frame number and capture timestamp to avoid
replacing earlier samples. Saving is explicit and bounded to one image per
command. To try OCR for contextual E/F labels, install `pytesseract` and the
Tesseract executable separately, then add `--vision-ocr`:

```powershell
python -m app.main --vision-image recordings/validation/live_capture.png --vision-ocr --save-vision-debug
```

Current vision limitations: player detection finds a yellow-clothing color
candidate, not a complete person; enemy detection deliberately reports unknown;
the HUD detector finds only a coarse magenta upper-left cue and does not read
health, Fever, Takedown, or score values; prompt OCR is optional and the saved
sample's prompt is clipped. Game status therefore remains `UNKNOWN`. No
detector sends game input.

## Phase 4A/4B: gameplay dataset collection and labeling

Initialize the dataset folders and class specification (safe to repeat; an
existing `dataset.yaml` is not overwritten):

```powershell
python -m app.main --dataset-init data/dataset
```

Before collecting, bring the game to the foreground yourself and make sure it
is visible and unobscured. At the verified 1536x864 game resolution, the
recommended profile captures monitor 1, region `0,0,1536,864`, with a 30 FPS
capture cap, one sample every 2 seconds, at most 120 sample attempts, and
optional deduplication enabled. The interval and limit can be changed to suit
the session. The recorder only captures screen pixels; it never activates the
game or sends keyboard/mouse/controller input.

```powershell
python -m app.main --dataset-capture --monitor 1 --region 0,0,1536,864 --fps 30 --sample-interval 2 --max-samples 120 --deduplicate --similarity-threshold 2 --dataset-session gameplay_YYYYMMDD_01 --dataset-split train --dataset-dir data/dataset
```

The maximum counts sample opportunities, including samples rejected as
duplicates; recording ends at that count. Use a new unique `--dataset-session`
value for every collection session. Set `--dataset-split` to `train`, `val`,
or `test` explicitly; `auto` assigns a whole session by stable hash, but with
few sessions this will not guarantee a 70/20/10 ratio. Assign whole continuous
gameplay sessions to one split only, and prioritize that over exact ratios.

Collect multiple bounded sessions when possible, for example normal gameplay,
combat-heavy scenes, multiple enemies, naturally occurring stunned/airborne
situations, and contextual prompt/takedown situations. Do not press inputs to
manufacture examples. The complete capture checklist is in
[data/dataset/LABELING.md](data/dataset/LABELING.md).

Frames are placed under `data/dataset/images/{train,val,test}`, metadata and
the append-only JSONL manifest under `data/dataset/metadata`, and session
records under `data/sessions`. No automatic vision detections are turned into
labels. A missing `.txt` means the image has not been manually reviewed yet.

### Candidate object classes

The class IDs are fixed in [data/dataset/dataset.yaml](data/dataset/dataset.yaml):

| ID | Name |
| ---: | --- |
| 0 | `player` |
| 1 | `enemy` |
| 2 | `stunned_enemy` |
| 3 | `airborne_enemy` |
| 4 | `action_prompt` |
| 5 | `takedown_indicator` |

Definitions and annotation guidance are in
[data/dataset/LABELING.md](data/dataset/LABELING.md). Label only visible,
manually verified instances. Keep a generic `enemy` label unless a special
stunned/airborne state is visually clear. HUD bars/values are not object
classes in this initial list.

### Manual labeling and YOLO format

Open each saved image in a manual bounding-box labeling tool configured with
the six classes above. Export one YOLO detection `.txt` file beside each image
in the matching `labels/{split}` directory, with the same basename. Each
non-empty line is:

```text
class_id x_center y_center width height
```

Coordinates are normalized to the image dimensions, in `[0,1]`; width and
height must be positive and the full box must fit the image. Empty label files
are valid for reviewed frames containing none of these classes. Do not use the
player color heuristic as ground truth.

Split by **whole session/continuous sequence**, targeting approximately 70%
train, 20% validation, and 10% test. Avoid putting near-identical neighboring
frames into different splits.

### Validate and preview labels

Validate every split, image, and available label file:

```powershell
python -m app.main --dataset-validate data/dataset
```

The JSON report lists total and per-split counts, labeled/unlabeled frames,
images with multiple classes, reviewed images with no target classes, per-class
object/frame counts, exact duplicate images, adjacent near-duplicate frame
pairs, invalid labels, session split leakage, errors, and warnings. Near
duplicates are counted only for neighboring frame sequence numbers within a
named capture session, using the recorder's default grayscale difference
threshold. Missing labels are reported as unlabeled warnings; malformed
labels, unknown classes, out-of-image boxes, corrupt images, duplicate
filenames, orphan labels, and detected session leakage are errors.

### Dataset quality review

Do not treat the existence of image files as training readiness. Before any
training decision, manually review labels and previews; check that every
important class has enough examples and visual variation; verify tight,
accurate boxes and difficult/edge cases; confirm session-level split
separation; and ensure no class is dominated by one nearly identical scene.
An empty label file counts as a reviewed image with no target classes; a
missing label file remains unreviewed.

Preview an image in the dataset by drawing only its manually supplied labels:

```powershell
python -m app.main --dataset-preview data/dataset/images/train/SESSION_frame_000001_TIMESTAMP.png
```

Or preview any image against an explicit label file:

```powershell
python -m app.main --dataset-preview path/to/frame.png --dataset-label path/to/frame.txt
```

Press `q` or `Esc` in the preview window to close it. Previewing never infers
labels or interacts with the game.

## Repository contents and capture privacy

This repository contains source code, tests, specifications, and empty dataset
directory placeholders. Local gameplay screenshots, generated vision
previews, and collected dataset images/labels/metadata are intentionally
excluded by `.gitignore`; captured game footage is not redistributed here.
The recorded live-capture and vision-verification results documented in
[GAME_SPEC.md](GAME_SPEC.md) were performed locally. To run the sample-image
commands above, provide your own local image path.

There is no model-training workflow in this phase. Collection and labeling
must remain read-only and manually verified. A future phase requires a
separate explicit decision; no control or autonomous-play behavior is
implemented.

## License

No license has been added. All rights remain with their respective owners;
publishing this repository does not grant permission to redistribute the game
or its assets.
