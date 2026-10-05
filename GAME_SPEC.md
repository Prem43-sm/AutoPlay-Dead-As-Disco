# Dead as Disco — Preliminary Game Specification

This is an initial investigation record, not a complete verified gameplay
specification. The controls below were reported by the user or supplied in a
user-provided functional design; they have not been independently
cross-checked. See [GAMEPLAY_ACTION_SPEC.md](GAMEPLAY_ACTION_SPEC.md) for the
full combat/action behavior and safety requirements, and
[docs/VISUAL_DETECTION_SPEC.md](docs/VISUAL_DETECTION_SPEC.md) for the
evidence-based visual detection and game-state proposal.

## Installation inspected

- The installation directory contains `Pagoda.exe`, an `Engine` directory, and
  a `Pagoda` directory.
- The packaged game executable found is
  `Pagoda/Binaries/Win64/PagodaSteam-Win64-Shipping.exe`.
- The only file found under `Engine/Config` is `StagedBuild_Pagoda.ini`
  (3 bytes). No plain-text gameplay or display settings were found in the
  inspected locations.
- Game content is packaged in `.pak` and `.ucas` files; those assets have not
  been unpacked or modified.
- Initial inspection found no running game process; a later live inspection
  found the game running in a window titled `Pagoda`.

## Gameplay specification

| Area | Finding |
| --- | --- |
| Movement | WASD (user-reported) |
| Basic attack | Left mouse button (user-reported) |
| Lock-on | Right mouse button targets an enemy (user-reported) |
| E | User reports that E performs a finisher when a defeated/stunned enemy displays a prompt. |
| F | Context-sensitive (user-confirmed): user reports it can perform a different-style finisher; the supplied action design also assigns it to Takedown when the skull/token indicator is available. Select behavior only from the visible prompt and confirmed target state; exact prompt-to-action mapping still needs verification. |
| Shift | The supplied action design specifies Shift + Attack for Fever Rush, which consumes Fever. Verify the actual input and required hold duration in-game. |
| Other combat inputs | The supplied action design proposes Heavy Kick (hold/release mouse button), Counter (mouse button), Dodge (key/button), Drumstick Toss (R), Dance Move (1), and combo/sprint/windmill sequences. Exact Heavy Kick, Counter, and Dodge bindings remain unknown. |
| Game states | Unknown; menu, gameplay, pause, death, and completion states need verification |
| Player, enemies, platforms, obstacles, projectiles, and goals | The supplied action design calls out enemy stun, airborne, and attack states; player position, movement, health, and action; Fever and Takedown resources; and environment hazards. Which of these can be reliably observed in-game remains unverified. |
| Available actions | See [GAMEPLAY_ACTION_SPEC.md](GAMEPLAY_ACTION_SPEC.md); action availability, timings, cooldowns, and success conditions require in-game verification. |
| Win condition | Unknown |
| Death condition | Unknown |
| Camera behavior | Unknown |
| Resolution and frame rate | Unknown; measure during gameplay |
| UI and HUD elements | Unknown; inspect during gameplay |

## Next investigation

Verify which on-screen prompts select each F action and confirm the unknown
bindings in-game. Inspect additional menu, pause, death, and completion frames
before describing those states. The live gameplay capture below was used for
read-only vision validation; the detectors remain preliminary and unreliable
for full gameplay.

## LIVE CAPTURE VERIFICATION

- Actual game resolution: **1536x864 pixels (captured game client frame)**
- Observed capture FPS: **26.53 average FPS over 90 frames, with 30 FPS target and preview disabled**
- Game window dimensions: **1536x864 pixels (client area, measured while Pagoda was foreground)**
- HUD visibility: **visible (health icons/meter and enemy count are present)**
- Player visibility: **visible**
- Enemy visibility: **visible (enemy and health bar are present)**
- Action prompt visibility: **partially visible at the right edge; prompt text is clipped and not verified**

The original captured sample and its metadata are stored locally at
`recordings/validation/live_capture.png` and
`recordings/validation/live_capture.json`; generated gameplay captures are
intentionally excluded from the public repository. This verifies that the
monitor-relative capture rectangle can acquire the current game client at its
measured position and dimensions. Capture still reads screen pixels, not an
isolated game-window surface; the game must remain unobscured. The local live
gameplay image was the input for the initial read-only vision validation below.

## LIVE VISION VERIFICATION

Input image: local `recordings/validation/live_capture.png` (1536x864; not
distributed in this repository). The reported
confidence values are heuristic scores, not calibrated probabilities.

| Target | Detection status | Method | Confidence / limitations | Real-game verified |
| --- | --- | --- | --- | --- |
| Player | Yellow clothing candidate detected; full player **NOT RELIABLY DETECTED YET** | HSV yellow threshold + connected components | 0.588 heuristic. Box covers yellow garment pixels (119x92 at x=992,y=518), not the full body. Yellow scene highlights may cause false positives. | Yes, run on captured gameplay image |
| Enemy | **NOT RELIABLY DETECTED YET** | Interface only; no validated appearance model | No detections; enemy count remains unknown rather than zero. | Yes, detector run on captured gameplay image; no reliable detection |
| HUD | Coarse HUD-region candidate detected | Magenta pixel-density cue in the upper-left 30% x 25% region | 0.791 heuristic. Candidate region is coarse; background color may cause false positives. | Yes, run on captured gameplay image |
| Health | Value not detected | No heart-count/value decoder | `health: null`; visible health icons do not yet have a tested parser. | No |
| Fever | Value not detected | No bar measurement/parser | `fever: null`; HUD candidate does not establish its value. | No |
| Takedown | Availability not detected | No prompt/resource-state decoder | `takedown_available: null`; do not infer from F alone. | No |
| Prompt | **NOT RELIABLY DETECTED YET** | Optional OCR interface; disabled for the validation run | Prompt is clipped at the right edge. `prompt.visible` and key remain unknown. | Yes, frame inspected and pipeline run; no reliable prompt detection |

Game status remains `UNKNOWN`; the frame alone does not establish menu,
playing, paused, or dead transitions. Saved debug output is local under
`recordings/vision/validation/frame_000001_20261004T110200_378429.png` and
`recordings/vision/validation/frame_000001_20261004T110200_378429.json`; generated
vision previews are not distributed in this repository.

Current read: simple OpenCV heuristics are useful for visual cues and
debugging, but are not sufficient for reliable full-player, enemy, prompt, or
HUD-value detection. One captured frame is not a labeled training set, so
YOLO training is not justified yet. Collect and label varied gameplay frames
and evaluate conservative tracking/segmentation first; consider a trained
detector only if those measurements show it is needed.

## VISION DATASET REQUIREMENTS

### Required gameplay situations

Collect varied, manually reviewable sessions that include:

- Menu/loading and ordinary gameplay, with the game window unobscured.
- Different camera distances, angles, movement, lighting, and backgrounds.
- Player visible, partially occluded, and absent when those cases occur.
- Enemies at varied distances and poses, including clear ordinary, stunned,
  and airborne examples where naturally observable.
- Contextual prompts (including E/F cases) fully inside the capture region;
  retain the surrounding interaction context but do not infer action meaning.
- Clear positive and negative examples for action prompts and the takedown
  indicator. Do not press inputs to manufacture examples.

### Existing legacy YOLO label classes

These are the current fixed IDs in `dataset.yaml`, not the recommended final
detector classes. The proposed final list and HUD/state methods are specified
in [docs/VISUAL_DETECTION_SPEC.md](docs/VISUAL_DETECTION_SPEC.md). Do not
renumber or rewrite existing labels as part of this specification.

The current legacy YOLO classes are:

| ID | Class |
| ---: | --- |
| 0 | `player` |
| 1 | `enemy` |
| 2 | `stunned_enemy` |
| 3 | `airborne_enemy` |
| 4 | `action_prompt` |
| 5 | `takedown_indicator` |

Health/Fever bars and score are not currently object-detection classes;
their future parsing can be handled separately. See
[LABELING.md](data/dataset/LABELING.md) for label format and annotation rules.
Automatic yellow-color detections must not be copied into ground-truth labels.

### Split and labeling policy

Store images and same-basename YOLO labels under matching `train`, `val`, and
`test` directories. Target approximately 70/20/10 by **whole sessions or
continuous sequences**, not individual neighboring frames. The automatic
session-hash assignment is deterministic but approximate; review it when there
are only a few sessions. Missing labels indicate unreviewed frames; an empty
label file is a reviewed negative.

### Known limitations and current inventory

- Dataset capture session `gameplay_20261004_1702_main_01` collected **30**
  real gameplay images on 2026-10-04. The session is assigned wholly to
  `train`; collection used a 1536x864 region, 30 FPS capture cap, 2-second
  sampling interval, and a 30-sample limit. All 30 sample opportunities were
  saved; none were skipped by the optional similarity filter.
- Video session `gameplay_video_20261004_01` imported **398** frames from
  `Dead as Disco.mp4` on 2026-10-04. The 200.633-second source is 1918x1078 at
  30 FPS with 6,019 frames. It was sampled every 0.5 seconds; 402 sample
  opportunities were evaluated and 4 near-duplicates were skipped. The whole
  video session is assigned to `train`.
- Current dataset total: **428 images** (train **428**, validation **0**, test
  **0**). There are **43** existing label files/reviewed frames and one
  skipped frame; 384 images remain unreviewed. These are not sufficient for
  model training.
- Dataset validation reports **11 out-of-bounds boxes**, **0** exact or
  near-duplicate pairs, and no session split leakage. The label files were
  deliberately left unchanged; review the invalid boxes before any training.
- Current validated box counts are: player 39, enemy 64, stunned_enemy 8,
  airborne_enemy 0, action_prompt 10, takedown_indicator 7. Missing labels are
  not negative examples.
- Existing `recordings/validation/live_capture.png` is one real gameplay
  sample used for vision verification; it is outside the training dataset and
  has no manual labels.
- Both dataset sessions need further review, but pause new annotation until
  the proposed visual-state taxonomy is approved. The video frame set is
  useful for curation, not model training; independent validation/test
  sessions and reviewed label fixes remain required.
