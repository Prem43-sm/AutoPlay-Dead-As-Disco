# Dead as Disco Visual Detection and Game-State Specification

**Phase:** 4C.1 proposal  
**Evidence reviewed:** local gameplay PNGs, the supplied gameplay MP4, frame/session metadata, the existing game/action/labeling specifications, and the current `vision/` interfaces.  
**Scope:** read-only visual/game-state specification only. No controls, model training, automatic labels, or dataset changes are part of this document.

Phase 4C.2 prepares the isolated v2 schema and reviewer-gated migration
workflow; see [`DATASET_TAXONOMY_MIGRATION.md`](./DATASET_TAXONOMY_MIGRATION.md).

## Decision summary

- Keep the object detector small: the recommended next-schema YOLO classes are **`player` and `enemy` only**.
- Decode HUD values in fixed screen regions; use OCR for text/numbers, templates for stable icons, and multiple frames for action/state transitions.
- Treat prompt key (`E`, `F`, or `Shift`) and prompt meaning as separate facts. The key is visible in frames; the exact E/F action mapping is not sufficiently verified to infer from the key alone.
- Preserve the existing six class IDs and all current labels as-is. The two-class recommendation is a future schema proposal, not permission to edit or reinterpret current label files.
- **Do not resume labeling yet.** Approve the proposed schema first, manually resolve the existing invalid boxes in a separate reviewed task, and obtain independent validation/test sessions.

## Evidence and dataset audit

### What was inspected

- `data/dataset/images/train/`, `val/`, and `test/`: **428 PNGs**, all in `train`; `val` and `test` are empty.
- `data/dataset/metadata/manifest.jsonl`: one entry per dataset image; source-session and source/split metadata are present. The dataset images do not have adjacent per-frame JSON sidecars.
- `data/sessions/`: session records for both sources.
- `Dead as Disco.mp4`: local 200.633-second, 30 FPS, 1918x1078 source; 398 retained frames at approximately 0.5-second intervals (402 samples considered, 4 near-duplicates skipped).
- `gameplay_20261004_1702_main_01`: 30 captured frames at 1536x864, sampled about every 2 seconds.
- Local `recordings/validation/live_capture.png` and its vision-debug image, both 1536x864, plus their metadata.
- `GAME_SPEC.md`, `GAMEPLAY_ACTION_SPEC.md`, `data/dataset/LABELING.md`, `vision/`, and the dataset validator/test code.
- No screenshot ZIP was found in the project.

### Dataset and annotation facts

| Fact | Audit result |
|---|---|
| Image count / split | 428 total: train 428, val 0, test 0 |
| Session composition | 30 live-capture frames + 398 video-derived frames; both sessions assigned to train |
| Label files | 43; labeling state records 43 reviewed and 1 skipped |
| Remaining review | 384 images are neither reviewed nor skipped; the validator reports 385 images without a label file (the skipped frame also has no label file) |
| Valid YOLO boxes reported by validator | player 39; enemy 64; stunned_enemy 8; airborne_enemy 0; action_prompt 10; takedown_indicator 7 |
| Label quality | Validation reports **11 boxes outside image boundaries**. The corresponding files are in the first capture session. No files were changed to address this. |
| Split/duplicate checks | No session split leakage, exact duplicates, or near-duplicate pairs reported |
| Review suitability | Useful for initial combat review, but not an independent train/validation/test corpus |

The counts above are an audit snapshot, not a claim that every existing box has been quality-reviewed. Existing IDs in `data/dataset/dataset.yaml` remain:

| Existing ID | Existing name |
|---:|---|
| 0 | `player` |
| 1 | `enemy` |
| 2 | `stunned_enemy` |
| 3 | `airborne_enemy` |
| 4 | `action_prompt` |
| 5 | `takedown_indicator` |

These are the **legacy configured labels**, not the recommended final YOLO taxonomy below. Do not renumber, delete, or rewrite them during this specification phase.

### Visual evidence actually seen

In the inspected combat frames, the player and several ordinary enemies are visible, sometimes together and sometimes overlapping. Visible interface elements include:

- Heart-shaped player-health units at upper left.
- A segmented meter below the hearts; its resource meaning is **not verified**.
- Skull-style UI icons at upper left and near bottom center; their exact role/value mapping is **not verified**.
- `Enemy Count: n` text at the left.
- Pink/red health-like bars above enemies.
- Contextual boxed `E` and `F` key prompts near combatants, and a `Shift` key badge beside a starburst-like icon in one frame. Their exact action meanings are not inferred from appearance alone.
- A circular/reticle-like target marker, directional arrow graphics around enemy bars, and conspicuous red/orange combat effects. Their meanings as lock-on, target, or incoming-attack signals are not confirmed.
- Score and multiplier, beat count, and floating feedback such as `Perfect!`, `Perfect Dodge!`, `Knockout!`, and `Beat`.
- Combat poses/effects, multiple enemies, a prone-looking opponent, and a player in a run/attack-like pose.

The screenshots demonstrate an active arena-combat scene, not every game mode. In the reviewed samples, no menu, pause screen, loading screen, game-over/death screen, victory screen, explicit objective panel, distinct boss, or clearly airborne enemy was verified. A posture or one-frame effect is not sufficient to label a temporal state.

## Detection design rules

**Methods:** YOLO = object detection; HUD = fixed-region pixel/shape analysis; OCR = text/numeric recognition; template = icon matching; temporal = tracking/state transitions over frames; derived = computed from other detections; ignore = no justified detector yet.

Confidence thresholds below are proposed starting gates, **not measured model performance or calibrated probabilities**. Calibrate them on manually reviewed, session-disjoint data. If evidence is absent, clipped, stale, conflicting, or below threshold, report `unknown` rather than `false`.

| Element / state | Type | Method | Why / required visual evidence | Bbox label? | Temporal? | Initial confidence / acceptance requirement | Likely false positive | Likely false negative | Priority |
|---|---|---|---|---|---|---|---|---|---|
| Player | World object | YOLO | Full visible player silhouette; distinguish from enemy. Yellow clothing may aid review but is not a reliable detector by itself. | Yes, full visible body | No for presence; tracking for identity | box >= 0.75; >= 0.90 or track-confirmed before action gating | Yellow lights, VFX, similar clothing | Occlusion, overlap, edge crop, motion blur | CRITICAL |
| Enemy | World object | YOLO | Visible opponent silhouette; support count, target selection, and threat estimation. | Yes, each visible enemy | No for presence; yes for identity/threat | box >= 0.75; target >= 0.90 or stable track | Crowd silhouettes, player overlap, effects | Small/far/occluded enemies, overlapping fighters | CRITICAL |
| Enemy type / boss | World object attribute | Derived from appearance only after validated type cues | Samples show appearance variation, but no stable, verified boss/special-enemy identity. Keep generic `enemy` until a distinctive type is evidenced. | No separate box/class now | Usually yes | Unknown until a reproducible visual cue and validation set exist | Clothing/lighting variation mistaken for type | A special type has no unique visible cue | LOW |
| Stunned / takedown-eligible enemy | Enemy state | Temporal + prompt-to-track association | Bent/down pose and an `E` prompt appear together in a sample; exact state and E meaning still require verification. Require prompt spatially associated with a tracked enemy plus compatible pose/transition. | No separate YOLO class; enemy box only | Yes | enemy track >= 0.90, prompt >= 0.90, agreement in >= 2 frames; action meaning separately verified | Hit reaction, attack wind-up, recovery pose | Brief/occluded prompt, overlapping targets | HIGH |
| Airborne enemy | Enemy state | Temporal track/pose relative to floor | Needed for aerial takedown; no clear positive example verified in sampled screenshots and no valid legacy boxes exist. | No separate YOLO class | Yes | >= 0.90 pose/track state across >= 2 frames; otherwise unknown | Jumping/leaning pose, VFX displacement | Short airtime, occlusion, camera movement | HIGH |
| Knocked-out / dead enemy | Enemy state | Temporal + event OCR as supporting evidence | `Knockout!` feedback and prone-looking pose are visible, but neither alone proves persistent dead state or target identity. | No separate YOLO class | Yes | >= 0.90 track transition; text is corroboration only | Temporary knockdown/recovery or unrelated global text | Offscreen/overlapped body or absent feedback | MEDIUM |
| Throwable / interactable world object | World object | Ignore pending evidence | No clearly identified throwable/interactable prop is verified in the supplied frames. Do not turn UI reticles into world-object classes. | No | N/A | N/A | UI marker mistaken for an object | Small/occluded props | LOW |
| Player health | HUD value | HUD region + heart/icon segmentation | Count filled versus empty heart-shaped units at upper left; preserve raw crop and confidence. Do not convert to numeric HP until units/max are established. | No | Optional smoothing; required across health changes | >= 0.95 for exact count; else unknown | Pink VFX or clipped/animated heart mistaken as filled | Low contrast, animation, crop boundary | CRITICAL |
| Enemy health | HUD value | Per-enemy bar segmentation + association to enemy track | Pink/red bars above enemies are visible; estimate fraction only when the bar is clearly linked to a track. | No | Yes for association and change | bar >= 0.90, association >= 0.90; else null | Attack effects / other nearby bars | Overlap, partial bar, similar colors | HIGH |
| Segmented upper-left meter | HUD value, meaning unknown | HUD crop/template/pixel ratio | The segmented meter is visibly present; resource identity is not proven. Store raw normalized fill only until its meaning is confirmed. | No | Optional | >= 0.95 for fill geometry; semantic name remains unknown | Background lights and fill animation | Color shift, clipping, low resolution | MEDIUM |
| Fever amount / availability | HUD/resource state | HUD/template after mapping verification | Gameplay spec requires Fever for Fever Rush, but the observed meter/badge has no verified label-to-resource mapping. Do not infer available from color alone. | No | Yes for consumption/refill | >= 0.95 after icon/bar mapping is verified | Other resource meter or contextual badge | Small/animated fill or hidden badge | HIGH |
| Takedown resource / indicator | HUD/resource state | Icon template + HUD region; associate with prompt/target | Skull-like icons are visible in multiple positions; their exact semantics and threshold are not confirmed. | No | Yes for threshold/availability | >= 0.95 for icon; availability requires verified mapping and stable state | Decorative skull/UI effect | Empty/dim state or occluded icon | HIGH |
| Action key prompt (`E`, `F`, `Shift`) | UI prompt | OCR + icon/template in local crop | Boxed key glyphs are legible in reviewed frames. Read key exactly and retain prompt crop/position. | No YOLO box; OCR/template crop coordinates only | Persistence/expiry should be tracked | OCR exact key >= 0.90 plus prompt shape/template >= 0.90 | HUD text/subtitles resembling a key | Clipped prompt, blur, stylized glyph | CRITICAL |
| Prompt meaning and target association | Combat affordance | Derived from prompt text/icon + verified mapping + enemy track | Key alone is not action meaning. E/F are context-dependent per action spec; associate prompt with the nearest eligible target only if unambiguous. | No | Yes | All components >= 0.90; otherwise unknown | Prompt associated with wrong overlapping enemy | Ambiguous/clipped key, occlusion, no mapping | CRITICAL |
| Target / lock-on marker | Combat/UI indicator | Icon/template + temporal position association | Reticle-like marker and arrows around bars appear; whether these mean lock-on/target selection is unverified. | No | Yes | >= 0.90 after marker semantics are verified | Decorative targeting effects | Marker hidden by scene effects | MEDIUM |
| Incoming attack / threat cue | Combat indicator/state | Indicator template + enemy track motion/pose over time | Red/orange arcs and directional graphics are visible, but their exact attack-warning semantics require sequence review. Require a cue tied to an attacker, not color alone. | Label the cue crop only if needed; not a world YOLO class | Yes, short window | >= 0.95, attacker association >= 0.90, confirmation over consecutive high-rate frames | Combat VFX, target arrows, damage effect | Cue short/occluded or outside crop | CRITICAL |
| Enemy attacking | Enemy state | Temporal pose/motion + optional threat cue | Needed to decide defense. A single aggressive pose is not a confirmed attack; infer transition/trajectory from a clip. | No separate class | Yes | >= 0.90 sequence score; unknown on one-frame evidence | Idle/guard pose or player attack effect | Fast attack between samples | CRITICAL |
| Counter opportunity | Combat window | Temporal threat classifier + counterability rule | No dedicated counter prompt/window is verified. Only true when the attack is identified as counterable and the timing window is current. | No | Yes, high-rate frames | >= 0.95 plus validated timing rule; otherwise false/unknown is not interchangeable | Any incoming attack incorrectly marked counterable | Very short timing window | CRITICAL |
| Dodge availability / dodge state | Combat state | Temporal player pose/motion and validated cooldown/animation cue | No dedicated availability indicator is verified. `Perfect Dodge!` is post-event feedback, not proof of a pre-dodge window. | No | Yes | >= 0.90 sequence evidence; availability unknown until its cue is verified | Hit-stun or ordinary movement | Brief dodge or occluded player | HIGH |
| Perfect-dodge result / Flashback window | Combat event/window | OCR feedback + temporal player/enemy transition | `Perfect Dodge!` feedback is visible in a sample. Use as result evidence; require a separately learned/verified follow-up window before treating Flashback as available. | No | Yes | OCR >= 0.95 and event-to-track/time association; window requires validation | Feedback from a different event/target | Text occluded or too brief | HIGH |
| Finisher available | Combat affordance | OCR/template prompt + enemy state/track | E prompt is visible over a bent/downed opponent; confirm its exact function and target association in gameplay. | No | Yes | >= 0.95 prompt and >= 0.90 target/state association | E prompt is another contextual action | Prompt cropped/overlap | CRITICAL |
| Aerial takedown available | Combat affordance | Derived from verified F prompt + takedown resource + airborne enemy | No unambiguous aerial-takedown example verified. Require all prerequisites and verified prompt mapping. | No | Yes | Each prerequisite >= 0.95; fail closed otherwise | Ground takedown prompt mistaken for aerial | Rare/brief airborne transition | HIGH |
| Fever Rush available | Combat affordance | Verified resource detector + verified badge/prompt + state | A Shift badge beside a burst-like icon is visible, but meaning is not confirmed. Match it to the action spec only after in-game verification. | No | Yes | >= 0.95 for resource and prompt; no action meaning from Shift alone | Generic ability badge or unrelated resource | Badge absent/clipped | HIGH |
| Enemy count | HUD text | OCR in left-side HUD region; cross-check tracks | `Enemy Count: n` is visible; useful as an independent consistency signal, not a substitute for detecting visible enemies. | No | Optional smoothing | exact OCR >= 0.95; otherwise null | Crowd/background text | Small/clipped number, stylized font | MEDIUM |
| Score / multiplier | HUD text | OCR in upper-right HUD region | Score and `x` multiplier are visibly displayed; useful for progress/reward, not essential to immediate survival. | No | Optional | exact OCR >= 0.95 | Large animated score/effects | Partial crop or animation | LOW |
| Beats / combat feedback | HUD text/event | OCR + event parsing | Beat counts and feedback (`Perfect!`, `Knockout!`, etc.) are visible. Preserve raw text; derive only verified events. | No | Yes for event timing | exact text >= 0.95; event meaning must be mapped | Text overlays, repeated old feedback | Short-lived/fading text | MEDIUM |
| Player moving / sprinting / attacking / dodging / airborne | Player action state | Player track + optical flow/pose over consecutive frames | These actions are motion states, not stable object categories; current frames show action-like poses but do not prove which action from a still. | No separate class | Yes | >= 0.90 sequence classifier; action labels verified from clips | One pose resembles multiple moves | Motion blur, occlusion, camera motion | HIGH |
| Player alive / dead | Player/game state | Health HUD + temporal pose + screen-state evidence | Needed to gate actions. A missing player box is not death; hearts/status/screen transitions must agree. | No | Yes | >= 0.95; disagreement => unknown | Occlusion mistaken for death | Death transition missed between sparse frames | CRITICAL |
| Pause / menu / loading / playing / game-over / victory / cutscene | Game state | Screen-state classifier + templates/OCR + temporal stability | Combat frames support `PLAYING` only for those frames. None of the other states was verified in reviewed samples. | No | Yes, stable transition | >= 0.98 for terminal states, >= 0.95 otherwise; confirm on >= 2 frames unless explicit transition | Effects/black frame mistaken for loading/death | Novel UI or transition screen | CRITICAL |
| Objective / mission state | Game state / text | OCR + icon/template in verified region | No explicit objective panel/text verified in the reviewed samples; enemy count is not an objective. Leave null until found. | No | Optional | >= 0.95 exact text/icon | Background/score text | Small or transient objective text | LOW |
| Environment hazards / usable space | Scene state | Ignore for first detector; consider coarse segmentation later | Arena floor/boundary are visible, but no actionable hazard or throwable prop has been confirmed. | No | Maybe | N/A until concrete gameplay need/evidence | Lighting/reflections | Hazard blends into arena | LOW |

### Specialized detectors and temporal confirmation

- **HUD:** Fix regions only after resolution/UI scaling is measured. Detect heart/icon shapes and meter fill; emit raw crop, value, confidence, and `unknown` when ambiguous. Never classify a coarse magenta patch as a health/resource value.
- **OCR:** Restrict to HUD/prompt crops where possible. Preserve raw text, box, timestamp, and confidence. The current optional OCR prompt code extracts a standalone E/F key only; it does not establish action meaning.
- **Templates:** Good candidates are stable prompt frames, hearts, skull-like icons, and the Shift badge. Validate across animation frames, display scales, and video compression before relying on a match.
- **Temporal states:** Use timestamped tracks, not sparse stills, for attack/counter/dodge/stun/airborne/dead transitions. Existing video-derived images are spaced at about 0.5 s, too sparse to label short action windows or timing reliably.
- **State handling:** Missing detection is `unknown`, never a safe assumption of absence. For action-critical facts, abstain if targets/prompts overlap or if capture is stale.
- **Current code:** `vision/player_detector.py` finds a yellow-clothing candidate, not a full body; `enemy_detector.py` deliberately returns unknown; `hud_detector.py` only finds a coarse region cue; `prompt_detector.py` optionally OCRs E/F; `vision/state_estimator.py` assembles a partial state. These are foundations, not validated game detectors.

## Recommended final YOLO class list

| Proposed class | Why YOLO | Why not another method | Importance | Labeling difficulty |
|---|---|---|---|---|
| `player` | A full-body, variable-position world-object box is useful for localization and tracking. | Fixed-region HUD parsing cannot locate the player; color thresholding is brittle and currently returns only garment pixels. | Critical | Medium: yellow jacket helps, but VFX, occlusion, pose and edge crops complicate full-body boxes. |
| `enemy` | Multiple opponents move through the arena and overlap; per-instance boxes give the tracker a consistent basis. | HUD bars/prompts do not locate all enemies; enemy appearance is not a fixed UI element. | Critical | High: far/overlapping enemies, crowd, poses, effects and bar association need careful review. |

**Do not add** `stunned_enemy` or `airborne_enemy` as object classes: these are time-varying states of an enemy track, not different world-object identities. Do not add `action_prompt` or `takedown_indicator`: prompt glyphs and HUD icons are small, position/context-sensitive UI and are better served by OCR/templates plus region logic. Do not add boss/special-enemy classes until a stable visible identity is established. Add a class only if a held-out test demonstrates that another detector cannot meet a measured requirement.

This proposed two-class schema must use a **new, explicitly versioned dataset/config** if adopted. Do not change the current six-ID YAML or reinterpret existing labels as part of this phase.

## Proposed GameState schema

`null`/`unknown` means not observed or not trustworthy; `false` means a detector positively observed absence. Confidence is detector-specific and should be calibrated. Coordinates should be normalized to the captured game viewport, with source dimensions retained.

```yaml
frame:
  timestamp: ISO-8601
  frame_number: integer|null
  capture_fps: number|null
  width: integer
  height: integer
  stale: boolean

player:
  visible: true|false|null
  bbox: [x1, y1, x2, y2]|null
  confidence: number|null
  track_id: string|null
  position: [x, y]|null
  velocity: [vx, vy]|null
  alive: true|false|null
  hp: {current_units: number|null, max_units: number|null, confidence: number|null}
  action: idle|moving|sprinting|attacking|dodging|airborne|recovering|unknown

enemies:
  - track_id: string
    bbox: [x1, y1, x2, y2]
    confidence: number
    type: generic|verified_type|unknown
    hp_fraction: number|null
    state: idle|moving|attacking|stunned|airborne|downed|dead|unknown
    distance_to_player: number|null
    threat_direction: left|right|front|behind|unknown

combat:
  target_id: string|null
  prompt: {visible: true|false|null, key: string|null, meaning: string|null,
           target_id: string|null, confidence: number|null, expires_at: ISO-8601|null}
  incoming_attack: {detected: true|false|null, attacker_id: string|null,
                    direction: string|null, counterable: true|false|null,
                    confidence: number|null, expires_at: ISO-8601|null}
  counter_available: true|false|null
  dodge_available: true|false|null
  perfect_dodge: {occurred: true|false|null, flashback_window: true|false|null}
  takedown_available: true|false|null
  finisher_available: true|false|null
  aerial_takedown_available: true|false|null
  fever_available: true|false|null

hud:
  visible: true|false|null
  enemy_count: integer|null
  enemy_count_confidence: number|null
  fever_fraction: number|null
  score: integer|null
  multiplier: number|null
  beats: integer|null
  feedback_events: [{text: string, timestamp: ISO-8601, confidence: number}]
  raw_meter_values: {meter_1_fraction: number|null}

game:
  status: UNKNOWN|MENU|LOADING|PLAYING|PAUSED|DEAD|GAME_OVER|VICTORY|CUTSCENE
  objective: string|null
  status_confidence: number|null
```

The schema is a proposal, not a request to extend implementation now. Keep the raw detected prompt key, icon identity, and interpreted action as separate fields. Avoid assigning an enemy HP bar to a target unless association is confident.

## Action decision evidence map

The action names and proposed controls below come from `GAMEPLAY_ACTION_SPEC.md`; bindings, availability cues, and timing behavior remain unverified where that document says so. These are **visual preconditions only**, not instructions to execute input.

| Action | Visual information required before it can be considered safe |
|---|---|
| Normal attack | `game.status=PLAYING`; player positively alive and not in an incompatible action; valid enemy track in range/facing; no more urgent incoming threat. Range/facing must be validated from game footage. |
| Heavy Kick | Normal-attack target checks plus a verified Heavy Kick availability/control interpretation and a clear charge/release state. Exact binding/duration are unknown. |
| Counter | Alive player; a counterable incoming attack from an identified attacker; current counter window with high-confidence timing. Do not treat any enemy attack as counterable. |
| Dodge / Perfect Dodge | Alive player; incoming threat/attack trajectory or validated urgent cue; dodge availability if there is a cooldown; enough current frames to time the window. `Perfect Dodge!` confirms a result, not an opportunity. |
| Flashback | A confirmed Perfect Dodge event and a still-open, verified Flashback follow-up window; valid target if required. |
| Sprint | `PLAYING`, alive player, chosen direction/path clear enough to traverse, and no higher-priority immediate threat. Environment/hazard detection is currently unverified. |
| Sprint Attack | Confirmed sprint state; valid target in front and estimated range; no urgent defense. |
| Drumstick Toss | Valid target, distance/line of sight, and verified ranged-action availability. No throwable-object or ammunition/resource detector is currently established. |
| Rush Down | Confirmed Drumstick Toss event and active post-toss follow-up window; target still valid and player alive. |
| Fever Rush | Positively confirmed Fever resource availability and verified badge/action mapping; valid attack opportunity; no immediate defense priority. A Shift badge alone is insufficient. |
| Finisher (E) | Exact E prompt, prompt-to-target association, eligible enemy state, and player alive. Verify E's meaning before using the prompt semantically. |
| Alternate-style Finisher (F) | Exact F prompt plus a verified mapping to the alternate finisher and eligible target. Do not infer from F alone. |
| Takedown (F) | Verified takedown-available state, exact matching F prompt, eligible non-boss target, and reliable prompt/target association. The screenshot does not settle F's meaning. |
| Aerial Takedown (F) | All Takedown evidence plus confirmed airborne target and verified aerial-specific prompt/window. |
| Dance Move | `PLAYING`, alive player, no imminent threat or critical action, and enough open time/space. Optional and low priority. |
| Windmill | Alive player, verified target/range/space and an uninterrupted action window; state must support a multi-step hold/rotation/release sequence. |
| Lock-on / target selection | If used, positively detected enemy tracks and a verified target marker/selection cue; never assume a reticle proves lock-on. |

Across all actions: reject stale frames; prioritize confirmed survival threats over offense; keep unknown prerequisites unknown; and require a visible/temporal success signal before treating a move as completed.

## Dataset impact and annotation recommendation

Counts below are the validator's current valid-box counts, not all raw box lines. Existing images are useful for beginning a **review and curation** pass, not for model training or final state validation. The 0.5-second video sampling cannot serve as ground truth for short attack windows.

| Detector/state | Examples to collect/review (starter target) | Existing 428 likely useful? | More recording needed? | Rare/dedicated coverage |
|---|---|---|---|---|
| `player` YOLO box | Aim for 200-300 varied full-body instances across at least 4 independent sessions, including near/far, pose, edge, overlap, occlusion, and negative/absent frames. | Yes; 39 valid player boxes, but only 43 reviewed frames total. | Yes, for pose/background/session diversity and held-out sessions. | Include partial/cropped/occluded cases in ordinary capture; no forced pose. |
| `enemy` YOLO box | Aim for 300+ instances across single/multiple enemies, sizes, poses, overlap, and distance; review complete scenes, not only easy targets. | Yes; 64 valid boxes and visible crowds/multiple opponents. | Yes. | Dedicated multi-enemy footage if naturally encountered; include far/small enemies. |
| Enemy tracking/identity | Several complete clips per session with continuous, timestamped frames and occlusion/reappearance. | Some scene context, but current 2 FPS/0.5-second sampling is too sparse for reliable identity continuity. | Yes, high-rate short clips. | Include crossings/overlaps and camera motion. |
| Stunned/downed/eligible | At least 30 confirmed event sequences with pre/event/post frames and matched prompt/target; record negative hit-reaction/recovery examples. | Partly; 8 valid legacy `stunned_enemy` boxes and visible E-prompt candidates, semantics unverified. | Yes, with dense capture and verified outcomes. | Rare states may require a dedicated capture session, but only if they occur naturally; do not manufacture them. |
| Airborne enemy / aerial takedown | At least 25 distinct airborne transitions with ground, rise, apex, descent/recovery frames; annotate sequence state, not a new box class. | No verified positive sample; 0 valid legacy airborne boxes. | Yes. | Dedicated observation is required if naturally occurring; do not label ambiguous poses. |
| Action keys and meanings | At least 50 clear prompt appearances per observed key/action mapping, plus prompt-absent negatives; preserve nearby target/context and prompt expiry. | Yes; 10 valid prompt boxes and E/F/Shift examples in the inspected screenshots, but small sample and semantics not settled. | Yes, especially for each verified key/action pair and UI scaling. | Capture each rare prompt naturally; do not trigger it for collection. |
| Incoming attack/counter/dodge/perfect dodge | At least 50 labeled attack/dodge/counter event clips each, with high-rate frames immediately before/through/after the cue and negative near-miss examples. | Poor for timing: sparse frames can help discover cues only. `Perfect Dodge!` feedback appears in sampled material. | Yes, high-rate action clips. | Counterable vs non-counterable attacks and short windows need dedicated event review. |
| HP / enemy HP / resource meters | At least 100 reviewed crops per visible state band (full, partial, low/empty, changing/animated, occluded), tied to timestamps and session/UI scale. | Likely useful for template discovery; no value decoder or labels currently exist. | Yes, with multiple health/resource states and resolution checks. | Low-health/empty and transitions are rare; capture them only when they occur. |
| Takedown / Fever indicators | 50+ positive and 100+ negative crops for each mapped icon/state, across animation/fill levels and contexts. | Partly; 7 valid legacy takedown-indicator boxes, but semantics and thresholds require review. Shift badge is visible but Fever mapping unverified. | Yes. | Resource thresholds/consumption/refill require a dedicated natural observation session. |
| Enemy count / score / beats / feedback OCR | 100+ crops per text family across numbers, animation, occlusion, and no-text negatives; preserve raw OCR samples. | Yes; these strings/numbers are visibly present. | More varied recording needed to validate OCR across motion and screen scale. | Rare event messages should be captured if encountered. |
| Game-state screens | At least 100 reviewed frames per state and 20 transition clips for each state actually supported: menu, loading, playing, paused, dead/game-over, victory, cutscene. | No evidence for non-playing states in inspected combat frames. | Yes, separate scenes/sessions. | Menu/pause/completion/death need deliberate *human-led* collection only if approved; no automated game interaction. |
| Objectives / special enemy / throwable / hazards | First verify the element exists and has decision value; then collect at least 50 positive and 100 negative examples before proposing a detector. | No verified objective panel, boss identity, throwable, or actionable hazard in inspected material. | Unknown until verified. | Do not spend labeling time or add classes based on speculation. |

### Next data-collection sessions

1. **Ordinary gameplay, train:** varied player/enemy scale, movement, and multiple-enemy situations in a distinct session.
2. **Combat events, train:** short, timestamped/high-rate clips around naturally occurring attacks, prompts, stun/down, airborne, and recovery transitions. Existing 0.5-second samples are not fast enough for window labels.
3. **HUD/UI variation, train:** natural changes in heart units, enemy bars, meter fills, score/beat feedback, prompt display, and crop/resolution conditions.
4. **Independent validation gameplay:** a different complete session (not adjacent frames or a split of the current video) with comparable combat diversity.
5. **Independent test gameplay:** another whole session, captured later and kept untouched until evaluation.
6. **Non-combat/game-state coverage:** only if desired and permitted, a separate human-led capture of real menu/pause/loading/completion states; do not automate or force gameplay states for this phase.

Retain all frames from a continuous clip in one split. Use short dense clips for temporal labels and sparse frames only for static objects/HUD. Review split leakage and duplicate scenes before any training decision.

## Labeling go/no-go

**Manual labeling should not begin/resume now.** Reasons:

1. The current six-class set mixes persistent objects, transient enemy states, tiny UI prompts, and HUD indicators; the proposed two-class schema and separate HUD/state annotations need approval first.
2. The current dataset has 11 invalid out-of-bounds boxes. Fixing them requires explicit human review; no automatic clipping or label rewrite is authorized here.
3. 384 images remain pending review (plus one skipped); all 428 images and both capture sessions are in train, with no validation or test session.
4. Airborne, counter-window, incoming-attack semantics, game-over/menu states, and Fever/takedown mappings lack enough verified examples.

After schema approval, define a versioned annotation format for temporal states and HUD/OCR crops, resolve existing box errors without overwriting originals, then continue review under session-level splits. This specification does not authorize training or control implementation.
