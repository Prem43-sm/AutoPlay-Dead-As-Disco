# Dead as Disco — Gameplay Action and Combat Design

This document organizes the user's supplied functional combat specification
for future AI/controller work. It describes intended behavior, not a verified
game API: confirm controls, action windows, and move outcomes in-game before
enabling autonomous inputs. The presently reported controls and unverified
prompt-to-action mapping for F are tracked in [GAME_SPEC.md](GAME_SPEC.md).

## Core behavior

- Observe the game state and choose a safe, high-value action; do not spam
  attacks or trigger moves without their prerequisites.
- Prefer survival over offense and music synchronization. Ignore a beat when
  danger requires a defensive response.
- Each action needs a target/availability check, prerequisites, duration,
  cooldown or timing window, resource requirement, success signal,
  interruptibility, and any follow-up actions.
- Execute only when the required state is positively detected. Until vision
  supports that check, leave game-specific actions disabled rather than
  guessing.

## Candidate actions

| Action | Proposed input | Preconditions and behavior |
| --- | --- | --- |
| Normal attack | Press primary attack mouse button | Default melee action when a suitable enemy is in range and defense is not more urgent. Can build Takedown availability. |
| Heavy Kick | Hold and release its mouse button | Hold for the charge duration, then release to execute; a press alone is not the move. Exact button and duration are unknown. |
| Counter | Press designated counter mouse button | Use only during a detected counterable enemy attack/window. Prefer it over a normal attack when a successful counter opportunity is confirmed. Binding is unknown. |
| Dodge / Perfect Dodge | Press designated dodge input | Avoid an incoming attack when counter is unavailable or dodge is safer. Avoid unnecessary dodges; detect whether the result was a Perfect Dodge. Binding is unknown. |
| Flashback | Press Attack after a Perfect Dodge | Follow-up only during the confirmed Perfect Dodge opportunity/window. Never attempt without a successful Perfect Dodge. |
| Sprint | Hold/release sprint input | Hold for approach, escape, chase, or repositioning; release when no longer needed. Binding is unknown. |
| Sprint Attack | Sprint, then press Attack | Requires active sprint; use when the target is in front and the attack can connect, unless defense takes priority. |
| Drumstick Toss | Press R | Ranged option when distance and line of sight make it useful. |
| Rush Down | Press Attack after Drumstick Toss | Combo follow-up only during the valid post-toss window; not an independent attack. |
| Fever Rush | Hold Shift + Attack | Requires available Fever resource; consumes the Fever Bar. Use only when strategically worthwhile and not at the expense of immediate defense. |
| Finisher | Press E on a stunned enemy | Requires a confirmed stunned enemy and an available finisher prompt/window. Do not press E speculatively. |
| Alternate-style finisher | Press F when its distinct finisher prompt is shown | User reports that F can perform the same kind of finisher as E with a different hit style. The exact prompt and conditions are not yet verified. |
| Takedown | Press F when the Takedown prompt/skull-token is shown | The supplied design says attacking builds availability and describes an instant finisher for eligible non-boss enemies. F is context-sensitive according to the user; require the matching visible prompt and eligible target. |
| Aerial Takedown | Press F when the aerial Takedown prompt is shown for an airborne enemy with Takedown available | Requires both the airborne target and Takedown availability; prefer over a normal Takedown only when success is likely. Verify the on-screen prompt/action mapping. |
| Dance Move | Press 1 | Optional style/music action; only in a safe situation. |
| Windmill | Hold Attack, rotate more than 360 degrees, release Attack | Composite action. Track hold, rotation amount, and release; do not model as a single press. Confirm the required rotation behavior in-game. |

## Combo and action relationships

```text
Perfect Dodge -> Flashback
Sprint -> Attack -> Sprint Attack
Drumstick Toss -> valid follow-up window -> Attack -> Rush Down
Stunned enemy + finisher prompt -> E -> Finisher
Build Takedown availability -> matching prompt + eligible target -> F -> Takedown
Airborne enemy + available Takedown -> matching prompt -> F -> Aerial Takedown
Hold Attack + rotate >360 degrees -> release Attack -> Windmill
```

Every arrow represents a prerequisite or a timed follow-up, not an instruction
to execute blindly. The controller must verify that the follow-up remains
available before issuing it.

## Decision priorities

Use this as a contextual guide, not a fixed ranking:

1. Prevent imminent damage or death.
2. Take a confirmed Perfect Dodge or Counter opportunity.
3. Use the F action indicated by a confirmed prompt on an eligible target.
4. Use a confirmed Finisher on a stunned enemy.
5. Use Flashback during its confirmed post-Perfect-Dodge window.
6. Consider Fever Rush only when Fever is available and spending it is sound.
7. Continue valid combo follow-ups and Sprint Attacks.
8. Consider Heavy Kick or a normal attack when appropriate.
9. Reposition or Sprint if that is safer or improves the attack.
10. Use Dance Move or other style actions only when safe.

Re-evaluate based on enemy distance/type/attack state, player health and
current action, resources, target vulnerability/stun/airborne state, cooldowns
and combo windows, number of enemies, environment hazards, and overall danger.
Music beat and energy may influence style only after safety checks pass.

## State to estimate

- **Player:** position, movement direction/velocity, health, current action,
  sprint and attack state.
- **Enemies:** position/distance, type and health where visible, stun/airborne/
  attack/vulnerability state, attack direction.
- **Resources:** Fever amount/availability, Takedown indicator, cooldowns, and
  special-action availability.
- **Combat windows:** incoming attack, counter/dodge opportunity, Perfect
  Dodge result, finisher/Takedown prompt, and combo/follow-up availability.
- **Environment:** obstacles, walls, open movement space, and hazards.

These are desired state fields, not claims that they are already detectable.
Only consume fields after their vision/tracking source has been implemented
and tested.

## Controller contract

The input layer must support:

1. **Press:** issue a short, discrete input (for example attack, Counter, R,
   E, F, or 1).
2. **Hold/release:** reliably send both input-down and input-up transitions
   (for example Heavy Kick, Sprint, and Fever Rush).
3. **Composite sequence:** order inputs and movement, wait for prerequisites
   or follow-up windows, and allow safety interruption (for example Sprint
   Attack, Drumstick Toss -> Rush Down, Perfect Dodge -> Flashback, and
   Windmill).

Maintain explicit action state (for example `IDLE`, `SPRINTING`,
`HEAVY_KICK_CHARGING`, `DODGING`, `FLASHBACK_WINDOW`, `RUSH_DOWN_WINDOW`,
`FEVER_RUSH`, `FINISHER_AVAILABLE`, `TAKEDOWN_AVAILABLE`, `WINDMILL`, `DEAD`,
and `RECOVERING`). Always release held controls on cancellation, stop, or
emergency stop. Add the roadmap's F8 emergency-stop behavior before enabling
unattended gameplay.

## Verification needed before automation

- Verify which visible prompts select the alternate-style Finisher, Takedown,
  and Aerial Takedown for F; never infer the action from F alone.
- Verify Heavy Kick, Counter, Dodge, and Sprint bindings.
- Confirm whether Shift + Attack is Fever Rush and determine its hold/release
  behavior.
- Observe the prompts, durations, cooldowns, resource indicators, target
  eligibility, and success/failure signals for every move.
- Test press, hold/release, composite actions, and emergency release manually
  before using them in an autonomous loop.
