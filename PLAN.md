# Project Plan — Autonomous RC Car: ML/Sim Phase

## 1. Context & Goal

**Endgame:** a physical self-driving RC car — Raspberry Pi 5, Xbox controller, real-time
control (likely C++ later for the fast loop). Hardware is expensive, so **this phase is
ML-only, in the DonkeyCar simulator**.

**Hard constraint that shapes everything:** *the machine-learning code is ours*, written
from scratch — not DonkeyCar's built-in training. `gym-donkeycar` is used **only** as the
pipe to the simulator. The point is to honestly say "I built the autonomous driver from
scratch."

**Where the "from scratch" line sits (so the claim is defensible):** *we* design the network
architecture, the preprocessing/augmentation, the training regime, and the inference/control loop; the
framework (TF/Keras) provides autodiff and optimizers. That's the honest boundary — enough to answer
"did you write backprop?" without flinching, while not reinventing linear algebra.

**Approach (decided):** Behavioral Cloning (imitation learning). You drive the sim with an
Xbox controller; we record `(camera image → steering, throttle)` pairs; a custom CNN learns
to imitate you; that model then drives the car autonomously.

**This phase is done when:** the car completes laps in the sim on its own, driven by a model
we trained on our own recorded data, in code we wrote and understand.

---

## 2. Requirements Analysis

### Functional requirements
| ID | Requirement |
|----|-------------|
| FR1 | Connect to a running DonkeyCar sim and exchange image/action frames. |
| FR2 | Drive the sim manually with an Xbox controller (teleoperation). |
| FR3 | Record time-synchronized `(image, steering, throttle)` samples, with start/stop control. |
| FR4 | Persist recordings in a documented, reloadable on-disk format (our own, not DonkeyCar "tubs"). |
| FR5 | Load recordings → preprocess → augment → train/validation split. |
| FR6 | Define + train a **custom** CNN; save the model and its training metrics. |
| FR7 | Run the trained model to drive autonomously, with a throttle safety clamp. |
| FR8 | Provide smoke tests (env connects, model builds, dataset loads). |

### Non-functional requirements
| ID | Requirement | Why it matters |
|----|-------------|----------------|
| NFR1 | Sim access isolated behind one wrapper module. | Swap sim → real car later by rewriting one file. |
| NFR2 | All ML logic is ours. | The "from scratch" goal. |
| NFR3 | Config-driven + seeded. | Reproducible experiments; training and driving can't drift apart. |
| NFR4 | Model stays small / exportable. | Must eventually run in real time on a Pi 5 (TFLite later). |
| NFR5 | Real-time-capable inference loop. | Driving can't lag behind the sim. |
| NFR6 | Readable, commented code. | You need to understand and own it. |

### Platform constraints (WSL2 — the two real friction points)
- **Simulator (Unity binary):** run the **Windows** build on the host; Python (in WSL)
  connects over TCP to `host:9091`. With WSL *mirrored networking*, `127.0.0.1` works;
  otherwise point config at the Windows host IP. Avoids fighting Unity GPU support in WSL.
- **Input (keyboard-default, controller as bonus):** the input layer is one interface that emits a
  normalized `(steering, throttle)` in `−1..1`. WSL2 doesn't expose USB HID by default, and even after
  attaching with **`usbipd-win`** the gamepad often won't enumerate (the WSL kernel may lack
  `xpad`/joystick support, so no `/dev/input/js*` node appears). Therefore the **keyboard driver is the
  default**, made *pseudo-analog* — while a key is held, steering/throttle ramp smoothly toward the
  extreme instead of snapping to ±1, so labels stay continuous (see §3.5). The Xbox controller is a
  **spike in M0** (does it enumerate at all?); if it does, it's used for true analog input, if not we
  stay on keyboard. This model is a sim model regardless — the real car gets its own controller-driven
  data later (§8), so keyboard here does not affect real-life driving.
- **Stack:** Python 3.10/3.11, TensorFlow 2.x (Keras), `gym-donkeycar`, `pygame`, `opencv`, `numpy`.

---

## 3. Data Models

### 3.1 On-disk recording format
One folder per recording session ("run"):

```
data/
└── run_2026-09-21_16-30-00/
    ├── imgs/
    │   ├── 000000.jpg        # camera frames, zero-padded index
    │   ├── 000001.jpg
    │   └── ...
    ├── records.csv           # one row per frame (the labels)
    └── meta.json             # run-level metadata
```

**`records.csv` schema** (one row = one captured frame):

| column     | type  | range / units      | description |
|------------|-------|--------------------|-------------|
| `frame`    | int   | 0..N               | index; matches `imgs/{frame:06d}.jpg` |
| `image`    | str   | relative path      | e.g. `imgs/000123.jpg` |
| `steering` | float | −1.0 … +1.0        | human steering at capture (label) |
| `throttle` | float | −1.0 … +1.0        | human throttle at capture (label) |
| `timestamp`| float | unix seconds       | capture time (for frame-rate checks) |
| `speed`    | float | sim units (opt.)   | from sim `info`, for analysis |
| `cte`      | float | cross-track error (opt.) | from sim `info`, for analysis |

**`meta.json` fields:** `track`, `date`, `controller_name`, `image_h`, `image_w`,
`fps_target`, `frame_count`, `notes`. Records the conditions a run was captured under.

### 3.2 In-memory / training representation
- **Sample:** `(image, label)` where
  - `image`: float32 tensor `[H, W, 3]`, pixels scaled to `[0, 1]`.
  - `label`: float32 `[steering, throttle]`.
- **Dataset:** a `tf.data.Dataset` streaming `(image, label)` batches from the CSVs of all runs.
- **Split:** `VAL_SPLIT` (default 0.2) held out for validation — the split happens **by run (whole
  runs, or whole laps, held out)**, *not* per frame. Consecutive frames are near-identical, so a
  per-frame split (even with a large shuffle buffer) leaks near-duplicate frames across train/val:
  validation loss then looks great, fails to catch overfitting, and doesn't predict real driving.
  Holding out entire runs the model never trains on is the only honest validation signal. Shuffling
  still happens *within* the training set for batch diversity.

### 3.3 Preprocessing pipeline (the exact steps, in order)
Applied identically at **train** and **drive** time via one shared function — if they differ,
the car sees different inputs than it learned on and misbehaves. Steps:
1. **Decode** JPEG → uint8 `[120,160,3]`.
2. **(Optional) color space:** keep RGB for now. YUV is a common BC trick (luminance separated
   from color) — noted as a tuning lever, not default.
3. **Normalize:** cast to float32, divide by 255 → `[0,1]`. (Cheap, stable; done outside the model
   so training and inference match exactly.)
4. **Crop:** remove the top ~1/3 (sky/horizon) — done *inside* the model as a `Cropping2D` layer so
   it can never be forgotten at drive time. Carries no steering signal, only distraction.
- **Resize:** not needed — sim already outputs `120×160`. If the real camera differs later, resize
  is the one preprocessing change, and it lives in the shared function.

**Minor train/serve skew (noted, accepted):** frames are stored as JPEG (lossy), so the model trains on
JPEG-decoded images but drives on pristine sim frames. The effect is negligible for BC; if it ever
matters, the cheap fix is to JPEG-round-trip at drive time too (or store PNG at the cost of disk).

### 3.4 Augmentation (train only — never on val/drive)
Augmentation multiplies effective data and fights overfitting by showing plausible variations:
| technique | transform | label change | why |
|-----------|-----------|--------------|-----|
| horizontal flip | mirror image L↔R | **negate steering** | doubles data; balances left/right turn bias for free |
| brightness jitter | random ± brightness | none | robustness to lighting; helps future sim-to-real |
| (optional) small shifts | shift image horizontally | add proportional steering offset | teaches recovery from off-center — powerful but must get the label math right |
- Applied randomly per-sample each epoch (so the model rarely sees the exact same frame twice).
- **Not** applied to validation or at drive time — we evaluate/drive on real inputs only.

### 3.5 Label handling
- **Task type:** regression (continuous steering/throttle), so labels are floats, loss is MSE — not
  classification. (DonkeyCar offers a "categorical" binned variant; we're doing linear/regression.)
- **Ranges:** steering/throttle already live in `−1..1`, the network's natural output scale, so **no
  label normalization needed**. Keeping labels in the raw action space means model output feeds the
  sim directly.
- **Label imbalance (the big BC gotcha):** most frames are "driving straight" (steering ≈ 0), so a
  lazy model can score low loss by always outputting ~0 and never turning. Mitigations, in order of
  preference: (a) capture more cornering/recovery data in M1; (b) inspect the steering histogram in
  M2; (c) if still skewed, downsample near-zero-steering frames or weight sharp turns more.
- **Label smoothness (why input is pseudo-analog):** BC labels should be *continuous*. An analog stick
  gives values across the whole `−1..1` range; a raw keyboard gives only `−1/0/+1` ("bang-bang"), which
  teaches the model to only slam full-lock or go straight — jerky, weaves, no fine corrections. Hence the
  keyboard driver ramps values while a key is held (§2), approximating analog labels; a real controller,
  if it enumerates, gives true analog for free.

### 3.6 Contracts (must stay identical across collect / train / drive)
- **Image spec:** `120 × 160 × 3` (the sim camera's native output), normalized `[0,1]`.
- **Action:** `[steering, throttle]`, each `−1..1`, sent to the sim every step.
- **Model I/O:** in = image `[120,160,3]`; out = `[steering, throttle]`, produced by a **`tanh` output
  layer** so predictions are structurally bounded to `−1..1` (no out-of-range values feeding the sim; the
  drive-time throttle clamp in FR7/M4 is then a *safety* limit, not a range fix).
- **Capture-loop ordering (causality — get this right or every label is shifted):** pair the image the
  human *saw when deciding* with the action they took, i.e. record **before** stepping:
  `obs = reset(); loop { action = readInput(); record(obs, action); obs, _, _, info = step(action) }`.
  `gym-donkeycar`'s `step()` returns the *next* frame, so recording the returned image with that action
  would shift every `(image → action)` pair by one frame and quietly poison the whole dataset.
- **`config.py` is the single source of truth** for all of the above.

### 3.7 Open data decisions (worth a quick call before M2)
- **Predict throttle, or fix it?** Simpler/steadier BC models predict *steering only* and use
  a constant throttle. Recommendation: start by predicting both, but keep fixed-throttle as an
  easy fallback if the car is unstable.
- **Capture rate:** target ~20 Hz. Tune during M1 based on smoothness vs. data volume.
- **How much data:** aim for several clean laps in *both* directions plus some "recovery"
  driving (steer back from the edge) — quantity/quality guidance lives in M1.
- **Color space:** RGB vs. YUV (see 3.3 step 2) — leave RGB unless the model struggles.

---

## 4. System Architecture (module map)

Python-only for now, organized so a real-car path drops in cleanly later.

```
DonkeyCarSimML/
├── config.py                 # single source of truth: sim conn, image size, paths, hyperparams
├── requirements.txt
├── README.md                 # quickstart + WSL setup notes
├── src/
│   ├── sim/environment.py    # ONLY file that talks to gym-donkeycar; reset()/step()/close()
│   ├── sim/fakeenv.py        # in-memory fake with the SAME interface — headless tests, no sim needed
│   ├── input/controller.py   # input interface → normalized (steering, throttle); pseudo-analog keyboard driver (default) + optional Xbox driver
│   ├── data/
│   │   ├── collect.py        # teleop loop → writes runs (imgs/ + records.csv + meta.json)
│   │   └── dataset.py        # runs → tf.data.Dataset; shared preprocess; augmentation; split
│   ├── model/network.py      # OUR CNN (PilotNet-style), heavily commented; tanh output → [steering, throttle] in −1..1
│   ├── train.py              # build → fit (MSE) → checkpoint/early-stop → save + metrics
│   └── drive.py              # load model → inference loop → autonomous driving (throttle clamp)
├── data/    (gitignored)     # recorded runs
├── models/  (gitignored)     # trained models
└── tests/                    # smoke tests
```

**Dependency direction:** `collect`/`train`/`drive` depend on `sim`, `input`, `data`, `model`
— never on `gym-donkeycar` directly. That's what makes the eventual real-car swap a one-file change.

---

## 5. Roadmap (milestones)

Each milestone is independently checkable — we don't move on until its acceptance test passes.

### M0 — Environment & connectivity
- Create venv, install `requirements.txt`. **Known landmine:** `gym==0.21` fails to build under modern
  `pip`/`setuptools`; pin older `setuptools`/`wheel` first (or use `--no-build-isolation`) and confirm the
  exact `gym-donkeycar`↔`gym` pair that resolves together. Document the working steps in the README.
- Download + launch the sim on Windows; configure `SIM_HOST`/port.
- **Controller spike:** attach the Xbox pad via `usbipd-win` and check whether `/dev/input/js*` appears.
  If yes → true analog input available; if no → stay on the pseudo-analog keyboard driver (don't sink
  time into a custom WSL kernel this phase).
- **Deliverable:** `config.py`, a connection check.
- **Acceptance:** `reset()` returns a `120×160×3` image; a few `step()` calls succeed; controller-spike
  outcome recorded (works / keyboard-only).

### M1 — Teleoperation & data capture
- Implement the input interface: **pseudo-analog keyboard driver (default)**; wire the Xbox driver only
  if the M0 spike showed it enumerates. Both emit normalized `(steering, throttle)`.
- Collection loop with the correct capture-loop ordering (§3.6): drive → save frames + `records.csv` +
  `meta.json`; start/stop recording.
- Capture a first dataset: clean laps both directions **and deliberate recovery driving** (steer back
  from the edge). Recovery data is a *first-class requirement*, not optional — it's the primary defence
  against covariate shift (§7), the top reason BC cars fail to finish laps.
- **Deliverable:** `input/controller.py`, `data/collect.py`, first `data/run_*`.
- **Acceptance:** drive N laps; images + CSV written; reload a run and confirm counts/ranges.

### M2 — Data pipeline
- `dataset.py`: load all runs, shared preprocessing, augmentation, train/val split.
- Sanity checks: batch shapes/ranges; steering-distribution histogram (spot label imbalance).
- **Deliverable:** `data/dataset.py`.
- **Acceptance:** dataset yields correct shapes/ranges; a batch can be visualized.

### M3 — Model & training
- Define our CNN; training script with checkpointing, early stopping, metric logging.
- **Deliverable:** `model/network.py`, `train.py`, a saved `models/pilot.keras`.
- **Acceptance:** training runs end to end; validation loss decreases; model file saved.

### M4 — Autonomous driving
- Inference loop: image → model → action, with throttle safety clamp; optional manual override.
- **Deliverable:** `drive.py`.
- **Acceptance:** the model drives the car in the sim and completes track segments/laps.

### M5 — Evaluate & iterate
- Measure performance (laps completed, cross-track error); collect more data where it fails;
  retune/augment; document results.
- **Deliverable:** short results notes; improved model.
- **Acceptance:** consistent autonomous laps; a recorded before/after comparison.

---

## 6. Verification / Testing

- **Unit/smoke (`tests/`):** model builds and outputs shape `(2,)` bounded to `−1..1`; dataset loads a
  tiny synthetic run with correct shapes/ranges; modules import cleanly.
- **Headless loop tests via the fake env (`sim/fakeenv.py`):** because the sim sits behind one interface
  (NFR1), a small in-memory fake with the same `reset()/step()` lets the *collect → train → drive* loops
  run in CI without the Unity sim — testing the wiring, not just imports.
- **Integration (manual, needs sim):** `python -m src.sim.environment` connection check;
  a short collect → train → drive pass on a small dataset.
- **End-to-end acceptance:** the M4/M5 criteria above (autonomous laps).

---

## 7. Risks & Mitigations
- **WSL sim/controller access** → sim on Windows + TCP; controller via usbipd or keyboard fallback.
- **`gym-donkeycar` / `gym` version conflicts** → pin versions; document a fallback in README.
- **Poor driving data → poor model** (garbage in, garbage out) → M1 capture guidance: smooth
  driving, both directions, recovery laps; M2 label-distribution check.
- **Covariate shift / compounding error (the #1 reason BC cars don't finish):** at drive time the car
  reaches states the human never demonstrated, makes a worse choice, and spirals. Mitigation is
  deliberate recovery data in M1 (elevated to a first-class requirement) and, if needed, more targeted
  capture where the car fails (M5). Named explicitly so it's designed for, not discovered.
- **Overfitting / model won't generalize** → dropout, augmentation, val split, early stopping.
- **Sim-to-real gap** (future) → keep model small + preprocessing simple; out of scope now.

## 8. Out of Scope (future phases)
- Reinforcement-learning training path (architecture leaves room; not built now).
- Real hardware: Pi 5 deployment, `cpp/` fast control loop, TFLite export, sim-to-real transfer.

---

## 9. Learning Notes (plain-English concepts)

Short explanations of the "why" behind the design, so you can defend every choice.

**Behavioral cloning (BC).** The car learns to *imitate you*. We record what you saw (the camera
image) and what you did (steering/throttle) at each instant. The model is trained to reproduce your
action given your view. It's called "cloning" because it copies a human policy. Strength: simple and
effective. Weakness: it only knows situations you demonstrated — if it drifts somewhere you never
drove, it can be lost (hence *recovery* driving in the data).

**Why a CNN (Convolutional Neural Network).** Images are grids of pixels; nearby pixels form
edges/lines/shapes. A CNN slides small learnable filters across the image to detect those patterns,
building up from edges → lane markings → "the road bends left." It's the standard tool for turning an
image into a decision, and far more data-efficient on images than plain dense networks.

**Regression vs. classification.** We predict *continuous numbers* (a steering angle can be any value
in −1..1), which is **regression**. Classification would instead pick from fixed buckets ("hard left",
"straight", ...). Regression gives smoother control, which suits driving.

**MSE loss (Mean Squared Error).** "Loss" measures how wrong a prediction is; training nudges the
model to make loss smaller. MSE = average of (prediction − truth)². Squaring punishes big mistakes
more than small ones and is the natural fit for regression.

**Train / validation split.** We hold back some data the model never trains on (validation). If
training loss keeps dropping but validation loss rises, the model is **overfitting** — memorizing
frames instead of learning to drive. Validation is our honesty check.

**Overfitting & the tools against it.** Dropout (randomly ignore some neurons during training),
augmentation (show varied versions of frames), and early stopping (halt when validation stops
improving) all push the model to learn general road-following, not to memorize your exact laps.

**Epochs & batches.** One **epoch** = one pass over all training data. Data is fed in **batches**
(e.g. 64 frames) because it's faster and more stable than one frame at a time. We train for many
epochs until validation loss flattens.

**Normalization (÷255).** Pixels are 0–255; networks train more stably when inputs are small and
consistent (0–1). It's a tiny step with an outsized effect on training stability.

**The label-imbalance trap (see 3.5).** Because most driving is straight, a model can "cheat" by
always predicting ~0 steering and still look accurate on average — then fail every corner. Watching
the steering distribution and capturing enough cornering data is how we avoid it. This is the single
most common reason a first BC model won't turn.
