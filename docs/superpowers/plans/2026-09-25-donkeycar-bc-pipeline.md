# DonkeyCar Behavioral-Cloning Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, from scratch, the ML pipeline that lets a custom CNN drive the DonkeyCar simulator autonomously — teleop data capture, a shared preprocessing/augmentation pipeline, a custom PilotNet-style network, training, and an autonomous drive loop.

**Architecture:** A layered Python package where the only file that touches `gym-donkeycar` is `src/sim/environment.py`; everything else depends on our own interfaces. A `FakeEnvironment` with the same `reset()/step()` contract makes the collect → train → drive loops testable headless. One `config.py` is the single source of truth for the image/action contracts, paths, and hyperparameters, and one shared `preprocessImage` is used identically at train and drive time.

**Tech Stack:** Python 3.10/3.11, TensorFlow 2.x (Keras), `gym-donkeycar` (sim pipe only), `pygame` (input), `opencv-python` + `numpy` (image I/O), `pytest`.

**Spec:** `PLAN.md` (repo root — the user's chosen spec location). The plan argues from that spec; executors read both.

## Global Constraints

- **Code style (project CLAUDE.md):** identifiers use **camelCase** (functions, variables, methods) — including in Python. Indentation uses **8-column hard tabs** for statements, 4-column continuation indents (BSD KNF). The code blocks below use spaces for readability; the implementer converts leading indentation to hard tabs and keeps camelCase names verbatim.
- **Python:** 3.10 or 3.11 only.
- **Image contract:** every image is `120 × 160 × 3` (sim-native), normalized to float32 `[0, 1]`. Cropping (top `CROP_TOP` rows) happens **inside the model** as a `Cropping2D` layer, never in the shared preprocess.
- **Action contract:** `[steering, throttle]`, each in `−1.0 … +1.0`.
- **Model I/O:** in = image `[120,160,3]` (uint8 or float pre-normalized); out = `[steering, throttle]` via a **`tanh`** output layer (structurally bounded to `−1..1`).
- **Capture-loop causality:** always `record(obs, action)` **before** `env.step(action)`. The image paired with an action is the one the human saw when deciding.
- **Validation split is by whole run**, never per-frame (adjacent frames are near-duplicates; per-frame leaks).
- **`config.py` is the single source of truth**; no magic numbers duplicated in other modules. `setSeed()` is called at the start of every training/collection entry point for reproducibility.
- **Sim isolation:** only `src/sim/environment.py` may `import gym_donkeycar` / `import gym`. No other module references the simulator library.
- **Package layout:** `src/` is a package; every directory under it (and `src/` itself) has an `__init__.py`. Tests run from repo root with `pytest`.

## Review Focus

These are input classes/failure modes the spec implies that are most likely to bite a user; each has a test pinned to the owning task.

1. **Capture-loop off-by-one** — if `collectRun` records the frame returned *by* `step()` instead of the frame the human saw, every label is shifted one frame and all data is corrupt. → Task 7 asserts exact `(image, action)` pairing using an identifiable fake env.
2. **Single-run dataset** — a user collects one run and trains; a by-run split can't hold out a whole run from one run. Must fail loudly with a clear message, not silently train with an empty val set. → Task 6 tests the `< 2 runs` error.
3. **Empty / zero-frame run** — a run directory with no images (recording started and stopped instantly) must raise a clear error, not yield a NaN-producing empty dataset. → Task 6 tests the empty-run error.
4. **Throttle safety clamp** — the model can predict throttle above the safe limit; drive must clamp to `THROTTLE_MAX` before sending to the sim. → Task 10 tests clamping.
5. **Out-of-range model output** — actions fed to the sim must stay in `−1..1`; the `tanh` head guarantees it. → Task 8 asserts every output is within `[-1, 1]`.

---

## File Structure

Created by this plan:

- `config.py` — single source of truth (contracts, paths, hyperparams, `setSeed`).
- `src/__init__.py`, `src/sim/__init__.py`, `src/input/__init__.py`, `src/data/__init__.py`, `src/model/__init__.py` — package markers.
- `src/sim/environment.py` — `SimEnvironment`: the only file that talks to `gym-donkeycar`.
- `src/sim/fakeenv.py` — `FakeEnvironment`: same interface, deterministic synthetic frames (tests).
- `src/input/controller.py` — input interface + `KeyboardDriver` (pseudo-analog, default) + optional `XboxDriver`; pure `rampToward` helper.
- `src/data/preprocess.py` — `preprocessImage` (the one shared normalize step for train & drive).
- `src/data/augment.py` — `randomFlip`, `randomBrightness`, `applyAugmentation` (train only).
- `src/data/dataset.py` — run discovery, by-run split, `tf.data` assembly, imbalance warning.
- `src/data/collect.py` — teleop capture loop → writes `imgs/`, `records.csv`, `meta.json`.
- `src/model/network.py` — `buildModel`: our CNN with `tanh` head.
- `src/train.py` — `trainModel`: build → fit (MSE) → checkpoint/early-stop → save model + metrics.
- `src/drive.py` — `driveLoop`: load model → inference → clamp → step.
- `tests/…` — one test module per source module.
- `README.md` — quickstart + WSL/install troubleshooting (folded into Task 1 and Task 2).

Task dependency order (each builds only on earlier ones): config → sim/fakeenv → input → preprocess → augment → dataset → collect → network → train → drive.

---

### Task 1: Config + reproducibility (`config.py`)

**Files:**
- Create: `config.py`
- Create: `src/__init__.py` (empty), and empty `__init__.py` in `src/sim/`, `src/input/`, `src/data/`, `src/model/`
- Create: `README.md` (quickstart skeleton)
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: module-level constants and `setSeed(seed: int = SEED) -> None`. Key names other tasks rely on:
  `IMAGE_H=120, IMAGE_W=160, IMAGE_C=3, CROP_TOP=40, ACTION_DIM=2`,
  `DATA_DIR: Path, MODELS_DIR: Path, MODEL_PATH: Path, METRICS_PATH: Path`,
  `SIM_HOST: str, SIM_PORT: int, SIM_TRACK: str`,
  `FPS_TARGET=20, BATCH_SIZE=64, EPOCHS=40, LEARNING_RATE=1e-3, VAL_SPLIT=0.2, SHUFFLE_BUFFER=2000, SEED=42`,
  `THROTTLE_MAX=0.5, DEFAULT_THROTTLE=0.35`,
  `STEER_RATE=3.0, THROTTLE_RATE=2.0, RETURN_RATE=4.0` (pseudo-analog ramps, units/sec),
  `AUGMENT_FLIP=True, BRIGHTNESS_DELTA=0.2`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
import config

def testContractConstantsAreCorrect():
    assert config.IMAGE_H == 120
    assert config.IMAGE_W == 160
    assert config.IMAGE_C == 3
    assert config.ACTION_DIM == 2
    assert 0 < config.CROP_TOP < config.IMAGE_H

def testHyperparamRangesAreSane():
    assert 0.0 < config.VAL_SPLIT < 1.0
    assert config.BATCH_SIZE > 0
    assert 0.0 < config.THROTTLE_MAX <= 1.0

def testSetSeedIsDeterministic():
    import numpy as np
    config.setSeed(123)
    first = np.random.rand(5)
    config.setSeed(123)
    second = np.random.rand(5)
    assert np.allclose(first, second)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'config'`.

- [ ] **Step 3: Write minimal implementation**

```python
# config.py — single source of truth for the whole pipeline.
import os
import random
from pathlib import Path

# --- Simulator connection (WSL2: point at Windows host if mirrored net is off) ---
SIM_HOST = os.environ.get("SIM_HOST", "127.0.0.1")
SIM_PORT = int(os.environ.get("SIM_PORT", "9091"))
SIM_TRACK = os.environ.get("SIM_TRACK", "donkey-generated-track-v0")

# --- Image / action contracts (must match collect, train, drive) ---
IMAGE_H = 120
IMAGE_W = 160
IMAGE_C = 3
CROP_TOP = 40          # rows removed inside the model (sky/horizon ~ top 1/3)
ACTION_DIM = 2         # [steering, throttle]

# --- Paths ---
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "pilot.keras"
METRICS_PATH = MODELS_DIR / "metrics.json"

# --- Capture / control ---
FPS_TARGET = 20
THROTTLE_MAX = 0.5     # drive-time safety clamp on model throttle
DEFAULT_THROTTLE = 0.35  # used if predicting steering-only

# Pseudo-analog keyboard ramps (axis units per second)
STEER_RATE = 3.0
THROTTLE_RATE = 2.0
RETURN_RATE = 4.0      # how fast an axis decays to 0 when no key is held

# --- Training hyperparameters ---
BATCH_SIZE = 64
EPOCHS = 40
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.2
SHUFFLE_BUFFER = 2000
SEED = 42

# --- Augmentation ---
AUGMENT_FLIP = True
BRIGHTNESS_DELTA = 0.2


def setSeed(seed: int = SEED) -> None:
    """Seed python/numpy/tf so experiments are reproducible."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass  # tf not required for non-training entry points
```

Also create the empty `__init__.py` files and a `README.md` with a "## Quickstart" heading and a "## Troubleshooting install" placeholder heading (filled in Task 2).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add config.py src/ tests/test_config.py README.md
git commit -m "feat: add config single source of truth and package skeleton"
```

---

### Task 2: Sim wrapper + fake env (`src/sim/environment.py`, `src/sim/fakeenv.py`)

**Files:**
- Create: `src/sim/environment.py`
- Create: `src/sim/fakeenv.py`
- Modify: `README.md` (fill "Troubleshooting install")
- Test: `tests/sim/test_fakeenv.py`

**Interfaces:**
- Consumes: `config` (image shape, sim host/port/track).
- Produces: the environment contract shared by real and fake:
  - `reset() -> np.ndarray` — uint8 image `[120,160,3]`.
  - `step(action) -> tuple[np.ndarray, float, bool, dict]` — `(obs, reward, done, info)`; `action` is a 2-sequence `[steering, throttle]`; `info` contains at least `cte` and `speed`.
  - `close() -> None`.
  - `FakeEnvironment(frames: int = 100)` — deterministic; the returned frame encodes its step index in pixel `[0,0,0]` (value = `stepIndex % 256`) so tests can prove pairing; sets `done=True` after `frames` steps.

- [ ] **Step 1: Write the failing test**

```python
# tests/sim/test_fakeenv.py
import numpy as np
from src.sim.fakeenv import FakeEnvironment
import config

def testResetReturnsContractImage():
    env = FakeEnvironment(frames=10)
    obs = env.reset()
    assert obs.shape == (config.IMAGE_H, config.IMAGE_W, config.IMAGE_C)
    assert obs.dtype == np.uint8

def testStepReturnsTupleAndAdvancesIndex():
    env = FakeEnvironment(frames=10)
    env.reset()
    obs, reward, done, info = env.step([0.0, 0.3])
    assert obs.shape == (config.IMAGE_H, config.IMAGE_W, config.IMAGE_C)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert "cte" in info and "speed" in info

def testFrameEncodesStepIndexForPairingProofs():
    env = FakeEnvironment(frames=10)
    firstSeen = env.reset()
    assert int(firstSeen[0, 0, 0]) == 0          # reset frame is index 0
    obs, _, _, _ = env.step([0.0, 0.0])
    assert int(obs[0, 0, 0]) == 1                 # step returns the NEXT frame

def testDoneAfterConfiguredFrames():
    env = FakeEnvironment(frames=3)
    env.reset()
    dones = [env.step([0.0, 0.0])[2] for _ in range(3)]
    assert dones[-1] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/sim/test_fakeenv.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.sim.fakeenv'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/sim/fakeenv.py — deterministic stand-in for the sim; NO gym import.
import numpy as np
import config


class FakeEnvironment:
    """Same interface as SimEnvironment, for headless tests.

    The frame at step index i has pixel [0,0,0] == i % 256, so a test can
    prove which frame was paired with which action.
    """

    def __init__(self, frames: int = 100):
        self.maxFrames = frames
        self.stepIndex = 0

    def _frame(self, index: int) -> np.ndarray:
        img = np.full(
            (config.IMAGE_H, config.IMAGE_W, config.IMAGE_C),
            fill_value=(index % 256),
            dtype=np.uint8,
        )
        return img

    def reset(self) -> np.ndarray:
        self.stepIndex = 0
        return self._frame(0)

    def step(self, action):
        self.stepIndex += 1
        obs = self._frame(self.stepIndex)
        reward = 1.0
        done = self.stepIndex >= self.maxFrames
        info = {"cte": 0.0, "speed": float(action[1]) * 10.0}
        return obs, reward, done, info

    def close(self) -> None:
        pass
```

```python
# src/sim/environment.py — the ONLY module that talks to gym-donkeycar.
import gym
import gym_donkeycar  # noqa: F401  (registers the donkey envs)
import numpy as np
import config


class SimEnvironment:
    """Thin wrapper over gym-donkeycar exposing reset/step/close.

    Isolates the simulator so a real-car env can replace just this file later.
    """

    def __init__(self, host: str = config.SIM_HOST, port: int = config.SIM_PORT,
                 track: str = config.SIM_TRACK):
        conf = {"host": host, "port": port}
        self.env = gym.make(track, conf=conf)

    def reset(self) -> np.ndarray:
        obs = self.env.reset()
        return np.asarray(obs, dtype=np.uint8)

    def step(self, action):
        obs, reward, done, info = self.env.step(
            np.asarray([action[0], action[1]], dtype=np.float32)
        )
        return np.asarray(obs, dtype=np.uint8), float(reward), bool(done), dict(info)

    def close(self) -> None:
        self.env.close()
```

In `README.md`, fill the "## Troubleshooting install" section with the known `gym==0.21` build failure and the fix:

```markdown
## Troubleshooting install

`gym==0.21` fails to build under modern pip/setuptools. Install the build
tools it expects first, then the requirements:

    pip install "setuptools==65.5.0" "wheel==0.38.4"
    pip install --no-build-isolation -r requirements.txt

If `gym-donkeycar` and `gym` still conflict, pin the exact pair that resolves
together (see the versions printed by `pip install` and adjust requirements.txt).
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/sim/test_fakeenv.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Manual acceptance (requires the sim — M0)**

Launch the Windows sim, set `SIM_HOST`/`SIM_PORT`, then:

```bash
python -c "from src.sim.environment import SimEnvironment; e=SimEnvironment(); o=e.reset(); print(o.shape); [e.step([0.0,0.2]) for _ in range(5)]; e.close(); print('ok')"
```

Expected: prints `(120, 160, 3)` then `ok`. Record the controller-spike outcome (does `/dev/input/js*` appear after `usbipd` attach?) in the README.

- [ ] **Step 6: Commit**

```bash
git add src/sim/environment.py src/sim/fakeenv.py README.md tests/sim/test_fakeenv.py
git commit -m "feat: add sim wrapper and deterministic fake environment"
```

---

### Task 3: Input interface + pseudo-analog keyboard (`src/input/controller.py`)

**Files:**
- Create: `src/input/controller.py`
- Test: `tests/input/test_controller.py`

**Interfaces:**
- Consumes: `config` (ramp rates).
- Produces:
  - `rampToward(current: float, target: float, rate: float, dt: float) -> float` — pure; moves `current` toward `target` by at most `rate*dt`, clamped to `[-1, 1]`.
  - `InputReading` (dataclass): `steering: float, throttle: float, recording: bool, quit: bool`.
  - `InputDriver` (base): `poll(dt: float) -> InputReading`, `close() -> None`.
  - `KeyboardDriver(InputDriver)` — pygame-backed, uses `rampToward`; arrows/WASD ramp axes, `r` toggles recording, `q`/ESC quits.
  - `XboxDriver(InputDriver)` — optional; reads axes 0 (steer) and 5/2 (throttle) via pygame joystick; only constructed when a joystick is present.
  - `makeDriver() -> InputDriver` — returns `XboxDriver` if a joystick enumerates, else `KeyboardDriver`.

- [ ] **Step 1: Write the failing test** (pure ramp math — no pygame needed)

```python
# tests/input/test_controller.py
from src.input.controller import rampToward

def testRampMovesTowardTargetByRateTimesDt():
    # rate 2.0/s over 0.1s => +0.2 toward target
    assert abs(rampToward(0.0, 1.0, 2.0, 0.1) - 0.2) < 1e-9

def testRampDoesNotOvershootTarget():
    assert rampToward(0.9, 1.0, 5.0, 1.0) == 1.0

def testRampClampsToUnitRange():
    assert rampToward(0.95, 5.0, 100.0, 1.0) == 1.0
    assert rampToward(-0.95, -5.0, 100.0, 1.0) == -1.0

def testRampReturnsTowardZeroWhenTargetIsZero():
    assert abs(rampToward(0.5, 0.0, 4.0, 0.1) - 0.1) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/input/test_controller.py -v`
Expected: FAIL with `ImportError: cannot import name 'rampToward'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/input/controller.py
from dataclasses import dataclass
import config


def rampToward(current: float, target: float, rate: float, dt: float) -> float:
    """Move current toward target by at most rate*dt, clamped to [-1, 1]."""
    step = rate * dt
    if target > current:
        current = min(current + step, target)
    else:
        current = max(current - step, target)
    return max(-1.0, min(1.0, current))


@dataclass
class InputReading:
    steering: float
    throttle: float
    recording: bool
    quit: bool


class InputDriver:
    def poll(self, dt: float) -> InputReading:
        raise NotImplementedError

    def close(self) -> None:
        pass


class KeyboardDriver(InputDriver):
    """Pseudo-analog: holding a key ramps the axis; releasing decays to 0."""

    def __init__(self):
        import pygame
        self.pygame = pygame
        if not pygame.get_init():
            pygame.init()
        # a tiny window is required for pygame to receive key events
        self.screen = pygame.display.set_mode((320, 120))
        pygame.display.set_caption("DonkeyCar teleop — arrows/WASD, r=record, q=quit")
        self.steering = 0.0
        self.throttle = 0.0
        self.recording = False
        self.quit = False

    def poll(self, dt: float) -> InputReading:
        pg = self.pygame
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.quit = True
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_r:
                    self.recording = not self.recording
                elif event.key in (pg.K_q, pg.K_ESCAPE):
                    self.quit = True
        keys = pg.key.get_pressed()
        steerTarget = 0.0
        if keys[pg.K_LEFT] or keys[pg.K_a]:
            steerTarget = -1.0
        elif keys[pg.K_RIGHT] or keys[pg.K_d]:
            steerTarget = 1.0
        throttleTarget = 0.0
        if keys[pg.K_UP] or keys[pg.K_w]:
            throttleTarget = 1.0
        elif keys[pg.K_DOWN] or keys[pg.K_s]:
            throttleTarget = -1.0
        steerRate = config.STEER_RATE if steerTarget != 0.0 else config.RETURN_RATE
        throttleRate = config.THROTTLE_RATE if throttleTarget != 0.0 else config.RETURN_RATE
        self.steering = rampToward(self.steering, steerTarget, steerRate, dt)
        self.throttle = rampToward(self.throttle, throttleTarget, throttleRate, dt)
        return InputReading(self.steering, self.throttle, self.recording, self.quit)

    def close(self) -> None:
        self.pygame.quit()


class XboxDriver(InputDriver):
    """Optional analog driver; only build when a joystick is present."""

    def __init__(self):
        import pygame
        self.pygame = pygame
        if not pygame.get_init():
            pygame.init()
        pygame.joystick.init()
        self.js = pygame.joystick.Joystick(0)
        self.js.init()
        self.recording = False
        self.quit = False

    def poll(self, dt: float) -> InputReading:
        pg = self.pygame
        for event in pg.event.get():
            if event.type == pg.JOYBUTTONDOWN:
                if event.button == 0:      # A toggles recording
                    self.recording = not self.recording
                elif event.button == 1:    # B quits
                    self.quit = True
        steering = float(self.js.get_axis(0))
        # right trigger (axis 5) mapped from [-1,1] rest=-1 to throttle [0,1]
        rawThrottle = float(self.js.get_axis(5))
        throttle = (rawThrottle + 1.0) / 2.0
        return InputReading(
            max(-1.0, min(1.0, steering)),
            max(-1.0, min(1.0, throttle)),
            self.recording,
            self.quit,
        )

    def close(self) -> None:
        self.pygame.quit()


def makeDriver() -> InputDriver:
    """Xbox if a joystick enumerates, else pseudo-analog keyboard."""
    try:
        import pygame
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() > 0:
            return XboxDriver()
    except Exception:
        pass
    return KeyboardDriver()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/input/test_controller.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Manual acceptance (requires a display / M1)**

`python -c "from src.input.controller import KeyboardDriver; d=KeyboardDriver(); import time;\n\nprint('hold arrows; ctrl-c to stop')\n[print(round(d.poll(0.05).steering,2), round(d.poll(0.05).throttle,2)) or time.sleep(0.05) for _ in range(100)]"`
Expected: holding an arrow ramps the value smoothly toward ±1; releasing decays toward 0.

- [ ] **Step 6: Commit**

```bash
git add src/input/controller.py tests/input/test_controller.py
git commit -m "feat: add input interface with pseudo-analog keyboard driver"
```

---

### Task 4: Shared preprocessing (`src/data/preprocess.py`)

**Files:**
- Create: `src/data/preprocess.py`
- Test: `tests/data/test_preprocess.py`

**Interfaces:**
- Consumes: `config`.
- Produces: `preprocessImage(image) -> tf.Tensor` — accepts a uint8 image (numpy or tensor, shape `[120,160,3]`), returns float32 in `[0,1]`, same H/W/C. Uses tf ops so it drops directly into `tf.data`; at drive time call it and `.numpy()`. This is the ONE normalization shared by train and drive.

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_preprocess.py
import numpy as np
from src.data.preprocess import preprocessImage
import config

def testOutputIsFloatInUnitRange():
    img = np.full((config.IMAGE_H, config.IMAGE_W, config.IMAGE_C), 255, dtype=np.uint8)
    out = preprocessImage(img).numpy()
    assert out.dtype == np.float32
    assert out.max() <= 1.0 and out.min() >= 0.0
    assert np.allclose(out, 1.0)

def testShapeIsPreserved():
    img = np.zeros((config.IMAGE_H, config.IMAGE_W, config.IMAGE_C), dtype=np.uint8)
    out = preprocessImage(img).numpy()
    assert out.shape == (config.IMAGE_H, config.IMAGE_W, config.IMAGE_C)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_preprocess.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.data.preprocess'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/data/preprocess.py — the single normalize step used at train AND drive.
import tensorflow as tf


def preprocessImage(image):
    """uint8 [120,160,3] -> float32 [0,1]. Cropping happens inside the model."""
    image = tf.cast(image, tf.float32) / 255.0
    return image
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/data/test_preprocess.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/data/preprocess.py tests/data/test_preprocess.py
git commit -m "feat: add shared image preprocessing used by train and drive"
```

---

### Task 5: Augmentation (`src/data/augment.py`)

**Files:**
- Create: `src/data/augment.py`
- Test: `tests/data/test_augment.py`

**Interfaces:**
- Consumes: `config` (`BRIGHTNESS_DELTA`).
- Produces (all operate on `(image: tf.Tensor float32 [H,W,3], label: tf.Tensor [2])`):
  - `flipImageAndLabel(image, label) -> (image, label)` — mirror L↔R and **negate steering** (label[0]); throttle unchanged.
  - `randomBrightness(image, label, maxDelta) -> (image, label)` — jitter brightness, label unchanged, result clipped to `[0,1]`.
  - `applyAugmentation(image, label) -> (image, label)` — 50% chance flip, then brightness jitter. Train only.

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_augment.py
import numpy as np
import tensorflow as tf
from src.data.augment import flipImageAndLabel, randomBrightness

def testFlipNegatesSteeringAndMirrorsImage():
    img = tf.constant(np.arange(120*160*3, dtype=np.float32).reshape(120,160,3) / (120*160*3))
    label = tf.constant([0.4, 0.3], dtype=tf.float32)
    outImg, outLabel = flipImageAndLabel(img, label)
    # steering negated, throttle unchanged
    assert np.isclose(outLabel.numpy()[0], -0.4)
    assert np.isclose(outLabel.numpy()[1], 0.3)
    # image mirrored horizontally: column 0 of output == column last of input
    assert np.allclose(outImg.numpy()[:, 0, :], img.numpy()[:, -1, :])

def testBrightnessChangesPixelsNotLabelAndStaysInRange():
    img = tf.fill((120,160,3), 0.5)
    label = tf.constant([0.1, 0.2], dtype=tf.float32)
    outImg, outLabel = randomBrightness(img, label, 0.2)
    assert np.allclose(outLabel.numpy(), label.numpy())
    assert outImg.numpy().max() <= 1.0 and outImg.numpy().min() >= 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_augment.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.data.augment'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/data/augment.py — train-only augmentation. Never applied to val/drive.
import tensorflow as tf
import config


def flipImageAndLabel(image, label):
    """Mirror image L<->R and negate steering (label[0])."""
    flipped = tf.image.flip_left_right(image)
    steering = -label[0]
    throttle = label[1]
    return flipped, tf.stack([steering, throttle])


def randomBrightness(image, label, maxDelta=config.BRIGHTNESS_DELTA):
    """Jitter brightness; label unchanged; keep pixels in [0,1]."""
    image = tf.image.random_brightness(image, maxDelta)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


def applyAugmentation(image, label):
    """50% flip, then brightness jitter. Train split only."""
    if config.AUGMENT_FLIP:
        doFlip = tf.random.uniform([]) < 0.5
        image, label = tf.cond(
            doFlip,
            lambda: flipImageAndLabel(image, label),
            lambda: (image, label),
        )
    image, label = randomBrightness(image, label)
    return image, label
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/data/test_augment.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/data/augment.py tests/data/test_augment.py
git commit -m "feat: add train-only augmentation with correct flip label math"
```

---

### Task 6: Dataset assembly with by-run split (`src/data/dataset.py`)

**Files:**
- Create: `src/data/dataset.py`
- Test: `tests/data/test_dataset.py`

**Interfaces:**
- Consumes: `config`, `preprocessImage` (Task 4), `applyAugmentation` (Task 5).
- Produces:
  - `listRuns(dataDir) -> list[Path]` — sorted `run_*` dirs containing `records.csv`.
  - `readRunSamples(runDir) -> list[tuple[str, float, float]]` — `(absImagePath, steering, throttle)` rows.
  - `splitRunsByRun(runs, valSplit, seed) -> tuple[list[Path], list[Path]]` — holds out **whole runs**; raises `ValueError` if fewer than 2 runs.
  - `buildSamples(runs) -> tuple[list[str], np.ndarray]` — flat image paths + `float32 [N,2]` labels; raises `ValueError` if total frames == 0.
  - `makeDataset(paths, labels, training, batchSize, shuffleBuffer) -> tf.data.Dataset` — decodes JPEG, `preprocessImage`, augments **iff** `training`, shuffles+batches.
  - `loadDatasets(dataDir=config.DATA_DIR) -> tuple[tf.data.Dataset, tf.data.Dataset]` — the train/val pair.
  - `warnOnImbalance(labels) -> float` — returns fraction of |steering|<0.05 and prints a warning if > 0.7.

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_dataset.py
import csv, os
import numpy as np
import cv2
import pytest
import config
from src.data import dataset as ds

def _writeRun(root, name, frames, steering=0.5):
    runDir = os.path.join(root, name)
    os.makedirs(os.path.join(runDir, "imgs"))
    with open(os.path.join(runDir, "records.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "image", "steering", "throttle", "timestamp", "speed", "cte"])
        for i in range(frames):
            rel = f"imgs/{i:06d}.jpg"
            img = np.full((config.IMAGE_H, config.IMAGE_W, 3), 128, np.uint8)
            cv2.imwrite(os.path.join(runDir, rel), img)
            w.writerow([i, rel, steering, 0.3, 0.0, 5.0, 0.0])
    return runDir

def testSplitByRunHoldsOutWholeRuns(tmp_path):
    _writeRun(tmp_path, "run_a", 3)
    _writeRun(tmp_path, "run_b", 3)
    _writeRun(tmp_path, "run_c", 3)
    runs = ds.listRuns(tmp_path)
    train, val = ds.splitRunsByRun(runs, 0.34, seed=1)
    assert len(val) >= 1
    assert set(train).isdisjoint(set(val))   # no run in both

def testSplitRaisesWithSingleRun(tmp_path):
    _writeRun(tmp_path, "run_only", 5)
    runs = ds.listRuns(tmp_path)
    with pytest.raises(ValueError):
        ds.splitRunsByRun(runs, 0.2, seed=1)

def testBuildSamplesRaisesOnZeroFrames(tmp_path):
    _writeRun(tmp_path, "run_empty", 0)
    runs = ds.listRuns(tmp_path)
    with pytest.raises(ValueError):
        ds.buildSamples(runs)

def testDatasetYieldsContractBatch(tmp_path):
    _writeRun(tmp_path, "run_a", 4)
    _writeRun(tmp_path, "run_b", 4)
    runs = ds.listRuns(tmp_path)
    paths, labels = ds.buildSamples(runs)
    d = ds.makeDataset(paths, labels, training=False, batchSize=2, shuffleBuffer=8)
    images, lbls = next(iter(d))
    assert images.shape[1:] == (config.IMAGE_H, config.IMAGE_W, config.IMAGE_C)
    assert lbls.shape[1:] == (config.ACTION_DIM,)
    assert float(images.numpy().max()) <= 1.0 and float(images.numpy().min()) >= 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.data.dataset'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/data/dataset.py — runs -> tf.data. Split is BY RUN, never per-frame.
import csv
import random
from pathlib import Path
import numpy as np
import tensorflow as tf
import config
from src.data.preprocess import preprocessImage
from src.data.augment import applyAugmentation


def listRuns(dataDir) -> list:
    dataDir = Path(dataDir)
    runs = [p for p in sorted(dataDir.glob("run_*")) if (p / "records.csv").exists()]
    return runs


def readRunSamples(runDir) -> list:
    runDir = Path(runDir)
    samples = []
    with open(runDir / "records.csv", newline="") as f:
        for row in csv.DictReader(f):
            imgPath = str(runDir / row["image"])
            samples.append((imgPath, float(row["steering"]), float(row["throttle"])))
    return samples


def splitRunsByRun(runs, valSplit=config.VAL_SPLIT, seed=config.SEED):
    if len(runs) < 2:
        raise ValueError(
            "Need at least 2 runs to hold out a whole run for validation; "
            f"found {len(runs)}. Collect another run (M1)."
        )
    runs = list(runs)
    random.Random(seed).shuffle(runs)
    nVal = max(1, round(len(runs) * valSplit))
    valRuns = runs[:nVal]
    trainRuns = runs[nVal:]
    return trainRuns, valRuns


def buildSamples(runs):
    paths, labels = [], []
    for run in runs:
        for imgPath, steering, throttle in readRunSamples(run):
            paths.append(imgPath)
            labels.append([steering, throttle])
    if not paths:
        raise ValueError("No frames found in the given runs (zero-frame dataset).")
    return paths, np.asarray(labels, dtype=np.float32)


def _loadImage(path, label):
    raw = tf.io.read_file(path)
    img = tf.io.decode_jpeg(raw, channels=config.IMAGE_C)
    img = tf.ensure_shape(img, (config.IMAGE_H, config.IMAGE_W, config.IMAGE_C))
    return preprocessImage(img), label


def makeDataset(paths, labels, training, batchSize=config.BATCH_SIZE,
                shuffleBuffer=config.SHUFFLE_BUFFER):
    d = tf.data.Dataset.from_tensor_slices((paths, labels))
    d = d.map(_loadImage, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        d = d.map(applyAugmentation, num_parallel_calls=tf.data.AUTOTUNE)
        d = d.shuffle(shuffleBuffer, seed=config.SEED, reshuffle_each_iteration=True)
    d = d.batch(batchSize).prefetch(tf.data.AUTOTUNE)
    return d


def warnOnImbalance(labels) -> float:
    steering = np.asarray(labels)[:, 0]
    frac = float(np.mean(np.abs(steering) < 0.05))
    if frac > 0.7:
        print(f"WARNING: {frac:.0%} of frames are ~straight. Capture more "
              "cornering/recovery data or downsample straights (PLAN 3.5).")
    return frac


def loadDatasets(dataDir=config.DATA_DIR):
    runs = listRuns(dataDir)
    trainRuns, valRuns = splitRunsByRun(runs)
    trainPaths, trainLabels = buildSamples(trainRuns)
    valPaths, valLabels = buildSamples(valRuns)
    warnOnImbalance(trainLabels)
    trainDs = makeDataset(trainPaths, trainLabels, training=True)
    valDs = makeDataset(valPaths, valLabels, training=False)
    return trainDs, valDs
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/data/test_dataset.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/data/dataset.py tests/data/test_dataset.py
git commit -m "feat: add dataset with by-run split, imbalance check, tf.data pipeline"
```

---

### Task 7: Data collection loop (`src/data/collect.py`)

**Files:**
- Create: `src/data/collect.py`
- Test: `tests/data/test_collect.py`

**Interfaces:**
- Consumes: `config`, the env contract (Task 2), the input contract (`InputReading`, Task 3).
- Produces:
  - `collectRun(env, driver, dataDir, meta, maxSteps=None) -> Path` — runs the teleop loop. **Records `(obs, action)` before stepping** (causality). Writes `imgs/NNNNNN.jpg`, `records.csv` (schema per PLAN §3.1), `meta.json`. Only frames while `reading.recording` is True are saved. Stops on `reading.quit`, env `done`, or `maxSteps`.
  - `main()` — wires `SimEnvironment` + `makeDriver()` and calls `collectRun`.

- [ ] **Step 1: Write the failing test** (uses fake env + a scripted driver; proves pairing)

```python
# tests/data/test_collect.py
import csv, json, os
import numpy as np
from src.sim.fakeenv import FakeEnvironment
from src.input.controller import InputReading
from src.data.collect import collectRun

class ScriptedDriver:
    """Yields preset readings; records from the first poll."""
    def __init__(self, readings):
        self.readings = list(readings)
        self.i = 0
    def poll(self, dt):
        r = self.readings[min(self.i, len(self.readings) - 1)]
        self.i += 1
        return r
    def close(self):
        pass

def testCollectWritesFramesCsvAndMeta(tmp_path):
    env = FakeEnvironment(frames=100)
    readings = [InputReading(0.2 * i, 0.3, recording=True, quit=False) for i in range(3)]
    readings.append(InputReading(0.0, 0.0, recording=True, quit=True))
    driver = ScriptedDriver(readings)
    runDir = collectRun(env, driver, tmp_path, meta={"track": "t", "controller_name": "scripted"})
    assert os.path.isdir(os.path.join(runDir, "imgs"))
    with open(os.path.join(runDir, "records.csv")) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    with open(os.path.join(runDir, "meta.json")) as f:
        meta = json.load(f)
    assert meta["frame_count"] == 3

def testCausalityImageMatchesActionSeen(tmp_path):
    # FakeEnvironment frame i has pixel[0,0,0]==i. The human "sees" frame i
    # and applies action i, so recorded frame i must pair with action i.
    env = FakeEnvironment(frames=100)
    readings = [InputReading(float(i) / 10.0, 0.3, recording=True, quit=False) for i in range(3)]
    readings.append(InputReading(0.0, 0.0, recording=True, quit=True))
    driver = ScriptedDriver(readings)
    runDir = collectRun(env, driver, tmp_path, meta={"track": "t", "controller_name": "s"})
    import cv2
    with open(os.path.join(runDir, "records.csv")) as f:
        rows = list(csv.DictReader(f))
    for i, row in enumerate(rows):
        img = cv2.imread(os.path.join(runDir, row["image"]))
        assert int(img[0, 0, 0]) == i                     # frame i was the SEEN frame
        assert abs(float(row["steering"]) - i / 10.0) < 1e-6  # paired with action i
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_collect.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.data.collect'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/data/collect.py — teleop capture. Record BEFORE stepping (causality).
import csv
import json
import time
from datetime import datetime
from pathlib import Path
import cv2
import config


def collectRun(env, driver, dataDir, meta, maxSteps=None) -> Path:
    config.setSeed()
    dataDir = Path(dataDir)
    runName = "run_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    runDir = dataDir / runName
    imgsDir = runDir / "imgs"
    imgsDir.mkdir(parents=True, exist_ok=True)

    obs = env.reset()
    frame = 0
    period = 1.0 / config.FPS_TARGET
    csvPath = runDir / "records.csv"
    with open(csvPath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "image", "steering", "throttle",
                         "timestamp", "speed", "cte"])
        steps = 0
        while True:
            reading = driver.poll(period)
            if reading.quit:
                break
            if reading.recording:
                # obs is the frame the human is reacting to -> pair with this action.
                rel = f"imgs/{frame:06d}.jpg"
                cv2.imwrite(str(runDir / rel), obs)
                writer.writerow([frame, rel, reading.steering, reading.throttle,
                                 time.time(), "", ""])
                frame += 1
            obs, _, done, info = env.step([reading.steering, reading.throttle])
            steps += 1
            if done or (maxSteps is not None and steps >= maxSteps):
                break

    metaOut = {
        "track": meta.get("track", ""),
        "date": datetime.now().isoformat(),
        "controller_name": meta.get("controller_name", ""),
        "image_h": config.IMAGE_H,
        "image_w": config.IMAGE_W,
        "fps_target": config.FPS_TARGET,
        "frame_count": frame,
        "notes": meta.get("notes", ""),
    }
    with open(runDir / "meta.json", "w") as f:
        json.dump(metaOut, f, indent=2)
    return runDir


def main():
    from src.sim.environment import SimEnvironment
    from src.input.controller import makeDriver
    env = SimEnvironment()
    driver = makeDriver()
    try:
        runDir = collectRun(env, driver, config.DATA_DIR,
                            meta={"track": config.SIM_TRACK, "controller_name": type(driver).__name__})
        print(f"Saved run to {runDir}")
    finally:
        driver.close()
        env.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/data/test_collect.py -v`
Expected: PASS (2 tests) — including the causality proof.

- [ ] **Step 5: Manual acceptance (requires sim + input — M1)**

`python -m src.data.collect` → drive with the keyboard, toggle recording with `r`, quit with `q`. Confirm a `data/run_*` dir appears with matching image count and CSV rows; reload with `ds.readRunSamples` and check steering/throttle ranges.

- [ ] **Step 6: Commit**

```bash
git add src/data/collect.py tests/data/test_collect.py
git commit -m "feat: add teleop collection loop with correct capture-loop causality"
```

---

### Task 8: Our CNN (`src/model/network.py`)

**Files:**
- Create: `src/model/network.py`
- Test: `tests/model/test_network.py`

**Interfaces:**
- Consumes: `config` (image shape, crop, action dim).
- Produces: `buildModel(imageShape=(120,160,3), cropTop=config.CROP_TOP) -> tf.keras.Model` — Cropping2D → conv stack → dense → **`tanh`** output of size `ACTION_DIM`. Input expects float32 `[0,1]` images (preprocess already applied).

- [ ] **Step 1: Write the failing test**

```python
# tests/model/test_network.py
import numpy as np
from src.model.network import buildModel
import config

def testModelInputOutputShapes():
    model = buildModel()
    x = np.zeros((4, config.IMAGE_H, config.IMAGE_W, config.IMAGE_C), np.float32)
    y = model.predict(x, verbose=0)
    assert y.shape == (4, config.ACTION_DIM)

def testOutputsAreBoundedByTanh():
    model = buildModel()
    x = np.random.rand(8, config.IMAGE_H, config.IMAGE_W, config.IMAGE_C).astype(np.float32)
    y = model.predict(x, verbose=0)
    assert y.max() <= 1.0 and y.min() >= -1.0

def testModelStaysSmallForPi():
    model = buildModel()
    assert model.count_params() < 5_000_000   # NFR4: small/exportable
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/model/test_network.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.model.network'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/model/network.py — OUR CNN (PilotNet-style). Crop is in-model so it can
# never be forgotten at drive time. tanh head keeps outputs in [-1, 1].
import tensorflow as tf
from tensorflow.keras import layers, Model
import config


def buildModel(imageShape=(config.IMAGE_H, config.IMAGE_W, config.IMAGE_C),
               cropTop=config.CROP_TOP) -> Model:
    inputs = layers.Input(shape=imageShape, name="image")
    x = layers.Cropping2D(cropping=((cropTop, 0), (0, 0)), name="cropSky")(inputs)
    x = layers.Conv2D(24, 5, strides=2, activation="relu")(x)
    x = layers.Conv2D(36, 5, strides=2, activation="relu")(x)
    x = layers.Conv2D(48, 5, strides=2, activation="relu")(x)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(100, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(50, activation="relu")(x)
    outputs = layers.Dense(config.ACTION_DIM, activation="tanh", name="action")(x)
    return Model(inputs, outputs, name="pilotNet")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/model/test_network.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/model/network.py tests/model/test_network.py
git commit -m "feat: add custom PilotNet-style CNN with in-model crop and tanh head"
```

---

### Task 9: Training (`src/train.py`)

**Files:**
- Create: `src/train.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: `config`, `loadDatasets` (Task 6), `buildModel` (Task 8).
- Produces:
  - `trainModel(dataDir=config.DATA_DIR, epochs=config.EPOCHS, modelPath=config.MODEL_PATH, metricsPath=config.METRICS_PATH) -> dict` — compiles with MSE + Adam, fits with ModelCheckpoint + EarlyStopping, saves the best model and a metrics JSON `{"train_loss": [...], "val_loss": [...], "best_val_loss": float}`. Returns the metrics dict.
  - `main()` — CLI entry.

- [ ] **Step 1: Write the failing test** (tiny synthetic data, 1 epoch, asserts finite loss + saved files)

```python
# tests/test_train.py
import csv, os
import numpy as np
import cv2
import config
from src.train import trainModel

def _writeRun(root, name, frames):
    d = os.path.join(root, name); os.makedirs(os.path.join(d, "imgs"))
    with open(os.path.join(d, "records.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["frame","image","steering","throttle","timestamp","speed","cte"])
        for i in range(frames):
            rel = f"imgs/{i:06d}.jpg"
            cv2.imwrite(os.path.join(d, rel), np.random.randint(0,255,(config.IMAGE_H,config.IMAGE_W,3),np.uint8))
            w.writerow([i, rel, np.sin(i), 0.3, 0.0, 5.0, 0.0])

def testTrainSavesModelAndFiniteLoss(tmp_path):
    _writeRun(tmp_path, "run_a", 8); _writeRun(tmp_path, "run_b", 8)
    modelPath = tmp_path / "pilot.keras"; metricsPath = tmp_path / "metrics.json"
    metrics = trainModel(dataDir=tmp_path, epochs=1, modelPath=modelPath, metricsPath=metricsPath)
    assert modelPath.exists() and metricsPath.exists()
    assert np.isfinite(metrics["val_loss"][-1])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_train.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.train'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/train.py — build -> fit (MSE) -> checkpoint/early-stop -> save + metrics.
import json
from pathlib import Path
import tensorflow as tf
import config
from src.data.dataset import loadDatasets
from src.model.network import buildModel


def trainModel(dataDir=config.DATA_DIR, epochs=config.EPOCHS,
               modelPath=config.MODEL_PATH, metricsPath=config.METRICS_PATH) -> dict:
    config.setSeed()
    modelPath = Path(modelPath); metricsPath = Path(metricsPath)
    modelPath.parent.mkdir(parents=True, exist_ok=True)

    trainDs, valDs = loadDatasets(dataDir)
    model = buildModel()
    model.compile(optimizer=tf.keras.optimizers.Adam(config.LEARNING_RATE), loss="mse")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(modelPath), monitor="val_loss",
                                           save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                         restore_best_weights=True),
    ]
    history = model.fit(trainDs, validation_data=valDs, epochs=epochs, callbacks=callbacks)
    if not modelPath.exists():   # e.g. epochs too few for a checkpoint improvement
        model.save(str(modelPath))

    metrics = {
        "train_loss": [float(v) for v in history.history.get("loss", [])],
        "val_loss": [float(v) for v in history.history.get("val_loss", [])],
    }
    metrics["best_val_loss"] = min(metrics["val_loss"]) if metrics["val_loss"] else None
    with open(metricsPath, "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def main():
    metrics = trainModel()
    print(f"Best val loss: {metrics['best_val_loss']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_train.py -v`
Expected: PASS (1 test). (May take ~10–20s on CPU.)

- [ ] **Step 5: Manual acceptance (M3)**

After collecting real data: `python -m src.train`. Expected: training runs, validation loss decreases across epochs, `models/pilot.keras` + `models/metrics.json` written.

- [ ] **Step 6: Commit**

```bash
git add src/train.py tests/test_train.py
git commit -m "feat: add training with MSE, checkpointing, early stopping, metrics"
```

---

### Task 10: Autonomous drive loop (`src/drive.py`)

**Files:**
- Create: `src/drive.py`
- Test: `tests/test_drive.py`

**Interfaces:**
- Consumes: `config`, env contract (Task 2), `preprocessImage` (Task 4), a saved Keras model, `buildModel` (Task 8, for the test's tiny model).
- Produces:
  - `clampThrottle(throttle, maxThrottle=config.THROTTLE_MAX) -> float` — clamp to `[-maxThrottle, maxThrottle]`.
  - `predictAction(model, obs) -> tuple[float, float]` — `preprocessImage(obs)` → model → `(steering, clampThrottle(throttle))`. Uses the SAME preprocess as training.
  - `driveLoop(env, model, maxSteps=None) -> int` — reset, loop `predictAction` → `step`, stop on `done`/`maxSteps`; returns steps taken.
  - `main()` — loads `config.MODEL_PATH`, builds `SimEnvironment`, runs `driveLoop`.

- [ ] **Step 1: Write the failing test** (fake env + tiny model; asserts clamp + range + shared preprocess)

```python
# tests/test_drive.py
import numpy as np
from src.sim.fakeenv import FakeEnvironment
from src.model.network import buildModel
from src.drive import clampThrottle, predictAction, driveLoop
import config

def testClampThrottleLimitsMagnitude():
    assert clampThrottle(0.9, 0.5) == 0.5
    assert clampThrottle(-0.9, 0.5) == -0.5
    assert clampThrottle(0.2, 0.5) == 0.2

def testPredictActionReturnsBoundedClampedAction():
    model = buildModel()
    obs = np.random.randint(0, 255, (config.IMAGE_H, config.IMAGE_W, 3), np.uint8)
    steering, throttle = predictAction(model, obs)
    assert -1.0 <= steering <= 1.0
    assert abs(throttle) <= config.THROTTLE_MAX

def testDriveLoopRunsToDone():
    env = FakeEnvironment(frames=5)
    model = buildModel()
    steps = driveLoop(env, model, maxSteps=50)
    assert steps == 5   # fake env sets done after 5 steps
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_drive.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.drive'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/drive.py — load model -> inference -> clamp -> step. Same preprocess as train.
import numpy as np
import tensorflow as tf
import config
from src.data.preprocess import preprocessImage


def clampThrottle(throttle, maxThrottle=config.THROTTLE_MAX) -> float:
    return float(max(-maxThrottle, min(maxThrottle, throttle)))


def predictAction(model, obs):
    x = preprocessImage(obs)                 # identical to training
    x = tf.expand_dims(x, 0)
    steering, throttle = model.predict(x, verbose=0)[0]
    return float(steering), clampThrottle(throttle)


def driveLoop(env, model, maxSteps=None) -> int:
    obs = env.reset()
    steps = 0
    while True:
        steering, throttle = predictAction(model, obs)
        obs, _, done, _ = env.step([steering, throttle])
        steps += 1
        if done or (maxSteps is not None and steps >= maxSteps):
            break
    return steps


def main():
    from src.sim.environment import SimEnvironment
    model = tf.keras.models.load_model(str(config.MODEL_PATH))
    env = SimEnvironment()
    try:
        steps = driveLoop(env, model)
        print(f"Drove {steps} steps.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_drive.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Manual acceptance (M4 — the endgame)**

After training: `python -m src.drive`. Expected: the model drives the car in the sim and completes track segments/laps with throttle held within `THROTTLE_MAX`.

- [ ] **Step 6: Run the full suite + commit**

```bash
pytest -q
git add src/drive.py tests/test_drive.py
git commit -m "feat: add autonomous drive loop with throttle safety clamp"
```

---

## Post-plan (M5 — iterate, not a fixed build task)

M5 is a measure/collect/retune loop, not discrete code: run `drive`, note where it fails, capture targeted recovery data there (Task 7), retrain (Task 9), compare `metrics.json` and lap completion before/after. No new modules — it exercises the pipeline this plan builds.

---

## Self-Review Notes

- **Spec coverage:** FR1→Task 2; FR2→Task 3; FR3→Task 7; FR4→Tasks 6/7 (CSV+meta format, `readRunSamples`); FR5→Tasks 4/5/6; FR6→Tasks 8/9; FR7→Task 10; FR8→every task's tests + fake env (Task 2). NFR1→sim isolation constraint + fake env; NFR2→our code in preprocess/augment/network/train/drive; NFR3→`config` + `setSeed`; NFR4→Task 8 param-count test; NFR5→lightweight `predictAction`; NFR6→commented code blocks.
- **Placeholder scan:** no TBD/TODO; every code step carries real code.
- **Type consistency:** `preprocessImage`, `applyAugmentation`, `buildModel`, `loadDatasets`, `collectRun`, `predictAction`, `driveLoop`, `InputReading`, `rampToward`, `clampThrottle` are named identically wherever referenced across tasks.
- **Review Focus:** all five items have a pinned test (Tasks 7, 6, 6, 10, 8).
