# DonkeyCar Sim — Behavioral Cloning (ML/Sim phase)

A from-scratch behavioral-cloning pipeline that learns to drive the DonkeyCar
simulator. `gym-donkeycar` is used only as the pipe to the sim; the data
pipeline, CNN, training, and drive loop are ours.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for how the pipeline fits
together, `PLAN.md` for the full design and `docs/superpowers/plans/` for the
implementation plan.

## Quickstart

```bash
# 1. Create an isolated environment (needs python3-venv on Debian/Ubuntu)
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# 2. Launch the DonkeyCar sim on Windows, set the host if not mirrored:
#    export SIM_HOST=<windows-host-ip>   # default 127.0.0.1

# 3. Collect data (keyboard by default; r=record, q=quit):
#    See "Collecting driving data" below — recording is OFF until you press r.
python -m src.data.collect

# 4. Train (needs at least 2 non-empty runs — a whole run is held out
#    for validation, so a single run cannot be split):
python -m src.train

# 5. Drive autonomously:
python -m src.drive
```

Run the tests (no sim required — uses the fake environment):

```bash
pytest
```

## Collecting driving data

`python -m src.data.collect` starts a teleoperation session: you drive the car
in the sim, and each frame is paired with the steering/throttle **you** applied
at that moment. Those (image, action) pairs are the training set.

When collection starts, a small **teleop window** opens (title:
`DonkeyCar teleop — …`). pygame only receives key events while that window is
focused, so **click it first**. The sim window shows the car; the teleop window
captures your input.

**Controls**

| Key            | Action                                              |
| -------------- | --------------------------------------------------- |
| `↑` / `W`      | Throttle forward (ramps up while held)              |
| `↓` / `S`      | Throttle reverse / brake                            |
| `←` / `A`      | Steer left                                          |
| `→` / `D`      | Steer right                                         |
| `r`            | **Toggle recording on/off**                         |
| `q` / `Esc`    | Quit and save the run                               |

**Recording is OFF by default.** Nothing is saved until you press `r` — this is
the most common mistake (you drive a perfect lap, but the run is empty). The
window title reflects the current state:

- `● REC` — frames are being captured
- `paused` — driving, but **not** recording

If a session ends with **0 frames captured** (recording never toggled on), the
run is discarded and `collect` prints a warning instead of leaving an empty run
directory that would only fail later at train time.

**A good collection workflow**

1. Launch the sim, then run `python -m src.data.collect`.
2. Click the teleop window to focus it.
3. Press `r` (title shows `● REC`) and drive a clean lap. Press `r` again to
   pause across a mistake, then `r` to resume.
4. Press `q` to save. Confirm it saved: the run's `meta.json` shows
   `"frame_count"` > 0 (or `data/run_*/records.csv` has data rows).
5. **Repeat for at least 2 runs.** Training holds out a whole run for
   validation, so a single run cannot be split — `src.train` needs ≥ 2
   non-empty runs. Empty (0-frame) runs are ignored automatically, so a
   stray one won't break training; you just need two runs that actually
   contain frames.

Capture **varied** driving — corners and recovery from the edges, not just
straights. If most frames are near-zero steering, training warns about the
imbalance (the model learns to only go straight).

## Troubleshooting install

`gym-donkeycar` v22.11.06 hard-pins `gym==0.21`, and `gym 0.21.0` ships broken
metadata (`opencv-python (>=3.)`). pip **>= 24.1** validates metadata strictly
and discards that version, so resolution fails before the build even starts
(`ResolutionImpossible`, or `Ignoring version 0.21.0 of gym since it has invalid
metadata`). The `gym>=0.21,<0.26` range in `requirements.txt` can't help — the
transitive `gym==0.21` overrides it.

Downgrade pip first, then install the build tools `gym 0.21` expects, then the
requirements:

```bash
pip install "pip<24.1"                            # 24.0: lenient metadata parsing
pip install "setuptools==65.5.0" "wheel==0.38.4"  # gym 0.21 build tools
pip install --no-build-isolation -r requirements.txt
```

Keep pip on 24.0 for this venv — upgrading back to a newer pip makes `gym 0.21`
uninstallable again in a fresh environment.

If `gym-donkeycar` and `gym` still conflict, pin the exact pair that resolves
together (see the versions printed by `pip install` and adjust
`requirements.txt`).

## Controller on WSL2

The keyboard driver is the default. To try an Xbox controller, attach it with
`usbipd-win` and check whether `/dev/input/js*` appears. If it does not, the
WSL kernel likely lacks joystick support — stay on the pseudo-analog keyboard
driver; the model is a sim model either way.
