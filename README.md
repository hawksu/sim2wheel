# DonkeyCar Sim — Behavioral Cloning (ML/Sim phase)

A from-scratch behavioral-cloning pipeline that learns to drive the DonkeyCar
simulator. `gym-donkeycar` is used only as the pipe to the sim; the data
pipeline, CNN, training, and drive loop are ours.

See `PLAN.md` for the full design and `docs/superpowers/plans/` for the
implementation plan.

## Quickstart

```bash
# 1. Create an isolated environment (needs python3-venv on Debian/Ubuntu)
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# 2. Launch the DonkeyCar sim on Windows, set the host if not mirrored:
#    export SIM_HOST=<windows-host-ip>   # default 127.0.0.1

# 3. Collect data (keyboard by default; r=record, q=quit):
python -m src.data.collect

# 4. Train:
python -m src.train

# 5. Drive autonomously:
python -m src.drive
```

Run the tests (no sim required — uses the fake environment):

```bash
pytest
```

## Troubleshooting install

`gym==0.21` fails to build under modern pip/setuptools. Install the build
tools it expects first, then the requirements:

```bash
pip install "setuptools==65.5.0" "wheel==0.38.4"
pip install --no-build-isolation -r requirements.txt
```

If `gym-donkeycar` and `gym` still conflict, pin the exact pair that resolves
together (see the versions printed by `pip install` and adjust
`requirements.txt`).

## Controller on WSL2

The keyboard driver is the default. To try an Xbox controller, attach it with
`usbipd-win` and check whether `/dev/input/js*` appears. If it does not, the
WSL kernel likely lacks joystick support — stay on the pseudo-analog keyboard
driver; the model is a sim model either way.
