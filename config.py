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
THROTTLE_MAX = 0.5       # drive-time safety cap on throttle
# Drive throttle is derived from predicted steering, not the model's throttle
# head (keyboard capture makes those labels mostly 0/reverse):
#   max(THROTTLE_MIN, DEFAULT_THROTTLE * (1 - THROTTLE_STEER_GAIN * |steering|))
DEFAULT_THROTTLE = 0.35     # throttle on straights
THROTTLE_MIN = 0.2          # forward floor: never stall or reverse
THROTTLE_STEER_GAIN = 0.5   # how much to back off in corners

# Pseudo-analog keyboard ramps (axis units per second)
STEER_RATE = 3.0
THROTTLE_RATE = 2.0
RETURN_RATE = 4.0        # how fast an axis decays to 0 when no key is held

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
