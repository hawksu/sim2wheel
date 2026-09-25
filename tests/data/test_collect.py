import csv
import json
import os
import numpy as np
import cv2
import config
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
	with open(os.path.join(runDir, "records.csv")) as f:
		rows = list(csv.DictReader(f))
	for i, row in enumerate(rows):
		img = cv2.imread(os.path.join(runDir, row["image"]))
		assert int(img[0, 0, 0]) == i                          # frame i was the SEEN frame
		assert abs(float(row["steering"]) - i / 10.0) < 1e-6   # paired with action i


class ColorEnv:
	"""Returns a fixed distinct-color RGB frame (R != G != B); done after one step."""

	def __init__(self, rgb):
		self.rgb = rgb

	def reset(self):
		img = np.zeros((config.IMAGE_H, config.IMAGE_W, 3), np.uint8)
		img[:, :, 0] = self.rgb[0]
		img[:, :, 1] = self.rgb[1]
		img[:, :, 2] = self.rgb[2]
		return img

	def step(self, action):
		return self.reset(), 1.0, True, {"cte": 0.0, "speed": 0.0}

	def close(self):
		pass


def testCollectPreservesRgbChannelOrder(tmp_path):
	# The sim hands RGB frames. A recorded run read back at train time (via
	# tf.io.decode_jpeg, RGB) must keep the same channel order the drive path
	# feeds the model — otherwise the model trains and drives on swapped colors.
	import tensorflow as tf
	env = ColorEnv((200, 100, 50))
	driver = ScriptedDriver([InputReading(0.0, 0.3, recording=True, quit=False)])
	runDir = collectRun(env, driver, tmp_path, meta={"track": "t", "controller_name": "s"})
	with open(os.path.join(runDir, "records.csv")) as f:
		row = next(csv.DictReader(f))
	raw = tf.io.read_file(os.path.join(runDir, row["image"]))
	decoded = tf.io.decode_jpeg(raw, channels=3).numpy()
	r, g, b = int(decoded[0, 0, 0]), int(decoded[0, 0, 1]), int(decoded[0, 0, 2])
	# true frame is R=200, G=100, B=50; a BGR swap would return R~=50, B~=200.
	assert r > 130, f"red channel came back {r} (expected ~200; BGR swap gives ~50)"
	assert b < 130, f"blue channel came back {b} (expected ~50; BGR swap gives ~200)"
	assert abs(g - 100) <= 30
