import csv
import json
import os
import numpy as np
import cv2
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
