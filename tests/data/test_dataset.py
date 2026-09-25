import csv
import os
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
