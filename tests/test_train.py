import csv
import os
import numpy as np
import cv2
import config
from src.train import trainModel


def _writeRun(root, name, frames):
	d = os.path.join(root, name)
	os.makedirs(os.path.join(d, "imgs"))
	with open(os.path.join(d, "records.csv"), "w", newline="") as f:
		w = csv.writer(f)
		w.writerow(["frame", "image", "steering", "throttle", "timestamp", "speed", "cte"])
		for i in range(frames):
			rel = f"imgs/{i:06d}.jpg"
			cv2.imwrite(os.path.join(d, rel),
			            np.random.randint(0, 255, (config.IMAGE_H, config.IMAGE_W, 3), np.uint8))
			w.writerow([i, rel, np.sin(i), 0.3, 0.0, 5.0, 0.0])


def testTrainSavesModelAndFiniteLoss(tmp_path):
	_writeRun(tmp_path, "run_a", 8)
	_writeRun(tmp_path, "run_b", 8)
	modelPath = tmp_path / "pilot.keras"
	metricsPath = tmp_path / "metrics.json"
	metrics = trainModel(dataDir=tmp_path, epochs=1, modelPath=modelPath, metricsPath=metricsPath)
	assert modelPath.exists() and metricsPath.exists()
	assert np.isfinite(metrics["val_loss"][-1])
