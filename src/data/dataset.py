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
