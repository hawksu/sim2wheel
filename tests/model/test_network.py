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
