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
