import config


def testContractConstantsAreCorrect():
	assert config.IMAGE_H == 120
	assert config.IMAGE_W == 160
	assert config.IMAGE_C == 3
	assert config.ACTION_DIM == 2
	assert 0 < config.CROP_TOP < config.IMAGE_H


def testHyperparamRangesAreSane():
	assert 0.0 < config.VAL_SPLIT < 1.0
	assert config.BATCH_SIZE > 0
	assert 0.0 < config.THROTTLE_MAX <= 1.0


def testSetSeedIsDeterministic():
	import numpy as np
	config.setSeed(123)
	first = np.random.rand(5)
	config.setSeed(123)
	second = np.random.rand(5)
	assert np.allclose(first, second)
