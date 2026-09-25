import numpy as np
import tensorflow as tf
from src.data.augment import flipImageAndLabel, randomBrightness


def testFlipNegatesSteeringAndMirrorsImage():
	img = tf.constant(np.arange(120 * 160 * 3, dtype=np.float32).reshape(120, 160, 3) / (120 * 160 * 3))
	label = tf.constant([0.4, 0.3], dtype=tf.float32)
	outImg, outLabel = flipImageAndLabel(img, label)
	# steering negated, throttle unchanged
	assert np.isclose(outLabel.numpy()[0], -0.4)
	assert np.isclose(outLabel.numpy()[1], 0.3)
	# image mirrored horizontally: column 0 of output == column last of input
	assert np.allclose(outImg.numpy()[:, 0, :], img.numpy()[:, -1, :])


def testBrightnessChangesPixelsNotLabelAndStaysInRange():
	img = tf.fill((120, 160, 3), 0.5)
	label = tf.constant([0.1, 0.2], dtype=tf.float32)
	outImg, outLabel = randomBrightness(img, label, 0.2)
	assert np.allclose(outLabel.numpy(), label.numpy())
	assert outImg.numpy().max() <= 1.0 and outImg.numpy().min() >= 0.0
