# src/data/augment.py — train-only augmentation. Never applied to val/drive.
import tensorflow as tf
import config


def flipImageAndLabel(image, label):
	"""Mirror image L<->R and negate steering (label[0])."""
	flipped = tf.image.flip_left_right(image)
	steering = -label[0]
	throttle = label[1]
	return flipped, tf.stack([steering, throttle])


def randomBrightness(image, label, maxDelta=config.BRIGHTNESS_DELTA):
	"""Jitter brightness; label unchanged; keep pixels in [0,1]."""
	image = tf.image.random_brightness(image, maxDelta)
	image = tf.clip_by_value(image, 0.0, 1.0)
	return image, label


def applyAugmentation(image, label):
	"""50% flip, then brightness jitter. Train split only."""
	if config.AUGMENT_FLIP:
		doFlip = tf.random.uniform([]) < 0.5
		image, label = tf.cond(
		    doFlip,
		    lambda: flipImageAndLabel(image, label),
		    lambda: (image, label),
		)
	image, label = randomBrightness(image, label)
	return image, label
