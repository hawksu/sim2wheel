# src/data/preprocess.py — the single normalize step used at train AND drive.
import tensorflow as tf


def preprocessImage(image):
	"""uint8 [120,160,3] -> float32 [0,1]. Cropping happens inside the model."""
	image = tf.cast(image, tf.float32) / 255.0
	return image
