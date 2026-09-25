# src/model/network.py — OUR CNN (PilotNet-style). Crop is in-model so it can
# never be forgotten at drive time. tanh head keeps outputs in [-1, 1].
import tensorflow as tf
from tensorflow.keras import layers, Model
import config


def buildModel(imageShape=(config.IMAGE_H, config.IMAGE_W, config.IMAGE_C),
               cropTop=config.CROP_TOP) -> Model:
	inputs = layers.Input(shape=imageShape, name="image")
	x = layers.Cropping2D(cropping=((cropTop, 0), (0, 0)), name="cropSky")(inputs)
	x = layers.Conv2D(24, 5, strides=2, activation="relu")(x)
	x = layers.Conv2D(36, 5, strides=2, activation="relu")(x)
	x = layers.Conv2D(48, 5, strides=2, activation="relu")(x)
	x = layers.Conv2D(64, 3, activation="relu")(x)
	x = layers.Conv2D(64, 3, activation="relu")(x)
	x = layers.Flatten()(x)
	x = layers.Dropout(0.2)(x)
	x = layers.Dense(100, activation="relu")(x)
	x = layers.Dropout(0.2)(x)
	x = layers.Dense(50, activation="relu")(x)
	outputs = layers.Dense(config.ACTION_DIM, activation="tanh", name="action")(x)
	return Model(inputs, outputs, name="pilotNet")
