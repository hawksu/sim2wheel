import numpy as np
import pytest
import tensorflow as tf
from src.sim.fakeenv import FakeEnvironment
from src.model.network import buildModel
from src.drive import steeringThrottle, predictAction, driveLoop
import config


def testSteeringThrottleIsBaseOnStraights():
	assert steeringThrottle(0.0) == pytest.approx(config.DEFAULT_THROTTLE)


def testSteeringThrottleIsFloorAtFullLock():
	assert steeringThrottle(1.0) == pytest.approx(config.THROTTLE_MIN)
	assert steeringThrottle(-1.0) == pytest.approx(config.THROTTLE_MIN)


def testSteeringThrottleSlowsSymmetricallyInCorners():
	assert steeringThrottle(0.5) < steeringThrottle(0.0)
	assert steeringThrottle(0.5) == pytest.approx(steeringThrottle(-0.5))


def testSteeringThrottleStaysForwardAndBounded():
	for steering in np.linspace(-1.0, 1.0, 41):
		throttle = steeringThrottle(steering)
		assert config.THROTTLE_MIN <= throttle <= config.THROTTLE_MAX


def testPredictActionIgnoresModelThrottle():
	# The model learned reverse/zero throttle from keyboard capture; drive
	# must never pass that through to the sim.
	def reversingModel(x, training=False):
		return tf.constant([[0.0, -0.9]])
	obs = np.random.randint(0, 255, (config.IMAGE_H, config.IMAGE_W, 3), np.uint8)
	steering, throttle = predictAction(reversingModel, obs)
	assert steering == pytest.approx(0.0)
	assert throttle == pytest.approx(config.DEFAULT_THROTTLE)


def testPredictActionReturnsBoundedForwardAction():
	model = buildModel()
	obs = np.random.randint(0, 255, (config.IMAGE_H, config.IMAGE_W, 3), np.uint8)
	steering, throttle = predictAction(model, obs)
	assert -1.0 <= steering <= 1.0
	assert config.THROTTLE_MIN <= throttle <= config.THROTTLE_MAX


def testDriveLoopRunsToDone():
	env = FakeEnvironment(frames=5)
	model = buildModel()
	steps = driveLoop(env, model, maxSteps=50)
	assert steps == 5   # fake env sets done after 5 steps
