import numpy as np
from src.sim.fakeenv import FakeEnvironment
from src.model.network import buildModel
from src.drive import clampThrottle, predictAction, driveLoop
import config


def testClampThrottleLimitsMagnitude():
	assert clampThrottle(0.9, 0.5) == 0.5
	assert clampThrottle(-0.9, 0.5) == -0.5
	assert clampThrottle(0.2, 0.5) == 0.2


def testPredictActionReturnsBoundedClampedAction():
	model = buildModel()
	obs = np.random.randint(0, 255, (config.IMAGE_H, config.IMAGE_W, 3), np.uint8)
	steering, throttle = predictAction(model, obs)
	assert -1.0 <= steering <= 1.0
	assert abs(throttle) <= config.THROTTLE_MAX


def testDriveLoopRunsToDone():
	env = FakeEnvironment(frames=5)
	model = buildModel()
	steps = driveLoop(env, model, maxSteps=50)
	assert steps == 5   # fake env sets done after 5 steps
