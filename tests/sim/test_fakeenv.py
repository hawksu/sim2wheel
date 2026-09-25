import numpy as np
from src.sim.fakeenv import FakeEnvironment
import config


def testResetReturnsContractImage():
	env = FakeEnvironment(frames=10)
	obs = env.reset()
	assert obs.shape == (config.IMAGE_H, config.IMAGE_W, config.IMAGE_C)
	assert obs.dtype == np.uint8


def testStepReturnsTupleAndAdvancesIndex():
	env = FakeEnvironment(frames=10)
	env.reset()
	obs, reward, done, info = env.step([0.0, 0.3])
	assert obs.shape == (config.IMAGE_H, config.IMAGE_W, config.IMAGE_C)
	assert isinstance(reward, float)
	assert isinstance(done, bool)
	assert "cte" in info and "speed" in info


def testFrameEncodesStepIndexForPairingProofs():
	env = FakeEnvironment(frames=10)
	firstSeen = env.reset()
	assert int(firstSeen[0, 0, 0]) == 0          # reset frame is index 0
	obs, _, _, _ = env.step([0.0, 0.0])
	assert int(obs[0, 0, 0]) == 1                 # step returns the NEXT frame


def testDoneAfterConfiguredFrames():
	env = FakeEnvironment(frames=3)
	env.reset()
	dones = [env.step([0.0, 0.0])[2] for _ in range(3)]
	assert dones[-1] is True
