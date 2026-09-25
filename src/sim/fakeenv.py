# src/sim/fakeenv.py — deterministic stand-in for the sim; NO gym import.
import numpy as np
import config


class FakeEnvironment:
	"""Same interface as SimEnvironment, for headless tests.

	The frame at step index i has pixel [0,0,0] == i % 256, so a test can
	prove which frame was paired with which action.
	"""

	def __init__(self, frames: int = 100):
		self.maxFrames = frames
		self.stepIndex = 0

	def _frame(self, index: int) -> np.ndarray:
		img = np.full(
		    (config.IMAGE_H, config.IMAGE_W, config.IMAGE_C),
		    fill_value=(index % 256),
		    dtype=np.uint8,
		)
		return img

	def reset(self) -> np.ndarray:
		self.stepIndex = 0
		return self._frame(0)

	def step(self, action):
		self.stepIndex += 1
		obs = self._frame(self.stepIndex)
		reward = 1.0
		done = self.stepIndex >= self.maxFrames
		info = {"cte": 0.0, "speed": float(action[1]) * 10.0}
		return obs, reward, done, info

	def close(self) -> None:
		pass
