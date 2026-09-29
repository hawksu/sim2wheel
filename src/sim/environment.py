# src/sim/environment.py — the ONLY module that talks to gym-donkeycar.
import gym
# gym-donkeycar 1.0.13 (PyPI latest) registers its envs only when this
# submodule is imported; the top-level package __init__ is metadata-only.
import gym_donkeycar.envs.donkey_env  # noqa: F401  (registers the donkey envs)
import numpy as np
import config


class SimEnvironment:
	"""Thin wrapper over gym-donkeycar exposing reset/step/close.

	Isolates the simulator so a real-car env can replace just this file later.
	"""

	def __init__(self, host: str = config.SIM_HOST, port: int = config.SIM_PORT,
	             track: str = config.SIM_TRACK):
		conf = {"host": host, "port": port}
		self.env = gym.make(track, conf=conf)

	def reset(self) -> np.ndarray:
		obs = self.env.reset()
		return np.asarray(obs, dtype=np.uint8)

	def step(self, action):
		obs, reward, done, info = self.env.step(
		    np.asarray([action[0], action[1]], dtype=np.float32)
		)
		return np.asarray(obs, dtype=np.uint8), float(reward), bool(done), dict(info)

	def close(self) -> None:
		self.env.close()
