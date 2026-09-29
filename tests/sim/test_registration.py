# Regression guard: importing the sim wrapper must register the donkey gym envs.
# gym-donkeycar 1.0.13 (PyPI latest) registers envs only when
# gym_donkeycar.envs.donkey_env is imported, NOT on `import gym_donkeycar`.
# If someone "simplifies" that import back to the top-level package, this fails.
import gym
import config


def registeredEnvIds():
	registry = gym.envs.registry
	if hasattr(registry, "all"):          # classic gym API
		return {spec.id for spec in registry.all()}
	return set(registry.keys())           # newer registry mapping


def testSimImportRegistersConfiguredTrack():
	import src.sim.environment  # noqa: F401  (import triggers registration)
	assert config.SIM_TRACK in registeredEnvIds()
