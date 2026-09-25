# src/drive.py — load model -> inference -> clamp -> step. Same preprocess as train.
import tensorflow as tf
import config
from src.data.preprocess import preprocessImage


def clampThrottle(throttle, maxThrottle=config.THROTTLE_MAX) -> float:
	return float(max(-maxThrottle, min(maxThrottle, throttle)))


def predictAction(model, obs):
	x = preprocessImage(obs)                 # identical to training
	x = tf.expand_dims(x, 0)
	steering, throttle = model.predict(x, verbose=0)[0]
	return float(steering), clampThrottle(throttle)


def driveLoop(env, model, maxSteps=None) -> int:
	obs = env.reset()
	steps = 0
	while True:
		steering, throttle = predictAction(model, obs)
		obs, _, done, _ = env.step([steering, throttle])
		steps += 1
		if done or (maxSteps is not None and steps >= maxSteps):
			break
	return steps


def main():
	from src.sim.environment import SimEnvironment
	model = tf.keras.models.load_model(str(config.MODEL_PATH))
	env = SimEnvironment()
	try:
		steps = driveLoop(env, model)
		print(f"Drove {steps} steps.")
	finally:
		env.close()


if __name__ == "__main__":
	main()
