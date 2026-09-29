# src/data/collect.py — teleop capture. Record BEFORE stepping (causality).
import csv
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
import cv2
import config


def collectRun(env, driver, dataDir, meta, maxSteps=None) -> Path:
	config.setSeed()
	dataDir = Path(dataDir)
	runName = "run_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
	runDir = dataDir / runName
	imgsDir = runDir / "imgs"
	imgsDir.mkdir(parents=True, exist_ok=True)

	obs = env.reset()
	frame = 0
	period = 1.0 / config.FPS_TARGET
	csvPath = runDir / "records.csv"
	with open(csvPath, "w", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(["frame", "image", "steering", "throttle",
		                 "timestamp", "speed", "cte"])
		steps = 0
		while True:
			reading = driver.poll(period)
			if reading.quit:
				break
			if reading.recording:
				# obs is the frame the human is reacting to -> pair with this action.
				# The sim yields RGB; cv2.imwrite expects BGR, so convert first.
				# Otherwise training (tf.io.decode_jpeg, RGB) and drive (raw sim
				# RGB) would see channel-swapped colors -> broken train/serve contract.
				rel = f"imgs/{frame:06d}.jpg"
				cv2.imwrite(str(runDir / rel), cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
				writer.writerow([frame, rel, reading.steering, reading.throttle,
				                 time.time(), "", ""])
				frame += 1
			obs, _, done, info = env.step([reading.steering, reading.throttle])
			steps += 1
			if done or (maxSteps is not None and steps >= maxSteps):
				break

	if frame == 0:
		# No frames means recording was never toggled on. Fail loudly here
		# instead of leaving an empty run that only breaks much later at
		# train time ("zero-frame dataset"). Drop the empty run directory so
		# it can't pollute the by-run train/val split.
		print("\nWARNING: recorded 0 frames — nothing was captured.")
		print("Press 'r' in the teleop window to start recording, then drive.")
		shutil.rmtree(runDir, ignore_errors=True)
		return None

	metaOut = {
	    "track": meta.get("track", ""),
	    "date": datetime.now().isoformat(),
	    "controller_name": meta.get("controller_name", ""),
	    "image_h": config.IMAGE_H,
	    "image_w": config.IMAGE_W,
	    "fps_target": config.FPS_TARGET,
	    "frame_count": frame,
	    "notes": meta.get("notes", ""),
	}
	with open(runDir / "meta.json", "w") as f:
		json.dump(metaOut, f, indent=2)
	return runDir


def main():
	from src.sim.environment import SimEnvironment
	from src.input.controller import makeDriver
	env = SimEnvironment()
	driver = makeDriver()
	try:
		runDir = collectRun(env, driver, config.DATA_DIR,
		                    meta={"track": config.SIM_TRACK,
		                          "controller_name": type(driver).__name__})
		if runDir is None:
			print("No run saved (0 frames captured).")
		else:
			print(f"Saved run to {runDir}")
	finally:
		driver.close()
		env.close()


if __name__ == "__main__":
	main()
