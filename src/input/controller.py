# src/input/controller.py — one input interface, keyboard default + optional Xbox.
from dataclasses import dataclass
import config


def rampToward(current: float, target: float, rate: float, dt: float) -> float:
	"""Move current toward target by at most rate*dt, clamped to [-1, 1]."""
	step = rate * dt
	if target > current:
		current = min(current + step, target)
	else:
		current = max(current - step, target)
	return max(-1.0, min(1.0, current))


@dataclass
class InputReading:
	steering: float
	throttle: float
	recording: bool
	quit: bool


class InputDriver:
	def poll(self, dt: float) -> InputReading:
		raise NotImplementedError

	def close(self) -> None:
		pass


class KeyboardDriver(InputDriver):
	"""Pseudo-analog: holding a key ramps the axis; releasing decays to 0."""

	def __init__(self):
		import pygame
		self.pygame = pygame
		if not pygame.get_init():
			pygame.init()
		# a tiny window is required for pygame to receive key events
		self.screen = pygame.display.set_mode((320, 120))
		self.steering = 0.0
		self.throttle = 0.0
		self.recording = False
		self.quit = False
		self._updateCaption()

	def _updateCaption(self) -> None:
		# Surface record state in the window title so it's obvious whether
		# frames are being captured (0-frame runs are a common footgun).
		if self.recording:
			caption = "DonkeyCar teleop — ● REC — arrows/WASD, r=pause, q=quit"
		else:
			caption = "DonkeyCar teleop — paused — arrows/WASD, r=record, q=quit"
		self.pygame.display.set_caption(caption)

	def poll(self, dt: float) -> InputReading:
		pg = self.pygame
		for event in pg.event.get():
			if event.type == pg.QUIT:
				self.quit = True
			elif event.type == pg.KEYDOWN:
				if event.key == pg.K_r:
					self.recording = not self.recording
					self._updateCaption()
				elif event.key in (pg.K_q, pg.K_ESCAPE):
					self.quit = True
		keys = pg.key.get_pressed()
		steerTarget = 0.0
		if keys[pg.K_LEFT] or keys[pg.K_a]:
			steerTarget = -1.0
		elif keys[pg.K_RIGHT] or keys[pg.K_d]:
			steerTarget = 1.0
		throttleTarget = 0.0
		if keys[pg.K_UP] or keys[pg.K_w]:
			throttleTarget = 1.0
		elif keys[pg.K_DOWN] or keys[pg.K_s]:
			throttleTarget = -1.0
		steerRate = config.STEER_RATE if steerTarget != 0.0 else config.RETURN_RATE
		throttleRate = config.THROTTLE_RATE if throttleTarget != 0.0 else config.RETURN_RATE
		self.steering = rampToward(self.steering, steerTarget, steerRate, dt)
		self.throttle = rampToward(self.throttle, throttleTarget, throttleRate, dt)
		return InputReading(self.steering, self.throttle, self.recording, self.quit)

	def close(self) -> None:
		self.pygame.quit()


class XboxDriver(InputDriver):
	"""Optional analog driver; only build when a joystick is present."""

	def __init__(self):
		import pygame
		self.pygame = pygame
		if not pygame.get_init():
			pygame.init()
		pygame.joystick.init()
		self.js = pygame.joystick.Joystick(0)
		self.js.init()
		self.recording = False
		self.quit = False

	def poll(self, dt: float) -> InputReading:
		pg = self.pygame
		for event in pg.event.get():
			if event.type == pg.JOYBUTTONDOWN:
				if event.button == 0:      # A toggles recording
					self.recording = not self.recording
				elif event.button == 1:    # B quits
					self.quit = True
		steering = float(self.js.get_axis(0))
		# right trigger (axis 5) mapped from rest=-1..1 to throttle 0..1
		rawThrottle = float(self.js.get_axis(5))
		throttle = (rawThrottle + 1.0) / 2.0
		return InputReading(
		    max(-1.0, min(1.0, steering)),
		    max(-1.0, min(1.0, throttle)),
		    self.recording,
		    self.quit,
		)

	def close(self) -> None:
		self.pygame.quit()


def makeDriver() -> InputDriver:
	"""Xbox if a joystick enumerates, else pseudo-analog keyboard."""
	try:
		import pygame
		pygame.init()
		pygame.joystick.init()
		if pygame.joystick.get_count() > 0:
			return XboxDriver()
	except Exception:
		pass
	return KeyboardDriver()
