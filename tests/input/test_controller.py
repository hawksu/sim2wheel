from src.input.controller import rampToward


def testRampMovesTowardTargetByRateTimesDt():
	# rate 2.0/s over 0.1s => +0.2 toward target
	assert abs(rampToward(0.0, 1.0, 2.0, 0.1) - 0.2) < 1e-9


def testRampDoesNotOvershootTarget():
	assert rampToward(0.9, 1.0, 5.0, 1.0) == 1.0


def testRampClampsToUnitRange():
	assert rampToward(0.95, 5.0, 100.0, 1.0) == 1.0
	assert rampToward(-0.95, -5.0, 100.0, 1.0) == -1.0


def testRampReturnsTowardZeroWhenTargetIsZero():
	assert abs(rampToward(0.5, 0.0, 4.0, 0.1) - 0.1) < 1e-9
