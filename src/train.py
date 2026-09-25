# src/train.py — build -> fit (MSE) -> checkpoint/early-stop -> save + metrics.
import json
from pathlib import Path
import tensorflow as tf
import config
from src.data.dataset import loadDatasets
from src.model.network import buildModel


def trainModel(dataDir=config.DATA_DIR, epochs=config.EPOCHS,
               modelPath=config.MODEL_PATH, metricsPath=config.METRICS_PATH) -> dict:
	config.setSeed()
	modelPath = Path(modelPath)
	metricsPath = Path(metricsPath)
	modelPath.parent.mkdir(parents=True, exist_ok=True)

	trainDs, valDs = loadDatasets(dataDir)
	model = buildModel()
	model.compile(optimizer=tf.keras.optimizers.Adam(config.LEARNING_RATE), loss="mse")

	callbacks = [
	    tf.keras.callbacks.ModelCheckpoint(str(modelPath), monitor="val_loss",
	                                       save_best_only=True),
	    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
	                                     restore_best_weights=True),
	]
	history = model.fit(trainDs, validation_data=valDs, epochs=epochs, callbacks=callbacks)
	if not modelPath.exists():   # e.g. epochs too few for a checkpoint improvement
		model.save(str(modelPath))

	metrics = {
	    "train_loss": [float(v) for v in history.history.get("loss", [])],
	    "val_loss": [float(v) for v in history.history.get("val_loss", [])],
	}
	metrics["best_val_loss"] = min(metrics["val_loss"]) if metrics["val_loss"] else None
	with open(metricsPath, "w") as f:
		json.dump(metrics, f, indent=2)
	return metrics


def main():
	metrics = trainModel()
	print(f"Best val loss: {metrics['best_val_loss']}")


if __name__ == "__main__":
	main()
