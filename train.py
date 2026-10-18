import tensorflow as tf
from dataset import build_dataset
from model import build_model
from loss_function import focal_loss

train_ds = build_dataset("dataset", training=True)
val_ds = build_dataset("dataset", training=False)

model = build_model()

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=focal_loss(),
    metrics=["accuracy"]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
]

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    callbacks=callbacks
)

model.save("stego_model.h5")
