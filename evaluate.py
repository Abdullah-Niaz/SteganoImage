import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from dataset import build_dataset

model = tf.keras.models.load_model("stego_model.h5", compile=False)

test_ds = build_dataset("dataset", training=False)

y_true = []
y_pred = []

for x, y in test_ds:
    preds = model.predict(x)
    y_true.extend(y.numpy())
    y_pred.extend(preds.flatten())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

threshold = 0.35
y_bin = (y_pred > threshold).astype(int)

print("ROC AUC:", roc_auc_score(y_true, y_pred))
print(confusion_matrix(y_true, y_bin))
print(classification_report(y_true, y_bin, target_names=["Clean", "Infected"]))
