"""
Fasal Doctor: download your dataset and train a small crop-disease model.

Run it on your own computer or in free Google Colab (no GPU needed for small sets):
    pip install tensorflow
    FASAL_URL=https://your-site.com OWNER_PASSWORD=... python train.py

Output: fasal_model.tflite and labels.json (phone-friendly MobileNetV3-Small model).

Honest notes:
- About 30 clear photos per disease is the minimum (more is better); fewer than 30 are skipped.
- By default only photos YOU labelled are used. Farmer photos carry labels guessed by the AI,
  which can be wrong. Set INCLUDE_FARMERS=1 only after an expert has checked them.
- Test the model on photos it has never seen before trusting it. Do not use it for dosage advice
  without an agronomist's review.
"""
import os, re, json, getpass, urllib.request

BASE = (os.environ.get("FASAL_URL") or input("Site address (https://...): ")).rstrip("/")
PW = os.environ.get("OWNER_PASSWORD") or getpass.getpass("Owner password: ")
INCLUDE_FARMERS = os.environ.get("INCLUDE_FARMERS") == "1"
MIN_PER_CLASS = int(os.environ.get("MIN_PER_CLASS", "30"))
OUT = "dataset"


def call(path, data=None, token=None):
    req = urllib.request.Request(BASE + path, data=data)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    return urllib.request.urlopen(req, timeout=60).read()


token = json.loads(call("/api/admin/login", json.dumps({"password": PW}).encode()))["token"]
items = json.loads(call("/api/admin/cv?all=1", token=token))["items"]
os.makedirs(OUT, exist_ok=True)

counts = {}
for it in items:
    if it["source"] == "farmer" and not INCLUDE_FARMERS:
        continue
    label = re.sub(r"[^a-z0-9]+", "_", f'{it.get("crop","")}_{it["label"]}'.lower()).strip("_") or "unknown"
    d = os.path.join(OUT, label)
    os.makedirs(d, exist_ok=True)
    fp = os.path.join(d, it["id"] + ".jpg")
    if not os.path.exists(fp):
        open(fp, "wb").write(call("/api/admin/img/" + it["file"], token=token))
    counts[label] = counts.get(label, 0) + 1

print("Images per class:", counts)
for label, n in counts.items():
    if n < MIN_PER_CLASS:
        print(f"Skipping '{label}' ({n} photos, need {MIN_PER_CLASS}+)")
        import shutil; shutil.rmtree(os.path.join(OUT, label))
if sum(1 for n in counts.values() if n >= MIN_PER_CLASS) < 2:
    raise SystemExit("Need at least 2 classes with enough photos. Upload more reference photos first.")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SIZE, BATCH = (224, 224), 16
train = keras.utils.image_dataset_from_directory(OUT, validation_split=0.2, subset="training", seed=42, image_size=SIZE, batch_size=BATCH)
val = keras.utils.image_dataset_from_directory(OUT, validation_split=0.2, subset="validation", seed=42, image_size=SIZE, batch_size=BATCH)
classes = train.class_names
train, val = train.prefetch(tf.data.AUTOTUNE), val.prefetch(tf.data.AUTOTUNE)

base = keras.applications.MobileNetV3Small(input_shape=SIZE + (3,), include_top=False, weights="imagenet", pooling="avg")
base.trainable = False
aug = keras.Sequential([layers.RandomFlip("horizontal"), layers.RandomRotation(0.1), layers.RandomZoom(0.1)])
inp = keras.Input(SIZE + (3,))
x = base(aug(inp))
x = layers.Dropout(0.2)(x)
out = layers.Dense(len(classes), activation="softmax")(x)
model = keras.Model(inp, out)

model.compile(keras.optimizers.Adam(1e-3), "sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train, validation_data=val, epochs=10)

base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False
model.compile(keras.optimizers.Adam(1e-5), "sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train, validation_data=val, epochs=10)

loss, acc = model.evaluate(val)
print(f"Validation accuracy: {acc:.2%}")

tflite = tf.lite.TFLiteConverter.from_keras_model(model)
tflite.optimizations = [tf.lite.Optimize.DEFAULT]
open("fasal_model.tflite", "wb").write(tflite.convert())
json.dump(classes, open("labels.json", "w"), ensure_ascii=False)
print("Saved fasal_model.tflite and labels.json")
