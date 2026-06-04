import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import json

# -----------------------------
# CONFIG
# -----------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 8
DATASET_PATH = "model_1_dataset"

# -----------------------------
# LOAD DATASETS
# -----------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DATASET_PATH}/train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

valid_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DATASET_PATH}/valid",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DATASET_PATH}/test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("Classes:", class_names)
print("Number of classes:", num_classes)

# -----------------------------
# SAVE CLASS NAMES (IMPORTANT)
# -----------------------------
with open("crop_class_names.json", "w") as f:
    json.dump(class_names, f)

# -----------------------------
# DATA PREPROCESSING
# -----------------------------
AUTOTUNE = tf.data.AUTOTUNE

def preprocess(ds):
    return ds.map(
        lambda x, y: (preprocess_input(x), y),
        num_parallel_calls=AUTOTUNE
    ).prefetch(AUTOTUNE)

train_ds = preprocess(train_ds)
valid_ds = preprocess(valid_ds)
test_ds  = preprocess(test_ds)

# -----------------------------
# DATA AUGMENTATION (SAFE)
# -----------------------------
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# -----------------------------
# MODEL DEFINITION
# -----------------------------
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False  # Transfer Learning

model = models.Sequential([
    data_augmentation,
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation="softmax")
])

# -----------------------------
# COMPILE MODEL
# -----------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# TRAIN MODEL
# -----------------------------
history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS
)

# -----------------------------
# TEST EVALUATION
# -----------------------------
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# -----------------------------
# SAVE MODEL (SAFE FORMAT)
# -----------------------------
model.save("crop_identification_model.keras")

print("✅ Model training and saving completed successfully.")
