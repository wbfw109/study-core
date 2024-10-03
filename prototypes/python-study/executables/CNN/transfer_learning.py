# %%
import os
import pickle
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from IPython.core.interactiveshell import InteractiveShell
from keras import mixed_precision
from keras.api import Input, Model, layers
from keras.api.applications import MobileNetV3Small, mobilenet_v3
from keras.api.layers import GlobalAveragePooling2D

InteractiveShell.ast_node_interactivity = "all"

saved_path: Path = Path.home() / "build" / "model"
saved_path.mkdir(parents=True, exist_ok=True)

# %%
## Optimization
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.test.gpu_device_name()
# Check if TensorFlow can detect a GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

gpus = tf.config.list_physical_devices("GPU")
print(gpus)
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)  # 메모리 동적 할당
    except RuntimeError as e:
        print(e)
mixed_precision.set_global_policy("mixed_float16")

# %%

import tensorflow_datasets as tfds

BATCH_SIZE = 16
# model_name = "tf_flowers"
model_name = "cats_vs_dogs"

(train_ds, val_ds, test_ds), metadata = tfds.load(
    "cats_vs_dogs",
    split=["train[:20%]", "train[20%:25%]", "train[25%:]"],
    batch_size=32,
    with_info=True,
    as_supervised=True,
)

num_classes = metadata.features["label"].num_classes
get_label_name = metadata.features["label"].int2str

train_ds.cache()
train_ds.shuffle(buffer_size=1000)
val_ds.cache()
val_ds.shuffle(buffer_size=1000)

image, label = next(iter(train_ds))
print(np.array(image).shape)
plt.figure(figsize=(5, 5))
for i in range(10):
    plt.subplot(3, 4, i + 1)
    plt.imshow(image[i])
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.title(get_label_name(label[i]))
plt.show()


input_shape = (80, 80, 3)
# 💡 전이 학습을 위해 FC 층을 제거. (include_top=False)
base_model = MobileNetV3Small(
    input_shape=input_shape, include_top=False, weights="imagenet"
)

preprocess_input = mobilenet_v3.preprocess_input

inputs = Input(shape=input_shape)
x = preprocess_input(inputs)
# Mobilenet CNN model
x = layers.RandomFlip("horizontal_and_vertical")(x)
x = layers.RandomRotation(0.2)(x)
x = base_model(x, training=False)
x = GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
output = layers.Dense(num_classes, activation="softmax")(x)
model = Model(inputs, output)

model.summary()
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    batch_size=32,
    steps_per_epoch=20,
    epochs=10,
)

with open(f"{saved_path}/history_{model_name}", "wb") as file_pi:
    pickle.dump(history.history, file_pi)

    # Print model summary
    model.summary()

    # Save the model in Keras format
    model.save(f"{saved_path}/{model_name}.keras")  # Correct method to save the model

    # Optionally, using keras.saving.save_model (though the above line already saves it)
    keras.saving.save_model(model, f"{saved_path}/{model_name}.keras")  # Correct usage

test_ds = test_ds.cache().shuffle(buffer_size=1000)
test_loss, test_accuracy = model.evaluate(test_ds)

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
