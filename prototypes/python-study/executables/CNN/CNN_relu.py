# %%
import pickle
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np
from IPython.core.interactiveshell import InteractiveShell
from keras.api import Model, Sequential, layers
from keras.api.datasets import fashion_mnist

InteractiveShell.ast_node_interactivity = "all"

saved_path: Path = Path.home() / "build" / "model"
saved_path.mkdir(parents=True, exist_ok=True)
file_name_stem: str = "fashion_mnist_relu"
# %%
### Title: CNN
(f_image_train, f_label_train), (f_image_test, f_label_test) = fashion_mnist.load_data()
# normalized iamges
f_image_train, f_image_test = f_image_train / 255.0, f_image_test / 255.0


image_train = f_image_train[:2000, :, :]
label_train = f_label_train[:2000]
image_val = f_image_train[2000:2500, :, :]
label_val = f_label_train[2000:2500]
val_dataset = (image_val, label_val)

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

plt.figure(figsize=(10, 10))
for i in range(10):
    plt.subplot(3, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(f_image_train[i])
    plt.xlabel(class_names[f_label_train[i]])
    plt.show()

model = Sequential()
model.add(layers.Conv2D(32, (2, 2), activation="relu", input_shape=(28, 28, 1)))
model.add(layers.Conv2D(64, (2, 2), activation="relu"))
model.add(layers.Conv2D(128, (2, 2), 2, activation="relu"))
model.add(layers.Conv2D(32, (2, 2), activation="relu"))
model.add(layers.Conv2D(64, (2, 2), activation="relu"))
model.add(layers.Conv2D(128, (2, 2), 2, activation="relu"))

model.add(layers.Flatten())
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dense(10, activation="softmax"))

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)


history = model.fit(
    image_train, label_train, validation_data=val_dataset, epochs=10, batch_size=10
)
model.summary()

print(f"{saved_path}/{file_name_stem}_history")

with open(f"{saved_path}/{file_name_stem}_history", "wb") as file_pi:
    pickle.dump(history.history, file_pi)
model.save(f"{saved_path}/{file_name_stem}.keras")  # Correct method to save the model
keras.saving.save_model(model, f"{saved_path}/{file_name_stem}.keras")  # Correct usage

# Optionally, using keras.saving.save_model (though the above line already saves it)
loaded_model: Model = keras.saving.load_model(f"{saved_path}/{file_name_stem}.keras")

num = 10000
predict = model.predict(f_image_test[:num])
print(f_label_train[:num])
print(" * Prediction, ", np.argmax(predict, axis=1))

loss_, accuracy_ = loaded_model.evaluate(
    f_image_train[:num], f_label_test[:num], verbose=2
)
print(f"ReLU 모델의 테스트 정확도: {accuracy_ * 100:.2f}%")

# %%

##### Display metrics
history_relu = pickle.load(open(f"{saved_path}/fashion_mnist_relu_history", "rb"))
history_batch_sig = pickle.load(
    open(f"{saved_path}/fashion_mnist_batch_sig_history", "rb")
)
history_nobatch_sig = pickle.load(
    open(f"{saved_path}/fashion_mnist_nobatch_sig_history", "rb")
)

tra_acc_relu = history_relu["accuracy"]
tra_loss_relu = history_relu["loss"]
val_acc_relu = history_relu["val_accuracy"]
val_loss_relu = history_relu["val_loss"]

tra_acc_batch_sig = history_batch_sig["accuracy"]
tra_loss_batch_sig = history_batch_sig["loss"]
val_acc_batch_sig = history_batch_sig["val_accuracy"]
val_loss_batch_sig = history_batch_sig["val_loss"]


tra_acc_nobatch_sig = history_nobatch_sig["accuracy"]
tra_loss_nobatch_sig = history_nobatch_sig["loss"]
val_acc_nobatch_sig = history_nobatch_sig["val_accuracy"]
val_loss_nobatch_sig = history_nobatch_sig["val_loss"]


plt.subplot(1, 2, 1)
plt.title("accuracy")
plt.plot(range(len(tra_acc_relu)), tra_acc_relu, label="train (relu)")
plt.plot(range(len(val_acc_relu)), val_acc_relu, label="val (relu)")
plt.plot(range(len(tra_acc_batch_sig)), tra_acc_batch_sig, label="train (sig)")
plt.plot(range(len(val_acc_batch_sig)), val_acc_batch_sig, label="val (sig)")
plt.plot(range(len(tra_acc_nobatch_sig)), tra_acc_nobatch_sig, label="train (sig)")
plt.plot(range(len(val_acc_nobatch_sig)), val_acc_nobatch_sig, label="val (sig)")
plt.legend()
plt.subplot(1, 2, 2)
plt.title("loss")
plt.plot(range(len(tra_loss_relu)), tra_loss_relu, label="train (relu)")
plt.plot(range(len(val_loss_relu)), val_loss_relu, label="val (relu)")
plt.plot(range(len(tra_loss_batch_sig)), tra_loss_batch_sig, label="train (sig)")
plt.plot(range(len(val_loss_batch_sig)), val_loss_batch_sig, label="val (sig)")
plt.plot(range(len(tra_loss_nobatch_sig)), tra_loss_nobatch_sig, label="train (sig)")
plt.plot(range(len(val_loss_nobatch_sig)), val_loss_nobatch_sig, label="val (sig)")
plt.legend()

plt.show()

# %%
saved_path
