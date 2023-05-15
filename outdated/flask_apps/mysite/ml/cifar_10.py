# %% [markdown]
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
#
# # CNN
# Link: [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html "CIFAR-10 dataset")
# https://www.tensorflow.org/datasets/catalog/cifar10
# Dataset: https://keras.io/api/datasets/cifar10/#load_data-function
#
#
# # # Problem classification
#
#   - Convolutional neural network
#       - ⊂ Multilayer perceptron
#       - , Deep learning
#       - , Feedforward neural network
#       - ⊂ Artificial neural network
#
#   - Feature learning
#
#
# # # tensorboard
#   - $ tensorboard --logdir "/mnt/c/Users/wbfw109/MyDrive/Colab Notebooks/cifar_10/model-CNN-v3/logs"
#
# # # Todo: Is it possible to restore global_step in keras? and apply callback
#

import IPython
from IPython.core.interactiveshell import InteractiveShell
from enum import Enum
from pathlib import Path
import logging
import os
import sys
import subprocess
import ast
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# * setting
# ** common environment setting for each python version
class EnvironmentLocation(Enum):
    LOCAL = 1
    LOCAL_WITH_DOCKER = 2
    GOOGLE_COLAB = 11


problem_name: str = "cifar_10"
environment_location: EnvironmentLocation = EnvironmentLocation.GOOGLE_COLAB
# tesnorflow_version: str = 1 | 2
tensorflow_version_as_integer: int = 2

# + setting - fixed variables
installed_packages: list = [
    pip_list_as_json["name"]
    for pip_list_as_json in ast.literal_eval(
        subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format", "json"],
            capture_output=True,
            text=True,
        ).stdout
    )
]

# + setting - to install required packages and for default location
if environment_location == EnvironmentLocation.GOOGLE_COLAB:
    # if it is not local environment
    from google.colab import drive

    GOOGLE_COLAB_DRIVE_PATH: Path = Path("/content/drive")
    drive.mount(str(GOOGLE_COLAB_DRIVE_PATH))
    MY_GOOGLE_DRIVE_JUPYTER_PATH: Path = (
        GOOGLE_COLAB_DRIVE_PATH / "MyDrive/Colab_Notebooks"
    )
    SAVED_FOLDER_PATH: Path = MY_GOOGLE_DRIVE_JUPYTER_PATH

    # upgrade outdated pip
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
else:
    # if it local environment
    from mysite.config import CONFIG_CLASS

    if environment_location == EnvironmentLocation.LOCAL:
        SAVED_FOLDER_PATH: Path = CONFIG_CLASS.MACHINE_LEARNING_ROOT_PATH
    elif environment_location == EnvironmentLocation.LOCAL_WITH_DOCKER:
        SAVED_FOLDER_PATH: Path = CONFIG_CLASS.GOOGLE_DRIVE_APP_PATH / "Colab_Notebooks"

PROBLEM_PATH = SAVED_FOLDER_PATH / problem_name
SAVED_FOLDER_PATH.mkdir(exist_ok=True)
PROBLEM_PATH.mkdir(exist_ok=True)

# + setting - tensorfow version
if tensorflow_version_as_integer == 2:
    import tensorflow as tf
    import tensorflow.keras as keras
elif tensorflow_version_as_integer == 1:
    # python <= 3.7
    if environment_location == EnvironmentLocation.GOOGLE_COLAB:
        IPython.get_ipython().run_line_magic("tensorflow_version", "1.x")
    import tensorflow as tf

    tf.compat.v1.enable_eager_execution()
    import keras

tesnorflow_version: str = tf.__version__

if "keras-adabound" not in installed_packages:
    if environment_location == EnvironmentLocation.GOOGLE_COLAB:
        print(
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "keras-adabound"],
                capture_output=True,
                text=True,
            ).stdout
        )
    else:
        # if environment_location == LOCAL-LIKE Environment:
        subprocess.run("pipenv install keras-adabound")

from keras.callbacks import LearningRateScheduler
from keras_adabound import AdaBound

# + allow multiple print
InteractiveShell.ast_node_interactivity = "all"
# + set verbose level
logging.getLogger("tensorflow").setLevel(logging.ERROR)
os.environ["KMP_AFFINITY"] = "noverbose"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.autograph.set_verbosity(3)

# ** set variables
input_shape: tuple = (32, 32, 3)
batch_size: int = 64
validation_rate: float = 0.2
validation_batch_size: int = int(batch_size / ((1 - validation_rate) / validation_rate))
epochs: int = 100
adabound_final_lr: float = 0.1
adabound_gamma: float = 1e-3
weight_decay: float = 1e-4
is_amsgrad: bool = False

model_name: str = "model-CNN-v4"
MODEL_PATH: Path = PROBLEM_PATH / model_name
LOG_PATH: Path = MODEL_PATH / "logs"

tensorboard = keras.callbacks.TensorBoard(
    LOG_PATH, update_freq="batch", histogram_freq=1
)
checkpoint = keras.callbacks.ModelCheckpoint(
    filepath=str(MODEL_PATH),
    monitor="loss",
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
)
early_stopping = keras.callbacks.EarlyStopping(monitor="loss", patience=10)


def load_model(model_name):
    base_url = "http://download.tensorflow.org/models/object_detection/"
    model_file = model_name + ".tar.gz"
    model_dir = keras.utils.get_file(
        fname=model_name, origin=base_url + model_file, untar=True
    )

    model_dir = Path(model_dir) / "saved_model"

    model = tf.saved_model.load(str(model_dir))

    return model


def schedule_learning_rate(epoch):
    """Learning Rate Schedule
    Called automatically every epoch as part of callbacks during training.
    """
    learning_rate: float = 1e-2
    epoch += 1
    if epoch > 75:
        learning_rate = 1e-3
        epoch_multiplier = int((epoch - 60) / 20)
        learning_rate -= 1e-4 * epoch_multiplier

    print("Learning rate: ", learning_rate)
    return learning_rate


learning_rate_scheduler = LearningRateScheduler(schedule_learning_rate)
callbacks = [checkpoint, learning_rate_scheduler, tensorboard, early_stopping]

# %%
# * preprocess
class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
len_class: int = len(class_names)

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# one-hot encoding
y_train = keras.utils.to_categorical(y_train, len_class)
y_test = keras.utils.to_categorical(y_test, len_class)

# data augmentation
# cross-validation: Repeated random sub-sampling validation
image_generator = keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    horizontal_flip=True,
    validation_split=validation_rate,
    # ** not helped
    # zca_whitening=True,
    rotation_range=15,
    # width_shift_range=0.1,
    # height_shift_range=0.1,
)
image_generator.fit(x_train)

# %%
# * Verify the data
# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5, 5, i + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(x_train[i])
#     # The CIFAR labels happen to be arrays,
#     # which is why you need the extra index
#     plt.xlabel(class_names[y_train[i][0]])
# plt.show()

# %%
# * Create the convolutional base +
# * Add Dense layers on top

if not MODEL_PATH.exists():
    print("===== defining model layers")
    model = keras.models.Sequential()

    # 1
    model.add(
        keras.layers.Conv2D(
            input_shape=input_shape,
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            kernel_regularizer=keras.regularizers.l2(weight_decay),
        )
    )
    model.add(
        keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            kernel_regularizer=keras.regularizers.l2(weight_decay),
        )
    )
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ELU())
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.20))

    # 2
    model.add(
        keras.layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            kernel_regularizer=keras.regularizers.l2(weight_decay),
        )
    )
    model.add(
        keras.layers.Conv2D(
            filters=32,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="valid",
            kernel_regularizer=keras.regularizers.l2(weight_decay),
        )
    )
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ELU())
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.20))

    # 3
    # 2
    model.add(
        keras.layers.Conv2D(
            filters=256,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            kernel_regularizer=keras.regularizers.l2(weight_decay),
        )
    )
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ELU())
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.20))

    # classify
    model.add(keras.layers.GlobalAveragePooling2D())
    model.add(
        keras.layers.Dense(
            units=128,
            activation="elu",
        )
    )
    model.add(
        keras.layers.Dense(
            units=64,
            activation="elu",
        )
    )
    model.add(
        keras.layers.Dense(
            units=len_class,
            activation="softmax",
            kernel_initializer=keras.initializers.HeNormal(),
        )
    )

    # ** Compile and train the mode

    model.compile(
        optimizer=AdaBound(
            learning_rate=schedule_learning_rate(0),
            final_lr=adabound_final_lr,
            gamma=adabound_gamma,
            weight_decay=weight_decay,
            amsgrad=is_amsgrad,
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
else:
    # ** load previous model
    print("===== load model from existing file. instead of defining model layers")
    model: keras.models.Sequential = keras.models.load_model(
        str(MODEL_PATH), custom_objects={"AdaBound": AdaBound}
    )

model.summary()

# %%

model.fit(
    image_generator.flow(
        x_train, y_train, batch_size=batch_size, subset="training", shuffle=True
    ),
    validation_data=image_generator.flow(
        x_train,
        y_train,
        batch_size=validation_batch_size,
        subset="validation",
        shuffle=True,
    ),
    steps_per_epoch=int(len(x_train) / batch_size),
    epochs=epochs,
    callbacks=callbacks,
    use_multiprocessing=True,
)

# %%
# * Evaluate the model

model.load_weights(str(MODEL_PATH))

# ** Score trained model.
image_test_generator = keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
)

image_test_generator.fit(x_test)

scores = model.evaluate(
    image_test_generator.flow(x_test, y_test, batch_size=batch_size),
    use_multiprocessing=True,
)
print("Test loss:", scores[0])
print("Test accuracy:", scores[1])

# ** confusion matrix
predictions = model.predict(image_test_generator.flow(x_test))

class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

cm = confusion_matrix(y_test.argmax(1), predictions.argmax(1), normalize="all")
# plt.figure(figsize=(8,8))
# normalize
cmn = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(
    cmn,
    annot=True,
    fmt=".2f",
    xticklabels=class_names,
    yticklabels=class_names,
    cmap="YlGnBu",
)
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show(block=False)

# %%
# test predict

# cifar_10_model: keras.models.Sequential = keras.models.load_model(
#     str(MODEL_PATH), custom_objects={"AdaBound": AdaBound}
# )

# # %%
# #
# class_names = [
#     "airplane",
#     "automobile",
#     "bird",
#     "cat",
#     "deer",``
#     "dog",
#     "frog",
#     "horse",
#     "ship",
#     "truck",
# ]
# len_class: int = len(class_names)

# image_test_generator = keras.preprocessing.image.ImageDataGenerator(
#     featurewise_center=True,
#     featurewise_std_normalization=True,
# )

# for class_name in class_names:
#     resized_image = keras.preprocessing.image.load_img(
#         str(CONFIG_CLASS.IMAGE_PROCESSING_SOURCE_PATH / f"{class_name}.jpg"), target_size=(32, 32)
#     )
#     image_np = keras.preprocessing.image.img_to_array(resized_image)
#     image_np = np.expand_dims(image_np, axis=0)
#     image_test_generator.fit(image_np)

#     class_predicted = class_names[np.argmax(cifar_10_model.predict(image_test_generator.flow(image_np)), axis=-1).item()]
#     print(f"origin: {class_name} : pred: {class_predicted}")
"""
droptout 을 높이면 train accuracy 가 높아진다.
droptout 을 높이면 validation accuracy 가 dropout 수치와 비슷하게 높아진다.
    // drop out 높을수록 train accuracy 에 비해 validation accuracy 가 낮게 나옴. (0.5 ~ 0.3 까지 테스트 함.)
drop out 을 0.2 씩 두고 kernel_regular 추가
    loss: 0.3339 - accuracy: 0.9046 - val_loss: 0.4414 - val_accuracy: 0.8711
    Test loss: 0.4683757424354553
    Test accuracy: 0.8647000193595886
v7. http://mipal.snu.ac.kr/images/1/16/Dropout_ACCV2016.pdf  dropout 0.1 and locate dropout on pooling.
    loss: 0.1961 - accuracy: 0.9498 - val_loss: 0.4686 - val_accuracy: 0.8624
    Test loss: 0.49153760075569153
    Test accuracy: 0.8633000254631042
v1. dropout 0.15 씩으로 변경
    loss: 0.3969 - accuracy: 0.8848 - val_loss: 0.5381 - val_accuracy: 0.8356
    Test loss: 0.5524666905403137
    Test accuracy: 0.833299994468689
v2. dropout 0.05 씩으로 변경
    loss: 0.2487 - accuracy: 0.9406 - val_loss: 0.6065 - val_accuracy: 0.8344
    Test loss: 0.559441328048706
    Test accuracy: 0.8500999808311462

Todo: from skmultilearn.model_selection import iterative_train_test_split
    사이킷런 사용해서 stratify 나 이걸 사용해서 클래스 균등하게 밸리데이션 세트 나누기.
"""
