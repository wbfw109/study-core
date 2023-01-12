# %% [markdown]
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# To use Google Colab, add following code
#
# # Linear regression
# Link: [Wine dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality "UCI Machine Learning Repository")
# https://www.tensorflow.org/datasets/catalog/wine_quality
#
# # # Problem classification
#   - Linear regression
#       - ⊂ Multiple Linear regression
#
#   - Supervised learning
#   - Manual feature engineering
#
# # # requirements version
#   - tensorflow==1.15.2
#       > python==3.7
#
#
# # # tensorboard
#   - $ tensorboard --logdir /home/wbfw109/wbfw109_flask/ml/red_wine/
#
#
# # # Result
#   - conditions
#       > batch size is 32
#
#   - if learning_rate value equal and greater than 10 ** -3, it makes overshooting.
#
#   - if learning_rate value equal and less than 10 ** -6, it makes no changes.
#
#   - I think learning_rate 10 ** -5 is value for convergence
#
#   - accuracy with 0.5 threshold is 0.5591715976331361
#
# Todo: how to use other @tf.function by function name from loaded model
#
"""
migration guide tensorflow version to 1.1 from 2.x

    - changed code
        self.checkpoint: tf.train.Checkpoint = tf.train.Checkpoint(self)
        > self.checkpoint: tf.train.Checkpoint = tf.train.Checkpoint(model=self)

        normalized_red_wines: pandas.DataFrame = normalize_using_zscore(red_wines, columns=red_wines.columns.drop(Y_COLUMN), common_threshold=2.0)      
        > normalized_red_wines: pandas.DataFrame = red_wines

        current_loss = MultipleLinearModel.loss(y, self(x.tolist()))


    - added code
        tf.enable_eager_execution()

    - delete code (require change code instead of delete)
        <feature zscore part>
        <tfds part>
        <load model and preidct part> 

    - ???
        iterations_per_epoch: int = int(1599 / batch_size)
    
    - check_existing_data
        !ls -R {WBFW109_FLASK_PATH}


"""

import logging
import os
import time
from pathlib import Path

# %%
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas
import seaborn
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_datasets as tfds
from IPython.core.interactiveshell import InteractiveShell
from IPython.display import clear_output
from wbfw109.libs.utilities.machine_learning import get_features, normalize_using_zscore

# * setting
# allow multiple print
InteractiveShell.ast_node_interactivity = "all"
# set verbose leve fl
logging.getLogger("tensorflow").setLevel(logging.ERROR)
os.environ["KMP_AFFINITY"] = "noverbose"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.autograph.set_verbosity(3)

# tf.__version__

# %%
# * set variables

problem_name: str = "red_wine"
development_environment: str = "google_drive_app"

if development_environment == "colab":
    # if it is not local environment
    from google.colab import drive

    GOOGLE_COLAB_DRIVE_PATH: Path = Path("/content/drive")
    drive.mount(str(GOOGLE_COLAB_DRIVE_PATH))
    MY_GOOGLE_DRIVE_JUPYTER_PATH: Path = (
        GOOGLE_COLAB_DRIVE_PATH / "MyDrive/Colab_Notebooks"
    )
    SAVED_FOLDER_PATH: Path = MY_GOOGLE_DRIVE_JUPYTER_PATH
else:
    # if it local environment
    from mysite.config import CONFIG_CLASS

    if development_environment == "local":
        SAVED_FOLDER_PATH: Path = CONFIG_CLASS.MACHINE_LEARNING_ROOT_PATH
    elif development_environment == "google_drive_app":
        SAVED_FOLDER_PATH: Path = CONFIG_CLASS.GOOGLE_DRIVE_APP_PATH / "Colab Notebooks"

SAVED_FOLDER_PATH = SAVED_FOLDER_PATH / problem_name

# %%
# ** prepare

tfds.features.FeaturesDict(
    {
        "features": tfds.features.FeaturesDict(
            {
                "alcohol": tf.float32,
                "chlorides": tf.float32,
                "citric acid": tf.float32,
                "density": tf.float32,
                "fixed acidity": tf.float32,
                "free sulfur dioxide": tf.float32,
                "pH": tf.float32,
                "residual sugar": tf.float32,
                "sulphates": tf.float64,
                "total sulfur dioxide": tf.float32,
                "volatile acidity": tf.float32,
            }
        ),
        "quality": tf.int32,
    }
)

Y_COLUMN: str = "quality"
winequality_red_file = keras.utils.get_file(
    "winequality-red.csv",
    "https://raw.githubusercontent.com/wbfw109-park/light_resource/main/winequality-red.csv",
)
red_wines: pandas.DataFrame = pandas.read_csv(winequality_red_file, sep=";")

correlations: pandas.DataFrame = red_wines.corr()[Y_COLUMN].drop(Y_COLUMN)

# ** show summary of data
# red_wines.shape
# dataframe.head()
# red_wines.describe()
# correlations
# each_data = [seaborn.relplot(data=red_wines, x=x, y='quality') for x in list(red_wines.columns)[:-1]]

# %%
# * Feature Engineering Start
# ** normalization and outlier detection using standard deviation
normalized_red_wines: pandas.DataFrame = normalize_using_zscore(
    red_wines, columns=red_wines.columns.drop(Y_COLUMN), common_threshold=2.0
)
normalized_correlations: pandas.DataFrame = normalized_red_wines.corr()[Y_COLUMN].drop(
    Y_COLUMN
)

# f"[before normalization] shape: {red_wines.shape}, correlations:"
# correlations
# f"[after normalization] shape: {normalized_red_wines.shape}, correlations: "
# normalized_correlations

# ** taking features with correlation more than threashold as input x and quality as target variable y
filtered_features = get_features(normalized_correlations, threshold=0.1)
filtered_features_including_label = filtered_features.copy()
filtered_features_including_label.append(Y_COLUMN)
filtered_red_wines: pandas.DataFrame = normalized_red_wines[
    filtered_features_including_label
]
filtered_correlations: pandas.DataFrame = filtered_red_wines.corr()[Y_COLUMN].drop(
    Y_COLUMN
)

# f"[after filtering] shape: {filtered_red_wines.shape}, correlations: "
# filtered_correlations

# ** default dataset
train_data: pandas.DataFrame
test_data: pandas.DataFrame
train_data, test_data = np.split(
    filtered_red_wines, [int(0.7 * len(filtered_red_wines))]
)

red_wine_train_dataset: tf.data.Dataset = tf.data.Dataset.from_tensor_slices(
    dict(train_data)
)
red_wine_test_dataset: tf.data.Dataset = tf.data.Dataset.from_tensor_slices(
    dict(test_data)
)

test_data_y = test_data.pop(Y_COLUMN)
# * Feature Engineering End

# %%
# prepare fitting linear regression to training data.


class MultipleLinearModel(tf.Module):
    def __init__(self, features_columns: list[str], model_root_path: Path, **kwargs):
        super().__init__(**kwargs)

        # In practice, biases should be randomly initialized
        self.w_n: dict[str, tf.Variable] = {
            column: tf.Variable(
                tf.random.uniform(shape=[1], dtype=tf.float64),
                trainable=True,
            )
            for column in features_columns
        }

        self.b: tf.Variable = tf.Variable(
            tf.random.uniform(shape=[1], dtype=tf.float64),
            trainable=True,
            name="biases",
        )

        self.set_features_to_be_trained(features_columns)

        self.global_step: tf.Variable = tf.Variable(
            0, dtype=tf.int32, trainable=False, name="global_step"
        )

        self.model_root_path = model_root_path
        self.checkpoint_path: Path = model_root_path / "checkpoint"
        self.saved_model_path: Path = model_root_path / "saved_model"
        self._init_path()

        self.checkpoint: tf.train.Checkpoint = tf.train.Checkpoint(self)
        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint, str(self.checkpoint_path), max_to_keep=3
        )
        if (self.checkpoint_path / "checkpoint").exists():
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)

    # hypothesis results in y_pred action
    def __call__(self, x_n_tf: tf.Tensor):
        """it require to pre-process converting to Tensor objects with same the number of columns and order of columns before passing argument x_n_tf

        The shape of arguments must be a shape that can do matrix multiplication and results in one element.

        x_n_tf is preceded for compatibility with pandas.dataframe.

        + e.g. `code: evaluate and update parameters`
            loss_fn = lambda: MultipleLinearModel.loss_rmse(
                target_y=tf.cast(shuffled_batch[target_column], dtype=tf.float64),
                predicted_y=[
                    tf.reshape(
                        self(
                            tf.reshape(
                                [
                                    shuffled_batch[column][i]
                                    for column in self.features_columns
                                ],
                                shape=self.x_shape,
                            )
                        ),
                        shape=[],
                    )
                    for i in range(batch_size)
                ],
            )

            step_count = optimizer_sgd.minimize(
                loss=loss_fn,
                var_list=[
                    *[self.w_n[column] for column in self.features_columns],
                    self.b,
                ],
            ).numpy()

        + note:
            - To use keras.optimizers and trace trainable tf.Variables, tf.Variables must be used only in this function.
            - tf.GradientTape can not be used in tf.data.Dataset iterators and tf.queue.
                refer to the link: https://www.tensorflow.org/guide/autodiff#4_took_gradients_through_a_stateful_object
            - it can not use keras.metrics.RootMeanSquaredError with keras.optimizers at a same time.
                because they were originally created for keras keras.Model for High-level API. refer to the keras.Model.compile()
            - if you want to change columns to be trained, use function set_features_to_be_trained().
                for performance, __call__ not passes immutable columns argument.

        Args:
            x_n_tf (tf.Tensor): [description]
        """

        return tf.matmul(x_n_tf, self.w_n_by_filtered_columns) + self.b

    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float64)])
    def predict(self, x_n_tf):
        return tf.matmul(x_n_tf, self.w_n_by_filtered_columns) + self.b

    @tf.function()
    def get_features_names(self):
        return self.features_columns

    def _init_path(self) -> None:
        self.model_root_path.mkdir(exist_ok=True)
        self.checkpoint_path.mkdir(exist_ok=True)
        self.saved_model_path.mkdir(exist_ok=True)

    def set_features_to_be_trained(self, features_columns: list[str]) -> None:
        self.features_columns: list[str] = features_columns
        self.x_shape: tuple = (1, len(features_columns))
        self.w_n_by_filtered_columns = [
            self.w_n[column] for column in self.features_columns
        ]

    @staticmethod
    def loss_rmse(target_y, predicted_y):
        return tf.sqrt(
            tf.reduce_mean(tf.math.square(tf.math.subtract(target_y, predicted_y)))
        )

    def train_in_loop(
        self,
        train_dataset: tf.data.Dataset,
        target_column: str,
        learning_rate: float = 10**-3,
        epochs: int = 0,
        batch_size: int = 32,
        accuracy_fn: Callable = None,
        shows_loss: bool = False,
    ):
        """
        if learning_rate value equal and greater than 10 ** -3, it makes overshooting.
        if learning_rate value equal and less than 10 ** -6, it makes no changes.
        I think learning_rate 10 ** -5 is corner value
        """
        # ** pre-process
        iterations_per_epoch: int = int(len(train_dataset) / batch_size)
        optimizer_sgd = keras.optimizers.SGD(learning_rate=learning_rate)
        start_time = time.time()

        # ** process
        for iteration, shuffled_batch in enumerate(
            train_dataset.shuffle(buffer_size=batch_size, reshuffle_each_iteration=True)
            .batch(batch_size, drop_remainder=True)
            .repeat(epochs),
            start=1,
        ):
            # evaluate and update parameters
            loss_fn = lambda: MultipleLinearModel.loss_rmse(
                target_y=tf.cast(shuffled_batch[target_column], dtype=tf.float64),
                predicted_y=[
                    tf.reshape(
                        self(
                            tf.reshape(
                                [
                                    shuffled_batch[column][i]
                                    for column in self.features_columns
                                ],
                                shape=self.x_shape,
                            )
                        ),
                        shape=[],
                    )
                    for i in range(batch_size)
                ],
            )

            step_count = optimizer_sgd.minimize(
                loss=loss_fn,
                var_list=[
                    *self.w_n_by_filtered_columns,
                    self.b,
                ],
            ).numpy()

            # ** pro-process
            self.checkpoint_manager.save()

            print(f"completes: local steps = {step_count}")
            if shows_loss:
                current_loss = loss_fn()
                print(f"current loss = {current_loss}")

            if iteration % iterations_per_epoch == 0:
                clear_output(wait=True)
                tf.saved_model.save(self, str(self.saved_model_path))

                end_time = time.time()
                self.global_step.assign_add(1)
                print(f"completes: global steps = {self.global_step.numpy()}")
                print(f"time taken seconds: {end_time-start_time}")
                if accuracy_fn:
                    print(accuracy_fn())
                start_time = time.time()


multiple_linear_model = MultipleLinearModel(
    features_columns=filtered_features, model_root_path=SAVED_FOLDER_PATH
)

# test
get_test_true_y = lambda: tf.constant(test_data_y, dtype=tf.float64)
get_test_pred_y = lambda: [
    tf.reshape(
        multiple_linear_model(
            x_n_tf=tf.reshape(
                (
                    tf.convert_to_tensor(
                        [test_data_true_x_one[1]],
                        dtype=tf.float64,
                    )
                ),
                (1, len(multiple_linear_model.features_columns)),
            )
        ),
        [],
    )
    for test_data_true_x_one in test_data.iterrows()
]

test_y_difference_fn = lambda: tf.math.subtract(get_test_true_y(), get_test_pred_y())
test_y_mean_difference_fn = lambda: tf.reduce_mean(test_y_difference_fn())
test_y_difference_abs_fn = lambda: tf.math.abs(test_y_difference_fn())
test_y_logistic_threshold_fn = lambda: tf.math.divide(
    tf.math.count_nonzero(tf.math.less_equal(x=test_y_difference_abs_fn(), y=[0.5])),
    len(test_data_y),
)

# ??? red_wine_test_dataset 사용하기?

multiple_linear_model.train_in_loop(
    red_wine_train_dataset,
    target_column=Y_COLUMN,
    learning_rate=10**-5,
    epochs=0,
    # batch_size=32,
    # accuracy_fn=test_y_logistic_threshold_fn,
    # shows_loss=True,
)

# %%
# load model and preidct
import random

# input_column: np.ndarray = red_wines.columns.drop(Y_COLUMN)
filtered_features_copy_literal = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "chlorides",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]
version_1_features_copy = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "chlorides",
    "total sulfur dioxide",
    "density",
    "sulphates",
    "alcohol",
]

random_test_data_df = pandas.DataFrame.from_dict(
    {x: [random.random() + random.randint(0, 5)] for x in version_1_features_copy}
)

loaded_model = tf.saved_model.load(str(multiple_linear_model.saved_model_path))
loaded_model

# loaded_model.get_features_names()

loaded_model.signatures.keys()
infer = loaded_model.signatures["serving_default"]
infer.structured_outputs

result = infer(
    x_n_tf=tf.reshape(
        tf.convert_to_tensor(random_test_data_df.iloc[0], dtype=tf.float64),
        shape=(1, random_test_data_df.iloc[0].size),
    )
)
# convert to result into scalar value
list(result.values())[0].numpy().item()

#%%
# load model in tf1
# with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
#     saver: tf.train.Saver = tf.train.import_meta_graph(str(model_dir / 'model.ckpt.meta'))
#     saver.restore(sess, save_path=tf.train.latest_checkpoint(str(model_dir)))

#     model = tf.saved_model.load(sess, tags=None, export_dir)
