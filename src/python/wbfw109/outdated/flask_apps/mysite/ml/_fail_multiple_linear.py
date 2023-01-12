# %% [markdown]
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
#
# ! from https://www.tensorflow.org/guide/estimator
# ! Warning: Estimators are not recommended for new code. Estimators run v1.Session-style code which is more difficult to write correctly, and can behave unexpectedly, especially when combined with TF 2 code. Estimators do fall under our compatibility guarantees, but will receive no fixes other than security vulnerabilities. See the migration guide for details.
# ! this is deprecated. I cannot find predict after export saved_model and load.
#
# note: Problem
# > estimator train, evaluation, predict 의 속도가 매우 느리다.
# > 함수 실행마다, gpu 로딩 시 정보가 출력되어 느려진다.
# > 함수 실행마다 tf.event 의 기록을 파일로서 쌓는데 비활성화가 불가능하여 느려진다.
#
# # Linear regression
# Link: [Wine dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality "UCI Machine Learning Repository")
#
# # # Problem classification
#   - multiple linear regression
#   - Supervised_learning
#
#
# # # requirements version
#   - tensorflow==1.15.2
#       > python==3.7
#
# # # tensorboard
#   - $ tensorboard --logdir /home/wbfw109/wbfw109_flask/ml/red_wine/

import logging
import os
from pathlib import Path

import numpy as np
import pandas
import seaborn
import tensorflow as tf
from IPython.core.interactiveshell import InteractiveShell
from IPython.display import clear_output

# %%
from mysite.config import CONFIG_CLASS
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.losses import losses_utils
from wbfw109.libs.utilities.machine_learning import (
    get_coefficient_from_linear,
    get_features,
    normalize_using_zscore,
)

# * setting
# allow multiple print
InteractiveShell.ast_node_interactivity = "all"
# set verbose leve fl
logging.getLogger("tensorflow").setLevel(logging.ERROR)
os.environ["KMP_AFFINITY"] = "noverbose"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.autograph.set_verbosity(3)

# tf.__version__
# ** prepare
Y_COLUMN: str = "quality"
red_wines: pandas.DataFrame = pandas.read_csv(
    "https://raw.githubusercontent.com/wbfw109-park/light_resource/main/winequality-red.csv",
    sep=";",
)
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
filtered_red_wines = normalized_red_wines[filtered_features_including_label]
filtered_correlations: pandas.DataFrame = filtered_red_wines.corr()[Y_COLUMN].drop(
    Y_COLUMN
)

# f"[after filtering] shape: {filtered_red_wines.shape}, correlations: "
# filtered_correlations

# ** default dataset
x: pandas.DataFrame = filtered_red_wines[filtered_features]
y: pandas.Series = filtered_red_wines[Y_COLUMN]

# * Feature Engineering End

# %%
# prepare fitting linear regression to training data.
# default of keywards: label_dimension=1, optimizer='Ftrl', loss_reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE, sparse_combiner='sum')
WINE_PATH: Path = CONFIG_CLASS.MACHINE_LEARNING_ROOT_PATH / "red_wine"
CHECKPOINT_PATH: Path = WINE_PATH / "checkpoint"
SAVED_MODEL_PATH: Path = WINE_PATH / "saved_model"
warm_start_from: str = None
if WINE_PATH.exists():
    if CHECKPOINT_PATH.exists():
        warm_start_from = str(CHECKPOINT_PATH)
else:
    WINE_PATH.mkdir()
SAVED_MODEL_PATH.mkdir(exist_ok=True)

linear_regressor = tf.estimator.LinearRegressor(
    feature_columns=[
        tf.feature_column.numeric_column(filtered_feature)
        for filtered_feature in filtered_features
    ],
    model_dir=str(WINE_PATH),
    label_dimension=1,
    weight_column=None,
    optimizer="Ftrl",
    config=None,
    warm_start_from=warm_start_from,
    loss_reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE,
    sparse_combiner="sum",
)

# Input builders


def input_fn(features, labels, batch_size=1536):
    """An input function for training or evaluating"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    return dataset.batch(batch_size)


def predict_fn(features, batch_size=1536):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), None))

    # Returns tf.data.Dataset of (x, None) tuple.
    return dataset.batch(batch_size)


# %%
# training and test
LOOP_COUNT: int = 0

# evaluation_result
evaluation_results: list = []
for count in range(LOOP_COUNT):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=3
    )

    linear_regressor.train(input_fn=lambda: input_fn(x_train, y_train))

    evaluation_result: dict = linear_regressor.evaluate(
        input_fn=lambda: input_fn(x_test, y_test)
    )
    evaluation_result["accuarcy"] = accuracy_score(
        y_true=y,
        y_pred=[
            round(y_pred["predictions"].tolist()[0])
            for y_pred in linear_regressor.predict(input_fn=lambda: predict_fn(x))
        ],
    )
    evaluation_result["count"] = count
    clear_output(wait=True)
    evaluation_result

# %%
# * save model

required_feature_sepc = tf.feature_column.make_parse_example_spec(
    [
        tf.feature_column.numeric_column(filtered_feature)
        for filtered_feature in filtered_features
    ]
)
required_feature_sepc

serving_input_fn: tf.estimator.export.ServingInputReceiver = (
    tf.estimator.export.build_parsing_serving_input_receiver_fn(
        feature_spec=required_feature_sepc
    )
)

linear_regressor.export_saved_model(
    export_dir_base=str(SAVED_MODEL_PATH), serving_input_receiver_fn=serving_input_fn
)

# %%
# load test
import random

# input_column: np.ndarray = red_wines.columns.drop(Y_COLUMN)
input_columns = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

intput_data_from_user = {
    x: [random.random() + random.randint(0, 5)] for x in filtered_features
}
input_value = [value for _, value in intput_data_from_user.items()]

loaded_model = tf.saved_model.load(str(SAVED_MODEL_PATH / "1624505815"))
loaded_model
f = loaded_model.signatures["predict"]
input_data_tf = tf.convert_to_tensor(input_value)
input_data_tf
input_data_df = pandas.DataFrame.from_dict(input_value)


def predict_input_fn(x):
    example = tf.train.Example()
    example.features.feature["x"].float_list.value.extend([x])
    return loaded_model.signatures["predict"](
        examples=tf.constant([example.SerializeToString()])
    )


f
predict_input_fn(123)

# f(tf.convert_to_tensor(intput_data_from_user))

# TypeError: 'AutoTrackable' object is not callable
#   loaded_model(tf.convert_to_tensor(input_data)).numpy()

# InvalidArgumentError: cannot compute __inference_pruned_60021 as input #0(zero-based) was expected to be a string tensor but is a float tensor [Op:__inference_pruned_60021]
#   infer(tf.convert_to_tensor(input_data)).numpy()

# %%

saved_predict_fn = loaded_model.signatures["predict"]

vars(saved_predict_fn)


def predict_input_fn(test_df):
    """Convert your dataframe using tf.train.Example() and tf.train.Features()"""

    examples = []

    return tf.constant(examples)


input_value
saved_predict_fn(examples=tf.convert_to_tensor(input_value))
