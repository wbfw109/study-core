# %% [markdown]
# # Keras
# Link: [Save and load Keras models](https://www.tensorflow.org/guide/keras/save_and_serialize#saving_loading_only_the_models_weights_values "Keras - Save and load Keras models")
#

# Todo: How to save and load a model ~

# %%
from IPython.core.interactiveshell import InteractiveShell
import logging
import os
from pathlib import Path
import tensorflow as tf
import tensorflow.keras as keras
from keras_adabound import AdaBound

# ** setting
# allow multiple print
InteractiveShell.ast_node_interactivity = "all"
# set verbose level
logging.getLogger("tensorflow").setLevel(logging.ERROR)
os.environ["KMP_AFFINITY"] = "noverbose"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.autograph.set_verbosity(3)

problem_name: str = "cifar_10"
development_environment: str = "google_drive_app"
if development_environment == "colab":
    # if it is not local environment
    from google.colab import drive

    GOOGLE_COLAB_DRIVE_PATH: Path = Path("/", "content", "drive")
    drive.mount(str(GOOGLE_COLAB_DRIVE_PATH))
    MY_GOOGLE_DRIVE_JUPYTER_PATH: Path = (
        GOOGLE_COLAB_DRIVE_PATH / "MyDrive" / "Colab_Notebooks"
    )
    SAVED_FOLDER_PATH: Path = MY_GOOGLE_DRIVE_JUPYTER_PATH
else:
    # if it local environment
    from mysite.config import CONFIG_CLASS

    if development_environment == "local":
        SAVED_FOLDER_PATH: Path = CONFIG_CLASS.MACHINE_LEARNING_ROOT_PATH
    elif development_environment == "google_drive_app":
        SAVED_FOLDER_PATH: Path = CONFIG_CLASS.GOOGLE_DRIVE_APP_PATH / "Colab_Notebooks"

SAVED_FOLDER_PATH = SAVED_FOLDER_PATH / problem_name

# %%
# * Introduction
"""
A Keras model consists of multiple components:
    - The architecture, or configuration, which specifies what layers the model contain, and how they're connected.
    - A set of weights values (the "state of the model").
    - An optimizer (defined by compiling the model).
    - A set of losses and metrics (defined by compiling the model or calling add_loss() or add_metric()).

The Keras API makes it possible to save all of these pieces to disk at once, or to only selectively save some of them:
    - Saving everything into a single archive in the TensorFlow SavedModel format (or in the older Keras H5 format). This is the standard practice.
    - Saving the architecture / configuration only, typically as a JSON file.
    - Saving the weights values only. This is generally used when training the model.

; H5 format "in the older Keras H5 format"
"""

# %%
# * How to save and load a model
# * Setup
TEST_PATH = Path.home() / "test"
TEST_PATH.mkdir(exist_ok=True)

# Saving a Keras model:
# model = keras.models.Sequential()
# model.save(TEST_PATH)

# Loading the model back:
model = keras.models.load_model(
    str(SAVED_FOLDER_PATH / "model-CNN"), custom_objects={"AdaBound": AdaBound}
)

# %%
# * Whole-model saving & loading
"""
You can save an entire model to a single artifact. It will include:
    - The model's architecture/config
    - The model's weight values (which were learned during training)
    - The model's compilation information (if compile() was called)
    - The optimizer and its state, if any (this enables you to restart training where you left)

APIs
    - model.save() or tf.keras.models.save_model()
    - tf.keras.models.load_model()

There are two formats you can use to save an entire model to disk: the TensorFlow SavedModel format, and the older Keras H5 format.

note: ● The recommended format is SavedModel. It is the default when you use model.save().
"""
