# %% [markdown]

# https://github.com/tensorflow/models/tree/master/research/object_detection
#   > https://github.com/tensorflow/models/tree/master/research/object_detection/g3doc
#   > https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/
#       > https://github.com/sglvladi/TensorFlowObjectDetectionTutorial

# %%
from IPython.core.interactiveshell import InteractiveShell
import logging
import os
import tensorflow as tf

# ** setting
# allow multiple print
InteractiveShell.ast_node_interactivity = "all"
# set verbose level
logging.getLogger("tensorflow").setLevel(logging.ERROR)
os.environ["KMP_AFFINITY"] = "noverbose"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.autograph.set_verbosity(3)

# * models/research/object_detection/g3doc/
# https://github.com/tensorflow/models/tree/master/research/object_detection/g3doc

# ** models/research/object_detection/g3doc/using_your_own_dataset.md
"""
note; To use your own dataset in TensorFlow Object Detection API, you must convert it into the TFRecord file format.
This document outlines how to write a script to generate the TFRecord file.
"""

# ** models/research/object_detection/g3doc/configuring_jobs.md

# ..

# + Picking Model Parameters
"""
● There are a large number of model parameters to configure. The best settings will depend on your given application. Faster R-CNN models are better suited to cases where high accuracy is desired and latency is of lower priority.
Conversely, if processing time is the most important factor, SSD models are recommended.
Read our paper for a more detailed discussion on the speed vs accuracy tradeoff.

note: To help new users get started, sample model configurations have been provided in the object_detection/samples/configs folder.
The contents of these configuration files can be pasted into model field of the skeleton configuration.
Users should note that the num_classes field should be changed to a value suited for the dataset the user is training on.

...
"""
# Anchor box parameters
"""
Many object detection models use an anchor generator as a region-sampling strategy, which generates a large number of anchor boxes in a range of shapes and sizes, in many locations of the image. The detection algorithm then incrementally offsets the anchor box closest to the ground truth until it (closely) matches. You can specify the variety of and position of these anchor boxes in the anchor_generator config.

● Usually, the anchor configs provided with pre-trained checkpoints are designed for large/versatile datasets (COCO, ImageNet), in which the goal is to improve accuracy for a wide range of object sizes and positions. But in most real-world applications, objects are confined to a limited number of sizes. So adjusting the anchors to be specific to your dataset and environment can both improve model accuracy and reduce training time.

- Single Shot Detector (SSD) full model:
    ...
- SSD with Feature Pyramid Network (FPN) head:
    note: When using an FPN head, you must specify the anchor box size relative to the convolutional filter's stride length at a given pyramid level, using anchor_scale.
    ...




...

"""
