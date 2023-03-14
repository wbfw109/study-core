# %%
from __future__ import annotations

import random
import time
import unittest
from typing import Iterator, Optional

from IPython.core.interactiveshell import InteractiveShell
from wbfw109.libs.utilities.ipython import (  # type: ignore
    VisualizationManager,
    VisualizationRoot,
    display_data_frame_with_my_settings,
)

# + allow multiple print
InteractiveShell.ast_node_interactivity = "all"

# %doctest_mode

#%%

# bisection method

# if __name__ == "__main__" or VisualizationManager.central_control_state:
#     if VisualizationManager.central_control_state:
#         # Do not change this.
#         only_class_list = []
#     else:
#         only_class_list = []
#     VisualizationManager.call_root_classes(only_class_list)
