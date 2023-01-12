# %%
#

import math
from IPython import get_ipython

get_ipython().run_line_magic("matplotlib", "inline")
import numpy as np
from matplotlib import pyplot as plt

t = np.linspace(0, 2 * math.pi, 800)
a = np.sin(t)
plt.figure(figsize=(9, 6), dpi=75)
plt.plot(t, a, "r")
plt.show()
