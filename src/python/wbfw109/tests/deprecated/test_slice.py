# %%
import pandas as pd
import numpy as np

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

# on list type
tmp_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
tmp_list[0:7:2]

# on pandas.DataFrame type
a = np.random.normal(size=200)
b = np.random.uniform(size=200)
tmp_df = pd.DataFrame({"A": a, "B": b})
# tmp_df[::2]

feature = tmp_df.iloc[:, 0]
feature

featureA = tmp_df.iloc[:, 1]
featureA

# DataFrame.iloc[ parameter ]
#   parameter ::= [ row selector, [colmun selector]]
#   selector is integer position
