import pandas as pd
import matplotlib.pyplot as plt


"""
[How do I create plots in pandas?]
https://pandas.pydata.org/pandas-docs/stable/reference/frame.html#plotting

Apart from the default line plot when using the plot function, a number of alternatives are available to plot data.
Let’s use some standard Python to get an overview of the available plot methods:
[
    method_name
    for method_name in dir(air_quality.plot)
    if not method_name.startswith("_")
]
"""
