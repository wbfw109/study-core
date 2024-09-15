# %%
# Written in 📅 2024-09-15 05:06:05
# Enable all outputs in the Jupyter notebook environment

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"



data = [
    ["cheat_sheet-cpp", "standard_algorithms", "Quick_Overview", "algorithms_crop.png"],
    [
        "cheat_sheet-cpp",
        "standard_algorithms",
        "Algorithms_Gallery_…",
        "gallery-existence_queries.png",
    ],
    ["cheat_sheet-cpp", "standard_views", "std::_span_C++20", "span_crop.png"],
    [
        "cheat_sheet-cpp",
        "standard_views",
        "std::_string_view_C++17",
        "string_view_crop.png",
    ],
    [
        "cheat_sheet-cpp",
        "standard_views",
        "Composable_Range_Views_Ranges_C++20/23",
        "range_views_crop.png",
    ],
    [
        "cheat_sheet-cpp",
        "standard_randomness",
        "Standard_Random_Distributions_C++11",
        "random_distributions_crop.png",
    ],
    [
        "cheat_sheet-cpp",
        "standard_randomness",
        "Standard_Random_Sampling_Distributions_C++11",
        "random_sampling_distributions_crop.png",
    ],
]

# Process the data to slugify the third element in each sublist
for item in data:
    original_value = item[2]
    slugified_value = slugify(original_value)
    item[2] = slugified_value
    # print(item)
