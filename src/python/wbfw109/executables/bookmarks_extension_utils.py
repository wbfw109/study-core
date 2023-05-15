# %%
from __future__ import annotations


from IPython.core.interactiveshell import InteractiveShell

# + allow multiple print
InteractiveShell.ast_node_interactivity = "all"
# %doctest_mode

# from google.protobuf.internal import api_implementation
# print(api_implementation.Type())


# %% # title: Declaration and Crawled tab


start_line: int = 468  # inclusive
# end_line: int = 470  # exclusive
end_line: int = 1446  # exclusive
current_line: int = start_line


def get_bookmark_json_format(line_num: int) -> str:
    return f"""
        {{
            "line": {str(line_num)},
            "column": 1,
            "label": ""
        }},"""


a = "".join(get_bookmark_json_format(x) for x in range(start_line, end_line))
print(a)

# %%
