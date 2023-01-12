# %%
import itertools

import IPython
from IPython.core.interactiveshell import InteractiveShell

# + allow multiple print
InteractiveShell.ast_node_interactivity = "all"

a = "abcdef"
x = "fff_fff"
b = ""
c = "def"
"-".join(list(filter(None, [a, b])))
"-".join(list(filter(None, [a, c])))

# even if c is empty, str.join() includes c element
print("==== not using asterik * unpack operator")
"-".join(
    list(
        filter(
            None,
            [
                a,
                "_".join([x, b]),
            ],
        )
    )
)

print()
# but * (asterisk) unpack operator not includes c element
print("==== using asterik * unpack operator")
"_".join([x, *c])
"_".join([x, *[]])
"_".join([x, *["def"]])
print()

# but * (asterisk) unpack operator can not exclude list that includes non-empty elements and empty elements
"_".join([x, *["x", "", "y", "", "z"]])
"_".join(filter(None, [x, *["x", "", "y", "", "z"]]))
