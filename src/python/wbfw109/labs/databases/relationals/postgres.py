# %%
from __future__ import annotations


from IPython.core.interactiveshell import InteractiveShell

# + allow multiple print
InteractiveShell.ast_node_interactivity = "all"

# %doctest_mode

#%%
if __name__ == "__main__":
    import sqlalchemy

    sqlalchemy.__version__

# DB 는 테스트를.. fixture 만들어야 하나
# pytest 가 필요할가?

# TODO: 인덱싱, 파티셔닝,
