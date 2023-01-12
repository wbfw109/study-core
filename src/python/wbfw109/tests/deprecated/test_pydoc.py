"""
https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#cross-referencing-python-objects
https://www.markdownguide.org/basic-syntax/
"""


def test_sphinx_grammar() -> list:
    """
    - backtics shows code block: `a = 10`
    - :const:`False`.
    - :meth:`run` method.
    - :func:`kombu.compression.register`.
    - :exc:`~@MaxRetriesExceededError`.
    - :setting:`task_eager_propagates`.
    - ~@Ignore:
    """
    pass
