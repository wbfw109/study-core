# pytest --cov=src/backend/ --cov-report=html:./resources_readme/coverages/backend
def func(x):
    return x + 1


def test_answer():
    assert func(3) == 4
