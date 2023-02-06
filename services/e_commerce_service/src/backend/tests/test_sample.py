# pytest --cov=src/backend/ --cov-report=html:./resources/coverages/backend
# start dev branch
def func(x):
    return x + 1


def test_answer():
    assert func(3) == 4


def test_answer2():
    assert func(5) == 6
