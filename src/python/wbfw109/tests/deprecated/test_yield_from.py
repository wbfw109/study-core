# https://github.com/python/cpython/blob/3.9/Lib/test/test_generators.py
def counter(maximum):
    i = 0
    while i < maximum:
        val = yield i
        # If value provided, change counter
        if val is not None:
            i = val
        else:
            i += 1


counter_generator = counter(5)
for x in counter_generator:
    print(x)


def counter_subgenerator():
    yield from counter(5)


y = lambda: (yield from counter(5))

for x in counter_subgenerator():
    print(x)
for x in y():
    print(x)
