from backend import add

result = add.delay(4, 4)
print(type(result))
print(result.ready())
print(result.get())
print(result.ready())
