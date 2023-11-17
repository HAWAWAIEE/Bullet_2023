def hello():
    return 1,3

a = 2
b = 3
a,b+=hello()
print(a,b)