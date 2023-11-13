import numpy as np

a = [1,2,3,4]
b = np.array([3,4])
a[0:2] = b
print(a)