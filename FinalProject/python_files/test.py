import numpy as np

arr = [[[2, 3], [1, 6], [6, 1]], [[4, 1], [7, 2], [5, 1]]]
arr = np.array(arr)

print(arr[:, 0, 0])
print(arr[:, 1, 0])
print(arr[:, 0, 1])
print(arr[:, 1, 1])