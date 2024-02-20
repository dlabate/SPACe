import numpy as np

arr = np.random.randint(0, 4, (12, 12))
component_sizes = np.bincount(arr.ravel())
print(arr)
print(component_sizes)
print(component_sizes[arr])

print((component_sizes < 40)[arr])
arr[(component_sizes < 40)[arr]] = 0
print(arr)
