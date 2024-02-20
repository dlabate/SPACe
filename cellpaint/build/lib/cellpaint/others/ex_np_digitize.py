import numpy as np

a = np.random.randint(0, 10, (30, ))
bins = np.linspace(0, 10, 4)
b = np.digitize(a, bins=bins, right=True)
print(a)
print(bins)
print(b)