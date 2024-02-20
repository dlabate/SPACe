import numpy as np

a = np.random.randint(-3, 4, (4, 2, 3))
# b = np.add.reduceat(a, (0, 1, 3, 2), axis=2)
# print(a.shape, b.shape)
# print(a, 5*'\n', b)

b = np.median(a, axis=0)
e = np.nanmin(a, axis=(1, 2))
print(e.shape)
# c = np.repeat(b[np.newaxis], repeats=4, axis=0)
# print(a.shape, b.shape, c.shape)
# print('a')
# print(a, 2*'\n')
# print('b')
# print(b, '\n')
#
# print('c')
# print(c)
