import numpy as np
from scipy import ndimage



a = np.zeros((2, 10, 10, 3), dtype=int)
a[0, :3, :3] = 5
a[0, 3:6, 3:6] = 1
a[0, 0:4, 6:] = 2
a[0, 3:6, :3] = 3
# a[0, 8:, 1:4] = 4
# a[0, 7:, 7:] = 6

# a[1, 2:4, 2:4] = 10
# a[1, 7, 7] = 10
a[1, :2, :2] = 10
a[1, 5, 5] = 7

b = ndimage.find_objects(a)
print(a)
print(type(b))
print(len(b))
# print(b)
for item in b:
    print(item)
    print('\n')
# for item in b:
#     print(item[0])
#     print(item[0].start, item[0].stop, item[0].step)
#     # print(item.start)
#     print(a[item])
#     print('\n')


# a = np.random.randint(-4, 4, (5, 10))
# print(a)
# b = np.unique(a, return_inverse=True)
# print(b)