import numpy as np

aa = np.random.randint(0, 5, (8, 8))

unix = np.unique(aa)
unix = np.setdiff1d(unix, 0)

bb = aa == unix[:, None, None]
print(aa.shape, bb.shape)
print(aa)
print(unix)
print(bb.astype(np.uint8))



