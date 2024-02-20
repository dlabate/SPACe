import numpy as np

# your numpy array
aa = np.array([0, 50, 60, 0, 10, 20, 20, 30, 30, 30, 40, 40, 40, 40])

# call np.unique with return_inverse=True
unique_labels, new_labels = np.unique(aa, return_inverse=True)

# new_labels is now a numpy array with the same shape as aa, but with labels replaced with consecutive integers
print(new_labels)