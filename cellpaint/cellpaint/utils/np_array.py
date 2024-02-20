

def sub2ind(array_shape, rows, cols):
    return rows*array_shape[1] + cols


def ind2sub(array_shape, ind):
    rows = ind // array_shape[1]
    cols = ind % array_shape[1]  # or numpy.mod(ind.astype('int'), array_shape[1])
    return rows, cols