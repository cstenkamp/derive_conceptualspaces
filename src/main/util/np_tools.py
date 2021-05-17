import numpy as np
import math

def np_divide(first, second, buffersize=10000):
    """in pmi, we want to just say `df = df/expected`, however that gives a sigkill.
    so this function will do first = first / second, but split up"""
    assert first.shape == second.shape
    res = np.empty(first.shape)
    for rowsplit in range(1, math.ceil(first.shape[0]/buffersize)+1):
        from_row = (rowsplit-1)*buffersize
        to_row = rowsplit*buffersize
        for colsplit in range(1, math.ceil(first.shape[1]/buffersize)+1):
            from_col = (colsplit-1)*buffersize
            to_col = colsplit*buffersize
            res[from_row: to_row, from_col: to_col] = first[from_row: to_row, from_col: to_col] / second[from_row: to_row, from_col: to_col]
    return res


def np_log(arr, buffersize=5000):
    res = np.empty(arr.shape)
    for rowsplit in range(1, math.ceil(arr.shape[0]/buffersize)+1):
        from_row = (rowsplit-1)*buffersize
        to_row = rowsplit*buffersize
        for colsplit in range(1, math.ceil(arr.shape[1]/buffersize)+1):
            from_col = (colsplit-1)*buffersize
            to_col = colsplit*buffersize
            tmp = np.log(arr[from_row: to_row, from_col: to_col])
            tmp[np.isinf(tmp)] = 0.0  # log(0) = 0
            arr[from_row: to_row, from_col: to_col] = tmp
    return arr