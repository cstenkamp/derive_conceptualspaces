import numpy as np
import math
import gc

from tqdm import tqdm

def np_divide(first, second, buffersize=5000):
    """in pmi, we want to just say `df = df/expected`, however that gives a sigkill.
    so this function will do first = first / second, but split up"""
    assert first.shape == second.shape or second.shape == ()
    res = np.empty(first.shape)
    for rowsplit in tqdm(range(1, math.ceil(first.shape[0]/buffersize)+1), desc=f"NP-Dividing for {first.shape}"):
        from_row = (rowsplit-1)*buffersize
        to_row = rowsplit*buffersize
        for colsplit in range(1, math.ceil(first.shape[1]/buffersize)+1):
            from_col = (colsplit-1)*buffersize
            to_col = colsplit*buffersize
            divisor = second if second.shape == () else second[from_row: to_row, from_col: to_col]
            res[from_row: to_row, from_col: to_col] = first[from_row: to_row, from_col: to_col] / divisor
            del divisor
            gc.collect()
    return res


def np_log(arr, buffersize=2000):
    res = np.empty(arr.shape)
    for rowsplit in tqdm(range(1, math.ceil(arr.shape[0]/buffersize)+1), desc=f"NP-Log for {arr.shape}"):
        from_row = (rowsplit-1)*buffersize
        to_row = rowsplit*buffersize
        for colsplit in range(1, math.ceil(arr.shape[1]/buffersize)+1):
            from_col = (colsplit-1)*buffersize
            to_col = colsplit*buffersize
            tmp = np.log(arr[from_row: to_row, from_col: to_col])
            tmp[np.isinf(tmp)] = 0.0  # log(0) = 0
            res[from_row: to_row, from_col: to_col] = tmp
            del tmp
            gc.collect()
    return res