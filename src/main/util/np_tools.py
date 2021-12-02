import numpy as np
import math
import matplotlib.pyplot as plt

def show_hist(x, xlabel="Data", ylabel="Count", cutoff_percentile=95, **kwargs): # density=False shows counts
    #see https://stackoverflow.com/a/33203848/5122790
    #Freedman–Diaconis number of bins
    x = np.array(x)
    max_val = round(np.percentile(x, cutoff_percentile)) + 1
    old_max = x.max()
    x[x >= max_val] = max_val
    q25, q75 = np.percentile(x, [25, 75])
    bin_width = 2 * (q75 - q25) * len(x) ** (-1 / 3)
    bins = round((x.max() - x.min()) / bin_width)
    bins = min(bins, (x.max() - x.min()))
    fig, ax = plt.subplots()
    ax.hist(x, bins=bins, **kwargs)
    ax.set_xlim(0, max_val)
    ax.set_xticks(list(plt.xticks()[0][:-1]) + [max_val])
    ax.set_xticklabels([str(round(i)) for i in plt.xticks()[0]][:-1] + [f"{max_val}-{old_max}"], ha="right", rotation=45)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    plt.tight_layout()
    fig.show()

def np_divide(first, second, buffersize=10000):
    """in pmi, we want to just say `df = df/expected`, however that gives a sigkill.
    so this function will do first = first / second, but split up"""
    assert first.shape == second.shape or second.shape == ()
    res = np.empty(first.shape)
    for rowsplit in range(1, math.ceil(first.shape[0]/buffersize)+1):
        from_row = (rowsplit-1)*buffersize
        to_row = rowsplit*buffersize
        for colsplit in range(1, math.ceil(first.shape[1]/buffersize)+1):
            from_col = (colsplit-1)*buffersize
            to_col = colsplit*buffersize
            divisor = second if second.shape == () else second[from_row: to_row, from_col: to_col]
            res[from_row: to_row, from_col: to_col] = first[from_row: to_row, from_col: to_col] / divisor
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