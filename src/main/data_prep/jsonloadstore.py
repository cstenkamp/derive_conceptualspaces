"""https://stackoverflow.com/a/47626762/5122790"""
import json

import numpy as np

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, Struct):
            return ["Struct", obj.__dict__]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return ["np.ndarray", obj.tolist()]
        return json.JSONEncoder.default(self, obj)

def json_dump(*args, **kwargs):
    assert "cls" not in kwargs
    with open(args[1], "w") as wfile:
        return json.dump(args[0], wfile, *args[2:], cls=NumpyEncoder, **kwargs)

def json_dumps(*args, **kwargs):
    assert "cls" not in kwargs
    return json.dumps(*args, cls=NumpyEncoder, **kwargs)


def npify_rek(di):
    res = {}
    for k, v in di.items():
        if isinstance(v, list) and len(v) == 2 and v[0] == "np.ndarray":
            res[k] = np.asarray(v[1])
        elif isinstance(v, list) and len(v) == 2 and v[0] == "Struct":
            res[k] = Struct(**npify_rek(v[1]))
        elif isinstance(v, dict):
            res[k] = npify_rek(v)
        else: #TODO also lists?
            res[k] = v
    return res

def json_load(*args, **kwargs):
    with open(args[0], "r") as rfile:
        tmp = json.load(rfile, **kwargs)
    return npify_rek(tmp)