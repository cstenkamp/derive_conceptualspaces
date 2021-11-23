"""https://stackoverflow.com/a/47626762/5122790"""
import json
import os
import subprocess
from datetime import datetime

import numpy as np

from src.static import settings

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

def prepare_dump(*args, write_meta=True, **kwargs):
    assert "cls" not in kwargs
    if write_meta:
        content = {"git_hash": get_commithash(), "settings": get_settings(), "date": str(datetime.now()),
                   "content": args[0]}
        #TODO: also date, captured std-out, ...
    else:
        content = args[0]
    return content

def json_dump(*args, **kwargs):
    content = prepare_dump(*args, **kwargs)
    with open(args[1], "w") as wfile:
        return json.dump(content, wfile, *args[2:], cls=NumpyEncoder, **kwargs)

def json_dumps(*args, **kwargs):
    content = prepare_dump(*args, **kwargs)
    return json.dumps(content, *args[2:], cls=NumpyEncoder, **kwargs)


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

def get_commithash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("UTF-8").strip()
    except subprocess.CalledProcessError:
        if os.getenv("RUNNING_IN_DOCKER"):
            import socket
            return f"cont_commit: {os.getenv('CONTAINER_GIT_COMMIT', 'None')}, cont_hostname: {socket.gethostname()}"
    return "no_commit"
    #TODO: return something reasonable when no repo!

def get_settings():
    from types import ModuleType  # noqa: E402
    return {
        k: v
        for k, v in settings.__dict__.items()
        if (not k.startswith("_") and not callable(v) and not isinstance(v, ModuleType) and k.isupper())
    }

def json_load(*args, assert_meta=(), **kwargs):
    if isinstance(args[0], str):
        with open(args[0], "r") as rfile:
            tmp = json.load(rfile, **kwargs)
    else: #then it may be a sacred opened resource (https://sacred.readthedocs.io/en/stable/apidoc.html#sacred.Experiment.open_resource)
        tmp = json.load(args[0], **kwargs)
    if isinstance(tmp, dict) and all(i in tmp for i in ["git_hash", "settings", "content"]):
        for i in assert_meta:
            assert getattr(settings, i) == tmp["settings"][i], f"The setting {i} does not correspond to what was saved!"
        return npify_rek(tmp["content"])
    return npify_rek(tmp)