"""https://stackoverflow.com/a/47626762/5122790"""
import json
import os
import sys
import subprocess
from datetime import datetime
from os.path import isfile, splitext, dirname

import numpy as np

from derive_conceptualspace import settings


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
                   "env_vars": {k:v for k,v in os.environ.items() if k.startswith(settings.ENV_PREFIX) or k.startswith(settings.OVEWRITE_SETTINGS_PREFIX)}, "cmdargs": sys.argv, "content": args[0]}
        #TODO: also captured std-out, ...
    else:
        content = args[0]
    return content


def json_dump(*args, forbid_overwrite=True, **kwargs):
    content = prepare_dump(*args, **kwargs)
    fpath = str(args[1])
    if forbid_overwrite and isfile(fpath):
        for i in range(2, 999):
            fpath = splitext(fpath)[0]+f"_{i}"+splitext(fpath)[1]
            if not isfile(fpath): break
    with open(fpath, "w") as wfile:
        return json.dump(content, wfile, *args[2:], cls=NumpyEncoder, **kwargs)


def json_dumps(*args, **kwargs):
    content = prepare_dump(*args, **kwargs)
    return json.dumps(content, *args[2:], cls=NumpyEncoder, **kwargs)


def npify_rek(di):
    if isinstance(di, dict):
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
    elif isinstance(di, (list, set, tuple)):
        res = []
        for i in di:
            if isinstance(i, list) and len(i) == 2 and i[0] == "np.ndarray":
                res.append(np.asarray(i[1]))
            elif isinstance(i, list) and len(i) == 2 and i[0] == "Struct":
                res.append(Struct(**npify_rek(i[1])))
            elif isinstance(i, dict):
                res.append(npify_rek(i))
            else:  # TODO also lists?
                res.append(i)
        return res

def get_commithash():
    res = {}
    try:
        res["inner_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=dirname(__file__)).strip().decode("UTF-8").strip()
    except:
        pass
    if os.getenv("RUNNING_IN_DOCKER"):
        import socket
        res["cont_commit"] = os.getenv('CONTAINER_GIT_COMMIT', 'None')
        res["cont_hostname"] = socket.gethostname()
    return res


def get_settings():
    from types import ModuleType  # noqa: E402
    return {
        k: v
        for k, v in settings.__dict__.items()
        if (not k.startswith("_") and not callable(v) and not isinstance(v, ModuleType) and k.isupper() and not "PASSWORD" in k)
    }

def json_load(*args, assert_meta=(), return_meta=False, **kwargs):
    if isinstance(args[0], str):
        with open(args[0], "r") as rfile:
            tmp = json.load(rfile, **kwargs)
    else: #then it may be a sacred opened resource (https://sacred.readthedocs.io/en/stable/apidoc.html#sacred.Experiment.open_resource)
        tmp = json.load(args[0], **kwargs)
    if isinstance(tmp, dict) and all(i in tmp for i in ["git_hash", "settings", "content"]):
        for i in assert_meta:
            assert getattr(settings, i) == tmp["settings"][i], f"The setting {i} does not correspond to what was saved!"
        if return_meta:
            meta = {k:v for k,v in tmp.items() if k != "content"}
            return npify_rek(tmp["content"]), meta
        return npify_rek(tmp["content"])
    return npify_rek(tmp)