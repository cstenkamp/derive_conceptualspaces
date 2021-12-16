"""https://stackoverflow.com/a/47626762/5122790"""
import json
import os
import sys
import subprocess
from datetime import datetime
from os.path import isfile, splitext, dirname, join, isdir
from pprint import pprint

import numpy as np
import pandas as pd
from sklearn.manifold._mds import MDS

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
        elif hasattr(obj, "json_serialize"):
            return [obj.__class__.__name__, obj.json_serialize()]
        elif isinstance(obj, MDS):
            return Struct(**obj.__dict__) #let's return the dict of the MDS such that we can load it from json and its equal
        return json.JSONEncoder.default(self, obj)


def prepare_dump(*args, write_meta=True, **kwargs):
    assert "cls" not in kwargs
    if write_meta:
        content = {"git_hash": get_commithash(), "settings": get_settings(), "date": str(datetime.now()),
                   "env_vars": {k:v for k,v in os.environ.items() if k.startswith(settings.ENV_PREFIX+"_") or k.startswith(settings.OVEWRITE_SETTINGS_PREFIX+"_")}, "cmdargs": sys.argv, "content": args[0]}
        #TODO: also captured std-out, stored plots, ...
    else:
        content = args[0]
    return content


def json_dump(*args, forbid_overwrite=True, **kwargs):
    content = prepare_dump(*args, **kwargs)
    fpath = str(args[1])
    if forbid_overwrite and isfile(fpath):
        for i in range(2, 999):
            fpath = splitext(str(args[1]))[0]+f"_{i}"+splitext(fpath)[1]
            if not isfile(fpath): break
    with open(fpath, "w") as wfile:
        json.dump(content, wfile, *args[2:], cls=NumpyEncoder, **kwargs)
    return fpath


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



class JsonPersister():
    FORWARD_SETTINGS = ["pp_components", "translate_policy", "extraction_method", "quantification_measure", "space_dims", "dcm_quant_measure"]
    META_INF = ["n_samples"]
    DIR_STRUCT = [["n_samples"], ["pp_components", "translate_policy"], ["quantification_measure"], ["mds_dimensions"]]

    #TODO a very important thing this should do is not done yet: Er soll von der kompletten historie an required files auch die kompletten meta-infos speichern,
    # also das was json_store mitspeichert! Das muss alles an loaded_objects hängen!
    # - dementsprechend muss es auch klappen, dass wenn ich in extract_candidates als relevant_metainfo das model anhänge, ich auch in den postprocessed_descriptions
    #   noch sehe was das originale model war! original code:
    #     candidate_terms, meta_inf = json_load(join(base_dir, "candidate_terms.json"), return_meta=True)
    #     model = candidate_terms["model"]
    #     assert len(candidate_terms["candidate_terms"]) == len(descriptions), f"Candidate Terms: {len(candidate_terms['candidate_terms'])}, Descriptions: {len(descriptions)}"
    #     candidate_terms["candidate_terms"] = postprocess_candidates(candidate_terms, descriptions)
    #     return model, candidate_terms

    def __init__(self, in_dir, out_dir, ctx, add_relevantparams_to_filename=True):
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.ctx = ctx
        self.loaded_objects = {} #dict: {save_basename of loaded file: [object (None if loaded), filepath, [save_basename_of_file_that_needed_it_1, save_basename_of_file_that_needed_it_2,..]
        self.loaded_relevant_params = {}
        self.loaded_relevant_metainf = {}
        self.add_relevantparams_to_filename = add_relevantparams_to_filename

    def get_subdir(self, relevant_metainf, ignore_params=None):
        dirstruct = [[self.ctx.obj.get(i) or relevant_metainf.get(i) for i in dir if i not in (ignore_params or [])] for dir in self.DIR_STRUCT]
        return os.sep.join(["_".join([str(i) for i in j if i]) for j in dirstruct if j and any(i is not None for i in j)])

    def load(self, filename, save_basename, by_config=False, relevant_metainf=None, ignore_params=None, loader=None):
        #TODO er muss noch die meta-infos mitladen und speichern!!!
        #TODO und die previous relevant params forwarden
        relevant_metainf = relevant_metainf or {}
        ignore_params = ignore_params or []
        subdir = ""
        if not filename and by_config:
            subdir = self.get_subdir(relevant_metainf, ignore_params)
            assert isdir(join(self.in_dir, subdir))
            candidates = [join(path, name)[len(join(self.in_dir, subdir))+1:] for path, subdirs, files in os.walk(join(self.in_dir, subdir)) for name in files if name.startswith(save_basename)]
            assert candidates, f"No Candidate for {save_basename}! Subdir: {subdir}"
            if len(candidates) > 1:
                if all([splitext(i)[1] == ".json" for i in candidates]):
                    correct_cands = []
                    for cand in candidates:
                        tmp = json_load(join(self.in_dir, subdir, cand))
                        if all(tmp.get("relevant_metainf", {}).get(k) == v for k, v in {**self.loaded_relevant_metainf, **relevant_metainf}.items()) and \
                          all(tmp.get("relevant_params", {}).get(k) == v for k, v in self.loaded_relevant_params.items()) and \
                          all(self.ctx.obj.get(k) == tmp["relevant_params"][k] for k in set(self.FORWARD_SETTINGS) & set(tmp.get("relevant_params", {}).keys())):
                            correct_cands.append(cand)
                    candidates = correct_cands
            assert len(candidates) == 1, "TODO: still wrong?!"
            filename = candidates[0]
        if splitext(filename)[1] == ".csv":
            obj = pd.read_csv(join(self.in_dir, subdir, filename))
        elif splitext(filename)[1] == ".json":
            tmp = json_load(join(self.in_dir, subdir, filename))
            for k, v in tmp.get("loaded_files", {}).items():
                if k not in self.loaded_objects: self.loaded_objects[k] = v
                elif tmp["basename"] in v[2]:
                    self.loaded_objects[k][2].extend(v[2])
                    #the pp_descriptions are used in candidate_terms AND in postprocess_candidates. So when pp_cands loads stuff, it needs to note that pp_descriptions were used in boht.
            for k, v in tmp.get("relevant_params", {}).items():
                if k in self.loaded_relevant_params: assert self.loaded_relevant_params[k] == v
                else: self.loaded_relevant_params[k] = v
            for k, v in tmp.get("relevant_metainf", {}).items():
                if k in self.loaded_relevant_metainf: assert self.loaded_relevant_metainf[k] == v
                else: self.loaded_relevant_metainf[k] = v
            obj = tmp["object"] if "object" in tmp else tmp
        if loader is not None:
            obj = loader(**obj)
        for k, v in self.loaded_relevant_params.items():
            if k in self.ctx.obj: assert self.ctx.obj[k] == v
            else: self.ctx.obj[k] = v
        assert save_basename not in self.loaded_objects
        self.loaded_objects[save_basename] = (obj, join(self.in_dir, subdir, filename), ["this"])
        return obj

    def save(self, basename, /, relevant_params=None, relevant_metainf=None, **kwargs):
        relevant_params = relevant_params or []
        relevant_metainf = relevant_metainf or {}
        basename, ext = splitext(basename)
        filename = basename
        loaded_files = {k:[None, v[1], [i if i != "this" else basename for i in v[2]]] for k,v in self.loaded_objects.items()}
        if self.add_relevantparams_to_filename and relevant_params:
            filename += "_" + "_".join([str(self.ctx.obj[i]) for i in relevant_params])
        relevant_metainf = {**self.loaded_relevant_metainf, **relevant_metainf}
        assert all(self.ctx.obj[v] == k for v, k in self.loaded_relevant_params.items())
        relevant_params += self.loaded_relevant_params
        if self.DIR_STRUCT and all(i for i in self.DIR_STRUCT):
            subdir = self.get_subdir(relevant_metainf)
        os.makedirs(join(self.out_dir, subdir), exist_ok=True)
        obj = {"loaded_files": loaded_files, "relevant_params": {i: self.ctx.obj[i] for i in relevant_params}, "relevant_metainf": relevant_metainf, "basename": basename, "object": kwargs}
        name = json_dump(obj, join(self.out_dir, subdir, filename+ext))
        print(f"Saved under {name}")
        return name
