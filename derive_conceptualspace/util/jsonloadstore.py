"""https://stackoverflow.com/a/47626762/5122790"""
import re
import inspect
import json
import os
import sys
import subprocess
import warnings
from datetime import datetime
from os.path import isfile, splitext, dirname, join, isdir

import numpy as np
import pandas as pd
from sklearn.manifold import MDS, TSNE, Isomap

from derive_conceptualspace import settings
from derive_conceptualspace.settings import standardize_config_name, standardize_config_val
from misc_util.logutils import CustomIO

flatten = lambda l: [item for sublist in l for item in sublist]


########################################################################################################################
########################################################################################################################
# stuff to serialize to JSON

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
        elif isinstance(obj, (MDS, TSNE)):
            return Struct(**obj.__dict__) #let's return the dict of the MDS such that we can load it from json and its equal
        elif isinstance(obj, Isomap):
            tmp = Struct(**obj.__dict__)
            tmp.kernel_pca_ = Struct(**tmp.kernel_pca_.__dict__)
            tmp.nbrs_ = Struct(**tmp.nbrs_.__dict__)
            tmp.kernel_pca_._centerer = Struct(**tmp.kernel_pca_._centerer.__dict__)
            return tmp
        return json.JSONEncoder.default(self, obj)


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


########################################################################################################################
########################################################################################################################
# stuff to get meta-info

def get_all_info():
    info = {"git_hash": get_commithash(),
            "settings": get_settings(),
            "date": str(datetime.now()),
            "env_vars": get_envvars(),
            "startup_env_vars": settings.STARTUP_ENVVARS,
            "cmdargs": sys.argv
            }
    if isinstance(sys.stdout, CustomIO) and isinstance(sys.stderr, CustomIO):
        info["stdout"], info["stderr"] = sys.stdout.getvalue(), sys.stderr.getvalue()
    else:
        warnings.warn("Couldn't capture stdout and/or stderr as you didn't init CustomIO!")
    return info

def get_envvars():
    return {k:v for k,v in os.environ.items() if k.startswith(settings.ENV_PREFIX+"_")}

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


########################################################################################################################
########################################################################################################################
# json-dump

def prepare_dump(*args, write_meta=True, **kwargs):
    assert "cls" not in kwargs
    if write_meta:
        content = {"git_hash": get_commithash(), "settings": get_settings(), "date": str(datetime.now()),
                   "env_vars": {k:v for k,v in os.environ.items() if k.startswith(settings.ENV_PREFIX+"_")}, "cmdargs": sys.argv, "content": args[0]}
        #TODO: also stored plots, ...
    else:
        content = args[0]
    return content


def json_dump(*args, forbid_overwrite=True, **kwargs):
    content = prepare_dump(*args, **kwargs)
    kwargs = {k:v for k,v in kwargs.items() if k not in inspect.getfullargspec(prepare_dump).kwonlyargs}
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


########################################################################################################################
########################################################################################################################
# json-persister

def makelist(orig, basename):
    res = []
    for i in orig:
        if i == "this":
            i = basename
        if i not in res:
            res.append(i)
    return res

class format_dict(dict):
    def __init__(self, *args, **kwargs):
        super(format_dict, self).__init__(*args, **kwargs)
        self.used_keys = {}
    def __missing__(self, key):
        return "UNDEFINED"
    def __getitem__(self, item):
        item = standardize_config_name(item)
        res = super().__getitem__(item)
        if res != "UNDEFINED":
            self.used_keys[item] = res
        return res


class JsonPersister():
    #TODO a very important thing this should do is not done yet: Er soll von der kompletten historie an required files auch die kompletten meta-infos speichern,
    # also das was json_store mitspeichert! Das muss alles an loaded_objects hängen! Ich möchte wissen was der git-commit der jeweiligen benötigten files war!
    # - dementsprechend muss es auch klappen, dass wenn ich in extract_candidates als relevant_metainfo das model anhänge, ich auch in den postprocessed_descriptions
    #   noch sehe was das originale model war! original code:
    #     candidate_terms, meta_inf = json_load(join(base_dir, "candidate_terms.json"), return_meta=True)
    #     model = candidate_terms["model"]
    #     assert len(candidate_terms["candidate_terms"]) == len(descriptions), f"Candidate Terms: {len(candidate_terms['candidate_terms'])}, Descriptions: {len(descriptions)}"
    #     candidate_terms["candidate_terms"] = postprocess_candidates(candidate_terms, descriptions)
    #     return model, candidate_terms

    def __init__(self, in_dir, out_dir, ctx, forward_params, forward_meta_inf, incompletedirnames_to_filenames=True, dir_struct=None, strict_metainf_checking=True):
        self.incompletedirnames_to_filenames = incompletedirnames_to_filenames #means that if there is info that should be added to dirname, but not all for one level, then add to filename
        # self.forward_params = forward_params
        # self.forward_meta_inf = forward_meta_inf
        #TODO the FORWARD_META_INF here is not used - I can use it to automatically add this in the save-method if the respective keys are in the ctx.obj, such that I don't need to
        # explitly specify them when saving!
        self.dir_struct = dir_struct or []
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.ctx = ctx
        self.loaded_objects = {} #dict: {save_basename of loaded file: [object (None if loaded), filepath, [save_basename_of_file_that_needed_it_1, save_basename_of_file_that_needed_it_2,..]
        # self.loaded_relevant_params = {}
        # self.loaded_relevant_metainf = {}
        # self.default_metainf_getters = {}
        # self.strict_metainf_checking = strict_metainf_checking
        self.created_plots = {}
        # self.used_config = {}

    @property
    def loaded_influentials(self):
        loadeds = [i.get("metadata", {}).get("used_influentials", {}) for i in self.loaded_objects.values()]
        return {k:v for list_item in loadeds for (k,v) in list_item.items()}


    def dirname_vars(self, uptolevel=None):
        return re.findall(r'{(.*?)}', "".join(self.dir_struct[:uptolevel]))

    def get_subdir(self, influential_confs):
        if not (self.dir_struct and all(i for i in self.dir_struct)):
            return "", []
        di = format_dict(influential_confs)
        dirstruct = [d.format_map(di) for d in self.dir_struct] #that will have "UNDEFINED" from some point on
        fulfilled_dirs = len(dirstruct) if not (tmp := [i for i, el in enumerate(dirstruct) if "UNDEFINED" in el]) else tmp[0]
        used_keys = {standardize_config_name(i): di.used_keys[standardize_config_name(i)] for i in self.dirname_vars(fulfilled_dirs)}
        return (os.sep.join(dirstruct[:fulfilled_dirs]),
                used_keys, #what you used
                {k: v for k, v in di.used_keys.items() if k not in used_keys.keys()}, #what you should have used for filename but didn't
                {k:v for k, v in influential_confs.items() if k not in used_keys.keys()} #what you didn't use
                )

    # def get_subdir(self, relevant_metainf, ignore_params=None):
    #     if not (self.dir_struct and all(i for i in self.dir_struct)):
    #         return "", []
    #     di = format_dict({**{k:v for k,v in self.ctx.obj.items() if k not in (ignore_params or [])}, **relevant_metainf})
    #     dirstruct = [d.format_map(di) for d in self.dir_struct]
    #     fulfilled_dirs = len(dirstruct) if not (tmp := [i for i, el in enumerate(dirstruct) if "UNDEFINED" in el]) else tmp[0]
    #     used_params = [k for k in di.keys() if "{"+k+"}" in "".join(self.dir_struct[:fulfilled_dirs])] #"verbrauchte", damit die nicht mehr zum filename hinzugefügt werden müssen
    #     return os.sep.join(dirstruct[:fulfilled_dirs]), used_params


    # def get_file_by_config(self, subdir, relevant_metainf, save_basename):
    #     subdirlen = len(join(self.in_dir, subdir))+1 if str(subdir).endswith(os.sep) else len(join(self.in_dir, subdir))
    #     candidates = [join(path, name)[subdirlen:] for path, subdirs, files in
    #                   os.walk(join(self.in_dir, subdir)) for name in files if name.startswith(save_basename)]
    #     candidates = [i if not i.startswith(os.sep) else i[1:] for i in candidates]
    #     assert candidates, f"No Candidate for {save_basename}! Subdir: {subdir}"
    #     if len(candidates) == 1:
    #         return candidates
    #     elif len(candidates) > 1:
    #         if all([splitext(i)[1] == ".json" for i in candidates]):
    #             correct_cands = []
    #             for cand in candidates:
    #                 tmp = json_load(join(self.in_dir, subdir, cand))
    #                 if (all(tmp.get("relevant_metainf", {}).get(k, v) == v or v == "ANY" for k, v in {**self.loaded_relevant_metainf, **relevant_metainf}.items()) and
    #                         # all(tmp.get("relevant_params", {}).get(k) for k, v in self.loaded_relevant_params.items()) and #TODO was this necessary
    #                         all(self.ctx.obj.get(k) == tmp["relevant_params"][k] for k in set(self.forward_params) & set(tmp.get("relevant_params", {}).keys()))):
    #                     correct_cands.append(cand)
    #             return correct_cands

    def get_file_by_config(self, subdir, save_basename):
        """If no filename specified, recursively search for files whose name startswith save_basename, and then from
           the paths of the results, reverse-engineer the demanded configs of these files and then match them up"""
        correct_cands = []
        candidates = [join(path, name)[len(self.in_dir)+1:] for path, subdirs, files in os.walk(join(self.in_dir, subdir)) for name in files if name.startswith(save_basename)]
        for cand in candidates: #ich muss nicht reverse-matchen, ich kann die required confs nehmen, anwenden und schauen ob's gleich ist!
            demanded_confs = self.dirname_vars(cand.count(os.sep))
            dirstruct, _, _, nonapplied_args = self.get_subdir({standardize_config_name(i): self.ctx.get_config(i) for i in demanded_confs})
            if dirname(cand) == dirstruct:
                correct_cands.append(cand)
        if not correct_cands:
            raise FileNotFoundError(f"There is no candidate for {save_basename} with the current config!")
        assert len(correct_cands) == 1
        return correct_cands[0]


    def add_file_metas(self, file): #TODO overhaul 16.01.2022
        print()
        for k, v in file.get("loaded_files", {}).items(): #for all files that THAT file loaded
            if k in self.loaded_objects: #if that file is already loaded
                self.loaded_objects[k]["used_in"].extend(v["used_in"])  # jot down that that file was also used for THAT file
            else:
                self.loaded_objects[k] = v  #add those ones as dependency here
            # elif file["basename"] in v["used_in"]: #and if they are already a dependency (either here or in THAT file)
            #     self.loaded_objects[k]["used_in"].extend(v["used_in"])  #jot down that #TODO do I need this?!
        for k, v in file.get("used_influentials", {}).items():
            #TODO if a dependency used a value, you should add it to the context with MAXIMUM priority (and fail if already set from non-default)
            self.ctx.set_config(k, v, "dependency")

        # for k, v in file.get("loaded_files", {}).items():
        #     if k not in self.loaded_objects: self.loaded_objects[k] = v
        #     elif file["basename"] in v[2]:
        #         assert {k:v for k,v in self.loaded_objects[k][3].items()} == v[3]
        #         self.loaded_objects[k][2].extend(v[2])
        #     elif k in self.loaded_objects:
        #         self.loaded_objects[k][2].extend(v[2])
        # for k, v in file.get("relevant_params", {}).items():
        #     if k in self.loaded_relevant_params: assert self.loaded_relevant_params[k] == v
        #     else: self.loaded_relevant_params[k] = v
        # for k, v in file.get("relevant_metainf", {}).items():
        #     if k in self.loaded_relevant_metainf: assert self.loaded_relevant_metainf[k] == v
        #     else: self.loaded_relevant_metainf[k] = v
        #     if self.ctx.obj["strict_metainf_checking"]:
        #         assert k in complete_metainf, f"The file `{tmp['basename']}` required the relevant-meta-inf `{k}`, but you don't have a value for this!"
        #         assert complete_metainf[k] in [v, "ANY"], f"The file `{tmp['basename']}` required the relevant-meta-inf `{k}` to be `{v}`, but here it is `{complete_metainf[k]}`!"
        #     else:
        #         assert complete_metainf.get(k) in [None, v, "ANY"], f"The file `{tmp['basename']}` required the relevant-meta-inf `{k}` to be `{v}`, but here it is `{complete_metainf[k]}`!"
        # obj = tmp["object"] if "object" in tmp else tmp
        # obj_info = {**tmp.get("obj_info", {}), "relevant_params": tmp.get("relevant_params", {}), "relevant_metainf": tmp.get("relevant_metainf", {})}
        # assert all(self.used_config[k] == tmp["introduced_config"][k] for k in self.used_config.keys() & tmp.get("introduced_config", {}).keys())


    def check_file_metas(self, file):
        for k, v in file.get("loaded_files", {}).items():
            if file["basename"] in v["used_in"] and k in self.loaded_objects:  #ensure that the info of all currently loaded files corresponds to all of those the file used
                assert self.loaded_objects[k]["metadata"] == v["metadata"]
        for k, v in file.get("used_influentials", {}).items():
            assert self.ctx.get_config(k, silent=True) == standardize_config_val(k, v)  #check that all current influential settings are consistent with what the file had
        # for k, v in file.get("relevant_metainf", {}).items():
        #     if k in self.loaded_relevant_metainf: assert self.loaded_relevant_metainf[k] == v
        #     else: self.loaded_relevant_metainf[k] = v
        #     if self.ctx.obj["strict_metainf_checking"]:
        #         assert k in complete_metainf, f"The file `{tmp['basename']}` required the relevant-meta-inf `{k}`, but you don't have a value for this!"
        #         assert complete_metainf[k] in [v, "ANY"], f"The file `{tmp['basename']}` required the relevant-meta-inf `{k}` to be `{v}`, but here it is `{complete_metainf[k]}`!"
        #     else:
        #         assert complete_metainf.get(k) in [None, v, "ANY"], f"The file `{tmp['basename']}` required the relevant-meta-inf `{k}` to be `{v}`, but here it is `{complete_metainf[k]}`!"
        # obj = tmp["object"] if "object" in tmp else tmp
        # obj_info = {**tmp.get("obj_info", {}), "relevant_params": tmp.get("relevant_params", {}), "relevant_metainf": tmp.get("relevant_metainf", {})}
        # assert all(self.used_config[k] == tmp["introduced_config"][k] for k in self.used_config.keys() & tmp.get("introduced_config", {}).keys())
        #TODO overhaul 16.01.2022: do I need this????

    def load(self, filename, save_basename, /, loader=None, silent=False):
        filename = filename or self.get_file_by_config("", save_basename)
        if splitext(filename)[1] == ".csv":
            obj = pd.read_csv(join(self.in_dir, filename))
            full_metadata = {}
        elif splitext(filename)[1] == ".json":
            obj = json_load(join(self.in_dir, filename))
            full_metadata = {k: v for k,v in obj.items() if k not in ["object", "loaded_files"]} if "object" in obj else {}
            if not silent:
                self.add_file_metas(obj)
            self.check_file_metas(obj)
            if "object" in obj:
                obj = obj["object"]
            if loader is not None:
                obj = loader(**obj)
        if not silent:
            # for k, v in self.loaded_relevant_params.items():
            #     if k in self.ctx.obj: assert self.ctx.obj[k] == v, f"{k} is demanded to be {self.ctx.obj[k]}, but is {v} in {tmp['basename']} at {join(subdir, filename)}"
            #     #TODO better error message that tells which file is probably missing
            #     else: self.ctx.obj[k] = v
            assert save_basename not in self.loaded_objects
            self.loaded_objects[save_basename] = {"content": obj, "path": join(self.in_dir, filename), "used_in": ["this"], "metadata": full_metadata}
        return obj

    def save(self, basename, /, force_overwrite=False, ignore_confs=None, **kwargs):
        basename, ext = splitext(basename)
        filename = basename
        # if relevant_params is not None:
        #     relevant_params += self.loaded_relevant_params
        #     assert len(set(relevant_params)) == len(relevant_params)
        # else:
        #     relevant_params = [i for i in self.forward_params if i in self.ctx.obj]
        # relevant_metainf = {**self.loaded_relevant_metainf, **(relevant_metainf or {})}
        #TODO overhaul 16.01.2022: dass der sich hier loaded_relevant_metainf anschaut macht ja schon sinn!!
        used_influentials = {k: v for k, v in self.ctx.used_influential_confs().items() if k not in ignore_confs} if ignore_confs else self.ctx.used_influential_confs()
        relevant_confs = {**self.loaded_influentials, **used_influentials}
        if ignore_confs: relevant_confs = {k: v for k, v in relevant_confs.items() if k not in ignore_confs}
        subdir, _, shoulduse_infls, _ = self.get_subdir(relevant_confs)
        if self.incompletedirnames_to_filenames and shoulduse_infls:
            filename += "_"+("_".join(str(i) for i in shoulduse_infls.values()))
        loaded_files = {k: dict(path=v["path"], used_in=makelist(v["used_in"], basename), metadata=v["metadata"]) for k, v in self.loaded_objects.items()}
        # assert all(self.ctx.obj[v] == k for v, k in self.loaded_relevant_params.items()) #TODO overhaul 16.01.2022: add back?!
        os.makedirs(join(self.out_dir, subdir), exist_ok=True)
        obj = {"loaded_files": loaded_files, "used_influentials": used_influentials,  #TODO overhaul 16.01.2022: used_conf is superfluos!!
               "basename": basename, "obj_info": get_all_info(), "created_plots": self.created_plots,
               "used_config": (self.ctx.used_configs, self.ctx.toset_configs),
               "object": kwargs} #object should be last!!
        name = json_dump(obj, join(self.out_dir, subdir, filename+ext), write_meta=False, forbid_overwrite=not force_overwrite)
        new_influentials = {k: v for k, v in used_influentials.items() if k not in self.loaded_influentials}
        print(f"Saved under {name}. New Influential Config: {new_influentials}.")# Relevant Meta-Inf: {relevant_metainf}")
        return name

    def add_plot(self, title, data):
        self.created_plots[title] = json.dumps(data, cls=NumpyEncoder)

