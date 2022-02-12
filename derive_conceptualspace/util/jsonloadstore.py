"""https://stackoverflow.com/a/47626762/5122790"""
import inspect
import json
import os
import re
import subprocess
import sys
import warnings
from collections import ChainMap
from datetime import datetime, timedelta
from os.path import isfile, splitext, dirname, join, basename

import ijson
import numpy as np
import pandas as pd
from sklearn.manifold import MDS, TSNE, Isomap
from parse import parse
from decimal import Decimal

from derive_conceptualspace import settings
from derive_conceptualspace.settings import standardize_config_name, standardize_config_val
from misc_util.logutils import CustomIO
from misc_util.pretty_print import pretty_print as print, fmt

flatten = lambda l: [item for sublist in l for item in sublist]


class DifferentFileWarning(Warning):
    pass

class ConfigError(Exception):
    pass

class DependencyError(ConfigError):
    pass

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
        elif isinstance(obj, (np.floating, Decimal)):
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
            "default_settings": get_defaults(),
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


def get_defaults():
    from types import ModuleType  # noqa: E402
    return {
        k: v
        for k, v in settings.__dict__.items()
        if (not k.startswith("_") and not callable(v) and not isinstance(v, ModuleType) and k.isupper() and not "PASSWORD" in k)
    }


########################################################################################################################
########################################################################################################################
# json-dump

def json_dump(content, orig_fpath, *args, forbid_overwrite=True, **kwargs):
    fpath = str(orig_fpath)
    if forbid_overwrite and isfile(fpath):
        for i in range(2, 999):
            fpath = splitext(str(orig_fpath))[0]+f"_{i}"+splitext(fpath)[1]
            if not isfile(fpath): break
    with open(fpath, "w") as wfile:
        json.dump(content, wfile, *args, cls=NumpyEncoder, **kwargs)
    return fpath


def json_load(fname, **kwargs): #assert_meta=(), return_meta=False,
    if isinstance(fname, str):
        with open(fname, "r") as rfile:
            tmp = json.load(rfile, **kwargs)
    else: #then it may be a sacred opened resource (https://sacred.readthedocs.io/en/stable/apidoc.html#sacred.Experiment.open_resource)
        tmp = json.load(fname, **kwargs)
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
    # TODO es muss auch klappen, dass wenn ich in extract_candidates als relevant_metainfo das model anhänge, ich auch in den postprocessed_descriptions
    #   noch sehe was das originale model war! original code:
    #     candidate_terms, meta_inf = json_load(join(base_dir, "candidate_terms.json"), return_meta=True)
    #     model = candidate_terms["model"]
    #     assert len(candidate_terms["candidate_terms"]) == len(descriptions), f"Candidate Terms: {len(candidate_terms['candidate_terms'])}, Descriptions: {len(descriptions)}"
    #     candidate_terms["candidate_terms"] = postprocess_candidates(candidate_terms, descriptions)
    #     return model, candidate_terms

    # TODO the FORWARD_META_INF here is not used - I can use it to automatically add this in the save-method if the respective keys are in the ctx.obj, such that I don't need to explitly specify them when saving!
    def __init__(self, ctx, dir_struct, /, in_dir=None, out_dir=None, forward_meta_inf=None, incompletedirnames_to_filenames=True): #TODO overhaul 16.01.2022: add back meta-inf-stuff
        self.ctx = ctx
        self.in_dir = in_dir or ctx.get_config("base_dir")
        if not self.in_dir.endswith(os.sep): self.in_dir+=os.sep
        self.out_dir = out_dir or ctx.get_config("base_dir")
        if not self.out_dir.endswith(os.sep): self.out_dir+=os.sep
        self.dir_struct = dir_struct or []
        self.incompletedirnames_to_filenames = incompletedirnames_to_filenames #means that if there is info that should be added to dirname, but not all for one level, then add to filename
        self.loaded_objects = {} # dict: {save_basename of loaded file: [object (None if loaded), filepath, [save_basename_of_file_that_needed_it_1, save_basename_of_file_that_needed_it_2,..]
        self.created_plots = {}
        # self.forward_meta_inf = forward_meta_inf
        # self.loaded_relevant_metainf = {}

    @property
    def loaded_influentials(self):
        loadeds = [i.get("metadata", {}).get("used_influentials", {}) for i in self.loaded_objects.values()]
        return {k:v for list_item in loadeds for (k,v) in list_item.items()}

    def dirname_vars(self, uptolevel=None):
        return re.findall(r'{(.*?)}', "".join(self.dir_struct[:uptolevel]))

    def get_subdir(self, influential_confs):
        if not (self.dir_struct and all(i for i in self.dir_struct)):
            return "", [], {}, {}
        di = format_dict(influential_confs)
        dirstruct = [d.format_map(di) for d in self.dir_struct] #that will have "UNDEFINED" from some point on
        fulfilled_dirs = len(dirstruct) if not (tmp := [i for i, el in enumerate(dirstruct) if "UNDEFINED" in el]) else tmp[0]
        used_keys = {standardize_config_name(i): di.used_keys[standardize_config_name(i)] for i in self.dirname_vars(fulfilled_dirs)}
        return (os.sep.join(dirstruct[:fulfilled_dirs]),
                used_keys, #what you used
                {k: v for k, v in di.used_keys.items() if k not in used_keys.keys()},    #what you should have used for filename but didn't
                {k:v for k, v in influential_confs.items() if k not in used_keys.keys()} #what you didn't use
                )

    def get_file_config(self, filepath):
        if not isfile(filepath) and isfile(join(self.in_dir, filepath)):
            filepath = join(self.in_dir, filepath)
        with open(filepath) as rfile:
            used_conf = next(ijson.items(rfile, "used_influentials")) #next(ijson.items(rfile, "used_config"))[0]
            rfile.seek(0)
            used_files = next(ijson.items(rfile, "loaded_files"))
        used_conf = {k: v for k, v in used_conf.items() if k not in set(settings.MAY_DIFFER_IN_DEPENDENCIES)-set(standardize_config_name(i) for i in self.dirname_vars())}
        all_used_conf = dict(ChainMap(*(i["metadata"].get("used_influentials", {}) for i in used_files.values()), used_conf))
        return all_used_conf



    def get_file_by_config(self, subdir, save_basename, postfix=None):
        """If no filename specified, recursively search for files whose name startswith save_basename, and then for any candidates,
           check if there is a file with their config-keys and the config-values of this instance."""
        candidates = [join(path, name)[len(self.in_dir):] for path, subdirs, files in os.walk(join(self.in_dir, subdir)) for name in files if name.startswith(save_basename)]
        if postfix:
            candidates = [i for i in candidates if splitext(i)[0].endswith(postfix)]
        if not candidates: #TODO try best to get the required conf to tell in the exception
            raise FileNotFoundError(fmt(f"There is no candidate for {save_basename} with the current config."))
        correct_cands = set()
        for cand in candidates: #from the bad candidates I can even figure out the good ones
            demanded_config = {k: self.ctx.get_config(k, silent=True, default_false=True) for k in self.get_file_config(cand).keys()}
            correct_cands.add(os.sep.join(self.get_filepath(demanded_config, save_basename)))
            if self.ctx.get_config("DEBUG", silent=True): #if NOW debug=True, you may still load stuff for which debug=False
                correct_cands.add(os.sep.join(self.get_filepath({**demanded_config, "DEBUG": False}, save_basename)))
        correct_cands = flatten([[join(self.in_dir, dirname(i), j) for j in os.listdir(join(self.in_dir, dirname(i))) if j.startswith(basename(i))] for i in correct_cands])
        if postfix:
            correct_cands = [i for i in correct_cands if splitext(i)[0].endswith(postfix)]
        if len(correct_cands) > 1: # if there are two files that are equal except `self.get_file_config(i).get("DEBUG")`, take the one from `self.ctx.get_config("DEP_PREFERS_NONDEBUG", silent=True)`
            by_dirnamevars = [(','.join(sorted([k for k, v in self.get_file_config(i).items() if k in [standardize_config_name(i) for i in self.dirname_vars()]])), (i, self.get_file_config(i).get("DEBUG"))) for i in correct_cands]
            correct_cands = [{i[1][1]: i[1][0] for i in by_dirnamevars if i[0] == k}[not self.ctx.get_config("DEP_PREFERS_NONDEBUG", silent=True)] for k in set(j[0] for j in by_dirnamevars) if len([i[1] for i in by_dirnamevars if i[0] == k]) == 2]
        elif not correct_cands:
            possible_file = os.sep.join(self.get_filepath(demanded_config, save_basename))+".json"
            #command is then `(export $(cat $MA_SELECT_ENV_FILE | xargs) && PYTHONPATH=$(realpath .):$PYTHONPATH snakemake --cores 1 -p --directory $MA_DATA_DIR filepath)`
            raise FileNotFoundError(fmt(f"There is no candidate for {save_basename} with the current config. You may need the file *b*{possible_file}*b*"))
        assert len(correct_cands) == 1, f"Multiple file candidates: {', '.join(i.replace(self.in_dir, '') for i in correct_cands)}"
        return correct_cands[0]


        # demanded_confs = self.dirname_vars(cand.count(os.sep))
        # dirstruct = self.get_subdir({standardize_config_name(i): self.ctx.get_config(i, silent=True) for i in demanded_confs})[0]
        # if dirname(cand) == dirstruct:
        #     correct_cands.append((cand, parse(os.sep.join(settings.DIR_STRUCT[:cand.count(os.sep)]), dirname(cand)).named)) #2nd elem ist der Versuch die configs draus zu parsen
        # if self.ctx.get_config("DEBUG", silent=True): #if NOW debug=True, you may still load stuff for which debug=False (TODO: settings.DEPENDENCIES_PREFER_NODEBUG)
        #     if dirname(cand) == self.get_subdir({**{standardize_config_name(i): self.ctx.get_config(i, silent=True) for i in demanded_confs}, "DEBUG": False})[0] and cand not in correct_cands:
        #         correct_cands.append((cand, parse(os.sep.join(settings.DIR_STRUCT[:cand.count(os.sep)]), dirname(cand)).named))
        # if self.ctx.get_config("DEBUG", silent=True):
        #     if self.ctx.get_config("DEP_PREFERS_NONDEBUG", silent=True):
        #         correct_cands = [i for i in correct_cands if i[1] != {k: v if standardize_config_name(k) != "DEBUG" else "True" for k, v in correct_cands[-1][1].items()}]
        #     elif len(set(i[0] for i in correct_cands if {k:v for k,v in i[1].items() if standardize_config_name(k) != "DEBUG"})) > 1:
        #         correct_cands = [i for i in correct_cands if i[1] != {k: v if standardize_config_name(k) != "DEBUG" else "False" for k, v in correct_cands[-1][1].items()}]
        # if len(correct_cands) > 1 or strict_checking or True: #if there is still more than 1 candidate, partially load them to extract configs
        #     for cand in correct_cands:
        #         with open(join(self.in_dir, cand[0])) as rfile:
        #             if strict_checking:
        #                 full_conf = [k for k in ijson.items(rfile, "used_config")][0][0]
        #             else:
        #                 full_conf = dict([k for k in ijson.kvitems(rfile, "used_influentials")])
        #             diff_confs = [k for k, v in full_conf.items() if self.ctx.has_config(k) and self.ctx.get_config(k, silent=True) != v and k not in settings.MAY_DIFFER_IN_DEPENDENCIES]
        #             if diff_confs:
        #                 correct_cands = [i for i in correct_cands if i != cand]
        # correct_cands = list(set(i[0] for i in correct_cands))



    def add_file_metas(self, file): #TODO overhaul 16.01.2022
        for k, v in file.get("loaded_files", {}).items(): #for all files that THAT file loaded
            if k in self.loaded_objects: #if that file is already loaded
                self.loaded_objects[k]["used_in"].extend(v["used_in"])  # jot down that that file was also used for THAT file
            else:
                self.loaded_objects[k] = v  #add those ones as dependency here
            for k2, v2 in v.get("metadata", {}).get("used_influentials", {}).items():
                if not (self.ctx.has_config(k2) and k2 in settings.MAY_DIFFER_IN_DEPENDENCIES):
                    self.ctx.set_config(k2, v2, "dependency")
            # elif file["basename"] in v["used_in"]: #and if they are already a dependency (either here or in THAT file)
            #     self.loaded_objects[k]["used_in"].extend(v["used_in"])  #jot down that #TODO do I need this?!
        for k, v in file.get("used_influentials", {}).items():
            if not (self.ctx.has_config(k) and k in settings.MAY_DIFFER_IN_DEPENDENCIES):
                self.ctx.set_config(k, v, "dependency") #if a dependency used a value, that's maximum priority (and fail if already used)
        for cnf in file.get("forbidden_config", []):
            self.ctx.forbid_config(cnf)

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


    def check_file_metas(self, file, required_metainf=None):
        with warnings.catch_warnings():
            for warning in self.ctx.warn_filters if hasattr(self.ctx, "warn_filters") else []:
                warnings.filterwarnings("ignore", category=globals()[warning])

            if required_metainf is not None:
                if isinstance(required_metainf, list):
                    assert all(i in file["metainf"] for i in required_metainf)
                else:
                    raise NotImplementedError("TODO: custom checking of metainf if it's a dict!")
            for k, v in file.get("loaded_files", {}).items():
                for k2, v2 in v.get("metadata", {}).get("used_influentials", {}).items():
                    if k2 not in settings.MAY_DIFFER_IN_DEPENDENCIES and self.ctx.has_config(k2, include_default=False) and self.ctx.get_config(k2, silent=True) != standardize_config_val(k2, v2):
                        raise ValueError(f"config {k2} is supposed to be {self.ctx.get_config(k2, silent=True, silence_defaultwarning=True)} but dependency {k} requires it to be {v2}")
                        #TODO instead of the valuerror, print (and optionally directly perform) the commands to create it with these configs instead
                if file["basename"] in v["used_in"] and k in self.loaded_objects:  #ensure that the info of all currently loaded files corresponds to all of those the file used
                    if self.loaded_objects[k]["path"] != v["path"] or self.loaded_objects[k]["metadata"].get("obj_info") != v["metadata"].get("obj_info"):
                        warnings.warn(f"A different {k} was used for file {file['basename']}!", DifferentFileWarning)
                    if self.loaded_objects[k]["metadata"].get("used_config") != v["metadata"].get("used_config"):
                        diff_configs = {k2 for k2 in self.loaded_objects[k]["metadata"]["used_config"][0].keys()|v["metadata"]["used_config"][0].keys() if self.loaded_objects[k]["metadata"]["used_config"][0].get(k2) != v["metadata"]["used_config"][0].get(k2)}
                        if not diff_configs <= set(settings.MAY_DIFFER_IN_DEPENDENCIES):
                            raise DependencyError(f"Different settings were used in a dependency: {diff_configs - set(settings.MAY_DIFFER_IN_DEPENDENCIES)}")
            for k, v in file.get("used_influentials", {}).items():
                if k not in settings.MAY_DIFFER_IN_DEPENDENCIES:
                    if self.ctx.has_config(k): #check that all current influential settings are consistent with what the file had
                        if self.ctx.get_config(k, silent=True, silence_defaultwarning=True, default_false=True) != standardize_config_val(k, v):
                            raise DependencyError(f"Different settings were used in a dependency: {k} is {self.ctx.get_config(k)} here and {v} in a dependency!")
                    else:
                        self.ctx.set_config(k, v, "dependency") #ensure that IF we WOULD use this config, it must be equal to what it was in the dependency
                elif self.ctx.get_config(k, silent=True) != standardize_config_val(k, v):
                    print(f"The setting {k} was *r*{v}*r* in a dependency and is *b*{self.ctx.get_config(k, silent=True)}*b* here!")

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

    def load(self, filename, save_basename, /, loader=None, silent=False, required_metainf=None, return_metainf=False):
        filename = filename or self.get_file_by_config("", save_basename)
        if not isfile(join(self.in_dir, filename)) and isfile(join(self.in_dir, self.ctx.get_config("dataset"), filename)):
            filename = join(self.ctx.get_config("dataset"), filename)
        if splitext(filename)[1] == ".csv":
            obj = pd.read_csv(join(self.in_dir, filename))
            full_metadata = {}
        elif splitext(filename)[1] == ".json":
            orig = obj = json_load(join(self.in_dir, filename))
            full_metadata = {k: v for k,v in obj.items() if k not in ["object", "loaded_files"]} if "object" in obj else {}
            self.check_file_metas(obj, required_metainf) #must be first check than add, otherwise a dependency can just overwrite demanded params
            if not silent:
                self.add_file_metas(obj)
            if "object" in obj:
                obj = obj["object"]
            if loader is not None:
                if isinstance(obj, dict) and ((len(obj) == len(inspect.getfullargspec(loader).args) and all(i in obj for i in inspect.getfullargspec(loader).args)) or (len(inspect.getfullargspec(loader).args) == 0)):
                    obj = loader(**obj)
                else:
                    print(f"Demanded loader for {save_basename} doesn't apply - Trying to load without it.")
        if not silent:
            # for k, v in self.loaded_relevant_params.items():
            #     if k in self.ctx.obj: assert self.ctx.obj[k] == v, f"{k} is demanded to be {self.ctx.obj[k]}, but is {v} in {tmp['basename']} at {join(subdir, filename)}"
            #     #TODO better error message that tells which file is probably missing
            #     else: self.ctx.obj[k] = v
            assert save_basename not in self.loaded_objects
            self.loaded_objects[save_basename] = {"content": obj, "path": join(self.in_dir, filename), "used_in": ["this"], "metadata": full_metadata}
        if return_metainf:
            return obj, orig["metainf"]
        return obj

    def get_filepath(self, relevant_confs, filename):
        subdir, _, shoulduse_infls, _ = self.get_subdir(relevant_confs)
        if self.incompletedirnames_to_filenames and shoulduse_infls:
            filename += "_"+("_".join(str(i) for i in shoulduse_infls.values()))
        return subdir, filename

    def save(self, basename, /, force_overwrite=False, ignore_confs=None, metainf=None, overwrite_old=False, **kwargs):
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
        subdir, filename = self.get_filepath(relevant_confs, filename)
        loaded_files = {k: dict(path=v["path"], used_in=makelist(v["used_in"], basename), metadata=v["metadata"]) for k, v in self.loaded_objects.items()}
        # assert all(self.ctx.obj[v] == k for v, k in self.loaded_relevant_params.items()) #TODO overhaul 16.01.2022: add back?!
        os.makedirs(join(self.out_dir, subdir), exist_ok=True)
        runtime = int((datetime.now()-self.ctx._init_time).total_seconds()) #restore as string: str(timedelta(seconds=runtime))
        obj = {"loaded_files": loaded_files, "used_influentials": used_influentials,
               "basename": basename, "obj_info": get_all_info(), "created_plots": self.created_plots,
               "used_config": (self.ctx.used_configs, self.ctx.toset_configs), "metainf": metainf, "runtime": runtime,
               "forbidden_configs": self.ctx.forbidden_configs}
        if isinstance(metainf, dict) and "NEWLY_INTERRUPTED" in metainf:
            filename += "_INTERRUPTED"
            oldname = filename
        if bool(overwrite_old):
            if "NEWLY_INTERRUPTED" not in metainf:
                oldname = filename + "_INTERRUPTED"
                metainf["NOW_FINISHED"] = True
            obj = self.check_add_interrupted_overwrite_metas(obj, join(self.out_dir, subdir, oldname + ext), metainf["N_RUNS"]-1)
            os.remove(join(self.out_dir, subdir, oldname + ext))
            obj["PREV_RUN_INFO"][str(metainf["N_RUNS"]-1)]["metainf"] = overwrite_old #overwrite_old is old_metainf
        obj["object"] = kwargs #object should be last for ijson-loading!
        name = json_dump(obj, join(self.out_dir, subdir, filename+ext), forbid_overwrite=not force_overwrite)
        new_influentials = {k: v for k, v in used_influentials.items() if k not in self.loaded_influentials}
        print((f"Saved under {name}. \n"
               f"  Took {timedelta(seconds=runtime)}. \n"
               + ((f"  New Influential Config: {new_influentials}. \n") if new_influentials else "")
               + ((f"  Saved Meta-Inf: {metainf}. \n") if metainf else "")
               + ((f"  New forbidden Config: {', '.join(self.ctx.forbidden_configs)}. \n") if self.ctx.forbidden_configs else "")
              ).strip())
        #TODO telegram-send this string as well if telegram-functionality enabled
        return name

    def add_plot(self, title, data):
        self.created_plots[title] = json.dumps(data, cls=NumpyEncoder)

    def collected_metainf(self):
        per_file = {k: v.get("metadata", {}).get("metainf", {}) for k, v in self.loaded_objects.items() if v.get("metadata", {}).get("metainf")}
        tmp = {}
        [tmp.update(i) for i in per_file.values()]
        return tmp


    def check_add_interrupted_overwrite_metas(self, obj, filepath, last_run_nr):
        assert isfile(filepath)
        obj["PREV_RUN_INFO"] = {}
        for equalkey in ["loaded_files", "used_influentials", "used_config", "forbidden_configs", "obj_info", "created_plots", "runtime", "PREV_RUN_INFO"]:
            with open(filepath, "rb") as rfile:
                try:
                    if equalkey in ["used_config", "runtime"]:
                        loaded = [i for i in ijson.items(rfile, equalkey)][0]
                    else:
                        loaded = {k: tuple(v) if isinstance(v, list) else v for k,v in ijson.kvitems(rfile, equalkey)}
                except ijson.common.IncompleteJSONError:
                    with open(filepath, "r") as rfile:
                        loaded = json.load(rfile).get(equalkey, {}) #TODO having to do this sucks.
            if equalkey == "loaded_files":
                assert all(tuple((k2, v2) for k2, v2 in obj[equalkey][k].items() if k2 != "metadata") == tuple((k2, v2) for k2, v2 in loaded[k].items() if k2 != "metadata") for k in obj[equalkey].keys())
            elif equalkey == "used_config":
                assert all(loaded[0][k]==obj[equalkey][0][k] or tuple(loaded[0][k])==tuple(obj[equalkey][0][k]) for k in (loaded[0].keys()|set(obj[equalkey][0].keys()))-set(settings.NON_INFLUENTIAL_CONFIGS))
            elif equalkey in ["obj_info", "created_plots", "runtime"]:
                obj["PREV_RUN_INFO"].setdefault(str(last_run_nr), {})[equalkey] = loaded
            elif equalkey == "PREV_RUN_INFO":
                if len(loaded) > 0:
                    assert not any(i in obj["PREV_RUN_INFO"] for i in loaded.keys())
                    obj["PREV_RUN_INFO"].update(loaded)
            # elif equalkey == "used_influentials":
            #     assert all(obj[equalkey][k]==loaded[k] for k in obj[equalkey].keys() & loaded.keys())
            else:
                assert (not obj[equalkey] and not loaded) or obj[equalkey] == loaded
        return obj