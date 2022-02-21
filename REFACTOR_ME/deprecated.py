import os

def set_envvar(envvarname, value):
    if isinstance(value, bool):
        if value:
            os.environ[envvarname] = "1"
        else:
            os.environ[envvarname + "_FALSE"] = "1"
    else:
        os.environ[envvarname] = str(value)


def get_envvar(envvarname):
    if os.getenv(envvarname):
        tmp = os.environ[envvarname]
        if tmp.lower() == "none":
            return "none"
        elif tmp == "True":
            return True
        elif tmp == "False":
            return False
        elif tmp.isnumeric() and "DEFAULT_"+envvarname[len(ENV_PREFIX+"_"):] in globals() and isinstance(globals()["DEFAULT_"+envvarname[len(ENV_PREFIX+"_"):]], bool) and tmp in [0, 1, "0", "1"]:
            return bool(int(tmp))
        elif tmp.isnumeric():
            return int(tmp)
        elif all([i.isdecimal() or i in ".," for i in tmp]):
            return float(tmp)
        return tmp
    elif os.getenv(envvarname+"_FALSE"):
        return False
    return None



from functools import wraps
import sys

def notify_jsonpersister(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        res = fn(*args, **kwargs)
        if not kwargs.get("fordefault"):
            if hasattr(sys.stdout, "ctx") and "json_persister" in sys.stdout.ctx.obj:  # TODO getting the json_serializer this way is dirty as fuck!
                sys.stdout.ctx.obj["json_persister"].add_config(args[0].lower(), res)
        return res
    return wrapped



# @notify_jsonpersister
def get_setting(name, default_none=False, silent=False, set_env_from_default=False, stay_silent=False, fordefault=True):
    #!!! diese funktion darf NICHTS machen außer sys.stdout.ctx.get_config(name) returnen!!! alles an processing gehört in die get_config!!!
    if hasattr(sys.stdout, "ctx"):
        return sys.stdout.ctx.get_config(name)
    #TODO einige Dinge von der old version waren schon sinnvoll, zum beispiel das bescheid sagen wenn von default, gucken
    # was ich davon wieder haben möchte
    # if fordefault: #fordefault is used for click's default-values. In those situations, it it should NOT notify the json-persister!
    #     silent = True
    #     stay_silent = False
    #     set_env_from_default = False
    #     default_none = True
    #     # return "default" #("default", globals().get("DEFAULT_"+name, "NO_DEFAULT"))
    # suppress_further = True if not silent else True if stay_silent else False
    # if get_envvar(get_envvarname(name, assert_hasdefault=False)) is not None:
    #     return get_envvar(get_envvarname(name, assert_hasdefault=False)) if get_envvar(get_envvarname(name, assert_hasdefault=False)) != "none" else None
    # if "DEFAULT_"+get_envvarname(name, assert_hasdefault=False, without_prefix=True) in globals():
    #     if not silent and not get_envvar(get_envvarname(name, assert_hasdefault=False)+"_shutup"):
    #         print(f"returning setting for {name} from default value: {globals()['DEFAULT_'+name]}")
    #     if suppress_further and not get_envvar(get_envvarname(name, assert_hasdefault=False) + "_shutup"):
    #         set_envvar(get_envvarname(name, assert_hasdefault=False)+"_shutup", True)
    #     if set_env_from_default:
    #         set_envvar(get_envvarname(name, assert_hasdefault=False)+name, globals()['DEFAULT_'+name])
    #     return globals()["DEFAULT_"+name]
    # if default_none:
    #     return None
    # raise ValueError(f"There is no default-value for setting {name}, you have to explicitly pass it!")



def get_envvarname(config, assert_hasdefault=True, without_prefix=False):
    config = config.upper()
    if assert_hasdefault:
        assert "DEFAULT_"+config in globals(), f"there is no default value for {config}!"
    if without_prefix:
        return config
    return ENV_PREFIX+"_"+config

########################################################################################################################
########################################################################################################################
########################################################################################################################
#from json_persister:

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



def json_load(fname, **kwargs): #assert_meta=(), return_meta=False,
    if isinstance(fname, str):
        with open(fname, "r") as rfile:
            tmp = json.load(rfile, **kwargs)
    else: #then it may be a sacred opened resource (https://sacred.readthedocs.io/en/stable/apidoc.html#sacred.Experiment.open_resource)
        tmp = json.load(fname, **kwargs)
    # if isinstance(tmp, dict) and all(i in tmp for i in ["git_hash", "settings", "content"]):
    #     for i in assert_meta:
    #         assert getattr(settings, i) == tmp["settings"][i], f"The setting {i} does not correspond to what was saved!"
    #     if return_meta:
    #         meta = {k:v for k,v in tmp.items() if k != "content"}
    #         return npify_rek(tmp["content"]), meta
    #     return npify_rek(tmp["content"])
    return npify_rek(tmp)

########################################################################################################################
########################################################################################################################
########################################################################################################################
# calculating first quants_s and from that the SVMs in create-candidate-svm-step:

# print(f"Starting Multiprocessed with {ncpu} CPUs")
# # with WorkerPool(get_ncpu(), dcm, pgbar="Counting Terms") as pool:
# #     quants_s = pool.work(terms, lambda dcm, term: dcm.term_quants(term))
# with SkipContext() as skipped, Interruptible(terms, [[], None, None], metainf, continue_from=continue_from,
#                                              contains_mp=True, name="Counting") as iter:
#     with WorkerPool(ncpu, dcm, pgbar="Counting Terms", comqu=iter.comqu) as pool:
#         quants_s, interrupted = pool.work(iter.iterable, lambda dcm, term: dcm.term_quants(term))
#     quants_s, _, _ = iter.notify([quants_s, None, None], exception=interrupted)
#     if interrupted is not False:
#         return quants_s, None, None, metainf
# if skipped.args is not None:
#     quants_s, _, _ = skipped.args
# assert len(quants_s) == len(terms)