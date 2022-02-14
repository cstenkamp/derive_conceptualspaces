import re
import hashlib
import os
import random
from datetime import datetime
from os.path import join, dirname, abspath, isfile
import inspect

import click
import numpy as np
import yaml
from dotenv import load_dotenv

from derive_conceptualspace import settings
from derive_conceptualspace.load_data import dataset_specifics
from derive_conceptualspace.settings import (
    ENV_PREFIX,
    standardize_config,
    standardize_config_name,
    get_defaultsetting,
    CONF_PRIORITY
)
from derive_conceptualspace.util.base_changer import NDPlane
from derive_conceptualspace.util.desc_object import DescriptionList
from derive_conceptualspace.util.dtm_object import dtm_dissimmat_loader, dtm_loader
from derive_conceptualspace.util.jsonloadstore import JsonPersister
from derive_conceptualspace.create_spaces.preprocess_descriptions import PPComponents
from derive_conceptualspace.util.misc_architecture import merge_streams
from misc_util.logutils import CustomIO
from misc_util.object_wrapper import ObjectWrapper
from misc_util.pretty_print import pretty_print as print, fmt, TRANSLATOR, isnotebook, color


########################################################################################################################
########################################################################################################################
########################################################################################################################

def apply_dotenv_vars():
    """I want to be able to have env-vars in env-files (like `MA_DATA_BASE = $HOME/data`), and even nested env-vars (like `OTHER_VAR = {MA_DATA_BASE}/config`, so
    variables that refer to other variables defined im the same file. This does that."""
    if os.getenv(f"{ENV_PREFIX}_SELECT_ENV_FILE"):
        assert isfile(os.getenv(f"{ENV_PREFIX}_SELECT_ENV_FILE"))
        load_dotenv(os.getenv(f"{ENV_PREFIX}_SELECT_ENV_FILE"))
    curr_envvars = {k: v for k, v in os.environ.items() if k.startswith(ENV_PREFIX+"_")}
    curr_envvars = {k: os.path.expandvars(v) for k, v in curr_envvars.items()} #replace envvars
    curr_envvars = {k: os.path.expandvars(os.path.expandvars(re.sub(r"{([^-\s]*?)}", r"$\1", v))) for k, v in curr_envvars.items()} #replace {ENV_VAR} with $ENV_VAR and then apply them
    if (envfile := curr_envvars.get(ENV_PREFIX+"_"+"ENV_FILE")) and not isfile(envfile) and isfile(join(curr_envvars.get(ENV_PREFIX+"_"+"CONFIGDIR", ""), envfile)):
        curr_envvars[ENV_PREFIX+"_"+"ENV_FILE"] = join(curr_envvars.get(ENV_PREFIX+"_"+"CONFIGDIR"), envfile)
    for k, v in curr_envvars.items():
        os.environ[k] = v

def load_envfiles(for_dataset=None):
    # if for_dataset is not None it will set both the env-file and the dataset-variable from it
    # TODO the dataset may be siddata2022 for env-file siddata !
    assert isfile(os.environ["MA_SELECT_ENV_FILE"])
    load_dotenv(os.environ["MA_SELECT_ENV_FILE"])
    if for_dataset is not None:
        os.environ["MA_ENV_FILE"] = f"{for_dataset}.env"
    apply_dotenv_vars()
    assert isfile(os.environ["MA_ENV_FILE"])
    if for_dataset is not None:
        os.environ["MA_DATASET"] = for_dataset #if I define it before loading the MA_ENV_FILE it can still be overwritten by if it is in there
    load_dotenv(os.environ["MA_ENV_FILE"])


class CustomContext(ObjectWrapper):
    def __init__(self, orig_ctx):
        assert isinstance(orig_ctx, (click.Context, SnakeContext))
        super(CustomContext, self).__init__(orig_ctx)
        self.toset_configs = []
        self.used_configs = {}
        self.forbidden_configs = []
        self._initialized = False
        self._given_warnings = []
        if hasattr(orig_ctx, "post_init"):
            type(orig_ctx).post_init(self)

    @property
    def p(self):
        return self.obj["json_persister"]

    def reattach(self, ctx): #If I have this ontop of click's context, then for every subcommand call I have to reattach the click-context
        restore_attrs = {k: getattr(self, k) for k in self._wrapped.__dict__.keys() - click.core.Context(self.command).__dict__.keys()}
        self.wrapper_setattr("_wrapped", ctx)
        [setattr(self, k, v) for k, v in restore_attrs.items()]
        return self

    def set_debug(self):
        if self.get_config("DEBUG"):
            self.set_config("CANDIDATE_MIN_TERM_COUNT", 2, "force")
            print(f"Debug is active! #Items for Debug: {self.get_config('DEBUG_N_ITEMS', silent=True)}")
            if self.get_config("RANDOM_SEED", silence_defaultwarning=True):
                print(f"Using a random seed: {self.get_config('RANDOM_SEED')}")
                random.seed(self.get_config("RANDOM_SEED"))
        else:
            if self.get_config("SEED_ONLY_IN_DEBUG", silent=True, silence_defaultwarning=True):
                self.set_config("RANDOM_SEED", None, "force")
            elif self.get_config("RANDOM_SEED"):
                print(f"*r*Using a random seed ({self.get_config('RANDOM_SEED')}) even though DEBUG=False!*r*")

    def init_context(self, load_envfile=False, load_conffile=True): #works for both a click-Context and my custom one
        if not self._initialized:
            #first of all, load settings from env-vars and, if you have it by then, from config-file
            if load_envfile and os.environ.get(ENV_PREFIX+"_"+"ENV_FILE"):
                load_dotenv(os.environ.get(ENV_PREFIX+"_"+"ENV_FILE"))
            relevant_envvars = {k[len(ENV_PREFIX)+1:]: v for k, v in os.environ.items() if k.startswith(ENV_PREFIX+"_")}
            for param, val in relevant_envvars.items():
                if param.startswith("CONF_FORCE_"): #that's how snakemake enforces the config-file, in that situation conf-file has higher prio than env-var
                    self.set_config(param[len("CONF_FORCE_"):], val, "smk_wildcard")
                else:
                    self.set_config(param, val, "env_vars")
            if self.get_config("conf_file"):
                if load_conffile:
                    self.read_configfile()
                else:
                    print("The env-vars contain the path to a config-file, but it intentionally isn't loaded!")
            self.obj["dataset_class"] = dataset_specifics.load_dataset_class(self.get_config("dataset"))
            if hasattr(self.obj["dataset_class"], "configs"):
                for param, val in self.obj["dataset_class"].configs.items():
                    self.set_config(param, val, "dataset_class")
            if hasattr(self.obj["dataset_class"], "init"):
                self.obj["dataset_class"].init(self)
            CustomIO.init(self)
            self.obj["json_persister"] = JsonPersister(self, settings.DIR_STRUCT)
            self.set_debug()
            if self.has_config("base_dir", include_default=False):
                os.chdir(self.get_config("base_dir"))
            self._init_time = datetime.now()
            self._initialized = True

    def forbid_config(self, confkey, check_unused=True):
        if check_unused:
            assert confkey not in self.used_configs
        print(f"The config *r*{confkey}*r* is now forbidden!")
        self.forbidden_configs.append(confkey)


    def set_config(self, key, val, source, silent=False): #this is only a suggestion, it will only be finally set once it's accessed!
        key, val = standardize_config(key, val)
        if key in self.used_configs and val != self.used_configs[key]:
            raise ValueError(fmt(f"{source} is trying to overwrite config {key} with *r*{val}*r*, but it was already used with value *b*{self.used_configs[key]}*b*!"))
        self.toset_configs.append([key, val, source])
        existing_configs = list(zip(*[i for i in self.toset_configs if i[0] == key and i[2] not in ["defaults", "smk_args"]]))
        if existing_configs and len(set(existing_configs[1])) > 1 and existing_configs[0][0] not in settings.MAY_DIFFER_IN_DEPENDENCIES:
            #TODO this has become a mess. I originally only wanted this warning for dependency, but then expanded it for force and now it's BS. Overhaul this!!
            ordered_args = sorted(list(zip(*existing_configs[::-1][:2])), key=lambda x:CONF_PRIORITY.index(re.sub(r'\[.*?\]', '', x[0])))
            ordered_args = dict(sorted({v:k for k,v in list({v: k for k, v in ordered_args[::-1]}.items())}.items(), key=lambda x:CONF_PRIORITY.index(re.sub(r'\[.*?\]', '', x[0])))) # per value only keep the highest-priority-thing that demanded it
            if "dependency" in ordered_args and ordered_args["dependency"] != ordered_args.get("force", ordered_args["dependency"]):
                raise ValueError(f"A Dependency requires {existing_configs[0][0]} to be {dict(ordered_args)['dependency']} but your other config demands {[v for k,v in ordered_args.items() if k!='dependency'][0]}")
            # if "dataset_class" in ordered_args and bool([k for k, v in ordered_args.items() if v != ordered_args["dataset_class"]]): #if something of higher prio overwrites dataset_class
            #     raise ValueError(f"dataset_class requires {existing_configs[0][0]} to be {dict(ordered_args)['dataset_class']} but it will be overwritten by {[k for k, v in ordered_args.items() if v != ordered_args['dataset_class']]}")
            ordered_args = list(ordered_args.items())
            if f"{existing_configs[0][0]} from {ordered_args[1][1]} to {ordered_args[0][1]}" not in self._given_warnings:
                self._given_warnings.append(f"{existing_configs[0][0]} from {ordered_args[1][1]} to {ordered_args[0][1]}")
                if not (silent or (hasattr(self, "silent") and self.silent)):
                    print(f"{ordered_args[1][0]} demanded config {existing_configs[0][0]} to be *r*{ordered_args[1][1]}*r*, but {ordered_args[0][0]} overwrites it to *b*{ordered_args[0][1]}*b*")


    def pre_actualcommand_ops(self, torun_fn):
        assert not any(i in self.forbidden_configs for i in set(inspect.getfullargspec(torun_fn).args)-{"ctx", "context"})
        self.torun_fn_name = torun_fn.__name__ #with this I could eg. restrict settings to be asked for only by certain functions
        self.print_important_settings()

    @property
    def important_settings(self): #TODO be able to explicitly add important settings
        #important are: * those that have "ALL_" in settings * those that are part of the dirpath
        # * those are from "force", "smk_wildcard", "cmd_args"
        res = ([k[4:] for k in settings.__dict__ if k.startswith("ALL_")] + self.obj["json_persister"].dirname_vars() +
               [i[0] for i in self.toset_configs if i[2] in ["force", "smk_wildcard", "cmd_args"]])
        if self.get_config("DEBUG", silent=True, silence_defaultwarning=True): res.append("DEBUG_N_ITEMS")
        return list({standardize_config_name(k):None for k in res}.keys()) #unique but with order

    def print_important_settings(self):
        params = {i: self.get_config(i, silent=True) for i in self.important_settings if self.has_config(i)}
        params = dict(sorted(params.items(), key=lambda x:x[0]))
        default_params = {k[len("DEFAULT_"):]:v for k,v in settings.__dict__.items() if k in ["DEFAULT_"+i for i in params.keys()]}
        print(f"Running with the following settings [{self.settingshash}]: ", ", ".join([f"{k}: *{'b' if v==default_params.get(k) else 'r'}*{v}*{'b' if v==default_params.get(k) else 'r'}*" for k, v in params.items()]))

    @property
    def settingshash(self):
        influentials = {k: self.get_config(k, silent=True, silence_defaultwarning=True) for k in self.important_settings}
        hash = hashlib.sha512(str(sorted(influentials.items())).encode("UTF-8")).hexdigest()[:10]
        colors = [i for i in TRANSLATOR if i.startswith("*") and len(i) == 3]
        color = colors[int(hash, 16) % len(colors)]
        return f"{color}{hash}{color}"


    def has_config(self, key, include_default=True): #if there is a click-arg with "default=None", it will be EXPLICITLY set by default, ONLY that is included if include_default
        return key in self.used_configs or bool([i for i in self.toset_configs if i[0]==standardize_config_name(key) and (include_default or i[2] != "defaults")])

    def get_config(self, key, silent=False, silence_defaultwarning=False, default_false=False):
        key = standardize_config_name(key)
        if key in self.forbidden_configs:
            assert False, f"Config {key} is forbidden!"
        if key not in self.used_configs:
            conf_suggestions = [i[1:] for i in self.toset_configs if i[0] == key]
            final_conf = min([i for i in conf_suggestions], key=lambda x: CONF_PRIORITY.index(re.sub(r'\[.*?\]', '', x[1]))) if len(conf_suggestions) > 0 else [None, "defaults"]
            if final_conf[1] == "defaults":
                final_conf[0] = get_defaultsetting(key, silent=silence_defaultwarning, default_false=default_false)
            if silent:
                return final_conf[0]
            self.used_configs[key] = final_conf[0]
        if key == "MAX_NGRAM" and self.used_configs[key] not in ["None", None]:
            raise Exception("ALÖKFÖSDLKFJÖLSDKJFÖLKSDJFÖLKJSD")
        return self.used_configs[key]

    def read_configfile(self):
        if self.get_config("conf_file"):
            fname = join(os.getenv(f"{ENV_PREFIX}_CONFIGDIR", dirname(settings.__file__)), self.get_config("conf_file")) if not isfile(self.get_config("conf_file")) and join(os.getenv(f"{ENV_PREFIX}_CONFIGDIR", dirname(settings.__file__)), self.get_config("conf_file")) else self.get_config("conf_file")
            with open(fname, "r") as rfile:
                config = yaml.load(rfile, Loader=yaml.SafeLoader)
            if config.get("__perdataset__"):
                if config["__perdataset__"].get(self.get_config("dataset"), {}):
                    config.update(config.get("__perdataset__", {}).get(self.get_config("dataset"), {}))
                del config["__perdataset__"]
            for k, v in config.items():
                if isinstance(v, list): #IDEA: wenn v eine liste ist und wenn ein cmd-arg bzw env-var einen wert hat der damit consistent ist, nimm das arg
                    overwriters = [i[1:] for i in self.toset_configs if i[0]==standardize_config_name(k) and CONF_PRIORITY.index(re.sub(r'\[.*?\]', '', i[2])) < CONF_PRIORITY.index("conf_file")]
                    if overwriters and len(set([i[0] for i in overwriters])) > 1:
                        # assert len(overwriters) == 1 and overwriters[0][0] in v, "TODO: do this"
                        self.set_config(k, overwriters[0][0], "conf_file")
                    else:
                        self.set_config(k, v[0], "conf_file")
                else:
                    self.set_config(k, v, "conf_file")
            if not self.silent: print(f"Config-File {fname} loaded.")

    def used_influential_confs(self):
        tmp = {k: v for k, v in self.used_configs.items() if k not in settings.NON_INFLUENTIAL_CONFIGS}
        if not tmp.get("DEBUG") and "DEBUG_N_ITEMS" in tmp:
            del tmp["DEBUG_N_ITEMS"]
        return tmp

    def display_output(self, objname, ignore_err=False):
        txt = merge_streams(self.p.loaded_objects[objname]["metadata"]["obj_info"]["stdout"],
                            self.p.loaded_objects[objname]["metadata"]["obj_info"]["stderr"] if not ignore_err else "",
                            objname)
        if isnotebook():
            txt = txt.replace("\n", "<br>")
        for k, v in TRANSLATOR.items():
            if k != "end":
                txt = txt.replace(v, k)
        while color.END in txt: # rather than replacing with end, you should replace with whatever replacement comes before this
            sign_before = max({i: txt[:txt.find(color.END)].rfind(i) for i in set(TRANSLATOR.keys())-{"end"}}.items(), key=lambda x:x[1])[0]
            txt = txt.replace(color.END, sign_before, 1)
        return txt



########################################################################################################################
########################################################################################################################
########################################################################################################################

cluster_loader = lambda **di: dict(decision_planes={k: NDPlane(np.array(v[1][0]),v[1][1]) for k, v in di["decision_planes"].items()}, metrics=di["metrics"])


class SnakeContext():
    """In the Click-CLI there is a context that gets passed, this class mocks the relevant stuff for snakemake/jupyter notebooks"""
    def __init__(self, silent=False, warn_filters=None):
        self.obj = {}
        self.silent=silent
        self.warn_filters = warn_filters or []

    def post_init(self: CustomContext):
        self.set_config("base_dir", os.getcwd(), "smk_args", silent=self.silent)

    autoloader_di = dict(
        pp_descriptions=DescriptionList.from_json,
        dissim_mat=dtm_dissimmat_loader,
        filtered_dcm=dtm_loader,
        clusters=cluster_loader,
        title_languages=lambda **kwargs: kwargs["langs"],
        languages=lambda **kwargs: kwargs["langs"],
        raw_descriptions=None,
        postprocessed_candidates=None,
    )
    # TODO das autoloader_di ist schon ne Mischung von Code und Daten, aber wohin sonst damit?

    @staticmethod
    def loader_context(load_envfile=True, config=None, load_conffile=True, **kwargs): #for jupyter
        ctx = CustomContext(SnakeContext(**kwargs))
        for k, v in (config or {}).items():
            ctx.set_config(k, v, "force")
        ctx.init_context(load_envfile=load_envfile, load_conffile=load_conffile)
        if not kwargs.get("silent"):
            ctx.print_important_settings()
        return ctx

    def load(self, *whats, loaders=None):
        for what in whats:
            loader = (loaders or {}).get(what) or self.autoloader_di.get(what) or (lambda **kwargs: kwargs[what])
            self.obj[what] = self.obj["json_persister"].load(None, what, loader=loader)
        return tuple(self.obj[what] for what in whats) if len(whats) > 1 else self.obj[whats[0]]



########################################################################################################################
########################################################################################################################
########################################################################################################################

def load_lang_translate_files(ctx, json_persister, pp_components):
    pp_components = PPComponents.from_str(pp_components)
    description_languages = json_persister.load(None, "description_languages", loader=lambda langs: langs)
    title_languages = json_persister.load(None, "title_languages", loader=lambda langs: langs) if pp_components.add_title else None
    subtitle_languages = json_persister.load(None, "subtitle_languages", loader=lambda langs: langs) if pp_components.add_subtitle else None
    if ctx.get_config("translate_policy") == "translate":
        description_translations = json_persister.load(None, "description_translations", loader=lambda **kw: kw["translations"])
        title_translations = json_persister.load(None, "title_translations", loader=lambda **kw: kw["translations"]) if pp_components.add_title else None
        subtitle_translations = json_persister.load(None, "subtitle_translations", loader=lambda **kw: kw["translations"]) if pp_components.add_subtitle else None
    else:
        description_translations = title_translations = subtitle_translations = None
    languages = dict(description=description_languages, title=title_languages, subtitle=subtitle_languages)
    translations = dict(description=description_translations, title=title_translations, subtitle=subtitle_translations)
    return languages, translations