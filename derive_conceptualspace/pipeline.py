import os
import random

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
from misc_util.logutils import CustomIO, setup_logging
from misc_util.object_wrapper import ObjectWrapper
from misc_util.pretty_print import pretty_print as print, fmt


########################################################################################################################
########################################################################################################################
########################################################################################################################


class CustomContext(ObjectWrapper):
    def __init__(self, orig_ctx):
        assert isinstance(orig_ctx, (click.Context, SnakeContext))
        super(CustomContext, self).__init__(orig_ctx)
        self.toset_configs = []
        self.used_configs = {}
        self._initialized = False
        if hasattr(orig_ctx, "post_init"):
            type(orig_ctx).post_init(self)

    @property
    def p(self):
        return self.obj["json_persister"]

    def reattach(self, ctx): #If I have this ontop of click's context, then for every subcommand call I have to reattach the click-context
        toset, used, init = self.toset_configs, self.used_configs, self._initialized
        self.wrapper_setattr("_wrapped", ctx)
        self.toset_configs, self.used_configs, self._initialized = toset, used, init
        return self

    def set_debug(self):
        if self.get_config("DEBUG"):
            self.set_config("CANDIDATE_MIN_TERM_COUNT", 1, "force")
            print(f"Debug is active! #Items for Debug: {self.get_config('DEBUG_N_ITEMS')}")
            if self.get_config("RANDOM_SEED", silence_defaultwarning=True):
                print(f"Using a random seed: {self.get_config('RANDOM_SEED')}")
                random.seed(self.get_config("RANDOM_SEED"))

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
                    print("A config-file could be loaded, but intentionally isn't.")
            self.obj["dataset_class"] = dataset_specifics.load_dataset_class(self.get_config("dataset"))
            if hasattr(self.obj["dataset_class"], "configs"):
                for param, val in self.obj["dataset_class"].configs.items():
                    self.set_config(param, val, "dataset_class")
            CustomIO.init(self)
            self.obj["json_persister"] = JsonPersister(self, settings.DIR_STRUCT)
            self.set_debug()
            if self.has_config("base_dir", include_default=False):
                os.chdir(self.get_config("base_dir"))
            self._initialized = True

    def set_config(self, key, val, source): #this is only a suggestion, it will only be finally set once it's accessed!
        key, val = standardize_config(key, val)
        if key in self.used_configs and val != self.used_configs[key]:
            raise ValueError(fmt(f"You're trying to overwrite config {key} with *r*{val}*r*, but it was already used with value *b*{self.used_configs[key]}*b*!"))
        self.toset_configs.append([key, val, source])
        existing_configs = list(zip(*[i for i in self.toset_configs if i[0] == key and i[2] not in ["defaults", "smk_args"]]))
        if existing_configs and len(set(existing_configs[1])) > 1:
            ordered_args = sorted(list(zip(*existing_configs[::-1][:2])), key=lambda x:CONF_PRIORITY.index(x[0]))
            ordered_args = {v:k for k,v in list({v: k for k, v in ordered_args[::-1]}.items())[::-1]}
            if "dependency" in ordered_args and ordered_args["dependency"] != ordered_args.get("force", ordered_args["dependency"]):
                raise ValueError(f"A Dependency requires {existing_configs[0][0]} to be {dict(ordered_args)['dependency']} but your other config demands {ordered_args[1][1]}")
            ordered_args = list(ordered_args.items())
            print(f"{ordered_args[1][0]} demanded config {existing_configs[0][0]} to be *r*{ordered_args[1][1]}*r*, but {ordered_args[0][0]} overwrites it to *b*{ordered_args[0][1]}*b*")

    def pre_actualcommand_ops(self):
        self.print_important_settings()

    @property
    def important_settings(self): #TODO be able to explicitly add important settings
        #important are: * those that have "ALL_" in settings * those that are part of the dirpath
        return [k[4:] for k in settings.__dict__ if k.startswith("ALL_")] + self.obj["json_persister"].dirname_vars()

    def print_important_settings(self):
        params = {standardize_config_name(i): self.get_config(i, silent=True) for i in self.important_settings if self.has_config(i)}
        default_params = {k[len("DEFAULT_"):]:v for k,v in settings.__dict__.items() if k in ["DEFAULT_"+i for i in params.keys()]}
        print("Running with the following settings:", ", ".join([f"{k}: *{'b' if v==default_params.get(k) else 'r'}*{v}*{'b' if v==default_params.get(k) else 'r'}*" for k, v in params.items()]))

    def has_config(self, key, include_default=True): #if there is a click-arg with "default=None", it will be EXPLICITLY set by default, ONLY that is included if include_default
        return key in self.used_configs or bool([i for i in self.toset_configs if i[0]==standardize_config_name(key) and (include_default or i[2] != "defaults")])

    def get_config(self, key, silent=False, silence_defaultwarning=False):
        key = standardize_config_name(key)
        if key not in self.used_configs:
            conf_suggestions = [i[1:] for i in self.toset_configs if i[0] == key]
            final_conf = min([i for i in conf_suggestions], key=lambda x: CONF_PRIORITY.index(x[1])) if len(conf_suggestions) > 0 else [None, "defaults"]
            if final_conf[1] == "defaults":
                final_conf[0] = get_defaultsetting(key, silent=silence_defaultwarning)
            if silent:
                return final_conf[0]
            self.used_configs[key] = final_conf[0]
        return self.used_configs[key]

    def read_configfile(self):
        if self.get_config("conf_file"):
            with open(self.get_config("conf_file"), "r") as rfile:
                config = yaml.load(rfile, Loader=yaml.SafeLoader)
            for k, v in config.items():
                if isinstance(v, list): #IDEA: wenn v eine liste ist und wenn ein cmd-arg bzw env-var einen wert hat der damit consistent ist, nimm das arg
                    overwriters = [i[1:] for i in self.toset_configs if i[0]==standardize_config_name(k) and CONF_PRIORITY.index(i[2]) < CONF_PRIORITY.index("conf_file")]
                    if overwriters:
                        assert len(overwriters) == 1 and overwriters[0][0] in v, "TODO: do this"
                        self.set_config(k, overwriters[0][0], "conf_file")
                    else:
                        self.set_config(k, v[0], "conf_file")
                else:
                    self.set_config(k, v, "conf_file")

    def used_influential_confs(self):
        tmp = {k: v for k, v in self.used_configs.items() if k not in settings.NON_INFLUENTIAL_CONFIGS}
        if not tmp.get("DEBUG") and "DEBUG_N_ITEMS" in tmp:
            del tmp["DEBUG_N_ITEMS"]
        return tmp


########################################################################################################################
########################################################################################################################
########################################################################################################################

cluster_loader = lambda **di: dict(decision_planes={k: NDPlane(np.array(v[1][0]),v[1][1]) for k, v in di["decision_planes"].items()}, metrics=di["metrics"])

class SnakeContext():
    """In the Click-CLI there is a context that gets passed, this class mocks the relevant stuff for snakemake/jupyter notebooks"""
    def __init__(self):
        self.obj = {}

    def post_init(self):
        self.set_config("base_dir", os.getcwd(), "smk_args")

    autoloader_di = dict(
        pp_descriptions=DescriptionList.from_json,
        dissim_mat=dtm_dissimmat_loader,
        filtered_dcm=dtm_loader,
        clusters=cluster_loader,
        title_languages=lambda **kwargs: kwargs["langs"],
        languages=lambda **kwargs: kwargs["langs"],
        raw_descriptions=None,
    )
    # TODO das autoloader_di ist schon ne Mischung von Code und Daten, aber wohin sonst damit?

    @staticmethod
    def loader_context(load_envfile=True, config=None): #for jupyter
        ctx = CustomContext(SnakeContext())
        for k, v in (config or {}).items():
            ctx.set_config(k, v, "force")
        ctx.init_context(load_envfile=load_envfile)
        ctx.print_important_settings()
        return ctx

    def load(self, *whats):
        for what in whats:
            self.obj[what] = self.obj["json_persister"].load(None, what, loader=self.autoloader_di.get(what, lambda **kwargs: kwargs[what]))
