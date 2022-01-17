import os
from os.path import join
import random

import yaml

import numpy as np
from dotenv import load_dotenv

from misc_util.logutils import CustomIO, setup_logging
from misc_util.object_wrapper import ObjectWrapper
from derive_conceptualspace.settings import ENV_PREFIX, get_setting, standardize_config, standardize_config_name, get_defaultsetting, CONF_PRIORITY
from derive_conceptualspace.util.base_changer import NDPlane
from derive_conceptualspace.util.desc_object import DescriptionList
from derive_conceptualspace.util.dtm_object import dtm_dissimmat_loader, dtm_loader
from derive_conceptualspace.load_data import dataset_specifics
from misc_util.pretty_print import pretty_print as print
from derive_conceptualspace.util.jsonloadstore import JsonPersister
from derive_conceptualspace import settings

import click

########################################################################################################################
########################################################################################################################
########################################################################################################################


class CustomContext(ObjectWrapper):
    def __init__(self, orig_ctx):
        assert isinstance(orig_ctx, (click.Context, SnakeContext))
        super(CustomContext, self).__init__(orig_ctx)
        self.toset_configs = []
        self.used_configs = {}

    @property
    def p(self):
        return self.obj["json_persister"]

    def reattach(self, ctx): #If I have this ontop of click's context, then for every subcommand call I have to reattach the click-context
        toset, used = self.toset_configs, self.used_configs
        self.wrapper_setattr("_wrapped", ctx)
        self.toset_configs, self.used_configs = toset, used
        return self

    def init_context(self, load_envfile=False, load_conffile=True): #works for both a click-Context and my custom one
        if load_envfile and os.environ.get(ENV_PREFIX+"_"+"ENV_FILE"):
            load_dotenv(os.environ.get(ENV_PREFIX+"_"+"ENV_FILE"))
        #first of all, load settings from env-vars and, if you have it by then, from config-file
        relevant_envvars = {k[len(ENV_PREFIX)+1:]: v for k, v in os.environ.items() if k.startswith(ENV_PREFIX+"_")}
        for param, val in relevant_envvars.items():
            self.set_config(param, val, "env_vars")
        if load_conffile and self.get_config("conf_file"):
            self.read_configfile()
        # setup_logging(self.get_config("log"), self.get_config("logfile")) #TODO overhaul 16.01.2022: this looks shitty AF in snakemake (but add back for click!!)
        self.obj["dataset_class"] = dataset_specifics.load_dataset_class(self.get_config("dataset"))
        if hasattr(self.obj["dataset_class"], "configs"):
            for param, val in self.obj["dataset_class"].configs.items():
                self.set_config(param, val, "dataset_class")
        CustomIO.init(self)
        self.obj["json_persister"] = setup_json_persister(self)
        set_debug(self) #TODO overhaul 16.01.2022: make this a method as well
        if self.has_config("base_dir", include_default=False) and self.has_config("dataset", include_default=False):
            os.chdir(join(self.get_config("base_dir"), self.get_config("dataset")))

    def set_config(self, key, val, source): #this is only a suggestion, it will only be finally set once it's accessed!
        key, val = standardize_config(key, val)
        if key in self.used_configs and val != self.used_configs[key]:
            raise ValueError(f"You're trying to overwrite config {key} with *r*{val}*r*, but it was already used with value *b*{self.used_configs[key]}*b*!")
        self.toset_configs.append([key, val, source])
        existing_configs = list(zip(*[i for i in self.toset_configs if i[0] == key and i[2] != "defaults"]))
        if existing_configs and len(set(existing_configs[1])) > 1:
            ordered_args = sorted(list(zip(*existing_configs[::-1][:2])), key=lambda x:CONF_PRIORITY.index(x[0]))
            ordered_args = {v:k for k,v in list({v: k for k, v in ordered_args[::-1]}.items())[::-1]}
            if "dependency" in ordered_args and ordered_args["dependency"] != ordered_args.get("force", ordered_args["dependency"]):
                raise ValueError(f"A Dependency requires {existing_configs[0][0]} to be {dict(ordered_args)['dependency']} but your other config demands {ordered_args[1][1]}")
            ordered_args = list(ordered_args.items())
            print(f"{ordered_args[1][0]} demanded config {existing_configs[0][0]} to be *r*{ordered_args[1][1]}*r*, but {ordered_args[0][0]} overwrites it to *b*{ordered_args[0][1]}*b*")

    def pre_actualcommand_ops(self):
        #TODO overhaul 16.01.2022: finalize configs?!
        self.print_important_settings()

    @property
    def important_settings(self): #TODO be able to explicitly add important settings
        #important are: * those that have "ALL_" in settings * those that are part of the dirpath
        return [k[4:] for k in settings.__dict__ if k.startswith("ALL_")] + self.obj["json_persister"].dirname_vars()

    def print_important_settings(self):
        params = {standardize_config_name(i): self.get_config(i, silent=True) for i in self.important_settings if self.has_config(i)}
        default_params = {k[len("DEFAULT_"):]:v for k,v in settings.__dict__.items() if k in ["DEFAULT_"+i for i in params.keys()]}
        print("Running with the following settings:", ", ".join([f"{k}: *{'b' if v==default_params[k] else 'r'}*{v}*{'b' if v==default_params[k] else 'r'}*" for k, v in params.items()]))

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
            # config = {k: v if not isinstance(v, list) else v[0] for k, v in config.items()}
            # TODO statt v[0], wenn v eine liste ist und wenn ein cmd-arg bzw env-var einen wert hat der damit consistent ist, nimm das arg
            for k, v in config.items():
                self.set_config(k, v, "conf_file")

    def used_influential_confs(self):
        tmp = {k: v for k, v in self.used_configs.items() if k not in settings.NON_INFLUENTIAL_CONFIGS}
        if not tmp.get("DEBUG") and "DEBUG_N_ITEMS" in tmp:
            del tmp["DEBUG_N_ITEMS"]
        return tmp

########################################################################################################################
########################################################################################################################
########################################################################################################################


def get_jsonpersister_args():
    import derive_conceptualspace.settings
    all_params = [k[4:].lower() for k in derive_conceptualspace.settings.__dict__ if k.startswith("ALL_")]
    forward_meta_inf = ["n_samples", "faster_keybert", "candidate_min_term_count"]
    dir_struct = ["debug_{debug}",
                  "{pp_components}_{translate_policy}_minwords{min_words_per_desc}",
                  "embedding_{quantification_measure}",
                  "{embed_algo}_{embed_dimensions}d",
                  "{extraction_method}_{dcm_quant_measure}"]
    return all_params, forward_meta_inf, dir_struct

def setup_json_persister(ctx):
    all_params, forward_meta_inf, dir_struct = get_jsonpersister_args()
    return JsonPersister(join(ctx.get_config("base_dir"), ctx.get_config("dataset")),
                         join(ctx.get_config("base_dir"), ctx.get_config("dataset")), ctx,
                         forward_params = all_params, forward_meta_inf = forward_meta_inf, dir_struct = dir_struct,
                         )

def set_debug(ctx):
    if ctx.get_config("DEBUG"):
        ctx.set_config("CANDIDATE_MIN_TERM_COUNT", 1, "force")
        print(f"Debug is active! #Items for Debug: {ctx.get_config('DEBUG_N_ITEMS')}")
        if ctx.get_config("RANDOM_SEED", silence_defaultwarning=True):
            print(f"Using a random seed: {ctx.get_config('RANDOM_SEED')}")
            random.seed(ctx.get_config("RANDOM_SEED"))

########################################################################################################################
########################################################################################################################
########################################################################################################################

cluster_loader = lambda **di: dict(decision_planes={k: NDPlane(np.array(v[1][0]),v[1][1]) for k, v in di["decision_planes"].items()}, metrics=di["metrics"])

class SnakeContext():
    """In the Click-CLI there is a context that gets passed, this class mocks the relevant stuff for snakemake/jupyter notebooks"""
    autoloader_di = dict(
        pp_descriptions=DescriptionList.from_json,
        dissim_mat=dtm_dissimmat_loader,
        filtered_dcm=dtm_loader,
        clusters=cluster_loader,
        languages=lambda **kwargs: kwargs["langs"],
        title_languages=lambda **kwargs: kwargs["langs"],
        raw_descriptions=None,
    )
    # ignore_params_di = dict(
    #     pp_descriptions=["quantification_measure", "embed_dimensions"],
    #     filtered_dcm=["quantification_measure", "embed_dimensions"],
    #     embedding=["extraction_method", "dcm_quant_measure"],
    # )
    #TODO obvs this shouldn't be a ignore_params but a requires_params, but that would mean having to change the loader
    #TODO das autoloader_di und ignore_params_di sind schon ne Mischung von Code und Daten, aber wohin sonst damit?

    def __init__(self, cwd, load_params=False):
        self.obj = {"cwd": cwd}
        # if load_params: #TODO overhaul 16.01.2022: obvs this
        #     all_params = {i: get_setting(i.upper(), stay_silent=True, silent=True) for i in get_jsonpersister_args()[0]}
        #     for k, v in all_params.items():
        #         self.obj[k.lower()] = v


    # def load(self, *whats, relevant_metainf=None):
    #     for what in whats:
    #         self.obj[what] = self.obj["json_persister"].load(None, what, relevant_metainf=relevant_metainf,
    #                                                          loader=self.autoloader_di.get(what, lambda **kwargs: kwargs[what]),
    #                                                          ignore_params=self.ignore_params_di.get(what))









########################################################################################################################
########################################################################################################################
########################################################################################################################

if __name__ == "__main__":
    ctx = CustomContext(SnakeContext(cwd=os.getcwd()))
    ctx.init_context(load_envfile=True, load_conffile=False)
    print()