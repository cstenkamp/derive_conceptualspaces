from os.path import join

import numpy as np

from .settings import ENV_PREFIX, get_setting
import derive_conceptualspace.settings
from .util.base_changer import NDPlane
from .util.desc_object import DescriptionList
from .util.dtm_object import dtm_dissimmat_loader, dtm_loader
from .load_data import dataset_specifics
from misc_util.pretty_print import pretty_print as print

########################################################################################################################
########################################################################################################################
########################################################################################################################
from .util.jsonloadstore import JsonPersister


def print_settings():
    all_params = {i: get_setting(i.upper(), stay_silent=True, silent=True) for i in get_jsonpersister_args()[0]}
    default_params = {k[len("DEFAULT_"):].lower():v for k,v in derive_conceptualspace.settings.__dict__.items() if k in ["DEFAULT_"+i.upper() for i in all_params.keys()]}
    print("Running with the following settings:", ", ".join([f"{k}: *{'b' if v==default_params[k] else 'r'}*{v}*{'b' if v==default_params[k] else 'r'}*" for k, v in all_params.items()]))


def get_jsonpersister_args():
    import derive_conceptualspace.settings
    all_params = [k[4:].lower() for k in derive_conceptualspace.settings.__dict__ if k.startswith("ALL_")]
    forward_meta_inf = ["n_samples", "faster_keybert", "candidate_min_term_count"]
    dir_struct = ["{n_samples}_samples", "{pp_components}_{translate_policy}","{quantification_measure}_{embed_algo}_{embed_dimensions}d", "{extraction_method}_{dcm_quant_measure}"]
               # ["{n_samples}_samples", "preproc-{pp_components}_{translate_policy}", "{quantification_measure}_{embed_dimensions}dim",]
    return all_params, forward_meta_inf, dir_struct

def setup_json_persister(ctx, ignore_nsamples=False):
    all_params, forward_meta_inf, dir_struct = get_jsonpersister_args()
    json_persister = JsonPersister(join(ctx.obj["base_dir"], ctx.obj["dataset"]), join(ctx.obj["base_dir"], ctx.obj["dataset"]), ctx,
                                   forward_params = all_params, forward_meta_inf = forward_meta_inf, dir_struct = dir_struct,
                                   add_relevantparams_to_filename=ctx.obj.get("add_relevantparams_to_filename", True),
                                   strict_metainf_checking=ctx.obj["strict_metainf_checking"],
                                  )
    if ignore_nsamples:
        n_samples_getter = lambda: "ANY"
        cand_ntc_getter = lambda: "ANY"
    else:
        n_samples_getter = lambda: get_setting("DEBUG_N_ITEMS", silent=True) if get_setting("DEBUG", silent=True) else ctx.obj["dataset_class"].N_ITEMS
        cand_ntc_getter = lambda: get_setting("CANDIDATE_MIN_TERM_COUNT", silent=True)
    json_persister.default_metainf_getters = {"n_samples": n_samples_getter,
                                              "candidate_min_term_count": cand_ntc_getter,
                                              "prim_lambda": lambda: "ANY",
                                              "sec_lambda": lambda: "ANY",
                                              "max_ngram": lambda: get_setting("MAX_NGRAM", silent=True)}
    return json_persister

########################################################################################################################
########################################################################################################################
########################################################################################################################

cluster_loader = lambda **di: dict(clusters=di["clusters"], cluster_directions=di["cluster_directions"],
                                   decision_planes={k: NDPlane(np.array(v[1][0]),v[1][1]) for k, v in di["decision_planes"].items()}, metrics=di["metrics"])

class Context():
    """In the Click-CLI there is a context that gets passed, this class mocks the relevant stuff for snakemake/jupyter notebooks"""
    autoloader_di = dict(
        pp_descriptions=DescriptionList.from_json,
        dissim_mat=dtm_dissimmat_loader,
        filtered_dcm=dtm_loader,
        clusters=cluster_loader,
        languages=lambda **kwargs: kwargs["langs"],
        title_languages=lambda **kwargs: kwargs["title_langs"],
    )
    ignore_params_di = dict(
        pp_descriptions=["quantification_measure", "embed_dimensions"],
        filtered_dcm=["quantification_measure", "embed_dimensions"],
        embedding=["extraction_method", "dcm_quant_measure"],
    )
    #TODO obvs this shouldn't be a ignore_params but a requires_params, but that would mean having to change the loader
    #TODO das autoloader_di und ignore_params_di sind schon ne Mischung von Code und Daten, aber wohin sonst damit?

    def __init__(self, base_dir, dataset_name, load_params=False):
        self.obj = {"base_dir": base_dir or get_setting("BASE_DIR"),
                    "dataset": dataset_name or get_setting("DATASET_NAME"),
                    "add_relevantparams_to_filename": True,
                    "verbose": get_setting("VERBOSE"),
                    "strict_metainf_checking": get_setting("STRICT_METAINF_CHECKING")}
        self.auto_envvar_prefix = ENV_PREFIX
        self.obj["dataset_class"] = dataset_specifics.load_dataset_class(self.obj["dataset"])
        self.obj["json_persister"] = setup_json_persister(self)
        if load_params:
            all_params = {i: get_setting(i.upper(), stay_silent=True, silent=True) for i in get_jsonpersister_args()[0]}
            for k, v in all_params.items():
                self.obj[k.lower()] = v


    def load(self, *whats):
        for what in whats:
            self.obj[what] = self.obj["json_persister"].load(None, what, loader=self.autoloader_di.get(what, lambda **kwargs: kwargs[what]),
                                                             ignore_params=self.ignore_params_di.get(what))

    def print_settings(self):
        return print_settings() #TODO Once I use this very context in click, I can put the print_settings() here