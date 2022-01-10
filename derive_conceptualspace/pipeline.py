from os.path import join

import numpy as np

from misc_util.logutils import CustomIO
from .settings import ENV_PREFIX, get_setting, NORMALIFY_PARAMS
from derive_conceptualspace import settings
from .util.base_changer import NDPlane
from .util.desc_object import DescriptionList
from .util.dtm_object import dtm_dissimmat_loader, dtm_loader
from .load_data import dataset_specifics
from misc_util.pretty_print import pretty_print as print
from .util.jsonloadstore import JsonPersister

########################################################################################################################
########################################################################################################################
########################################################################################################################







def set_debug(ctx, use_auto_envvar_prefix=False):
    env_prefix = ctx.auto_envvar_prefix if use_auto_envvar_prefix else ENV_PREFIX
    if get_setting("DEBUG"):
        if not os.getenv(env_prefix+"_DEBUG_SET"):
            assert env_prefix+"_CANDIDATE_MIN_TERM_COUNT" not in os.environ
            os.environ[env_prefix + "_CANDIDATE_MIN_TERM_COUNT"] = "1"
            print(f"Debug is active! #Items for Debug: {get_setting('DEBUG_N_ITEMS')}")
            if get_setting("RANDOM_SEED", default_none=True): print("Using a random seed!")
        if get_setting("RANDOM_SEED", default_none=True):
            random.seed(get_setting("RANDOM_SEED"))
        assert os.environ[env_prefix + "_CANDIDATE_MIN_TERM_COUNT"] == "1"
        os.environ[env_prefix+"_DEBUG_SET"] = "1"




########################################################################################################################
########################################################################################################################
########################################################################################################################


def setup_json_persister(ctx):
    json_persister = JsonPersister(join(ctx.obj["base_dir"], ctx.obj["dataset"]), join(ctx.obj["base_dir"], ctx.obj["dataset"]),
                                   ctx, settings.FNAME_PARAMS, settings.ASSERT_PARAMS, settings.DIR_STRUCT,
                                   add_relevantparams_to_filename = ctx.obj.get("add_relevantparams_to_filename", True),
                                   strict_metainf_checking=ctx.obj["strict_metainf_checking"],
                                  )
    # if ignore_nsamples:
    #     n_samples_getter = lambda: "ANY"
    #     cand_ntc_getter = lambda: "ANY"
    # else:
    #     n_samples_getter = lambda: get_setting("DEBUG_N_ITEMS", silent=True) if get_setting("DEBUG", silent=True) else "full"
    #     cand_ntc_getter = lambda: get_setting("CANDIDATE_MIN_TERM_COUNT", silent=True)
    # json_persister.default_metainf_getters = {"n_samples": n_samples_getter,
    #                                           "candidate_min_term_count": cand_ntc_getter}
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
        raw_descriptions=None,
    )
    ignore_params_di = dict(
        pp_descriptions=["quantification_measure", "embed_dimensions"],
        filtered_dcm=["quantification_measure", "embed_dimensions"],
        embedding=["extraction_method", "dcm_quant_measure"],
    )
    # relevant_params = dict(
    #     pp_descriptions = ["pp_components", "translate_policy"],
    #     dissim_mat = ["pp_components", "translate_policy", "quantification_measure"],
    #     embedding = ["pp_components", "translate_policy", "quantification_measure", "embed_algo", "embed_dimensions"],
    #     candidate_terms = ["pp_components", "translate_policy", "extraction_method"],
    #     postprocessed_candidates = ["pp_components", "translate_policy", "extraction_method"],
    #     filtered_dcm = ["pp_components", "translate_policy", "extraction_method", "dcm_quant_measure"],
    #     clusters = ["pp_components", "translate_policy", "quantification_measure", "embed_algo", "embed_dimensions", "extraction_method", "dcm_quant_measure"],
    # )

    #TODO obvs this shouldn't be a ignore_params but a requires_params, but that would mean having to change the loader
    #TODO das autoloader_di und ignore_params_di sind schon ne Mischung von Code und Daten, aber wohin sonst damit?

    def __init__(self, base_dir, dataset_name, load_params=False):
        self.obj = {"base_dir": base_dir or get_setting("BASE_DIR"),
                    "dataset": dataset_name or get_setting("DATASET_NAME"),
                    "add_relevantparams_to_filename": True,
                    "verbose": get_setting("VERBOSE"),
                    "strict_metainf_checking": get_setting("STRICT_METAINF_CHECKING")}
        self.auto_envvar_prefix = ENV_PREFIX
        init_context(self)
        if load_params:
            all_params = {i: get_setting(i.upper(), stay_silent=True, silent=True) for i in get_jsonpersister_args()[0]}
            for k, v in all_params.items():
                self.obj[k.lower()] = v


    def load(self, *whats):
        for what in whats:
            self.obj[what] = self.obj["json_persister"].load(None, what, loader=self.autoloader_di.get(what, lambda **kwargs: kwargs[what]),
                                                             ignore_params=self.ignore_params_di.get(what))


def init_context(ctx): #works for both a click-Context and my custom one
    ctx.obj["dataset_class"] = dataset_specifics.load_dataset_class(ctx.obj["dataset"])
    CustomIO.init(ctx)
    ctx.obj["json_persister"] = setup_json_persister(ctx)
    set_debug(ctx)


def print_settings():
    print("TODO")
    # all_params = {i: get_setting(i.upper(), stay_silent=True, silent=True) for i in settings.FNAME_PARAMS}
    # default_params = {k[len("DEFAULT_"):].lower():v for k,v in settings.__dict__.items() if k in ["DEFAULT_"+i.upper() for i in all_params.keys()]}
    # print("Running with the following settings:", ", ".join([f"{k}: *{'b' if v==default_params[k] else 'r'}*{v}*{'b' if v==default_params[k] else 'r'}*" for k, v in all_params.items()]))

