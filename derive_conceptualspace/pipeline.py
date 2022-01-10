from os.path import join

import numpy as np

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

normalify = lambda txt: "".join([i for i in txt.lower() if i.isalpha() or i in "_"])

def cast_config(k, v):
    if k.upper() in NORMALIFY_PARAMS:
        v = normalify(v)
    if isinstance(v, str) and v.isnumeric():
        v = int(v)
    if "DEFAULT_" + k.upper() in settings.__dict__ and isinstance(settings.__dict__["DEFAULT_" + k.upper()], bool) and v in [0, 1]:
        v = bool(v)
    return v




def loadstore_settings_envvars(ctx, use_auto_envvar_prefix=False):
    """auto_envvar_prefix only works for options, not for arguments. So this function overwrites ctx.params & ctx.obj
       from env-vars (if they have the correct prefix), and also SETS these env-vars from the cmd-args such that they
       can be accessed using get_setting() """
    env_prefix = ctx.auto_envvar_prefix if use_auto_envvar_prefix else ENV_PREFIX
    #the auto_envvar_prefix always gets appended the subcommand, I don't want that generally though.
    for param, val in ctx.params.items():
        val = cast_config(val)
        ctx.obj[param] = val
        envvarname = env_prefix+"_"+param.upper().replace("-","_")
        # https://github.com/pallets/click/issues/714#issuecomment-651389598
        if (envvar := get_envvar(envvarname)) is not None and envvar != ctx.params[param]:
            print(f"The param {param} used to be *r*{ctx.params[param]}*r*, but is overwritten by an env-var to *b*{envvar}*b*")
            ctx.params[param] = envvar
            ctx.obj[param] = envvar
        else:
            set_envvar(envvarname, ctx.obj[param])



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

def print_settings():
    all_params = {i: get_setting(i.upper(), stay_silent=True, silent=True) for i in get_jsonpersister_args()[0]}
    default_params = {k[len("DEFAULT_"):].lower():v for k,v in derive_conceptualspace.settings.__dict__.items() if k in ["DEFAULT_"+i.upper() for i in all_params.keys()]}
    print("Running with the following settings:", ", ".join([f"{k}: *{'b' if v==default_params[k] else 'r'}*{v}*{'b' if v==default_params[k] else 'r'}*" for k, v in all_params.items()]))


def get_jsonpersister_args():
    import derive_conceptualspace.settings
    all_params = [k[4:].lower() for k in derive_conceptualspace.settings.__dict__ if k.startswith("ALL_")]
    forward_meta_inf = ["n_samples", "faster_keybert", "candidate_min_term_count"]
    dir_struct = ["debug_{debug}",
                  "{pp_components}_{translate_policy}_minwords{min_words_per_desc}",
                  "{quantification_measure}_{embed_algo}_{embed_dimensions}d",
                  "{extraction_method}_{dcm_quant_measure}"]
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
        n_samples_getter = lambda: get_setting("DEBUG_N_ITEMS", silent=True) if get_setting("DEBUG", silent=True) else "full"
        cand_ntc_getter = lambda: get_setting("CANDIDATE_MIN_TERM_COUNT", silent=True)
    json_persister.default_metainf_getters = {"n_samples": n_samples_getter,
                                              "candidate_min_term_count": cand_ntc_getter}
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