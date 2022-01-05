import inspect
from functools import wraps
import os
from os.path import join, dirname, basename, abspath
import random
import logging
from datetime import datetime
from time import sleep
import sys

if abspath(join(dirname(__file__), "../..")) not in sys.path:
    sys.path.append(abspath(join(dirname(__file__), "../..")))

import click

from misc_util.telegram_notifier import telegram_notify
from misc_util.logutils import setup_logging, CustomIO
from misc_util.pretty_print import pretty_print as print

from derive_conceptualspace.util.desc_object import DescriptionList
from derive_conceptualspace.settings import (
    ALL_TRANSLATE_POLICY, ALL_QUANTIFICATION_MEASURE, ALL_EXTRACTION_METHOD, ALL_DCM_QUANT_MEASURE, ALL_EMBED_ALGO, ALL_DCM_QUANT_MEASURE,
    ENV_PREFIX, NORMALIFY_PARAMS,
    get_setting, set_envvar, get_envvar,
)
from derive_conceptualspace.util.jsonloadstore import JsonPersister
from derive_conceptualspace.create_spaces.translate_descriptions import (
    full_translate_titles as translate_titles_base,
    full_translate_descriptions as translate_descriptions_base,
    # count_translations as count_translations_base
)
from derive_conceptualspace.extract_keywords.postprocess_candidates import (
    postprocess_candidateterms as postprocess_candidateterms_base,
)
from derive_conceptualspace.extract_keywords.keywords_main import (
    extract_candidateterms as extract_candidateterms_base,
    create_filtered_doc_cand_matrix as create_filtered_doc_cand_matrix_base,
)
from derive_conceptualspace.create_spaces.preprocess_descriptions import (
    preprocess_descriptions_full as preprocess_descriptions_base,
)
from derive_conceptualspace.create_spaces.spaces_main import (
    create_dissim_mat as create_dissim_mat_base,
)
from derive_conceptualspace.create_spaces.create_embedding import (
    create_embedding as create_embedding_base,
)
from derive_conceptualspace.semantic_directions.create_candidate_svm import (
    create_candidate_svms as create_candidate_svms_base
)
from derive_conceptualspace.util.dtm_object import dtm_dissimmat_loader, dtm_loader

logger = logging.getLogger(basename(__file__))
flatten = lambda l: [item for sublist in l for item in sublist]

########################################################################################################################
########################################################################################################################
########################################################################################################################
#cli helpers & main

normalify = lambda txt: "".join([i for i in txt.lower() if i.isalpha() or i in "_"])

def loadstore_settings_envvars(ctx, use_auto_envvar_prefix=False):
    """auto_envvar_prefix only works for options, not for arguments. So this function overwrites ctx.params & ctx.obj
       from env-vars (if they have the correct prefix), and also SETS these env-vars from the cmd-args such that they
       can be accessed using get_setting() """
    env_prefix = ctx.auto_envvar_prefix if use_auto_envvar_prefix else ENV_PREFIX
    #the auto_envvar_prefix always gets appended the subcommand, I don't want that generally though.
    for param, val in ctx.params.items():
        if param.upper() in NORMALIFY_PARAMS:
            val = normalify(val)
        ctx.obj[param] = val
        envvarname = env_prefix+"_"+param.upper().replace("-","_")
        # https://github.com/pallets/click/issues/714#issuecomment-651389598
        if (envvar := get_envvar(envvarname)) is not None and envvar != ctx.params[param]:
            print(f"The param {param} used to be {ctx.params[param]}, but is overwritten by an env-var to {envvar}")
            ctx.params[param] = envvar
            ctx.obj[param] = envvar
        else:
            set_envvar(envvarname, ctx.obj[param])

def get_jsonpersister_args():
    import derive_conceptualspace.settings
    all_params = [k[4:].lower() for k in derive_conceptualspace.settings.__dict__ if k.startswith("ALL_")]
    forward_meta_inf = ["n_samples", "faster_keybert", "candidate_min_term_count"]
    dir_struct = ["{n_samples}_samples", "{pp_components}_{translate_policy}","{quantification_measure}_{embed_algo}_{embed_dimensions}d", "{extraction_method}_{dcm_quant_measure}"]
               # ["{n_samples}_samples", "preproc-{pp_components}_{translate_policy}", "{quantification_measure}_{embed_dimensions}dim",]
    return all_params, forward_meta_inf, dir_struct


def setup_json_persister(ctx, ignore_nsamples=False):
    all_params, forward_meta_inf, dir_struct = get_jsonpersister_args()
    json_persister = JsonPersister(ctx.obj["base_dir"], ctx.obj["base_dir"], ctx,
                                   forward_params = all_params, forward_meta_inf = forward_meta_inf, dir_struct = dir_struct,
                                   add_relevantparams_to_filename=ctx.obj.get("add_relevantparams_to_filename", True),
                                   strict_metainf_checking=ctx.obj["strict_metainf_checking"],
                                  )
    if ignore_nsamples:
        n_samples_getter = lambda: "ANY"
        cand_ntc_getter = lambda: "ANY"
    else:
        n_samples_getter = lambda: get_setting("DEBUG_N_ITEMS", silent=True) if get_setting("DEBUG", silent=True) else 7588 #TODO don't hard-code this!
        cand_ntc_getter = lambda: get_setting("CANDIDATE_MIN_TERM_COUNT", silent=True)
    json_persister.default_metainf_getters = {"n_samples": n_samples_getter,
                                              "candidate_min_term_count": cand_ntc_getter,
                                              "prim_lambda": lambda: "ANY",
                                              "sec_lambda": lambda: "ANY",
                                              "max_ngram": lambda: get_setting("MAX_NGRAM", silent=True)}
    return json_persister


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



def click_pass_add_context(fn):
    @click.pass_context
    @wraps(fn)
    def wrapped(*args, **kwargs):
        assert isinstance(args[0], click.Context)
        for k, v in kwargs.items():
            assert k not in args[0].obj
            args[0].obj[k] = v
        ctx = args[0]
        nkw = {k:v for k,v in {**kwargs, **ctx.obj}.items() if k in set(inspect.getfullargspec(fn).args)-{"ctx", "context"}}
        loadstore_settings_envvars(ctx)
        if isinstance(ctx.command, click.Command) and not isinstance(ctx.command, click.Group): #print settings
            import derive_conceptualspace.settings
            all_params = {i: get_setting(i.upper(), stay_silent=True, silent=True) for i in get_jsonpersister_args()[0]}
            default_params = {k[len("DEFAULT_"):].lower():v for k,v in derive_conceptualspace.settings.__dict__.items() if k in ["DEFAULT_"+i.upper() for i in all_params.keys()]}
            print("Running with the following settings:", ", ".join([f"{k}: *{'b' if v==default_params[k] else 'r'}*{v}*{'b' if v==default_params[k] else 'r'}*" for k, v in all_params.items()]))
        res = fn(*args, **nkw)
        if isinstance(ctx.command, click.Group):
            if ctx.obj["notify_telegram"] == True:
                if not isinstance(cli.get_command(ctx, ctx.invoked_subcommand), click.Group):
                    ctx.command.get_command(ctx, ctx.invoked_subcommand).callback = telegram_notify(only_terminal=False, only_on_fail=False, log_start=True)(ctx.command.get_command(ctx, ctx.invoked_subcommand).callback)
        return res
    return wrapped


@click.group()
@click.argument("base-dir", type=str)
@click.option("--verbose/--no-verbose", default=True, help="default: True")
@click.option("--debug/--no-debug", default=lambda: get_setting("DEBUG"), help=f"If True, many functions will only run on a few samples, such that everything should run really quickly. Default: {get_setting('DEBUG')}")
@click.option("--log", type=str, default="INFO", help="log-level for logging-module. one of [DEBUG, INFO, WARNING, ERROR, CRITICAL]")
@click.option("--logfile", type=str, default="", help="logfile to log to. If not set, it will be logged to standard stdout/stderr")
@click.option("--notify-telegram/--no-notify-telegram", default=False, help="If you want to get telegram-notified of start & end of the command")
@click.option("--strict-metainf-checking/--strict-metainf-checking", default=lambda: get_setting("STRICT_METAINF_CHECKING"), help=f"If True, all subsequent steps of the pipeline must excplitly state which meta-info of the previous steps they demand")
@click_pass_add_context
def cli(ctx):
    CustomIO.init(ctx)
    print("Starting up at", datetime.now().strftime("%d.%m.%Y, %H:%M:%S"))
    setup_logging(ctx.obj["log"], ctx.obj["logfile"])
    set_debug(ctx)
    ctx.obj["json_persister"] = setup_json_persister(ctx)


@cli.resultcallback()
def process_result(*args, **kwargs):
    """gets executed after the actual code. Prints time for bookkeeping"""
    print("Done at", datetime.now().strftime("%d.%m.%Y, %H:%M:%S"))

########################################################################################################################
########################################################################################################################
########################################################################################################################
#prepare descriptions

# @cli.command()
# @click_pass_add_context
# def count_translations(ctx, ...):
#     return count_translations_base(ctx.obj["base_dir"], mds_basename=mds_basename, descriptions_basename=descriptions_basename)

@cli.command()
@click.option("--pp-components", type=str, default=lambda: get_setting("PP_COMPONENTS", silent=True))
@click.option("--translate-policy", type=click.Choice(ALL_TRANSLATE_POLICY, case_sensitive=False), default=lambda: get_setting("TRANSLATE_POLICY", silent=True))
@click.option("--raw-descriptions-file", type=str, default="kurse-beschreibungen.csv")
@click.option("--title-languages-file", type=str, default="title_languages.json")
@click.option("--title-translations-file", type=str, default="translated_titles.json")
@click_pass_add_context
def translate_titles(ctx, pp_components, translate_policy, raw_descriptions_file, title_languages_file, title_translations_file):
    raw_descriptions = ctx.obj["json_persister"].load(raw_descriptions_file, "raw_descriptions", ignore_params=["pp_components", "translate_policy"])
    translate_titles_base(raw_descriptions, pp_components, translate_policy, title_languages_file, title_translations_file, ctx.obj["json_persister"])
    #no need to save, that's done inside the function.


@cli.command()
@click.option("--translate-policy", type=click.Choice(ALL_TRANSLATE_POLICY, case_sensitive=False), default=lambda: get_setting("TRANSLATE_POLICY", silent=True))
@click.option("--raw-descriptions-file", type=str, default="kurse-beschreibungen.csv")
@click.option("--languages-file", type=str, default="languages.json")
@click.option("--translations-file", type=str, default="translated_descriptions.json")
@click_pass_add_context
def translate_descriptions(ctx, translate_policy, raw_descriptions_file, languages_file, translations_file):
    raw_descriptions = ctx.obj["json_persister"].load(raw_descriptions_file, "raw_descriptions", ignore_params=["pp_components", "translate_policy"])
    translate_descriptions_base(raw_descriptions, translate_policy, languages_file, translations_file, ctx.obj["json_persister"])
    #no need to save, that's done inside the function.


@cli.command()
@click.option("--pp-components", type=str, default=lambda: get_setting("PP_COMPONENTS", silent=True))
@click.option("--translate-policy", type=click.Choice(ALL_TRANSLATE_POLICY, case_sensitive=False), default=lambda: get_setting("TRANSLATE_POLICY", silent=True))
@click.option("--raw-descriptions-file", type=str, default="kurse-beschreibungen.csv")
@click.option("--languages-file", type=str, default="languages.json")
@click.option("--translations-file", type=str, default="translated_descriptions.json")
@click.option("--title-languages-file", type=str, default="title_languages.json")
@click.option("--title-translations-file", type=str, default="translated_titles.json")
@click.option("--max-ngram", type=int, default=lambda: get_setting("MAX_NGRAM"))
@click_pass_add_context
def preprocess_descriptions(ctx, raw_descriptions_file, languages_file, translations_file, title_languages_file, title_translations_file):
    raw_descriptions = ctx.obj["json_persister"].load(raw_descriptions_file, "raw_descriptions", ignore_params=["pp_components", "translate_policy"])
    languages = ctx.obj["json_persister"].load(languages_file, "languages", ignore_params=["pp_components", "translate_policy"], loader=lambda langs: langs)
    translations = ctx.obj["json_persister"].load(translations_file, "translated_descriptions", ignore_params=["pp_components", "translate_policy"], force_overwrite=True, loader=lambda **kwargs: kwargs["translations"])
    title_languages = ctx.obj["json_persister"].load(title_languages_file, "title_languages", ignore_params=["pp_components", "translate_policy"], loader=lambda title_langs: title_langs)
    title_translations = ctx.obj["json_persister"].load(title_translations_file, "translated_titles", ignore_params=["pp_components", "translate_policy"], force_overwrite=True, loader=lambda **kwargs: kwargs["title_translations"])
    descriptions, metainf = preprocess_descriptions_base(raw_descriptions, ctx.obj["pp_components"], ctx.obj["translate_policy"], languages, translations, title_languages, title_translations)
    #TODO[e] depending on translate_policy and pp_compoments, title_languages etc may allowed to be empty
    ctx.obj["json_persister"].save("pp_descriptions.json", descriptions=descriptions, relevant_metainf=metainf)


########################################################################################################################
########################################################################################################################
########################################################################################################################
#create-spaces group

@cli.group()
@click.option("--pp-components", type=str, default=lambda: get_setting("PP_COMPONENTS", silent=True))
@click.option("--translate-policy", type=click.Choice(ALL_TRANSLATE_POLICY, case_sensitive=False), default=lambda: get_setting("TRANSLATE_POLICY", silent=True))
@click.option("--quantification-measure", type=click.Choice(ALL_QUANTIFICATION_MEASURE, case_sensitive=False), default=lambda: get_setting("QUANTIFICATION_MEASURE", silent=True))
@click_pass_add_context
def create_spaces(ctx):
    """[group] CLI base to create the spaces from texts"""
    pass


@create_spaces.command()
@click_pass_add_context
def create_dissim_mat(ctx):
    pp_descriptions = ctx.obj["json_persister"].load(None, "pp_descriptions", ignore_params=["quantification_measure"], loader=DescriptionList.from_json)
    quant_dtm, dissim_mat, metainf = create_dissim_mat_base(pp_descriptions, ctx.obj["quantification_measure"], ctx.obj["verbose"])
    ctx.obj["json_persister"].save("dissim_mat.json", quant_dtm=quant_dtm, dissim_mat=dissim_mat, relevant_metainf=metainf)


@create_spaces.command()
@click.option("--embed-dimensions", type=int, default=lambda: get_setting("EMBED_DIMENSIONS", silent=True))
@click.option("--embed-algo", type=click.Choice(ALL_EMBED_ALGO, case_sensitive=False), default=lambda: get_setting("EMBED_ALGO", silent=True))
@click_pass_add_context
def create_embedding(ctx):
    dissim_mat = ctx.obj["json_persister"].load(None, "dissim_mat", ignore_params=["embed_dimensions"], loader=dtm_dissimmat_loader)
    pp_descriptions = ctx.obj["json_persister"].load(None, "pp_descriptions", ignore_params=["quantification_measure"], loader=DescriptionList.from_json, silent=True) \
                      if ctx.obj["verbose"] else None
    embedding = create_embedding_base(dissim_mat, ctx.obj["embed_dimensions"], ctx.obj["embed_algo"], ctx.obj["verbose"], pp_descriptions=pp_descriptions)
    ctx.obj["json_persister"].save("embedding.json", embedding=embedding)

########################################################################################################################
########################################################################################################################
########################################################################################################################
#prepare-candidateterms group

@cli.group()
@click.option("--pp-components", type=str, default=lambda: get_setting("PP_COMPONENTS", silent=True))
@click.option("--translate-policy", type=click.Choice(ALL_TRANSLATE_POLICY, case_sensitive=False), default=lambda: get_setting("TRANSLATE_POLICY", silent=True))
@click.option("--extraction-method", type=click.Choice(ALL_EXTRACTION_METHOD, case_sensitive=False), default=lambda: get_setting("EXTRACTION_METHOD", silent=True))
@click_pass_add_context
def prepare_candidateterms(ctx):
    """[group] CLI base to extract candidate-terms from texts"""
    ctx.obj["pp_descriptions"] = ctx.obj["json_persister"].load(None, "pp_descriptions", loader=DescriptionList.from_json)


@prepare_candidateterms.command()
@click_pass_add_context
def extract_candidateterms_stanfordlp(ctx):
    raise NotImplementedError()
    # names, descriptions, _, _ = load_mds(join(SID_DATA_BASE, f"siddata_names_descriptions_mds_20.json"))
    # nlp = download_activate_stanfordnlp(DATA_BASE, ["english", "german"])
    # print(stanford_extract_nounphrases(nlp, descriptions[1]))


@prepare_candidateterms.command()
@click.option("--faster-keybert/--no-faster-keybert", default=lambda: get_setting("FASTER_KEYBERT"))
@click.option("--max-ngram", default=lambda: get_setting("MAX_NGRAM"))
@click_pass_add_context
# @telegram_notify(only_terminal=True, only_on_fail=False, log_start=True)
def extract_candidateterms(ctx, max_ngram):
    #TODO if not NGRAMS_IN_EMBEDDING and extraction_method in tfidf/ppmi, you have to re-extract, otherwise you won't get n-grams
    candidateterms, relevant_metainf = extract_candidateterms_base(ctx.obj["pp_descriptions"], ctx.obj["extraction_method"], max_ngram, ctx.obj["faster_keybert"], verbose=ctx.obj["verbose"])
    ctx.obj["json_persister"].save("candidate_terms.json", candidateterms=candidateterms, relevant_metainf=relevant_metainf)


@prepare_candidateterms.command()
@click_pass_add_context
def postprocess_candidateterms(ctx):
    ctx.obj["candidate_terms"] = ctx.obj["json_persister"].load(None, "candidate_terms")
    postprocessed_candidates = postprocess_candidateterms_base(ctx.obj["candidate_terms"], ctx.obj["pp_descriptions"], ctx.obj["extraction_method"])
    ctx.obj["json_persister"].save("postprocessed_candidates.json", postprocessed_candidates=postprocessed_candidates)


@prepare_candidateterms.command()
@click.option("--candidate-min-term-count", type=int, default=lambda: get_setting("CANDIDATE_MIN_TERM_COUNT"))
@click.option("--dcm-quant-measure", type=click.Choice(ALL_DCM_QUANT_MEASURE, case_sensitive=False), default=lambda: get_setting("DCM_QUANT_MEASURE", silent=True))
@click.option("--use-ndocs-count/--no-use-ndocs-count", default=lambda: get_setting("CANDS_USE_NDOCS_COUNT"))
@click_pass_add_context
# @telegram_notify(only_terminal=True, only_on_fail=False, log_start=True)
def create_filtered_doc_cand_matrix(ctx, candidate_min_term_count, use_ndocs_count):
    # TODO missing options here: `tag-share` (chap. 4.2.1 of [VISR12])
    # TODO Do I need pp_descriptions except for verbosity? -> if not, make loading silent
    ctx.obj["postprocessed_candidates"] = ctx.obj["json_persister"].load(None, "postprocessed_candidates", loader = lambda **args: args["postprocessed_candidates"])
    filtered_dcm = create_filtered_doc_cand_matrix_base(ctx.obj["postprocessed_candidates"], ctx.obj["pp_descriptions"], min_term_count=candidate_min_term_count,
                                                    dcm_quant_measure=ctx.obj["dcm_quant_measure"], use_n_docs_count=use_ndocs_count, verbose=ctx.obj["verbose"])
    ctx.obj["json_persister"].save("filtered_dcm.json", relevant_metainf={"candidate_min_term_count": candidate_min_term_count}, doc_term_matrix=filtered_dcm)


########################################################################################################################
########################################################################################################################
########################################################################################################################
# generate-conceptualspace group


@cli.group()
@click.option("--pp-components", type=str, default=lambda: get_setting("PP_COMPONENTS", silent=True))
@click.option("--translate-policy", type=click.Choice(ALL_TRANSLATE_POLICY, case_sensitive=False), default=lambda: get_setting("TRANSLATE_POLICY", silent=True))
@click.option("--quantification-measure", type=click.Choice(ALL_QUANTIFICATION_MEASURE, case_sensitive=False), default=lambda: get_setting("QUANTIFICATION_MEASURE", silent=True))
@click.option("--embed-dimensions", type=int, default=lambda: get_setting("EMBED_DIMENSIONS", silent=True))
@click.option("--embed-algo", type=click.Choice(ALL_EMBED_ALGO, case_sensitive=False), default=lambda: get_setting("EMBED_ALGO", silent=True))
@click.option("--extraction-method", type=click.Choice(ALL_EXTRACTION_METHOD, case_sensitive=False), default=lambda: get_setting("EXTRACTION_METHOD", silent=True))
@click.option("--dcm-quant-measure", type=click.Choice(ALL_DCM_QUANT_MEASURE, case_sensitive=False), default=lambda: get_setting("DCM_QUANT_MEASURE", silent=True))
@click_pass_add_context
def generate_conceptualspace(ctx):
    """[group] CLI base to create the actual conceptual spaces"""
    ctx.obj["pp_descriptions"] = ctx.obj["json_persister"].load(None, "pp_descriptions", loader=DescriptionList.from_json, ignore_params=["quantification_measure", "embed_dimensions"])
    ctx.obj["filtered_dcm"] = ctx.obj["json_persister"].load(None, "filtered_dcm", loader=dtm_loader, ignore_params=["quantification_measure", "embed_dimensions"])
    ctx.obj["embedding"] = ctx.obj["json_persister"].load(None, "embedding", ignore_params=["extraction_method", "dcm_quant_measure"], loader=lambda **args: args["embedding"])
    assert ctx.obj["embedding"].embedding_.shape[0] == len(ctx.obj["filtered_dcm"].dtm), f'The Doc-Candidate-Matrix contains {len(ctx.obj["filtered_dcm"].dtm)} items But your embedding has {ctx.obj["embedding"].embedding_.shape[0] } descriptions!'


@generate_conceptualspace.command()
@click.option("--prim-lambda", type=float, default=lambda: get_setting("PRIM_LAMBDA"))
@click.option("--sec-lambda", type=float, default=lambda: get_setting("SEC_LAMBDA"))
@click_pass_add_context
def create_candidate_svm(ctx):
    clusters, cluster_directions, decision_planes, metrics = create_candidate_svms_base(ctx.obj["filtered_dcm"], ctx.obj["embedding"], ctx.obj["pp_descriptions"], verbose=ctx.obj["verbose"],
                                                                                        prim_lambda=ctx.obj["prim_lambda"], sec_lambda=ctx.obj["sec_lambda"])
    ctx.obj["json_persister"].save("clusters.json", clusters=clusters, cluster_directions=cluster_directions, decision_planes=decision_planes, metrics=metrics,
                                   relevant_metainf={"prim_lambda": ctx.obj["prim_lambda"], "sec_lambda": ctx.obj["sec_lambda"]})
    #TODO hier war dcm = DocTermMatrix(json_load(join(ctx.obj["base_dir"], dcm_filename), assert_meta=("CANDIDATE_MIN_TERM_COUNT", "STANFORDNLP_VERSION"))), krieg ich das wieder hin?

#TODO I can do something like autodetect_relevant_params und metainf im json_persister


@generate_conceptualspace.command()
@click_pass_add_context
def show_data_info(ctx):
    from graphviz import Digraph
    import git
    if get_setting("DEBUG"):
        print(f"Looking at data generated in Debug-Mode for {get_setting('DEBUG_N_ITEMS')} items!")
    ctx.obj["clusters"] = ctx.obj["json_persister"].load(None, "clusters")
    print(f"Data lies at *b*{ctx.obj['json_persister'].in_dir}*b*")
    print("Settings:", ", ".join([f"{k}: *b*{v}*b*" for k, v in ctx.obj["json_persister"].loaded_relevant_params.items()]))
    print("Relevant Metainfo:", ", ".join([f"{k}: *b*{v}*b*" for k, v in ctx.obj["json_persister"].loaded_relevant_metainf.items()]))
    data_dirs = {k: v[1].replace(ctx.obj["json_persister"].in_dir, "data_dir/") for k, v in ctx.obj["json_persister"].loaded_objects.items()}
    print("Directories:\n ", "\n  ".join(f"{k.rjust(max(len(i) for i in data_dirs))}: {v}" for k,v in data_dirs.items()))
    dependencies = {k: set([i for i in v[2] if i != "this"]) for k,v in ctx.obj["json_persister"].loaded_objects.items()}
    #figuring out when a new param was first necessary
    param_intro = {k: v[3].get("relevant_params") if v[3] else None for k, v in ctx.obj["json_persister"].loaded_objects.items()}
    newparam = {}
    for key, val in {k: list(v.keys()) for k, v in param_intro.items() if v}.items():
        for elem in val:
            if elem not in flatten(newparam.values()):
                newparam.setdefault(key, []).append(elem)
    #/figuring out when a new param was first necessary
    dot = Digraph()
    for key in dependencies:
        add_txt = "\n  ".join([f"{el}: {ctx.obj['json_persister'].loaded_relevant_params[el]}" for el in newparam.get(key, [])])
        dot.node(key, key+("\n\n  "+add_txt if add_txt else ""))
    dot.edges([[k, e] for k, v in dependencies.items() for e in v])
    # print(dot.source) #TODO save to file
    if ctx.obj["verbose"]:
        dot.render(view=True)
    commits = {k2:v2 for k2,v2 in {k: v[3]["git_hash"]["inner_commit"] if isinstance(v[3], dict) and "git_hash" in v[3] else None for k,v in ctx.obj["json_persister"].loaded_objects.items()}.items() if v2 is not None}
    if len(set(commits.values())) == 1:
        print(f"All Parts from commit {list(commits.values())[0]}")
    #ob alle vom gleichem commit, wenn ja welcher, und die letzten 2-3 commit-messages davor
    git_hist = list(git.Repo(".", search_parent_directories=True).iter_commits("main", max_count=20))
    commit_num = [ind for ind, i in enumerate(git_hist) if i.hexsha == list(commits.values())[0]][0]
    messages = [i.message.strip() for i in git_hist[commit_num:commit_num+5]]
    tmp = []
    for msg in messages:
        if msg not in tmp: tmp.append(msg)
    print("Latest commit messages:\n  ", "\n   ".join(tmp))
    dates = {k2:v2 for k2,v2 in {k: v[3]["date"] if isinstance(v[3], dict) and "date" in v[3] else None for k,v in ctx.obj["json_persister"].loaded_objects.items()}.items() if v2 is not None}
    print("Dates:\n ", "\n  ".join(f"{k.rjust(max(len(i) for i in dates))}: {v}" for k,v in dates.items()))
    output = {k: merge_streams(v[3].get("stdout", ""), v[3].get("stderr", ""), k) for k, v in ctx.obj["json_persister"].loaded_objects.items()}
    print()


def merge_streams(s1, s2, for_):
    format = sys.stdout.date_format if isinstance(sys.stdout, CustomIO) else CustomIO.DEFAULT_DATE_FORMAT
    if not s1 and not s2:
        return ""
    def make_list(val):
        res = []
        for i in val.split("\n"):
            try: res.append([datetime.strptime(i[:len(datetime.now().strftime(format))], format), i[len(datetime.now().strftime(format))+1:]])
            except ValueError: res[-1][1] += "\n"+i
        return res
    s1 = make_list(s1) if s1 else []
    s2 = make_list(s2) if s2 else []
    return "\n".join([i[1] for i in sorted(s1+s2, key=lambda x:x[0])])

@generate_conceptualspace.command()
@click_pass_add_context
def rank_courses_saldirs(ctx):
    from derive_conceptualspace.util.base_changer import NDPlane
    import numpy as np
    from itertools import combinations
    from tqdm import tqdm

    cluster_loader = lambda **di: dict(clusters=di["clusters"], cluster_directions=di["cluster_directions"],
                                       decision_planes={k: NDPlane(np.array(v[1][0]),v[1][1]) for k, v in di["decision_planes"].items()}, metrics=di["metrics"])
    ctx.obj["clusters"] = ctx.obj["json_persister"].load(None, "clusters", loader=cluster_loader)
    ctx.obj["pp_descriptions"].add_embeddings(ctx.obj["embedding"].embedding_)
    _, _, decision_planes, metrics = ctx.obj["clusters"].values()
    existinds = {k: set(v) for k, v in ctx.obj["filtered_dcm"].term_existinds(use_index=False).items()}
    for k, v in metrics.items():
        metrics[k]["existinds"] = existinds[k]
        metrics[k]["decision_plane"] = decision_planes[k]

    # TODO this is only bc in debug i set the min_existinds to 1
    metrics = {k:v for k,v in metrics.items() if len(v["existinds"]) >= 25}
    decision_planes = {k:v for k,v in decision_planes.items() if k in metrics}
    existinds = {k:v for k,v in existinds.items() if k in metrics}
    #/that

    good_candidates = dict([i for i in sorted(metrics.items(), key=lambda x:x[1]["accuracy"], reverse=True) if i[1]["accuracy"] > 0.9 and i[1]["precision"] > 0.2])
    semi_candidates = dict([i for i in sorted(metrics.items(), key=lambda x:x[1]["accuracy"], reverse=True) if i[1]["accuracy"] > 0.6 and i[1]["precision"] > 0.1 and i[1]["recall"] > 0.6 and i[0] not in good_candidates])
    print()

    #jetzt will ich: Die Candidates gruppieren, die einen hohen overlap haben in welchen Texten sie vorkommen.
    # Also, wenn "a1" und "a2" in den beschreibungen von mostly den selben Kursen vorkommen, werden sie germergt.
    #TODO: was AUCH GEHT: Statt die Liste der Kurse pro Keyword anzugucken kann ich auch die Liste der Keywords pro Kurs angucken
    # und wenn da ein (dann muss es aber extremer sein) overlap ist alle keywords von kurs1 zu kurs2 hinzufügen und vice versa,
    # das wäre im Grunde die explizite version von den latent methoden LSI etc

    # combs = list(combinations(existinds.keys(), 2))
    all_overlaps = {}
    for nkey1, (key1, inds1) in enumerate(tqdm(existinds.items(), desc="Checking all overlaps")):
        n1 = len(inds1)
        for key2, inds2 in list(existinds.items())[nkey1+1:]:
            overlap_percentages = (n12 := len(inds1 & inds2)) / n1, n12 / len(inds2), n12 / (n1+len(inds2)), n1/len(ctx.obj["pp_descriptions"]), len(inds2)/len(ctx.obj["pp_descriptions"])
            all_overlaps[(key1, key2)] = overlap_percentages
    for val in range(3):
        ar = np.array([i[val] for i in all_overlaps.values()])
        print(f"[{val}]: Mean Overlap of respective exist-indices: {ar[ar>0].mean()*100:.2f}% for those with any overlap, {ar.mean()*100:.2f}% overall")
    merge_candidates = [[k[0], k[1]] for k,v in all_overlaps.items() if v[2] > 0.45 and max(v[3], v[4]) < 0.1]


@generate_conceptualspace.command()
@click_pass_add_context
def run_lsi_gensim(ctx, filtered_dcm):
    """as in [VISR12: 4.2.1]"""
    # TODO options here:
    # * if it should filter AFTER the LSI
    from gensim import corpora
    from gensim.models import LsiModel

    if ctx.obj["verbose"]:
        filtered_dcm.show_info(descriptions=ctx.obj["pp_descriptions"])
        if get_setting("DCM_QUANT_MEASURE") != "binary":
            warnings.warn("VISR12 say it works best with binary!")

    filtered_dcm.add_pseudo_keyworddocs()
    dictionary = corpora.Dictionary([list(filtered_dcm.all_terms.values())])
    print("Start creating the LSA-Model with MORE topics than terms...")
    lsamodel_manytopics = LsiModel(doc_term_matrix, num_topics=len(all_terms) * 2, id2word=dictionary)
    print("Start creating the LSA-Model with FEWER topics than terms...")
    lsamodel_lesstopics = LsiModel(filtered_dcm.dtm, num_topics=len(filtered_dcm.all_terms)//10, id2word=dictionary)
    print()
    import matplotlib.cm; import matplotlib.pyplot as plt
    # TODO use the mpl_tools here as well to also save plot!
    plt.imshow(lsamodel_lesstopics.get_topics()[:100,:200], vmin=lsamodel_lesstopics.get_topics().min(), vmax=lsamodel_lesstopics.get_topics().max(), cmap=matplotlib.cm.get_cmap("coolwarm")); plt.show()



@generate_conceptualspace.command()
@click_pass_add_context
# @telegram_notify(only_terminal=True, only_on_fail=False, log_start=True)
def run_lsi(ctx, filtered_dcm):
    """as in [VISR12: 4.2.1]"""
    from sklearn.decomposition import TruncatedSVD
    from scipy.spatial.distance import cosine
    import numpy as np
    from tqdm import tqdm
    if ctx.obj["verbose"]:
        filtered_dcm.show_info(descriptions=ctx.obj["pp_descriptions"])
        if get_setting("DCM_QUANT_MEASURE") != "binary":
            warnings.warn("VISR12 say it works best with binary!")
    orig_len = len(filtered_dcm.dtm)
    filtered_dcm.add_pseudo_keyworddocs()
    #https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
    svd = TruncatedSVD(n_components=100, random_state=get_setting("RANDOM_SEED"))
    transformed = svd.fit_transform(filtered_dcm.as_csr().T)
    desc_psdoc_dists = np.zeros([orig_len, len(filtered_dcm.all_terms)])
    for desc in tqdm(range(orig_len)):
        for psdoc in range(orig_len, len(filtered_dcm.dtm)):
            desc_psdoc_dists[desc, psdoc-len(filtered_dcm.all_terms)] = cosine(transformed[desc], transformed[psdoc])
            if desc_psdoc_dists[desc, psdoc-len(filtered_dcm.all_terms)] < 0.1:
                print()
    good_fits = np.where(desc_psdoc_dists.min(axis=1) < 0.1)[0]
    for ndesc, keyword in zip(good_fits, np.argmin(desc_psdoc_dists[good_fits], axis=1)):
        print(f"*b*{filtered_dcm.all_terms[keyword]}*b*", ctx.obj["pp_descriptions"]._descriptions[ndesc], )
#
#
#
#
# def rank_courses_saldirs_base(descriptions, clusters):
#     assert hasattr(descriptions[0], "embedding")
#     print()



########################################################################################################################
########################################################################################################################
########################################################################################################################


if __name__ == '__main__':
    cli(auto_envvar_prefix=ENV_PREFIX, obj={})



